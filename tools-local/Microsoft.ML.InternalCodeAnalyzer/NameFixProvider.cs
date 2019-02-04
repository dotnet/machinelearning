// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Composition;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CodeActions;
using Microsoft.CodeAnalysis.CodeFixes;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Rename;

namespace Microsoft.ML.InternalCodeAnalyzer
{
    // This is somewhat difficult. The trouble is, if a name is in a bad state, it is
    // actually rather difficult to come up with a general procedure to "fix" it. We
    // instead focus on the common case where a name is right according to *some* sort
    // of regular scheme, and focus on that.

    using Debug = System.Diagnostics.Debug;

    [ExportCodeFixProvider(LanguageNames.CSharp, Name = nameof(NameFixProvider)), Shared]
    public sealed class NameFixProvider : CodeFixProvider
    {
        private const string PrivateTitle = "Fix name";

        private static ImmutableArray<string> _fixable = ImmutableArray.Create(
            NameAnalyzer.PrivateFieldName.Id, NameAnalyzer.GeneralName.Id,
            ParameterVariableNameAnalyzer.Id, TypeParamNameAnalyzer.Id);
        private static ImmutableHashSet<string> _fixableSet = ImmutableHashSet<string>.Empty.Union(_fixable);

        private static Regex _sections = new Regex(
            @"(?:\p{Nd}\p{Ll}*)|" + // Numbers we consider a separate token.
            @"(?:\p{Lu}+(?!\p{Ll}))|" + // Completely upper case sections.
            @"(?:\p{Lu}\p{Ll}+)|" + // Title cased word.
            @"(?:\p{Ll}+)"); // Lower case word.

        public override ImmutableArray<string> FixableDiagnosticIds => _fixable;

        public override FixAllProvider GetFixAllProvider()
            => WellKnownFixAllProviders.BatchFixer;

        public override async Task RegisterCodeFixesAsync(CodeFixContext context)
        {
            var diagnostic = context.Diagnostics.FirstOrDefault(d => _fixableSet.Contains(d.Id));
            if (diagnostic == null)
                return;

            string originalName = diagnostic.Properties[NameAnalyzer.NameProperty];
            string desiredNameStr = diagnostic.Properties[NameAnalyzer.DesiredNameProperty];
            NameType desiredName;
            if (!Enum.TryParse(desiredNameStr, out desiredName))
                return;

            var root = await context.Document.GetSyntaxRootAsync(context.CancellationToken);
            var token = root.FindToken(diagnostic.Location.SourceSpan.Start);
            if (token.Text != originalName)
                return;

            string docName = context.Document.Name;
            if (docName.Length - 3 == originalName.Length && docName.EndsWith(".cs", StringComparison.OrdinalIgnoreCase)
                && context.Document.Name.StartsWith(originalName, StringComparison.OrdinalIgnoreCase))
            {
                // So this is an entity like "FooBarBiz" in a file named "FooBarBiz.cs".
                // We can continue to warn on these, but registering a *fix* for them would
                // be inappropriate, since while the Roslyn API allows us to rename items like,
                // these, we cannot change the file name.
                return;
            }

            Func<string, string> renamer = null;
            switch (desiredName)
            {
                case NameType.UnderScoreCamelCased:
                    renamer = RenameUnderscore;
                    break;
                case NameType.CamelCased:
                    renamer = RenameCamelCase;
                    break;
                case NameType.PascalCased:
                    renamer = RenamePascal;
                    break;
                case NameType.IPascalCased:
                    renamer = RenameInterface;
                    break;
                case NameType.TPascalCased:
                    renamer = RenameTypeParam;
                    break;
                default:
                    Debug.Assert(!Enum.IsDefined(typeof(NameType), desiredName));
                    break;
            }

            context.RegisterCodeFix(CodeAction.Create(PrivateTitle,
                c => RenameAsync(context.Document, token.Parent, originalName, renamer, c), diagnostic.Id), diagnostic);
        }

        private async Task<Solution> RenameAsync(Document document,
            SyntaxNode identifier, string name, Func<string, string> renamer, CancellationToken cancellationToken)
        {
            // Get the symbol representing the type to be renamed.
            var semanticModel = await document.GetSemanticModelAsync(cancellationToken);
            ISymbol typeSymbol = semanticModel.GetDeclaredSymbol(identifier, cancellationToken);

            string newName = renamer(name);

            // Produce a new solution that has all references to that type renamed, including the declaration.
            var originalSolution = document.Project.Solution;
            var optionSet = originalSolution.Workspace.Options;
            var newSolution = await Renamer.RenameSymbolAsync(document.Project.Solution, typeSymbol, newName, optionSet, cancellationToken).ConfigureAwait(false);

            // Return the new solution with the now-uppercase type name.
            return newSolution;
        }

        private IEnumerable<string> ExtractSections(string name)
        {
            foreach (Match match in _sections.Matches(name))
                yield return match.Value;
        }

        private string RenameUnderscore(string name) => RenameCamelCore(name, "_");
        private string RenameCamelCase(string name) => RenameCamelCore(name, "");
        private string RenameTypeParam(string name) => RenamePascalPrefixCore(name, "T");
        private string RenameInterface(string name) => RenamePascalPrefixCore(name, "I");
        private string RenamePascal(string name) => RenamePascalPrefixCore(name, "");

        private string RenameCamelCore(string name, string prefix)
        {
            if (string.IsNullOrEmpty(name))
                return prefix;
            StringBuilder sb = new StringBuilder(prefix);
            foreach (var section in ExtractSections(name))
            {
                if (sb.Length == prefix.Length)
                    sb.Append(section.ToLowerInvariant());
                else
                    AppendTitleCase(sb, section);
            }
            return sb.ToString();
        }

        private string RenamePascalPrefixCore(string name, string prefix)
        {
            if (string.IsNullOrEmpty(name))
                return prefix;
            StringBuilder sb = new StringBuilder(prefix);
            bool first = true;
            foreach (var section in ExtractSections(name))
            {
                if (first)
                {
                    first = false;
                    if (prefix == section)
                        continue;
                }
                AppendTitleCase(sb, section);
            }
            return sb.ToString();
        }

        private void AppendTitleCase(StringBuilder builder, string token)
        {
            if (string.IsNullOrEmpty(token))
                return;
            if (token.Length == 2 && char.IsUpper(token[0]) && char.IsUpper(token[1]))
            {
                builder.Append(token);
                return;
            }
            // Further special casing for things like: IO, UI?
            builder.Append(char.ToUpperInvariant(token[0]));
            builder.Append(token.Substring(1).ToLowerInvariant());
        }
    }
}
