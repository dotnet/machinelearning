// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Microsoft.ML.CodeAnalyzer
{
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class SingleVariableDeclarationAnalyzer : DiagnosticAnalyzer
    {
        private const string Category = "Declaration";
        public const string DiagnosticId = "MSML_SingleVariableDeclaration";

        private const string Title = "Have only a single variable present per declaration";
        private const string Format = "Variables '{0}' were all part of a single declaration, and should be broken up";
        private const string Description =
            "We prefer to have one variable per declaration.";

        private static DiagnosticDescriptor Rule =
            new DiagnosticDescriptor(DiagnosticId, Title, Format, Category,
                DiagnosticSeverity.Warning, isEnabledByDefault: true, description: Description);

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
            ImmutableArray.Create(Rule);

        public override void Initialize(AnalysisContext context)
        {
            context.RegisterSyntaxNodeAction(Analyze, SyntaxKind.VariableDeclaration);
        }

        private static void Analyze(SyntaxNodeAnalysisContext context)
        {
            var node = (VariableDeclarationSyntax)context.Node;
            var vars = node.Variables;
            if (vars.Count <= 1 || node.Parent.IsKind(SyntaxKind.ForStatement))
                return;
            string jointVariableNames = string.Join("', '", vars.Select(v => v.Identifier.Text));
            var diagnostic = Diagnostic.Create(Rule, context.Node.GetLocation(), jointVariableNames);
            context.ReportDiagnostic(diagnostic);
        }
    }
}