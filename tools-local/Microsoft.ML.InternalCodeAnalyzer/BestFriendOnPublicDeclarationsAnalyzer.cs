// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Microsoft.ML.InternalCodeAnalyzer
{
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class BestFriendOnPublicDeclarationsAnalyzer : DiagnosticAnalyzer
    {
        private const string Category = "Access";
        internal const string DiagnosticId = "MSML_BestFriendOnPublicDeclaration";

        private const string Title = "Public declarations should not have " + AttributeName + " attribute.";
        private const string Format = "The " + AttributeName + " should not be applied to publicly visible members.";

        private const string Description =
            "The " + AttributeName + " attribute is not valid on public identifiers.";

        private static readonly DiagnosticDescriptor Rule =
            new DiagnosticDescriptor(DiagnosticId, Title, Format, Category,
                DiagnosticSeverity.Warning, isEnabledByDefault: true, description: Description);

        private const string AttributeName = "Microsoft.ML.BestFriendAttribute";

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
            ImmutableArray.Create(Rule);

        public override void Initialize(AnalysisContext context)
        {
            context.EnableConcurrentExecution();
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);

            context.RegisterCompilationStartAction(CompilationStart);
        }

        private void CompilationStart(CompilationStartAnalysisContext context)
        {
            var list = new List<string> { AttributeName, "Microsoft.ML.Internal.CpuMath.Core.BestFriendAttribute" };

            foreach (var attributeName in list)
            {
                var attribute = context.Compilation.GetTypeByMetadataName(attributeName);

                if (attribute == null)
                    continue;

                context.RegisterSymbolAction(c => AnalyzeCore(c, attribute), SymbolKind.NamedType, SymbolKind.Method, SymbolKind.Field, SymbolKind.Property);
            }
        }

        private void AnalyzeCore(SymbolAnalysisContext context, INamedTypeSymbol attributeType)
        {
            if (context.Symbol.DeclaredAccessibility != Accessibility.Public)
                return;

            var attribute = context.Symbol.GetAttributes().FirstOrDefault(a => Equals(a.AttributeClass, attributeType));
            if (attribute == null)
                return;

            var diagnostic = Diagnostic.Create(Rule, attribute.ApplicationSyntaxReference.GetSyntax().GetLocation(), context.Symbol.Name);
            context.ReportDiagnostic(diagnostic);
        }
    }
}
