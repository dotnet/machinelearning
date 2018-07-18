// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Microsoft.ML.CodeAnalyzer
{
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class InterfaceNameAnalyzer : DiagnosticAnalyzer
    {
        private const string Category = "Naming";

        public const string Id = "MSML_InterfaceName";
        private const string Title = "Interface name not standard";
        private const string Format = "Interface name '{0}' not standard";
        private const string Description =
            "Interface names should start with 'I' and the remainder PascalCased.";

        private static DiagnosticDescriptor Rule =
            new DiagnosticDescriptor(Id, Title, Format, Category,
                DiagnosticSeverity.Warning, isEnabledByDefault: true, description: Description);

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
            ImmutableArray.Create(Rule);

        public override void Initialize(AnalysisContext context)
        {
            context.RegisterSyntaxNodeAction(Analyze, SyntaxKind.InterfaceDeclaration);
        }

        private static void Analyze(SyntaxNodeAnalysisContext context)
        {
            var node = (InterfaceDeclarationSyntax)context.Node;
            var identifier = node.Identifier;
            var name = identifier.Text;
            if (name == null || (name.StartsWith("I") && Utils.NameIsGood(name, 1, true)))
                return;
            context.ReportDiagnostic(NameAnalyzer.CreateDiagnostic(Rule, identifier, NameType.IPascalCased));
        }
    }
}