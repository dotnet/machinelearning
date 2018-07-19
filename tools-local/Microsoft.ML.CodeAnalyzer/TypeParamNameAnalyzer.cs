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
    public sealed class TypeParamNameAnalyzer : DiagnosticAnalyzer
    {
        private const string Category = "Naming";

        public const string Id = "MSML_TypeParamName";
        private const string Title = "Type parameter name not standard";
        private const string Format = "Type parameter name '{0}' not standard";
        private const string Description =
            "Type parameter names should start with 'T' and the remainder PascalCased.";

        private static DiagnosticDescriptor Rule =
            new DiagnosticDescriptor(Id, Title, Format, Category,
                DiagnosticSeverity.Warning, isEnabledByDefault: true, description: Description);

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
            ImmutableArray.Create(Rule);

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.RegisterSyntaxNodeAction(Analyze, SyntaxKind.TypeParameter);
        }

        private static void Analyze(SyntaxNodeAnalysisContext context)
        {
            var node = (TypeParameterSyntax)context.Node;
            var identifier = node.Identifier;
            var name = identifier.Text;
            if (name == null || (name.StartsWith("T") && Utils.NameIsGood(name, 1, true)))
                return;
            context.ReportDiagnostic(NameAnalyzer.CreateDiagnostic(Rule, identifier, NameType.TPascalCased));
        }
    }
}