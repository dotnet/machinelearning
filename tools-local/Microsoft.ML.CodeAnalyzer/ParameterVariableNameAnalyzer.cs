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
    public sealed class ParameterVariableNameAnalyzer : DiagnosticAnalyzer
    {
        private const string Category = "Naming";

        public const string Id = "MSML_ParameterLocalVarName";
        private const string Title = "Parameter or local variable name not standard";
        private const string Format = "{1} name '{0}' not standard";
        private const string Description =
            "Parameter and local variable names should be lowerCamelCased.";

        private static DiagnosticDescriptor Rule =
            new DiagnosticDescriptor(Id, Title, Format, Category,
                DiagnosticSeverity.Warning, isEnabledByDefault: true, description: Description);

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
            ImmutableArray.Create(Rule);

        public override void Initialize(AnalysisContext context)
        {
            context.RegisterSyntaxNodeAction(AnalyzeParameter, SyntaxKind.Parameter);
            context.RegisterSyntaxNodeAction(AnalyzeLocal, SyntaxKind.LocalDeclarationStatement);
        }

        private static void AnalyzeParameter(SyntaxNodeAnalysisContext context)
        {
            var node = (ParameterSyntax)context.Node;
            AnalyzeCore(context, node.Identifier, "parameter");
        }

        private static void AnalyzeLocal(SyntaxNodeAnalysisContext context)
        {
            var node = (LocalDeclarationStatementSyntax)context.Node;
            foreach (var dec in node.DescendantNodesAndSelf().Where(s => s.IsKind(SyntaxKind.VariableDeclarator)))
                AnalyzeCore(context, ((VariableDeclaratorSyntax)dec).Identifier, "local variable");
        }

        private static void AnalyzeCore(SyntaxNodeAnalysisContext context, SyntaxToken identifier, string type)
        {
            var name = identifier.Text;
            if (name == null || Utils.NameIsGood(name, 0, false))
                return;
            context.ReportDiagnostic(NameAnalyzer.CreateDiagnostic(Rule, identifier, NameType.CamelCased, type));
        }
    }
}