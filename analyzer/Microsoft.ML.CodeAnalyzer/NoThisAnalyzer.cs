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
    public sealed class NoThisAnalyzer : DiagnosticAnalyzer
    {
        private const string Category = "Language";
        public const string DiagnosticId = "MSML_NoThis";

        private const string Title = "Do not use 'this' keyword for member access";
        private const string Format = "Do not use 'this' keyword for member access";
        private const string Description =
            "While 'this' is acceptable for, say, passing this object somewhere else, or " +
            "invoking another constructor, one should refer to members directly without using 'this'. " +
            "An exception is if the member being addressed is written in camelCase.";

        private static DiagnosticDescriptor Rule =
            new DiagnosticDescriptor(DiagnosticId, Title, Format, Category,
                DiagnosticSeverity.Warning, isEnabledByDefault: true, description: Description);

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
            ImmutableArray.Create(Rule);

        public override void Initialize(AnalysisContext context)
        {
            context.RegisterSyntaxNodeAction(Analyze, SyntaxKind.ThisExpression);
        }

        private static void Analyze(SyntaxNodeAnalysisContext context)
        {
            // We want to allow passing "this" object.
            var node = context.Node;
            var parent = node.Parent as MemberAccessExpressionSyntax;
            if (parent == null)
                return;
            // The rationale behind disallowing "this" is that if member names are always
            // _camelCased or PascalCased, there is no potential for confusion with camelCased
            // parameter or variable names. However, in the case where the member name is
            // camelCased due to special exceptions, we allow "this".
            string name = parent.Name.ToString();
            if (!string.IsNullOrEmpty(name) && char.IsLower(name[0]))
                return;

            var diagnostic = Diagnostic.Create(Rule, node.GetLocation());
            context.ReportDiagnostic(diagnostic);
        }
    }
}
