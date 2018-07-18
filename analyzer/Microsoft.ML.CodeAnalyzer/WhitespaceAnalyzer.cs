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
    public sealed class WhitespaceAnalyzer : DiagnosticAnalyzer
    {
        private const string Category = "Whitespace";

        public const string MultiLineDiagnosticId = "MSML_NoMultiNewLines";
        private static DiagnosticDescriptor _multilineRule = new DiagnosticDescriptor(MultiLineDiagnosticId,
            "Multiple successive newlines detected", "Multiple successive newlines detected", Category,
            DiagnosticSeverity.Warning, isEnabledByDefault: true, description: "Do not have multiple blank lines in succession.");

        public const string TrailingSpaceDiagnosticId = "MSML_NoTailingSpace";
        private static DiagnosticDescriptor _trailingSpaceRule = new DiagnosticDescriptor(TrailingSpaceDiagnosticId,
            "Trailing whitespace detected", "Trailing whitespace detected", Category,
            DiagnosticSeverity.Warning, isEnabledByDefault: true, description: "Do not end your lines with trailing whitespace.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
            ImmutableArray.Create(_multilineRule, _trailingSpaceRule);

        public override void Initialize(AnalysisContext context)
        {
            context.RegisterSemanticModelAction(Analyze);
        }

        private static void Analyze(SemanticModelAnalysisContext context)
        {
            var model = context.SemanticModel;
            var tree = model.SyntaxTree;
            if (!tree.HasCompilationUnitRoot)
                return;
            var root = (CompilationUnitSyntax)tree.GetRoot(context.CancellationToken);

            int linesSinceContent = 0;
            SyntaxTrivia? lastTrivia = null;

            var allTrivia = root.DescendantTrivia().ToArray();

            foreach (var trivia in root.DescendantTrivia())
            {
                bool continuous = lastTrivia?.FullSpan.End == trivia.FullSpan.Start;
                if (!continuous)
                    linesSinceContent = 0;
                switch (trivia.Kind())
                {
                    case SyntaxKind.WhitespaceTrivia:
                        break;
                    case SyntaxKind.EndOfLineTrivia:
                        if (continuous && lastTrivia.Value.IsKind(SyntaxKind.WhitespaceTrivia))
                            context.ReportDiagnostic(Diagnostic.Create(_trailingSpaceRule, lastTrivia.Value.GetLocation()));
                        if (++linesSinceContent > 2)
                            context.ReportDiagnostic(Diagnostic.Create(_multilineRule, trivia.GetLocation()));
                        break;
                    default:
                        continue;
                }
                lastTrivia = trivia;
            }
        }
    }
}
