// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.FindSymbols;

namespace Microsoft.ML.LibraryAnalyzer
{
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class PigstyDelegateAnalyzer : DiagnosticAnalyzer
    {
        internal const string Category = "Bring the funk";

        internal static class Funky
        {
            public const string Id = "MSML_FunkyTown";
            private const string Title = "Welcome to funky town!!";
            private const string Format = "Method {0} is super funky";
            private const string Description =
                "Welcome yet again to funky town!";

            internal static DiagnosticDescriptor Rule =
                new DiagnosticDescriptor(Id, Title, Format, Category,
                    DiagnosticSeverity.Warning, isEnabledByDefault: true, description: Description);
        }

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
            ImmutableArray.Create(Funky.Rule);

        public override void Initialize(AnalysisContext context)
        {
            //context.RegisterSyntaxNodeAction(AnalyzeMethod, SyntaxKind.InvocationExpression);
            context.RegisterSemanticModelAction(AnalyzeSymantics);
        }

        private static void AnalyzeMethod(SyntaxNodeAnalysisContext context)
        {
            var node = (InvocationExpressionSyntax)context.Node;
            var diag = Diagnostic.Create(Funky.Rule, node.GetLocation(), node.GetText());
            context.ReportDiagnostic(diag);
        }

        private static void AnalyzeSymantics(SemanticModelAnalysisContext context)
        {
            var model = context.SemanticModel;

            // Possible leads...
            // https://stackoverflow.com/questions/44243024/find-all-references-of-a-symbol-that-is-attributed-with-some-special-attribute
            // https://stackoverflow.com/questions/31861762/finding-all-references-to-a-method-with-roslyn?noredirect=1&lq=1
        }
    }
}
