// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Concurrent;
using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Microsoft.ML.InternalCodeAnalyzer
{
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class RelaxTestNamingSuppressor : DiagnosticSuppressor
    {
        private const string Id = "MSML_RelaxTestNaming";
        private const string SuppressedDiagnosticId = "VSTHRD200";
        private const string Justification = "Asynchronous test methods do not require the 'Async' suffix.";

        internal static readonly SuppressionDescriptor Rule =
            new SuppressionDescriptor(Id, SuppressedDiagnosticId, Justification);

        public override ImmutableArray<SuppressionDescriptor> SupportedSuppressions { get; } = ImmutableArray.Create(Rule);

        public override void ReportSuppressions(SuppressionAnalysisContext context)
        {
            if (!(context.Compilation.GetTypeByMetadataName("Xunit.FactAttribute") is { } factAttribute))
            {
                return;
            }

            var knownTestAttributes = new ConcurrentDictionary<INamedTypeSymbol, bool>();

            foreach (var diagnostic in context.ReportedDiagnostics)
            {
                // The diagnostic is reported on the test method
                if (!(diagnostic.Location.SourceTree is { } tree))
                {
                    continue;
                }

                var root = tree.GetRoot(context.CancellationToken);
                var node = root.FindNode(diagnostic.Location.SourceSpan, getInnermostNodeForTie: true);

                var semanticModel = context.GetSemanticModel(tree);
                var declaredSymbol = semanticModel.GetDeclaredSymbol(node, context.CancellationToken);
                if (declaredSymbol is IMethodSymbol method
                    && method.IsTestMethod(knownTestAttributes, factAttribute))
                {
                    context.ReportSuppression(Suppression.Create(Rule, diagnostic));
                }
            }
        }
    }
}
