// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Microsoft.ML.InternalCodeAnalyzer
{
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class BestFriendAnalyzer : DiagnosticAnalyzer
    {
        private const string Category = "Access";
        internal const string DiagnosticId = "MSML_NoBestFriendInternal";

        private const string Title = "Cross-assembly internal access requires referenced item to have " + AttributeName + " attribute.";
        private const string Format = "Access of '{0}' is a cross assembly internal " +
            "reference, and the declaring assembly wants these accesses to be on something " +
            "with the attribute " + AttributeName + ".";
        private const string Description =
            "The identifier indicated is defined as an internal member of an assembly that has the " +
            AssemblyAttributeName + " assembly-level attribute set. Even with friend access to that " +
            "assembly, such a usage requires that the item have the " + AttributeName + " on it.";

        private static readonly DiagnosticDescriptor Rule =
            new DiagnosticDescriptor(DiagnosticId, Title, Format, Category,
                DiagnosticSeverity.Warning, isEnabledByDefault: true, description: Description);

        private const string AttributeName = "Microsoft.ML.BestFriendAttribute";
        private const string AssemblyAttributeName = "Microsoft.ML.WantsToBeBestFriendsAttribute";

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
            ImmutableArray.Create(Rule);

        public override void Initialize(AnalysisContext context)
        {
            // This analyzer reports violations in all code (including generated code)
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.Analyze | GeneratedCodeAnalysisFlags.ReportDiagnostics);
            context.EnableConcurrentExecution();

            context.RegisterSemanticModelAction(Analyze);
        }

        private void AnalyzeCore(SemanticModelAnalysisContext context, string attributeName, string assemblyAttributeName)
        {
            var model = context.SemanticModel;
            var comp = model.Compilation;

            // Get the symbols of the key types we are analyzing. If we can't find either
            // of them there is no point in going further.
            var bestFriendAttributeType = comp.GetTypeByMetadataName(attributeName);
            if (bestFriendAttributeType == null)
                return;
            var wantsToBeBestFriendsAttributeType = comp.GetTypeByMetadataName(assemblyAttributeName);
            if (wantsToBeBestFriendsAttributeType == null)
                return;

            var myAssembly = comp.Assembly;
            var assemblyHasAttrMap = new Dictionary<IAssemblySymbol, bool>();

            int count = 0;
            foreach (var node in model.SyntaxTree.GetRoot().DescendantNodes(n => !n.IsKind(SyntaxKind.UsingDirective)))
            {
                count++;
                switch (node.Kind())
                {
                    case SyntaxKind.BaseConstructorInitializer:
                    case SyntaxKind.IdentifierName:
                        break;
                    default:
                        continue;
                }
                var symbol = model.GetSymbolInfo(node).Symbol;
                if (symbol == null)
                    continue;
                var symbolAssembly = symbol.ContainingAssembly;
                if (Equals(symbolAssembly, myAssembly))
                    continue;
                switch (symbol.DeclaredAccessibility)
                {
                    case Accessibility.Friend:
                    case Accessibility.ProtectedAndFriend:
                        break;
                    default:
                        continue;
                }
                // We now know that this is a friend reference in a different assembly. Check whether
                // this is an assembly we've determined that wants best friends, before continuing
                // further.
                if (!assemblyHasAttrMap.TryGetValue(symbolAssembly, out bool assemblyWantsBestFriends))
                {
                    // It's the first of seeing the assembly containing symbol. A key-value pair is added into assemblyHasAttrMap to
                    // indicate if that assembly includes an attribute WantsToBeBestFriends. If an assembly has WantsToBeBestFriends then
                    // its associated value would be true.
                    assemblyWantsBestFriends = symbolAssembly.GetAttributes().Any(a => Equals(a.AttributeClass, wantsToBeBestFriendsAttributeType));
                    assemblyHasAttrMap[symbolAssembly] = assemblyWantsBestFriends;
                }
                if (!assemblyWantsBestFriends)
                    continue;
                if (symbol.GetAttributes().Any(a => Equals(a.AttributeClass, bestFriendAttributeType)))
                {
                    // You're not just a friend, you're my best friend!
                    continue;
                }

                var diagnostic = Diagnostic.Create(Rule, node.GetLocation(), symbol.Name);
                context.ReportDiagnostic(diagnostic);
            }
        }

        private void Analyze(SemanticModelAnalysisContext context)
        {
            AnalyzeCore(context, "Microsoft.ML.BestFriendAttribute", "Microsoft.ML.WantsToBeBestFriendsAttribute");
            AnalyzeCore(context, "Microsoft.ML.Internal.CpuMath.Core.BestFriendAttribute", "Microsoft.ML.Internal.CpuMath.Core.WantsToBeBestFriendsAttribute");
        }
    }
}
