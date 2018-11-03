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

        private const string Title = "Cross-assembly internal access requires.";
        private const string Format = "Access of '{0}' is a cross assembly internal " +
            "reference, and the declaring assembly wants these accesses to be on something " +
            "with the attribute " + AttributeName + ".";
        private const string Description =
            "The ML.NET .";

        private static DiagnosticDescriptor Rule =
            new DiagnosticDescriptor(DiagnosticId, Title, Format, Category,
                DiagnosticSeverity.Warning, isEnabledByDefault: true, description: Description);

        private const string AttributeName = "Microsoft.ML.BestFriendAttribute";
        private const string AssemblyAttributeName = "Microsoft.ML.WantsToBeBestFriendsAttribute";

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
            ImmutableArray.Create(Rule);

        public override void Initialize(AnalysisContext context)
        {
            context.RegisterSemanticModelAction(Analyze);
        }

        private void Analyze(SemanticModelAnalysisContext context)
        {
            var model = context.SemanticModel;
            var comp = model.Compilation;

            // Get the symbols of the key types we are analyzing. If we can't find any of them there is
            // no point in going further.
            var attrType = comp.GetTypeByMetadataName(AttributeName);
            if (attrType == null)
                return;
            var assemblyAttrType = comp.GetTypeByMetadataName(AssemblyAttributeName);
            if (assemblyAttrType == null)
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
                if (symbolAssembly == myAssembly)
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
                    assemblyWantsBestFriends = symbolAssembly.GetAttributes().Any(a => a.AttributeClass == assemblyAttrType);
                    assemblyHasAttrMap[symbolAssembly] = assemblyWantsBestFriends;
                }
                if (!assemblyWantsBestFriends)
                    continue;
                if (symbol.GetAttributes().Any(a => a.AttributeClass == attrType))
                {
                    // You're not just a friend, you're my best friend!
                    continue;
                }

                var diagnostic = Diagnostic.Create(Rule, node.GetLocation(), symbol.Name);
                context.ReportDiagnostic(diagnostic);
            }
        }
    }
}
