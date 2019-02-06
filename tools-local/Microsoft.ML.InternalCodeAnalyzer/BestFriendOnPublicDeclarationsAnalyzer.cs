// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
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
            "The " + AttributeName + " attribute is only valid on internal identifiers.";

        private static DiagnosticDescriptor Rule =
            new DiagnosticDescriptor(DiagnosticId, Title, Format, Category,
                DiagnosticSeverity.Warning, isEnabledByDefault: true, description: Description);

        private const string AttributeName = "Microsoft.ML.BestFriendAttribute";

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
            ImmutableArray.Create(Rule);

        public override void Initialize(AnalysisContext context)
        {
            context.RegisterSemanticModelAction(Analyze);
        }

        private void AnalyzeCore(SemanticModelAnalysisContext context, string attributeName)
        {
            var model = context.SemanticModel;
            var comp = model.Compilation;

            // Get the symbols of the key types we are analyzing. If we can't find it
            // there is no point in going further.
            var bestFriendAttributeType = comp.GetTypeByMetadataName(attributeName);
            if (bestFriendAttributeType == null)
                return;

            foreach (var node in model.SyntaxTree.GetRoot().DescendantNodes(n => !n.IsKind(SyntaxKind.UsingDirective)))
            {
                switch (node.Kind())
                {
                    case SyntaxKind.ClassDeclaration:
                    case SyntaxKind.StructDeclaration:
                    case SyntaxKind.InterfaceDeclaration:
                    case SyntaxKind.EnumDeclaration:
                    case SyntaxKind.MethodDeclaration:
                    case SyntaxKind.ConstructorDeclaration:
                    case SyntaxKind.PropertyDeclaration:
                    case SyntaxKind.DelegateDeclaration:
                        var declaredSymbol = model.GetDeclaredSymbol(node);
                        if (declaredSymbol == null)
                            continue;

                        VerifySymbol(context, declaredSymbol, bestFriendAttributeType);
                        break;
                    case SyntaxKind.FieldDeclaration: // field declaration is a little more complicated as it needs to be decomposed
                        foreach (var variable in ((FieldDeclarationSyntax)node).Declaration.Variables)
                        {
                            var fieldSymbol = model.GetDeclaredSymbol(variable);
                            VerifySymbol(context, fieldSymbol, bestFriendAttributeType);
                        }
                        break;
                    default:
                        continue;
                }
            }
        }

        private static void VerifySymbol(SemanticModelAnalysisContext context, ISymbol symbol,
            INamedTypeSymbol bestFriendAttributeType)
        {
            if (symbol.DeclaredAccessibility != Accessibility.Public)
                return;

            var attribute = symbol.GetAttributes().FirstOrDefault(a => a.AttributeClass == bestFriendAttributeType);
            if (attribute != null)
            {
                var diagnostic = Diagnostic.Create(Rule, attribute.ApplicationSyntaxReference.GetSyntax().GetLocation(), symbol.Name);
                context.ReportDiagnostic(diagnostic);
            }
        }

        private void Analyze(SemanticModelAnalysisContext context)
        {
            AnalyzeCore(context, "Microsoft.ML.BestFriendAttribute");
            AnalyzeCore(context, "Microsoft.ML.Internal.CpuMath.Core.BestFriendAttribute");
        }
    }
}