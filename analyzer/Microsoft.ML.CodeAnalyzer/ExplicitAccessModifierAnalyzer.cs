// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CSharp.RuntimeBinder;

namespace Microsoft.ML.CodeAnalyzer
{
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class ExplicitAccessModifierAnalyzer : DiagnosticAnalyzer
    {
        private const string Category = "Declaration";
        public const string DiagnosticId = "MSML_ExplicitAccessModifiers";

        private const string Title = "Must have explicit access modifiers, as the first modifiers";
        private const string Format = "Symbol '{0}' did not have access modifiers as the leading modifiers";
        private const string Description =
            "All symbols should have an explicit access modifier, and the access modifiers should " +
            "come before all other modifiers.";

        private static DiagnosticDescriptor Rule =
            new DiagnosticDescriptor(DiagnosticId, Title, Format, Category,
                DiagnosticSeverity.Warning, isEnabledByDefault: true, description: Description);

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
            ImmutableArray.Create(Rule);

        public override void Initialize(AnalysisContext context)
        {
            // We omit get/set accessors, variable declarations, etc.
            context.RegisterSyntaxNodeAction(Analyze, SyntaxKind.FieldDeclaration, SyntaxKind.PropertyDeclaration,
                SyntaxKind.ClassDeclaration, SyntaxKind.InterfaceDeclaration, SyntaxKind.StructDeclaration,
                SyntaxKind.DelegateDeclaration, SyntaxKind.EnumDeclaration,
                SyntaxKind.ConstructorDeclaration, SyntaxKind.MethodDeclaration, SyntaxKind.IndexerDeclaration,
                SyntaxKind.OperatorDeclaration);
        }

        internal static bool IsAccessorMod(SyntaxToken token)
        {
            return token.IsKind(SyntaxKind.PrivateKeyword) || token.IsKind(SyntaxKind.InternalKeyword) ||
                token.IsKind(SyntaxKind.ProtectedKeyword) || token.IsKind(SyntaxKind.PublicKeyword);
        }

        private static void Analyze(SyntaxNodeAnalysisContext context)
        {
            // So it's come to this... I have to think there's a better way to get modifiers,
            // no matter what they're on? Ideally these would have implemented some sort of
            // common interface, like `IHaveModifiers` or something, but, they don't.
            var node = context.Node;
            // If the parent is an interface declaration, these can't have explicit access modifiers.
            if (node.Parent.Kind() == SyntaxKind.InterfaceDeclaration)
                return;

            dynamic dynNode = context.Node;
            try
            {
                ExplicitInterfaceSpecifierSyntax expSyntax = dynNode.ExplicitInterfaceSpecifier;
                if (expSyntax != null)
                    return;
            }
            catch (RuntimeBinderException) { }

            SyntaxTokenList mods = dynNode.Modifiers;
            bool anyAccessor = false;
            bool anyAccessorAfterNon = false;
            bool isStatic = false;

            // You can have multiple accessor declarations. Make sure they appear before any other modifiers.
            if (mods.Count > 0)
            {
                anyAccessor = IsAccessorMod(mods[0]);
                isStatic = mods[0].IsKind(SyntaxKind.StaticKeyword);
                for (int i = 1; i < mods.Count; ++i)
                {
                    var mod = mods[i];
                    if (IsAccessorMod(mod))
                    {
                        anyAccessor = true;
                        anyAccessorAfterNon |= !IsAccessorMod(mods[i - 1]);
                    }
                    isStatic |= mod.IsKind(SyntaxKind.StaticKeyword);
                }
            }
            if (anyAccessor && !anyAccessorAfterNon)
                return;
            if (isStatic && node.IsKind(SyntaxKind.ConstructorDeclaration))
                return; // Static constructors can't have fields.

            // Great, now we have to find the identifier.

            SyntaxToken idToken;
            switch (node.Kind())
            {
                case SyntaxKind.FieldDeclaration:
                    idToken = ((FieldDeclarationSyntax)node).Declaration.Variables.FirstOrDefault().Identifier;
                    break;
                case SyntaxKind.OperatorDeclaration:
                    idToken = ((OperatorDeclarationSyntax)node).OperatorToken;
                    break;
                default:
                    idToken = dynNode.Identifier;
                    break;
            }

            var diagnostic = Diagnostic.Create(Rule, idToken.GetLocation(), idToken.Text);
            context.ReportDiagnostic(diagnostic);
        }
    }
}
