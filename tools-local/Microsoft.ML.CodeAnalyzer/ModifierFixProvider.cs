// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Immutable;
using System.Collections.Generic;
using System.Composition;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CodeActions;
using Microsoft.CodeAnalysis.CodeFixes;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Microsoft.ML.CodeAnalyzer
{
    [ExportCodeFixProvider(LanguageNames.CSharp, Name = nameof(ModifierFixProvider)), Shared]
    public sealed class ModifierFixProvider : CodeFixProvider
    {
        private const string Title = "Have access modifiers , put new first";
        private static string Id => ModifierAnalyzer.AccessModifierDiagnostic.Id;
        private static string NewId => ModifierAnalyzer.NewModifierDiagnostic.Id;

        public override ImmutableArray<string> FixableDiagnosticIds => ImmutableArray.Create(Id, NewId);

        public override FixAllProvider GetFixAllProvider()
            => WellKnownFixAllProviders.BatchFixer;

        /// <summary>
        /// Nested types are private by default, at the namespace level types are internal.
        /// </summary>
        private SyntaxKind IsNested(SyntaxNode node)
        {
            while (node.Parent != null)
            {
                node = node.Parent;
                if (node.IsKind(SyntaxKind.ClassDeclaration) || node.IsKind(SyntaxKind.StructDeclaration))
                    return SyntaxKind.PrivateKeyword;
            }
            return SyntaxKind.InternalKeyword;
        }

        private static void RegisterFix<T>(CodeFixContext context, Diagnostic diag, SyntaxTokenList mods,
            Func<SyntaxTokenList, T> withModifier, SyntaxKind def, T node)
            where T : SyntaxNode
        {
            context.RegisterCodeFix(CodeAction.Create(Title,
                c => Fix(context.Document, mods, withModifier, def, node, c), Id), diag);
            context.RegisterCodeFix(CodeAction.Create(Title,
                c => Fix(context.Document, mods, withModifier, def, node, c), NewId), diag);
        }

        private static IEnumerable<SyntaxToken> WithAccessorFirst(SyntaxTokenList mods, SyntaxKind def)
        {
            bool anyAccessor = false;
            var defTok = SyntaxFactory.Token(def);

            foreach (var mod in mods)
            {
                if (mod.IsKind(SyntaxKind.NewKeyword))
                    yield return mod.WithTriviaFrom(defTok);
            }

            foreach (var mod in mods)
            {
                if (ModifierAnalyzer.IsAccessorMod(mod))
                {
                    anyAccessor = true;
                    yield return mod.WithTriviaFrom(defTok);
                }
            }
            if (!anyAccessor)
                yield return SyntaxFactory.Token(def);
            foreach (var mod in mods)
            {
                if (!ModifierAnalyzer.IsAccessorMod(mod) && !mod.IsKind(SyntaxKind.NewKeyword))
                    yield return mod.WithTriviaFrom(defTok);
            }
        }

        private static async Task<Document> Fix<T>(Document document, SyntaxTokenList mods,
            Func<SyntaxTokenList, T> withModifier, SyntaxKind def, T node, CancellationToken cancellationToken)
            where T : SyntaxNode
        {
            var newMods = SyntaxFactory.TokenList(WithAccessorFirst(mods, def));

            var head = newMods[0];
            var tree = await document.GetSyntaxTreeAsync(cancellationToken);
            var root = await tree.GetRootAsync(cancellationToken);
            SyntaxNode newRoot;

            if (mods.Count > 0)
            {
                newMods = newMods.Replace(head, head.WithTriviaFrom(mods[0]));
                var tail = newMods[newMods.Count - 1];
                newMods = newMods.Replace(tail, tail.WithTrailingTrivia(mods[mods.Count - 1].TrailingTrivia));
                newRoot = root.ReplaceNode(node, withModifier(newMods));
            }
            else if (node.HasLeadingTrivia)
            {
                newMods = newMods.Replace(head, head.WithLeadingTrivia(node.GetLeadingTrivia()));
                T moddedNode = node.WithoutLeadingTrivia();
                moddedNode = (T)withModifier.GetMethodInfo().Invoke(moddedNode, new object[] { newMods });
                newRoot = root.ReplaceNode(node, moddedNode);
            }
            else
                newRoot = root.ReplaceNode(node, withModifier(newMods));

            return document.WithSyntaxRoot(newRoot);
        }

        public override async Task RegisterCodeFixesAsync(CodeFixContext context)
        {
            const SyntaxKind privateAc = SyntaxKind.PrivateKeyword;
            var diagnostic = context.Diagnostics.FirstOrDefault(d => d.Id == Id || d.Id == NewId);
            if (diagnostic == null)
                return;
            var root = await context.Document.GetSyntaxRootAsync(context.CancellationToken).ConfigureAwait(false);
            var diagnosticSpan = diagnostic.Location.SourceSpan;
            SyntaxToken identifier = root.FindToken(diagnosticSpan.Start);
            SyntaxNode parent;
            if (!identifier.IsKind(SyntaxKind.IdentifierToken))
            {
                // Maybe it's an operator?
                parent = identifier.Parent.FirstAncestorOrSelf<OperatorDeclarationSyntax>();
                if (parent == null)
                    return;
            }
            else
            {
                parent = identifier.Parent;
                if (parent.IsKind(SyntaxKind.VariableDeclarator))
                    parent = parent.FirstAncestorOrSelf<FieldDeclarationSyntax>();
            }

            // I don't see a Roslyn convenience to find the default access modifier?
            switch (parent.Kind())
            {
                case SyntaxKind.FieldDeclaration:
                    var field = (FieldDeclarationSyntax)parent;
                    RegisterFix(context, diagnostic, field.Modifiers, field.WithModifiers, privateAc, field);
                    break;
                case SyntaxKind.PropertyDeclaration:
                    var prop = (PropertyDeclarationSyntax)parent;
                    RegisterFix(context, diagnostic, prop.Modifiers, prop.WithModifiers, privateAc, prop);
                    break;
                case SyntaxKind.ClassDeclaration:
                    var cls = (ClassDeclarationSyntax)parent;
                    RegisterFix(context, diagnostic, cls.Modifiers, cls.WithModifiers, IsNested(parent), cls);
                    break;
                case SyntaxKind.InterfaceDeclaration:
                    var inter = (InterfaceDeclarationSyntax)parent;
                    RegisterFix(context, diagnostic, inter.Modifiers, inter.WithModifiers, IsNested(parent), inter);
                    break;
                case SyntaxKind.StructDeclaration:
                    var strct = (StructDeclarationSyntax)parent;
                    RegisterFix(context, diagnostic, strct.Modifiers, strct.WithModifiers, IsNested(parent), strct);
                    break;
                case SyntaxKind.DelegateDeclaration:
                    var del = (DelegateDeclarationSyntax)parent;
                    RegisterFix(context, diagnostic, del.Modifiers, del.WithModifiers, IsNested(parent), del);
                    break;
                case SyntaxKind.EnumDeclaration:
                    var enm = (EnumDeclarationSyntax)parent;
                    RegisterFix(context, diagnostic, enm.Modifiers, enm.WithModifiers, IsNested(parent), enm);
                    break;
                case SyntaxKind.ConstructorDeclaration:
                    var cons = (ConstructorDeclarationSyntax)parent;
                    RegisterFix(context, diagnostic, cons.Modifiers, cons.WithModifiers, privateAc, cons);
                    break;
                case SyntaxKind.MethodDeclaration:
                    var meth = (MethodDeclarationSyntax)parent;
                    RegisterFix(context, diagnostic, meth.Modifiers, meth.WithModifiers, privateAc, meth);
                    break;
                case SyntaxKind.IndexerDeclaration:
                    var ind = (IndexerDeclarationSyntax)parent;
                    RegisterFix(context, diagnostic, ind.Modifiers, ind.WithModifiers, privateAc, ind);
                    break;
                case SyntaxKind.OperatorDeclaration:
                    var op = (OperatorDeclarationSyntax)parent;
                    RegisterFix(context, diagnostic, op.Modifiers, op.WithModifiers, privateAc, op);
                    break;
                default:
                    // Possible?
                    return;
            }
        }
    }
}