// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Immutable;
using System.Composition;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CodeActions;
using Microsoft.CodeAnalysis.CodeFixes;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Microsoft.ML.CodeAnalyzer
{
    using Debug = System.Diagnostics.Debug;

    [ExportCodeFixProvider(LanguageNames.CSharp, Name = nameof(ContractsCheckNameofFixProvider)), Shared]
    public sealed class ContractsCheckNameofFixProvider : CodeFixProvider
    {
        private const string Title = "Try to introduce nameof";
        /// <summary>
        /// Id of the diagnostic, and equivalence id of the fix.
        /// </summary>
        private string Id => ContractsCheckAnalyzer.NameofDiagnostic.Id;

        public override ImmutableArray<string> FixableDiagnosticIds => ImmutableArray.Create(Id);

        public override FixAllProvider GetFixAllProvider()
            => WellKnownFixAllProviders.BatchFixer;

        public override async Task RegisterCodeFixesAsync(CodeFixContext context)
        {
            var diagnostic = context.Diagnostics.FirstOrDefault(d => d.Id == Id);
            if (diagnostic == null)
                return;
            var root = await context.Document.GetSyntaxRootAsync(context.CancellationToken).ConfigureAwait(false);

            var diagnosticSpan = diagnostic.Location.SourceSpan;

            // Find the name/paramName argument identified by the diagnostic.
            var nameArg = root.FindToken(diagnosticSpan.Start).Parent.FirstAncestorOrSelf<ArgumentSyntax>();
            string nameArgValue = (nameArg.Expression as LiteralExpressionSyntax)?.Token.ValueText;
            // If not a string literal, or not a valid identifier, there really is very little we can do. Suggest nothing.
            if (nameArgValue == null || !SyntaxFacts.IsValidIdentifier(nameArgValue))
                return;

            Debug.Assert(nameArg.Parent.Parent.IsKind(SyntaxKind.InvocationExpression));
            var invokeExp = (InvocationExpressionSyntax)nameArg.Parent.Parent;
            var member = invokeExp.Expression as MemberAccessExpressionSyntax;
            var methodName = member.ToString(); // Something like CheckParam, ExceptUserArg...

            // Check whether this is a simple case, that is, this string has the same text as some token.
            var argList = (ArgumentListSyntax)nameArg.Parent;

            // One of the most common checks are checks for value. Check whether this is the case.
            // If it is, we don't even have to resort to extracting the semantic model.
            argList.Arguments[0].Expression.GetText();
            if (nameArg.NameColon == null && argList.Arguments.Count >= 2 &&
                argList.Arguments[1] == nameArg && argList.Arguments[0].Expression.ToString() == nameArgValue)
            {
                context.RegisterCodeFix(CodeAction.Create(Title,
                    c => StringReplace(context.Document, nameArgValue, nameArg, c), Id), diagnostic);
                return;
            }
            // Check all symbols used in the Check/Except argument. Let's see if there's a match.
            // In the event of ambiguity, we choose the shortest one, figuring that the least complex
            // might be the most likely.
            int shortestSymbol = int.MaxValue;
            ExpressionSyntax bestSymbol = null;
            var sameNameNodes = argList.DescendantTokens().Where(tok => tok.Text == nameArgValue)
                .Select(p => p.Parent).Where(n => n.IsKind(SyntaxKind.IdentifierName));
            foreach (var node in sameNameNodes)
            {
                SyntaxNode candidate = node;
                var pk = node.Parent.Kind();
                if (pk == SyntaxKind.SimpleMemberAccessExpression)
                {
                    var parentAccess = (MemberAccessExpressionSyntax)node.Parent;
                    candidate = parentAccess.Expression == node ? node : parentAccess;
                }
                else if (pk == SyntaxKind.QualifiedName)
                {
                    // A little weird, but if you have class Z nested in Y, nested in X, then typeof(X.Y.Z) will
                    // be a series of qualified names, but nameof(X.Y.Z) will be a series of simple member accesses.
                    // nameof(X.Y.Z) if phrased as qualified names will not work.
                    candidate = SyntaxFactory.ParseExpression(node.Parent.ToString());
                }

                if (candidate.Span.Length < shortestSymbol)
                {
                    bestSymbol = (ExpressionSyntax)candidate;
                    shortestSymbol = candidate.Span.Length;
                }
            }

            if (bestSymbol != null)
            {
                context.RegisterCodeFix(CodeAction.Create(Title,
                    c => ExpressionReplace(context.Document, bestSymbol, nameArg, c), Id), diagnostic);
                return;
            }

            // No luck within the check statement itself. Next check the parameter list of this method or constructor.
            SyntaxNode temp = nameArg;
            while (temp != null && !temp.IsKind(SyntaxKind.MethodDeclaration) && !temp.IsKind(SyntaxKind.ConstructorDeclaration))
                temp = temp.Parent;

            ParameterSyntax argParam = null;
            if (temp != null)
            {
                var paramList = (temp as MethodDeclarationSyntax)?.ParameterList
                    ?? ((ConstructorDeclarationSyntax)temp).ParameterList;
                foreach (var param in paramList.Parameters)
                {
                    if (param.Identifier.ToString() == nameArgValue)
                    {
                        context.RegisterCodeFix(CodeAction.Create(Title,
                            c => StringReplace(context.Document, nameArgValue, nameArg, c), Id), diagnostic);
                        return;
                    }
                    // A hack, but whatever works.
                    string paramTypeString = param.Type.ToString();
                    if (argParam == null && (paramTypeString == "Arguments" || paramTypeString == "Column"))
                        argParam = param;
                }
            }
            // All else has failed. The last is to try to get information from any Arguments object, if present.
            if (argParam != null)
            {
                var semanticModel = await context.Document.GetSemanticModelAsync(context.CancellationToken);
                var type = semanticModel.GetTypeInfo(argParam.Type, context.CancellationToken).Type;
                var argName = argParam.Identifier.ToString();
                if (type != null && !(type is IErrorTypeSymbol))
                {
                    //var m = type.GetMembers().Cast<IFieldSymbol>;
                    foreach (IFieldSymbol s in type.GetMembers().Where(p => p.Kind == SymbolKind.Field))
                    {
                        if (!s.CanBeReferencedByName)
                            continue;
                        AttributeData attr = s.GetAttributes().FirstOrDefault(a => a.AttributeClass.Name == "ArgumentAttribute");
                        if (attr == null)
                            continue;
                        if (s.Name == nameArgValue)
                        {
                            context.RegisterCodeFix(CodeAction.Create(Title,
                                c => StringReplace(context.Document, argName + "." + s.Name, nameArg, c), Id), diagnostic);
                            return;
                        }
                        var shortPair = attr.NamedArguments.FirstOrDefault(p => p.Key == "ShortName");
                        var shortName = shortPair.Value.Value as string;
                        if (shortName == null)
                            continue;
                        if (shortName.Split(',').Contains(nameArgValue))
                        {
                            context.RegisterCodeFix(CodeAction.Create(Title,
                                c => StringReplace(context.Document, argName + "." + s.Name, nameArg, c), Id), diagnostic);
                            return;
                        }
                    }
                }
            }
        }

        private async Task<Document> StringReplace(Document document, string name, ArgumentSyntax nameArg, CancellationToken cancellationToken)
        {
            var nameofExp = SyntaxFactory.ParseExpression($"nameof({name})").WithTriviaFrom(nameArg);
            var tree = await document.GetSyntaxTreeAsync(cancellationToken);
            var root = await tree.GetRootAsync(cancellationToken);
            var newRoot = root.ReplaceNode(nameArg.Expression, nameofExp);
            return document.WithSyntaxRoot(newRoot);
        }

        private async Task<Document> ExpressionReplace(Document document, SyntaxNode exp, ArgumentSyntax nameArg, CancellationToken cancellationToken)
        {
            var nameofExp = (InvocationExpressionSyntax)SyntaxFactory.ParseExpression($"nameof(a)").WithTriviaFrom(nameArg);
            var newNameofExp = nameofExp.ReplaceNode(nameofExp.ArgumentList.Arguments[0].Expression, exp.WithoutTrivia());

            var tree = await document.GetSyntaxTreeAsync(cancellationToken);
            var root = await tree.GetRootAsync(cancellationToken);
            var newRoot = root.ReplaceNode(nameArg.Expression, newNameofExp);
            return document.WithSyntaxRoot(newRoot);
        }
    }
}
