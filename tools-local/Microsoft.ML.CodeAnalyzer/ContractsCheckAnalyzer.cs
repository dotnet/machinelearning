// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Microsoft.ML.CodeAnalyzer
{
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class ContractsCheckAnalyzer : DiagnosticAnalyzer
    {
        // Detecting that a syntax call is actually on a particular method is computationally
        // intensive, so once we detect that we're on Contracts methods, we put all the methods
        // here.

        private const string Category = "Contracts";

        internal static class NameofDiagnostic
        {
            public const string Id = "MSML_ContractsNameUsesNameof";
            private const string Title = "Contracts argument for names is not a nameof";
            private const string Format = "Call to '{0}' should use nameof(...) for {1} argument, but instead used '{2}'";
            private const string Description =
                "For Contracts.Checks or Excepts with some form of parameter name, unless that " +
                "argument is a nameof(...) expression there's almost certainly something wrong.";

            internal static DiagnosticDescriptor Rule =
                new DiagnosticDescriptor(Id, Title, Format, Category,
                    DiagnosticSeverity.Warning, isEnabledByDefault: true, description: Description);
        }

        internal static class ExceptionDiagnostic
        {
            public const string Id = "MSML_ContractsExceptAsExpression";
            private const string Title = "Contracts.Except used as expression";
            private const string Format = "Something should be done with the exception created by '{0}'";
            private const string Description =
                "Contracts.Except and similar methods do not themselves throw, but provide an " +
                "exception that can be thrown. This call did nothing with the exception.";

            internal static DiagnosticDescriptor Rule =
                new DiagnosticDescriptor(Id, Title, Format, Category,
                    DiagnosticSeverity.Warning, isEnabledByDefault: true, description: Description);
        }

        internal static class SimpleMessageDiagnostic
        {
            public const string Id = "MSML_ContractsCheckMessageNotLiteralOrIdentifier";
            private const string Title = "Contracts.Check argument for message may involve formatting";
            private const string Format = "On call to '{0}' message '{1}' could not be identified as being either a string literal or variable";

            internal static DiagnosticDescriptor Rule =
                new DiagnosticDescriptor(Id, Title, Format, Category,
                    DiagnosticSeverity.Warning, isEnabledByDefault: true,
                    description: Descriptions.ContractsCheckMessageNotLiteralOrIdentifier);
        }

        internal static class DecodeMessageWithLoadContextDiagnostic
        {
            public const string Id = "MSML_NoMessagesForLoadContext";
            private const string Title = "Contracts.Check argument for message may involve formatting";
            private const string Format = "On call to '{0}' message '{1}' was provided, but this method had a ModelLoadContext";

            internal static DiagnosticDescriptor Rule =
                new DiagnosticDescriptor(Id, Title, Format, Category,
                    DiagnosticSeverity.Warning, isEnabledByDefault: true,
                    description: Descriptions.NoMessagesForLoadContext);
        }

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
            ImmutableArray.Create(
                NameofDiagnostic.Rule, ExceptionDiagnostic.Rule, SimpleMessageDiagnostic.Rule,
                DecodeMessageWithLoadContextDiagnostic.Rule);

        private static HashSet<string> _targetSet = new HashSet<string>(new[]
        {
            "Check", "CheckUserArg", "CheckParam", "CheckParamValue", "CheckRef", "CheckValue",
            "CheckNonEmpty", "CheckNonWhiteSpace", "CheckDecode", "CheckIO", "CheckAlive", "CheckValueOrNull",
            "Except", "ExceptUserArg", "ExceptParam", "ExceptParamValue", "ExceptValue", "ExceptEmpty",
            "ExceptWhiteSpace", "ExceptDecode", "ExceptIO", "ExceptNotImpl", "ExceptNotSupp",
        });

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.RegisterSyntaxNodeAction(Analyze, SyntaxKind.InvocationExpression);
        }

        /// <summary>
        /// Returns an array parallel to <paramref name="parameters"/> that contains
        /// the arguments in <paramref name="invocation"/>. If named parameters are used
        /// then this is not necessarily the same. Note that in the event that there are
        /// more arguments than parameters (e.g., via a <c>params</c> variable length
        /// parameter) only the first match for the parameter is recorded.
        /// </summary>
        private static ArgumentSyntax[] ParallelArgs(
            ImmutableArray<IParameterSymbol> parameters,
            InvocationExpressionSyntax invocation)
        {
            ArgumentSyntax[] args = new ArgumentSyntax[parameters.Length];
            var syntaxArgs = invocation.ArgumentList.Arguments;
            for (int i = 0; i < syntaxArgs.Count; ++i)
            {
                var arg = syntaxArgs[i];
                int index = -1;
                if (arg.NameColon == null)
                    index = i;
                else
                {
                    string nameColonText = arg.NameColon.Name.ToString();
                    for (int p = 0; p < parameters.Length; ++p)
                    {
                        if (parameters[p].Name == nameColonText)
                        {
                            index = p;
                            break;
                        }
                    }
                }
                if (0 <= index && index < args.Length && args[index] == null)
                    args[index] = arg;
            }
            return args;
        }

        private static bool NameIsNameof(ExpressionSyntax exp)
        {
            var invokeExp = exp as InvocationExpressionSyntax;
            return invokeExp != null && invokeExp.Expression.ToString() == "nameof";
        }

        private static bool IsGoodMessage(SyntaxNodeAnalysisContext context, ExpressionSyntax exp)
        {
            if (exp.IsKind(SyntaxKind.AddExpression))
            {
                // These sorts of string concatenation things always wind up being compile
                // time constants, from what I can tell from ildasm.
                var binExp = (BinaryExpressionSyntax)exp;
                return IsGoodMessage(context, binExp.Left) && IsGoodMessage(context, binExp.Right);
            }

            if (exp.IsKind(SyntaxKind.SimpleMemberAccessExpression))
            {
                var access = (MemberAccessExpressionSyntax)exp;
                var field = context.SemanticModel.GetSymbolInfo(access).Symbol as IFieldSymbol;
                return field?.IsConst ?? false;
            }

            if (exp.IsKind(SyntaxKind.InvocationExpression))
                return ((InvocationExpressionSyntax)exp).Expression.ToString() == "nameof";

            return exp.IsKind(SyntaxKind.StringLiteralExpression) || exp.IsKind(SyntaxKind.IdentifierName);
        }

        private static bool HasModelLoadContext(SyntaxNode node)
        {
            while (node != null && !node.IsKind(SyntaxKind.MethodDeclaration) && !node.IsKind(SyntaxKind.ConstructorDeclaration))
                node = node.Parent;
            if (node == null)
                return false;
            var enclosingParams = ((node as MethodDeclarationSyntax)?.ParameterList
                ?? ((ConstructorDeclarationSyntax)node).ParameterList).Parameters;
            foreach (var param in enclosingParams)
            {
                // It is possible that this may mislead us slightly, since there could be another
                // unrelated type called ModelLoadContext, or someone could have type aliasing, or
                // some other complicating factor that will defeat this simple check. With some
                // additional computational load, we could access the semantic model for this.
                if (param.Type.ToString() == "ModelLoadContext")
                    return true;
            }
            return false;
        }

        private static void Analyze(SyntaxNodeAnalysisContext context)
        {
            var invocation = (InvocationExpressionSyntax)context.Node;
            if (!(invocation.Expression is MemberAccessExpressionSyntax access))
                return;
            var name = access.Name.ToString();
            // Do the quick checks first on the name.
            bool isCheck = false;
            bool isExcept = false;
            if ((!(isCheck = name.StartsWith("Check")) && !(isExcept = name.StartsWith("Except"))) || !_targetSet.Contains(name))
                return;
            // Now that we've verified we're approximately in the right neighborhood, do a more
            // in depth semantic analysis to verify we're targetting the right sort of object.
            var symbolInfo = context.SemanticModel.GetSymbolInfo(invocation);
            if (!(symbolInfo.Symbol is IMethodSymbol methodSymbol))
                return;
            var containingSymbolName = methodSymbol.ContainingSymbol.ToString();
            // The "internal" version is one used by some projects that want to benefit from Contracts,
            // but for some reason cannot reference MLCore.
            if (containingSymbolName != "Microsoft.ML.Runtime.Contracts" &&
                containingSymbolName != "Microsoft.ML.Runtime.Internal.Contracts")
            {
                return;
            }
            if (isExcept && invocation.Parent.IsKind(SyntaxKind.ExpressionStatement))
            {
                context.ReportDiagnostic(Diagnostic.Create(
                    ExceptionDiagnostic.Rule, invocation.GetLocation(), name));
            }

            var parameters = methodSymbol.Parameters;
            var args = ParallelArgs(parameters, invocation);

            for (int i = 0; i < parameters.Length; ++i)
            {
                if (args[i] == null)
                    continue;
                var arg = args[i];
                var parameter = parameters[i];

                switch (parameter.Name)
                {
                    case "paramName":
                    case "name":
                        if (!NameIsNameof(arg.Expression))
                        {
                            context.ReportDiagnostic(Diagnostic.Create(
                                NameofDiagnostic.Rule, arg.GetLocation(), name, parameter.Name, arg.Expression));
                        }
                        break;
                    case "msg":
                        if (isCheck && !IsGoodMessage(context, arg.Expression))
                        {
                            context.ReportDiagnostic(Diagnostic.Create(
                                SimpleMessageDiagnostic.Rule, arg.GetLocation(), name, arg.Expression));
                        }
                        if ((name == "CheckDecode" || name == "ExceptDecode") && HasModelLoadContext(invocation))
                        {
                            context.ReportDiagnostic(Diagnostic.Create(
                                DecodeMessageWithLoadContextDiagnostic.Rule, arg.GetLocation(), name, arg.Expression));
                        }
                        break;
                    default:
                        break;
                }
            }
        }
    }
}
