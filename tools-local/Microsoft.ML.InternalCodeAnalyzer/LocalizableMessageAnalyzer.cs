// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Operations;

namespace Microsoft.ML.InternalCodeAnalyzer
{
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class LocalizableMessageAnalyzer : DiagnosticAnalyzer
    {
        private const string Category = "Globalization";

        internal static class HardCodedMessageDiagnostic
        {
            public const string Id = "MSML_LocalizableMessage";
            private const string Title = "Message must come from a resource";
            private const string Format = "The message passed to '{0}' must come from a localizable resource";
            private const string Description =
                "Exception and validation messages must be stored in resource files so they can be localized.";

            internal static readonly DiagnosticDescriptor Rule =
                new DiagnosticDescriptor(Id, Title, Format, Category,
                    DiagnosticSeverity.Error, isEnabledByDefault: true, description: Description);
        }

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
            ImmutableArray.Create(HardCodedMessageDiagnostic.Rule);

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();

            context.RegisterOperationAction(AnalyzeInvocation, OperationKind.Invocation);
            context.RegisterOperationAction(AnalyzeObjectCreation, OperationKind.ObjectCreation);
        }

        private static void AnalyzeInvocation(OperationAnalysisContext context)
        {
            var invocation = (IInvocationOperation)context.Operation;
            if (!IsContractsMethod(invocation.TargetMethod) ||
                SymbolEqualityComparer.Default.Equals(
                    context.ContainingSymbol.ContainingType,
                    invocation.TargetMethod.ContainingType))
            {
                return;
            }

            foreach (var argument in invocation.Arguments)
            {
                if (argument.Parameter?.Name == "msg")
                    AnalyzeMessage(context, argument, invocation.TargetMethod.Name);
            }
        }

        private static void AnalyzeObjectCreation(OperationAnalysisContext context)
        {
            var creation = (IObjectCreationOperation)context.Operation;
            if (!DerivesFromException(creation.Type, context.Compilation))
                return;

            foreach (var argument in creation.Arguments)
            {
                if (argument.Parameter?.Name == "message")
                    AnalyzeMessage(context, argument, creation.Constructor.ContainingType.Name);
            }
        }

        private static void AnalyzeMessage(
            OperationAnalysisContext context,
            IArgumentOperation argument,
            string targetName)
        {
            if (!IsHardCoded(argument.Value))
                return;

            context.ReportDiagnostic(Diagnostic.Create(
                HardCodedMessageDiagnostic.Rule,
                argument.Syntax.GetLocation(),
                targetName));
        }

        private static bool IsHardCoded(IOperation operation)
        {
            operation = UnwrapConversion(operation);

            if (operation is INameOfOperation)
                return false;

            if (operation.ConstantValue.HasValue && operation.ConstantValue.Value is string)
                return true;

            if (operation is IInterpolatedStringOperation ||
                operation is IBinaryOperation binary && binary.OperatorKind == BinaryOperatorKind.Add)
            {
                return true;
            }

            if (operation is IInvocationOperation invocation)
            {
                if (invocation.TargetMethod.Name != "Format" ||
                    invocation.TargetMethod.ContainingType.SpecialType != SpecialType.System_String)
                {
                    return false;
                }

                foreach (IArgumentOperation argument in invocation.Arguments)
                {
                    if (argument.Parameter?.Name == "format")
                        return IsHardCoded(argument.Value);
                }
            }

            return false;
        }

        private static IOperation UnwrapConversion(IOperation operation)
        {
            while (operation is IConversionOperation conversion)
                operation = conversion.Operand;

            return operation;
        }

        private static bool IsContractsMethod(IMethodSymbol method)
        {
            switch (method.Name)
            {
                case "Check":
                case "CheckUserArg":
                case "CheckParam":
                case "CheckParamValue":
                case "CheckRef":
                case "CheckValue":
                case "CheckNonEmpty":
                case "CheckNonWhiteSpace":
                case "CheckDecode":
                case "CheckIO":
                case "CheckAlive":
                case "CheckValueOrNull":
                case "Except":
                case "ExceptUserArg":
                case "ExceptParam":
                case "ExceptParamValue":
                case "ExceptValue":
                case "ExceptEmpty":
                case "ExceptWhiteSpace":
                case "ExceptDecode":
                case "ExceptIO":
                case "ExceptNotImpl":
                case "ExceptNotSupp":
                case "ExceptSchemaMismatch":
                    break;
                default:
                    return false;
            }

            string containingType = method.ContainingType.ToDisplayString();
            return containingType == "Microsoft.ML.Runtime.Contracts" ||
                containingType == "Microsoft.ML.Internal.CpuMath.Core.Contracts" ||
                containingType == "Microsoft.ML.Internal.DataView.Contracts";
        }

        private static bool DerivesFromException(ITypeSymbol type, Compilation compilation)
        {
            INamedTypeSymbol exceptionType = compilation.GetTypeByMetadataName("System.Exception");
            return exceptionType != null &&
                compilation.ClassifyCommonConversion(type, exceptionType).IsImplicit;
        }
    }
}
