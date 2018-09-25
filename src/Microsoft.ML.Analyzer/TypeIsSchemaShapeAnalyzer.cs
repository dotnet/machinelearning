// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Microsoft.ML.Analyzer
{
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class TypeIsSchemaShapeAnalyzer : DiagnosticAnalyzer
    {
        internal static class ShapeDiagnostic
        {
            private const string Category = "Type Check";
            public const string Id = "MSML_TypeShouldBeSchemaShape";
            private const string Title = "The type is not a schema shape";
            private const string Format = "Type{0} is neither a PipelineColumn nor a ValueTuple.";
            internal const string Description =
                "Within statically typed pipeline elements of ML.NET, the shape of the schema is determined by a type. " +
                "A valid type is either an instance of one of the PipelineColumn subclasses (for example, Scalar<bool> " +
                "or something like that), or a ValueTuple containing only valid types. (So, ValueTuples containing " +
                "other value tuples are fine, so long as they terminate in a PipelineColumn subclass.)";

            internal static DiagnosticDescriptor Rule =
                new DiagnosticDescriptor(Id, Title, Format, Category,
                    DiagnosticSeverity.Error, isEnabledByDefault: true, description: Description);
        }

        internal static class ShapeParameterDiagnostic
        {
            private const string Category = "Type Check";
            public const string Id = "MSML_TypeParameterShouldBeSchemaShape";
            private const string Title = "The type is not a schema shape";
            private const string Format = "Type parameter {0} is not marked with [IsShape] or appropriate type constraints.";
            internal const string Description = ShapeDiagnostic.Description + " " +
                "If using type parameters when interacting with the statically typed pipelines, the type parameter ought to be " +
                "constrained in such a way that it, either by applying the [IsShape] attribute or by having type constraints to " +
                "indicate that it is valid, for example, constraining the type to descend from PipelineColumn.";

            internal static DiagnosticDescriptor Rule =
                new DiagnosticDescriptor(Id, Title, Format, Category,
                    DiagnosticSeverity.Error, isEnabledByDefault: true, description: Description);
        }

        private const string AttributeName = "Microsoft.ML.Data.StaticPipe.IsShapeAttribute";
        private const string LeafTypeName = "Microsoft.ML.Data.StaticPipe.Runtime.PipelineColumn";

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
            ImmutableArray.Create(ShapeDiagnostic.Rule, ShapeParameterDiagnostic.Rule);

        public override void Initialize(AnalysisContext context)
        {
            context.RegisterSemanticModelAction(Analyze);
        }

        private void Analyze(SemanticModelAnalysisContext context)
        {
            // We start with the model, then do the the method invocations.
            // We could have phrased it as RegisterSyntaxNodeAction(Analyze, SyntaxKind.InvocationExpression),
            // but this seemed more inefficient since getting the model and fetching the type symbols every
            // single time seems to incur significant cost. The following invocation is somewhat more awkward
            // since we must iterate over the invocation syntaxes ourselves, but this seems to be worthwhile.
            var model = context.SemanticModel;
            var comp = model.Compilation;

            // Get the symbols of the key types we are analyzing. If we can't find any of them there is
            // no point in going further.
            var attrType = comp.GetTypeByMetadataName(AttributeName);
            if (attrType == null)
                return;
            var leafType = comp.GetTypeByMetadataName(LeafTypeName);
            if (leafType == null)
                return;

            // This internal helper method recursively determines whether an attributed type parameter
            // has a valid type. It is called externally from the loop over invocations.
            bool CheckType(ITypeSymbol type, out string path, out ITypeSymbol problematicType)
            {
                if (type.TypeKind == TypeKind.TypeParameter)
                {
                    var typeParam = (ITypeParameterSymbol)type;
                    path = null;
                    problematicType = null;
                    // Does the type parameter have the attribute that triggers a check?
                    if (type.GetAttributes().Any(attr => attr.AttributeClass == attrType))
                        return true;
                    // Are any of the declared constraint types OK?
                    if (typeParam.ConstraintTypes.Any(ct => CheckType(ct, out string ctPath, out var ctProb)))
                        return true;
                    // Well, probably not good then. Let's call it a day.
                    problematicType = typeParam;
                    return false;
                }
                else if (type.IsTupleType)
                {
                    INamedTypeSymbol nameType = (INamedTypeSymbol)type;
                    var tupleElems = nameType.TupleElements;

                    for (int i = 0; i < tupleElems.Length; ++i)
                    {
                        var e = tupleElems[i];
                        if (!CheckType(e.Type, out string innerPath, out problematicType))
                        {
                            path = e.Name ?? $"Item{i + 1}";
                            if (innerPath != null)
                                path += "." + innerPath;
                            return false;
                        }
                    }
                    path = null;
                    problematicType = null;
                    return true;
                }
                else
                {
                    for (var rt = type; rt != null; rt = rt.BaseType)
                    {
                        if (rt == leafType)
                        {
                            path = null;
                            problematicType = null;
                            return true;
                        }
                    }
                    path = null;
                    problematicType = type;
                    return false;
                }
            }

            foreach (var invocation in model.SyntaxTree.GetRoot().DescendantNodes().OfType<InvocationExpressionSyntax>())
            {
                var symbolInfo = model.GetSymbolInfo(invocation);
                if (!(symbolInfo.Symbol is IMethodSymbol methodSymbol))
                {
                    // Should we perhaps skip when there is a method resolution failure? This is often but not always a sign of another problem.
                    if (symbolInfo.CandidateReason != CandidateReason.OverloadResolutionFailure || symbolInfo.CandidateSymbols.Length == 0)
                        continue;
                    methodSymbol = symbolInfo.CandidateSymbols[0] as IMethodSymbol;
                    if (methodSymbol == null)
                        continue;
                }
                // Analysis only applies to generic methods.
                if (!methodSymbol.IsGenericMethod)
                    continue;
                // Scan the type parameters for one that has our target attribute.
                for (int i = 0; i < methodSymbol.TypeParameters.Length; ++i)
                {
                    var par = methodSymbol.TypeParameters[i];
                    var attr = par.GetAttributes();
                    if (attr.Length == 0)
                        continue;
                    if (!attr.Any(a => a.AttributeClass == attrType))
                        continue;
                    // We've found it. Check the type argument to ensure it is of the appropriate type.
                    var p = methodSymbol.TypeArguments[i];
                    if (CheckType(p, out string path, out ITypeSymbol problematicType))
                        continue;

                    if (problematicType.Kind == SymbolKind.TypeParameter)
                    {
                        var diagnostic = Diagnostic.Create(ShapeParameterDiagnostic.Rule, invocation.GetLocation(), problematicType.Name);
                        context.ReportDiagnostic(diagnostic);
                    }
                    else
                    {
                        path = path == null ? "" : " of item " + path;
                        var diagnostic = Diagnostic.Create(ShapeDiagnostic.Rule, invocation.GetLocation(), path);
                        context.ReportDiagnostic(diagnostic);
                    }
                }
            }
        }
    }
}
