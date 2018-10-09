// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
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
            private const string Format = "Type{0} is neither a PipelineColumn nor a ValueTuple, nor a class of an allowed form.";
            internal const string Description =
                "Within statically typed pipeline elements of ML.NET, the shape of the schema is determined by a 'shape' type. " +
                "A valid 'shape' type is either an instance of one of the PipelineColumn subclasses (for example, Scalar<bool> " +
                "or something like that), or a ValueTuple containing only valid 'shape' types, or a class whose only publicly " +
                "accessible members are methods, a single constructor, and properties that are valid 'shape' types themselves, " +
                "and that have either get and set accessors and the single constructor takes no parameters, or that has get only " +
                "property accessors and the constructor takes as many parameters as there are properties. (So, ValueTuples " +
                "containing other value tuples are fine, so long as they terminate in a PipelineColumn subclass.)";

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

        internal static class ShapeClassDiagnosticConstructor
        {
            private const string Category = "Type Check";
            public const string Id = "MSML_SchemaShapeClassShouldHaveOnePublicConstructor";
            private const string Title = "The class does not have exactly one public constructor.";
            private const string Format = "Member's type {0} does not have exactly one public constructor.";
            internal const string Description = ShapeDiagnostic.Description + " " +
                "This type does not have exactly one public constructor.";

            internal static DiagnosticDescriptor Rule =
                new DiagnosticDescriptor(Id, Title, Format, Category,
                    DiagnosticSeverity.Error, isEnabledByDefault: true, description: Description);
        }

        internal static class ShapeClassDiagnosticField
        {
            private const string Category = "Type Check";
            public const string Id = "MSML_SchemaShapeClassShouldHaveNoPublicFields";
            private const string Title = "The class should not have publicly accessible fields.";
            private const string Format = "Type {0} has publicly accessible field {1}.";
            internal const string Description = ShapeDiagnostic.Description + " " +
                "This type has publicly accessible fields.";

            internal static DiagnosticDescriptor Rule =
                new DiagnosticDescriptor(Id, Title, Format, Category,
                    DiagnosticSeverity.Error, isEnabledByDefault: true, description: Description);
        }

        internal static class ShapeClassDiagnosticGettable
        {
            private const string Category = "Type Check";
            public const string Id = "MSML_SchemaShapeClassGettableProperty";
            private const string Title = "All properties should be gettable.";
            private const string Format = "Type {0} has property {1} without a public getter.";
            internal const string Description = ShapeDiagnostic.Description + " " +
                "This type has a property without a getter.";

            internal static DiagnosticDescriptor Rule =
                new DiagnosticDescriptor(Id, Title, Format, Category,
                    DiagnosticSeverity.Error, isEnabledByDefault: true, description: Description);
        }

        internal static class ShapeClassDiagnosticNoArgsSettable
        {
            private const string Category = "Type Check";
            public const string Id = "MSML_SchemaShapeClassWithParameterlessConstructorSettableProperties";
            private const string Title = "If the class has a constructor with no parameters, all properties should be settable.";
            private const string Format = "Type {0} has property {1} that is not settable.";
            internal const string Description = ShapeDiagnostic.Description + " " +
                "This type has a parameterless constructor, but a field that is not settable.";

            internal static DiagnosticDescriptor Rule =
                new DiagnosticDescriptor(Id, Title, Format, Category,
                    DiagnosticSeverity.Error, isEnabledByDefault: true, description: Description);
        }

        internal static class ShapeClassDiagnosticArgsSettable
        {
            private const string Category = "Type Check";
            public const string Id = "MSML_SchemaShapeClassWithParameterfulConstructorSettableProperties";
            private const string Title = "If the class has a constructor with parameters, but some properties are settable.";
            private const string Format = "Type {0} has property {1} that is settable.";
            internal const string Description = ShapeDiagnostic.Description + " " +
                "This type has a constructor with parameters, but some of the properties also have setters.";

            internal static DiagnosticDescriptor Rule =
                new DiagnosticDescriptor(Id, Title, Format, Category,
                    DiagnosticSeverity.Error, isEnabledByDefault: true, description: Description);
        }

        internal static class ShapeClassDiagnosticCorrespondence
        {
            private const string Category = "Type Check";
            public const string Id = "MSML_SchemaShapeClassConstructorAndPropertiesCorrespond";
            private const string Title = "If the class has a constructor with parameters, there ought to be a one to one correspondence between the parameters and the properties.";
            private const string Format = "Type {0} appears to not have an exact correspondence among the number or type of constructor parameters and properties.";
            internal const string Description = ShapeDiagnostic.Description + " " +
                "This type has a constructor with parameters, but it appears that the number or types of the properties do not correspond to the parameters in the constructor.";

            internal static DiagnosticDescriptor Rule =
                new DiagnosticDescriptor(Id, Title, Format, Category,
                    DiagnosticSeverity.Error, isEnabledByDefault: true, description: Description);
        }

        private const string AttributeName = "Microsoft.ML.StaticPipe.IsShapeAttribute";
        private const string LeafTypeName = "Microsoft.ML.StaticPipe.PipelineColumn";

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
            ImmutableArray.Create(ShapeDiagnostic.Rule, ShapeParameterDiagnostic.Rule, ShapeClassDiagnosticConstructor.Rule, ShapeClassDiagnosticField.Rule,
                ShapeClassDiagnosticGettable.Rule, ShapeClassDiagnosticNoArgsSettable.Rule, ShapeClassDiagnosticArgsSettable.Rule, ShapeClassDiagnosticCorrespondence.Rule);

        public override void Initialize(AnalysisContext context)
        {
            context.RegisterSemanticModelAction(Analyze);
        }

        private enum SpecificError
        {
            None,
            General,
            TypeParam,
            Constructor,
            Field,

            PropGettable,
            PropNoArgSettable,
            PropArgSettable,
            PropNoCorrespondence,
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
            bool CheckType(ITypeSymbol type, out string path, out ITypeSymbol problematicType, out SpecificError specificError)
            {
                // Assume it's OK.
                path = null;
                problematicType = null;
                specificError = SpecificError.None;

                if (type.Kind == SymbolKind.ErrorType)
                    return true; // We at least should not complain, so we don't get in the way of whatever the real problem is.

                if (type.TypeKind == TypeKind.TypeParameter)
                {
                    var typeParam = (ITypeParameterSymbol)type;
                    // Does the type parameter have the attribute that triggers a check?
                    if (type.GetAttributes().Any(attr => attr.AttributeClass == attrType))
                        return true;
                    // Are any of the declared constraint types OK? If they're OK, we're OK.
                    if (typeParam.ConstraintTypes.Any(ct => CheckType(ct, out string ctPath, out var ctProb, out var ctSpecificError)))
                        return true;
                    // Well, probably not good then. Let's call it a day.
                    specificError = SpecificError.TypeParam;
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
                        if (!CheckType(e.Type, out string innerPath, out problematicType, out specificError))
                        {
                            path = e.Name ?? $"Item{i + 1}";
                            if (innerPath != null)
                                path += "." + innerPath;
                            return false;
                        }
                    }
                    return true;
                }
                else if (type.IsReferenceType)
                {
                    // First check to see if it is a pipeline column. If it is we can stop.
                    for (var rt = type; rt != null; rt = rt.BaseType)
                    {
                        if (rt == leafType)
                            return true;
                    }

                    // Next check if it's a reference type.
                    var members = type.GetMembers();

                    // First find the constructor.
                    IMethodSymbol constructor = null;
                    foreach (var method in members.OfType<IMethodSymbol>())
                    {
                        if (method.DeclaredAccessibility != Accessibility.Public)
                            continue;

                        if (method.MethodKind != MethodKind.Constructor)
                            continue;
                        if (constructor != null)
                        {
                            problematicType = type;
                            specificError = SpecificError.Constructor;
                            return false;
                        }
                        constructor = method;
                    }

                    if (constructor == null)
                    {
                        problematicType = type;
                        specificError = SpecificError.Constructor;
                        return false;
                    }

                    // Determine the parameters of the constructor, if any.
                    var t2c = new Dictionary<ITypeSymbol, int>();
                    foreach (var prm in constructor.Parameters)
                    {
                        t2c.TryGetValue(prm.Type, out int cnt);
                        t2c[prm.Type] = cnt + 1;
                    }
                    bool needsSetters = constructor.Parameters.Length == 0;

                    // Next iterate over the members.
                    foreach (var member in members)
                    {
                        // Only care about public members, and ignore methods.
                        if (member.DeclaredAccessibility != Accessibility.Public || member.Kind == SymbolKind.Method || member.IsStatic)
                            continue;

                        if (member.Kind == SymbolKind.Field)
                        {
                            path = member.Name;
                            problematicType = type;
                            specificError = SpecificError.Field;
                            return false;
                        }

                        if (member.Kind == SymbolKind.Property)
                        {
                            var propSymbol = (IPropertySymbol)member;
                            if (!CheckType(propSymbol.Type, out string innerPath, out problematicType, out specificError))
                            {
                                path = propSymbol.Name;
                                if (innerPath != null)
                                    path += "." + innerPath;
                                return false;
                            }

                            // Make sure the property is gettable.
                            if (propSymbol.GetMethod?.DeclaredAccessibility != Accessibility.Public)
                            {
                                path = propSymbol.Name;
                                problematicType = type;
                                specificError = SpecificError.PropGettable;
                                return false;
                            }
                            if (constructor.Parameters.Length > 0)
                            {
                                if (t2c.TryGetValue(propSymbol.Type, out int count))
                                {
                                    t2c[propSymbol.Type] = count - 1;
                                    if (count == 1)
                                        t2c.Remove(propSymbol.Type);
                                }
                                else
                                {
                                    // Couldn't find a corresponding parameter in the constructor.
                                    path = propSymbol.Name;
                                    problematicType = type;
                                    specificError = SpecificError.PropNoCorrespondence;
                                    return false;
                                }
                                if (propSymbol.SetMethod?.DeclaredAccessibility == Accessibility.Public)
                                {
                                    path = propSymbol.Name;
                                    problematicType = type;
                                    specificError = SpecificError.PropArgSettable;
                                    return false;
                                }
                            }
                            else if (propSymbol.SetMethod?.DeclaredAccessibility != Accessibility.Public)
                            {
                                path = propSymbol.Name;
                                problematicType = type;
                                specificError = SpecificError.PropNoArgSettable;
                                return false;
                            }
                        }
                    }
                    // Finally check that *every* parameter in the constructor has been covered.
                    if (t2c.Count > 0)
                    {
                        // Some parameters in the constructor were uncovered.
                        problematicType = type;
                        specificError = SpecificError.PropNoCorrespondence;
                        return false;
                    }
                    return true;
                }
                problematicType = type;
                specificError = SpecificError.General;
                return false;
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
                    if (CheckType(p, out string path, out ITypeSymbol problematicType, out SpecificError error))
                        continue;

                    Diagnostic diagnostic;
                    switch (error)
                    {
                        case SpecificError.TypeParam:
                            diagnostic = Diagnostic.Create(ShapeParameterDiagnostic.Rule, invocation.GetLocation(), problematicType.Name);
                            break;
                        case SpecificError.Constructor:
                            diagnostic = Diagnostic.Create(ShapeClassDiagnosticConstructor.Rule, invocation.GetLocation(), problematicType.Name);
                            break;
                        case SpecificError.Field:
                            diagnostic = Diagnostic.Create(ShapeClassDiagnosticField.Rule, invocation.GetLocation(), problematicType.Name, path);
                            break;

                        case SpecificError.PropGettable:
                            diagnostic = Diagnostic.Create(ShapeClassDiagnosticGettable.Rule, invocation.GetLocation(), problematicType.Name, path);
                            break;
                        case SpecificError.PropArgSettable:
                            diagnostic = Diagnostic.Create(ShapeClassDiagnosticArgsSettable.Rule, invocation.GetLocation(), problematicType.Name, path);
                            break;
                        case SpecificError.PropNoArgSettable:
                            diagnostic = Diagnostic.Create(ShapeClassDiagnosticNoArgsSettable.Rule, invocation.GetLocation(), problematicType.Name, path);
                            break;
                        case SpecificError.PropNoCorrespondence:
                            diagnostic = Diagnostic.Create(ShapeClassDiagnosticCorrespondence.Rule, invocation.GetLocation(), problematicType.Name);
                            break;

                        case SpecificError.General:
                        default: // Whoops. Just pretend it's a general error.
                            path = path == null ? "" : " of item " + path;
                            diagnostic = Diagnostic.Create(ShapeDiagnostic.Rule, invocation.GetLocation(), path);
                            break;
                    }
                    context.ReportDiagnostic(diagnostic);
                }
            }
        }
    }
}
