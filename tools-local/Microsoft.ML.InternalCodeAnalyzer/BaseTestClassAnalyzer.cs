﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Concurrent;
using System.Collections.Immutable;
using System.Diagnostics;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Microsoft.ML.InternalCodeAnalyzer
{
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class BaseTestClassAnalyzer : DiagnosticAnalyzer
    {
        private const string Category = "Test";
        internal const string DiagnosticId = "MSML_ExtendBaseTestClass";

        private const string Title = "Test classes should be derived from BaseTestClass";
        private const string Format = "Test class '{0}' should extend BaseTestClass.";
        private const string Description =
            "Test classes should be derived from BaseTestClass.";

        private static DiagnosticDescriptor Rule =
            new DiagnosticDescriptor(DiagnosticId, Title, Format, Category,
                DiagnosticSeverity.Warning, isEnabledByDefault: true, description: Description);

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics => ImmutableArray.Create(Rule);

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();

            context.RegisterCompilationStartAction(AnalyzeCompilation);
        }

        private void AnalyzeCompilation(CompilationStartAnalysisContext context)
        {
            if (!(context.Compilation.GetTypeByMetadataName("Xunit.FactAttribute") is { } factAttribute))
            {
                return;
            }

            var analyzerImpl = new AnalyzerImpl(context.Compilation, factAttribute);
            context.RegisterSymbolAction(analyzerImpl.AnalyzeNamedType, SymbolKind.NamedType);
        }

        private sealed class AnalyzerImpl
        {
            private readonly Compilation _compilation;
            private readonly INamedTypeSymbol _factAttribute;
            private readonly INamedTypeSymbol _baseTestClass;
            private readonly ConcurrentDictionary<INamedTypeSymbol, bool> _knownTestAttributes = new ConcurrentDictionary<INamedTypeSymbol, bool>();

            public AnalyzerImpl(Compilation compilation, INamedTypeSymbol factAttribute)
            {
                _compilation = compilation;
                _factAttribute = factAttribute;
                _baseTestClass = _compilation.GetTypeByMetadataName("Microsoft.ML.TestFramework.BaseTestClass");
            }

            public void AnalyzeNamedType(SymbolAnalysisContext context)
            {
                var namedType = (INamedTypeSymbol)context.Symbol;
                if (namedType.TypeKind != TypeKind.Class)
                    return;

                if (ExtendsBaseTestClass(namedType))
                    return;

                var hasTestMethod = false;
                foreach (var member in namedType.GetMembers())
                {
                    if (member is IMethodSymbol method && IsTestMethod(method))
                    {
                        hasTestMethod = true;
                        break;
                    }
                }

                if (!hasTestMethod)
                    return;

                context.ReportDiagnostic(Diagnostic.Create(Rule, namedType.Locations[0], namedType));
            }

            private bool IsTestMethod(IMethodSymbol method)
            {
                foreach (var attribute in method.GetAttributes())
                {
                    if (IsTestAttribute(attribute.AttributeClass))
                        return true;
                }

                return false;
            }

            private bool IsTestAttribute(INamedTypeSymbol attributeClass)
            {
                if (_knownTestAttributes.TryGetValue(attributeClass, out var isTest))
                    return isTest;

                return _knownTestAttributes.GetOrAdd(attributeClass, ExtendsFactAttribute(attributeClass));
            }

            private bool ExtendsBaseTestClass(INamedTypeSymbol namedType)
            {
                if (_baseTestClass is null)
                    return false;

                for (var current = namedType; current is object; current = current.BaseType)
                {
                    if (Equals(current, _baseTestClass))
                        return true;
                }

                return false;
            }

            private bool ExtendsFactAttribute(INamedTypeSymbol namedType)
            {
                Debug.Assert(_factAttribute is object);
                for (var current = namedType; current is object; current = current.BaseType)
                {
                    if (Equals(current, _factAttribute))
                        return true;
                }

                return false;
            }
        }
    }
}
