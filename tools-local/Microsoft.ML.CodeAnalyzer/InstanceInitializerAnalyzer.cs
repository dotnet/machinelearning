// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Immutable;
using System.Linq;
using System.Reflection;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Microsoft.ML.CodeAnalyzer
{
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class InstanceInitializerAnalyzer : DiagnosticAnalyzer
    {
        private const string Category = "Declaration";
        public const string DiagnosticId = "MSML_NoInstanceInitializers";

        private const string Title = "No initializers on instance fields or properties";
        private const string Format = "Member {0} has a {1} initialier outside the constructor";
        private const string Description =
            "All instance fields or properties should be initialized in a constructor, not in the field";

        private static DiagnosticDescriptor Rule =
            new DiagnosticDescriptor(DiagnosticId, Title, Format, Category,
                DiagnosticSeverity.Warning, isEnabledByDefault: true, description: Description);

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
            ImmutableArray.Create(Rule);

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.RegisterSymbolAction(AnalyzeField, SymbolKind.Field);
            context.RegisterSymbolAction(AnalyzeProperty, SymbolKind.Property);
        }

        private static void AnalyzeField(SymbolAnalysisContext context)
        {
            var symbol = (IFieldSymbol)context.Symbol;
            // Constant or static field initializers are desirable. If implicitly
            // declared, then we can't very well ask the developer to fix.
            if (symbol.IsConst || symbol.IsStatic || symbol.IsImplicitlyDeclared)
                return;
            // Exempt argument attributes from the test. Note that because we cannot
            // depend on the Microsoft.ML source itself, we have to identify this class by name.
            if (symbol.GetAttributes().Any(i => i.AttributeClass.Name == "ArgumentAttribute"))
                return;

            var typeInfo = symbol.GetType().GetTypeInfo();
            var hasInitProp = typeInfo.GetDeclaredProperty("HasInitializer");
            if (hasInitProp?.PropertyType != typeof(bool))
                return;
            bool hasInit = (bool)hasInitProp.GetValue(symbol);
            if (!hasInit)
                return;
            var diagnostic = Diagnostic.Create(Rule, symbol.Locations[0], symbol.Name, "field");
            context.ReportDiagnostic(diagnostic);
        }

        private static void AnalyzeProperty(SymbolAnalysisContext context)
        {
            var symbol = (IPropertySymbol)context.Symbol;
            if (symbol.IsAbstract || symbol.IsImplicitlyDeclared || symbol.IsStatic)
                return;
            var syntaxRefs = symbol.DeclaringSyntaxReferences;
            if (syntaxRefs.IsEmpty)
                return;
            var syntax = syntaxRefs[0].GetSyntax();
            if (!syntax.ChildNodes().Any(s => s.IsKind(SyntaxKind.EqualsValueClause)))
                return;

            var diagnostic = Diagnostic.Create(Rule, symbol.Locations[0], symbol.Name, "property");
            context.ReportDiagnostic(diagnostic);
        }
    }
}
