// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Microsoft.ML.InternalCodeAnalyzer
{
    internal enum NameType
    {
        UnderScoreCamelCased, // E.g., _myPrivateField
        CamelCased,           // E.g., myAwesomeParameter
        PascalCased,          // E.g., AwesomeClass
        IPascalCased,         // E.g., IEnumerableStuff
        TPascalCased,         // E.g., TDictArg
    }

    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class NameAnalyzer : DiagnosticAnalyzer
    {
        internal const string Category = "Naming";

        internal const string NameProperty = "Original";
        internal const string DesiredNameProperty = "Desired";

        internal static Diagnostic CreateDiagnostic(DiagnosticDescriptor rule, SyntaxToken identifier, NameType desired, params object[] args)
        {
            string text = identifier.Text;
            var props = ImmutableDictionary<string, string>.Empty
                .Add(NameProperty, text).Add(DesiredNameProperty, desired.ToString());
            if (args == null || args.Length == 0)
                return Diagnostic.Create(rule, identifier.GetLocation(), props, text);
            var newArgs = new object[args.Length + 1];
            Array.Copy(args, 0, newArgs, 1, args.Length);
            newArgs[0] = text;
            return Diagnostic.Create(rule, identifier.GetLocation(), props, newArgs);
        }

        internal static class PrivateFieldName
        {
            public const string Id = "MSML_PrivateFieldName";
            private const string Title = "Private field name not _camelCased";
            private const string Format = "Private field name '{0}' not _camelCased";
            private const string Description =
                "Private fields should have an _ prefix and be _lowerCamelCased, unless they are const.";

            internal static DiagnosticDescriptor Rule =
                new DiagnosticDescriptor(Id, Title, Format, Category,
                    DiagnosticSeverity.Warning, isEnabledByDefault: true, description: Description);
        }

        internal static class GeneralName
        {
            public const string Id = "MSML_GeneralName";
            private const string Title = "This name should be PascalCased";
            private const string Format = "Identifier '{0}' not PascalCased";
            private const string Description =
                "Identifier names other than parameters, local variables, private non-const fields, interfaces, and type parameters should be PascalCased.";

            internal static DiagnosticDescriptor Rule =
                new DiagnosticDescriptor(Id, Title, Format, Category,
                    DiagnosticSeverity.Warning, isEnabledByDefault: true, description: Description);
        }

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
            ImmutableArray.Create(PrivateFieldName.Rule, GeneralName.Rule);

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.RegisterSyntaxNodeAction(AnalyzeField, SyntaxKind.FieldDeclaration);
            context.RegisterSyntaxNodeAction(AnalyzeField, SyntaxKind.EventFieldDeclaration);
            context.RegisterSyntaxNodeAction(AnalyzeClass, SyntaxKind.ClassDeclaration);
            context.RegisterSyntaxNodeAction(AnalyzeEnum, SyntaxKind.EnumDeclaration);
            context.RegisterSyntaxNodeAction(AnalyzeEnumMember, SyntaxKind.EnumMemberDeclaration);
            context.RegisterSyntaxNodeAction(AnalyzeEvent, SyntaxKind.EventDeclaration);
            context.RegisterSyntaxNodeAction(AnalyzeMethod, SyntaxKind.MethodDeclaration);
            context.RegisterSyntaxNodeAction(AnalyzeNamespace, SyntaxKind.NamespaceDeclaration);
            context.RegisterSyntaxNodeAction(AnalyzeProperty, SyntaxKind.PropertyDeclaration);
            context.RegisterSyntaxNodeAction(AnalyzeStruct, SyntaxKind.StructDeclaration);
        }

        private static void AnalyzeClass(SyntaxNodeAnalysisContext context)
            => CheckPascal(((ClassDeclarationSyntax)context.Node).Identifier, context);

        private static void AnalyzeEnum(SyntaxNodeAnalysisContext context)
            => CheckPascal(((EnumDeclarationSyntax)context.Node).Identifier, context);

        private static void AnalyzeEnumMember(SyntaxNodeAnalysisContext context)
            => CheckPascal(((EnumMemberDeclarationSyntax)context.Node).Identifier, context);

        private static void AnalyzeEvent(SyntaxNodeAnalysisContext context)
            => CheckPascal(((EventDeclarationSyntax)context.Node).Identifier, context);

        private static void AnalyzeMethod(SyntaxNodeAnalysisContext context)
        {
            var node = (MethodDeclarationSyntax)context.Node;
            if (ModifierContainsExtern(node.Modifiers))
                return;
            CheckPascal(node.Identifier, context);
        }

        private static void AnalyzeNamespace(SyntaxNodeAnalysisContext context)
        {
            var node = ((NamespaceDeclarationSyntax)context.Node);
            var name = node.Name;
            // This is annoying. I can't figure out how to do this. When I get a namespace,
            // this seems to trigger for *every* part of a namespace name, not just once for one.
            //foreach (var id in name.DescendantTokens().Where(tok => tok.IsKind(SyntaxKind.IdentifierToken)))
            //    CheckPascal(id, context);
        }

        private static void AnalyzeProperty(SyntaxNodeAnalysisContext context)
            => CheckPascal(((PropertyDeclarationSyntax)context.Node).Identifier, context);

        private static void AnalyzeStruct(SyntaxNodeAnalysisContext context)
        {
            CheckPascal(((StructDeclarationSyntax)context.Node).Identifier, context);
        }

        private static bool ModifierContainsExtern(SyntaxTokenList modifiers)
        {
            return modifiers.Any(token => token.IsKind(SyntaxKind.ExternKeyword));
        }

        private static void CheckPascal(SyntaxToken token, SyntaxNodeAnalysisContext context)
        {
            if (!Utils.NameIsGood(token.Text, 0, true))
                context.ReportDiagnostic(CreateDiagnostic(GeneralName.Rule, token, NameType.PascalCased));
        }

        private static bool CheckUnderscore(string name)
            => !string.IsNullOrEmpty(name) && name.StartsWith("_", StringComparison.OrdinalIgnoreCase) && Utils.NameIsGood(name, 1, false);

        private static void AnalyzeField(SyntaxNodeAnalysisContext context)
        {
            var node = (BaseFieldDeclarationSyntax)context.Node;

            bool isConst = false;
            bool isPrivate = true; // Fields are private by default.
            foreach (var mod in node.Modifiers)
            {
                if (mod.IsKind(SyntaxKind.ConstKeyword))
                    isConst = true;
                else if (mod.IsKind(SyntaxKind.PublicKeyword) || mod.IsKind(SyntaxKind.ProtectedKeyword) || mod.IsKind(SyntaxKind.InternalKeyword))
                    isPrivate = false;
            }
            foreach (var variable in node.Declaration.Variables)
            {
                var identifier = variable.Identifier;
                var name = identifier.Text;
                if (!isPrivate)
                {
                    CheckPascal(identifier, context);
                    continue;
                }

                // Private consts are a little bit funny. Sometimes it makes sense to have them
                // be _camelCased, but often it's good to have them be PascalCased. We have decided
                // that going forward they will be standardized as PascalCased, but *at the moment*
                // we do not diagnose it as an error if they are _camelCased. At some point we will.
                if (CheckUnderscore(name) || (isConst && Utils.NameIsGood(name, 0, true)))
                    continue;
                var diagnostic = Diagnostic.Create(PrivateFieldName.Rule, identifier.GetLocation(), name);
                context.ReportDiagnostic(CreateDiagnostic(PrivateFieldName.Rule, identifier,
                    isConst ? NameType.PascalCased : NameType.UnderScoreCamelCased));
            }
        }
    }
}
