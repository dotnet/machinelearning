// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CodeFixes;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Testing;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Testing;
using Microsoft.CodeAnalysis.Testing.Verifiers;

namespace Microsoft.ML.CodeAnalyzer.Tests.Helpers
{
    internal static class CSharpCodeFixVerifier<TAnalyzer, TCodeFix>
        where TAnalyzer : DiagnosticAnalyzer, new()
        where TCodeFix : CodeFixProvider, new()
    {
        public static DiagnosticResult Diagnostic()
                => CSharpCodeFixVerifier<TAnalyzer, TCodeFix, XUnitVerifier>.Diagnostic();

        public static DiagnosticResult Diagnostic(string diagnosticId)
            => CSharpCodeFixVerifier<TAnalyzer, TCodeFix, XUnitVerifier>.Diagnostic(diagnosticId);

        public static DiagnosticResult Diagnostic(DiagnosticDescriptor descriptor)
            => CSharpCodeFixVerifier<TAnalyzer, TCodeFix, XUnitVerifier>.Diagnostic(descriptor);

        public static async Task VerifyAnalyzerAsync(string source, params DiagnosticResult[] expected)
        {
            var test = new Test
            {
                TestCode = source,
            };

            test.ExpectedDiagnostics.AddRange(expected);
            await test.RunAsync();
        }

        public static Task VerifyCodeFixAsync(string source, string fixedSource)
            => VerifyCodeFixAsync(source, DiagnosticResult.EmptyDiagnosticResults, fixedSource);

        public static Task VerifyCodeFixAsync(string source, DiagnosticResult expected, string fixedSource)
            => VerifyCodeFixAsync(source, new[] { expected }, fixedSource);

        public static async Task VerifyCodeFixAsync(string source, DiagnosticResult[] expected, string fixedSource)
        {
            var test = new Test
            {
                TestCode = source,
                FixedCode = fixedSource,
            };

            test.ExpectedDiagnostics.AddRange(expected);
            await test.RunAsync();
        }

        internal class Test : CSharpCodeFixTest<TAnalyzer, TCodeFix, XUnitVerifier>
        {
            public Test()
            {
                TestState.AdditionalReferences.Add(AdditionalMetadataReferences.StandardReference);
                TestState.AdditionalReferences.Add(AdditionalMetadataReferences.RuntimeReference);
                TestState.AdditionalReferences.Add(AdditionalMetadataReferences.CSharpSymbolsReference);
                TestState.AdditionalReferences.Add(AdditionalMetadataReferences.MSDataDataViewReference);
                TestState.AdditionalReferences.Add(AdditionalMetadataReferences.MLNetCoreReference);
                TestState.AdditionalReferences.Add(AdditionalMetadataReferences.MLNetDataReference);
                TestState.AdditionalReferences.Add(AdditionalMetadataReferences.MLNetStaticPipeReference);

                SolutionTransforms.Add((solution, projectId) =>
                {
                    if (LanguageVersion != null)
                    {
                        var parseOptions = (CSharpParseOptions)solution.GetProject(projectId).ParseOptions;
                        solution = solution.WithProjectParseOptions(projectId, parseOptions.WithLanguageVersion(LanguageVersion.Value));
                    }

                    return solution;
                });
            }

            public LanguageVersion? LanguageVersion { get; set; }
        }
    }
}
