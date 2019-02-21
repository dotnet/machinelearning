// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Testing;
using Microsoft.ML.CodeAnalyzer.Tests.Helpers;
using Xunit;
using VerifyCS = Microsoft.ML.CodeAnalyzer.Tests.Helpers.CSharpCodeFixVerifier<
    Microsoft.ML.InternalCodeAnalyzer.BestFriendAnalyzer,
    Microsoft.CodeAnalysis.Testing.EmptyCodeFixProvider>;

namespace Microsoft.ML.InternalCodeAnalyzer.Tests
{
    public sealed class BestFriendTest
    {
        // We do things in this somewhat odd way rather than just referencing the Core assembly directly,
        // because we certainly want the best friend attribute itself to be internal, but the assembly
        // we build dynamically as part of the test cannot be signed, and so cannot itself be a friend
        // of the core assembly (even if we were in a mood to pollute the core assembly with friend
        // declarations to enable this one test). We instead compile the same source, as part of this
        // dummy assembly. The type name will be the same so the same analyzer will work.
        private readonly Lazy<string> SourceAttribute = TestUtils.LazySource("BestFriendAttribute.cs");
        private readonly Lazy<string> SourceDeclaration = TestUtils.LazySource("BestFriendDeclaration.cs");
        private readonly Lazy<string> SourceUser = TestUtils.LazySource("BestFriendUser.cs");

        [Fact]
        public async Task BestFriend()
        {
            // The setup to this one is a bit more involved than many of the analyzer tests,
            // because in this case we have to actually set up *two* assemblies, where the
            // first considers the second a friend. But, setting up this dependency structure
            // so that things actually compile to the point where the analyzer can actually do
            // its work is rather involved.

            var expected = new DiagnosticResult[] {
                VerifyCS.Diagnostic().WithLocation(10, 31).WithArguments("A"),
                VerifyCS.Diagnostic().WithLocation(11, 31).WithArguments("A"),
                VerifyCS.Diagnostic().WithLocation(11, 33).WithArguments("My"),
                VerifyCS.Diagnostic().WithLocation(14, 33).WithArguments("Awhile"),
                VerifyCS.Diagnostic().WithLocation(15, 33).WithArguments("And"),
                VerifyCS.Diagnostic().WithLocation(18, 13).WithArguments("A"),
                VerifyCS.Diagnostic().WithLocation(18, 25).WithArguments("A"),
                new DiagnosticResult("CS0122", DiagnosticSeverity.Error).WithLocation(23, 21).WithMessage("'D.D(float)' is inaccessible due to its protection level"),
                VerifyCS.Diagnostic().WithLocation(25, 13).WithArguments("IA"),
                VerifyCS.Diagnostic().WithLocation(25, 23).WithArguments("IA"),
                VerifyCS.Diagnostic().WithLocation(32, 38).WithArguments(".ctor"),
                VerifyCS.Diagnostic().WithLocation(38, 38).WithArguments(".ctor"),
            };

            VerifyCS.Test test = null;
            test = new VerifyCS.Test
            {
                LanguageVersion = LanguageVersion.CSharp7_2,
                TestState =
                {
                    Sources = { SourceUser.Value },
                    AdditionalReferences = { MetadataReference.CreateFromFile(typeof(Console).Assembly.Location) },
                },
                SolutionTransforms =
                {
                    (solution, projectId) =>
                    {
                        var projectA = solution.AddProject("ProjectA", "ProjectA", LanguageNames.CSharp);
                        projectA = projectA.AddDocument("BestFriendAttribute.cs", SourceAttribute.Value).Project;
                        projectA = projectA.AddDocument("BestFriendDeclaration.cs", SourceDeclaration.Value).Project;
                        projectA = projectA.WithParseOptions(((CSharpParseOptions)projectA.ParseOptions).WithLanguageVersion(LanguageVersion.CSharp7_2));
                        projectA = projectA.WithCompilationOptions(projectA.CompilationOptions.WithOutputKind(OutputKind.DynamicallyLinkedLibrary));
                        projectA = projectA.WithMetadataReferences(solution.GetProject(projectId).MetadataReferences.Concat(test.TestState.AdditionalReferences));
                        solution = projectA.Solution;

                        solution = solution.AddProjectReference(projectId, new ProjectReference(projectA.Id));
                        solution = solution.WithProjectAssemblyName(projectId, "ProjectB");

                        return solution;
                    },
                },
            };

            // Remove these assemblies, or an additional definition of BestFriendAttribute could exist and the
            // compilation will not be able to locate a single definition for use in the analyzer.
            test.TestState.AdditionalReferences.Remove(AdditionalMetadataReferences.MLNetCoreReference);
            test.TestState.AdditionalReferences.Remove(AdditionalMetadataReferences.MLNetDataReference);
            test.TestState.AdditionalReferences.Remove(AdditionalMetadataReferences.MLNetStaticPipeReference);

            test.Exclusions &= ~AnalysisExclusions.GeneratedCode;
            test.ExpectedDiagnostics.AddRange(expected);
            await test.RunAsync();
        }
    }
}
