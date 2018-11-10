// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.ML.CodeAnalyzer.Tests.Helpers;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Reflection;
using Xunit;

namespace Microsoft.ML.InternalCodeAnalyzer.Tests
{
    public sealed class BestFriendTest : DiagnosticVerifier<BestFriendAnalyzer>
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
        public void BestFriend()
        {
            // The setup to this one is a bit more involved than many of the analyzer tests,
            // because in this case we have to actually set up *two* assemblies, where the
            // first considers the second a friend. But, setting up this dependency structure
            // so that things actually compile to the point where the analyzer can actually do
            // its work is rather involved.
            Solution solution = null;
            var projA = CreateProject("ProjectA", ref solution, SourceDeclaration.Value);
            var projB = CreateProject("ProjectB", ref solution, SourceUser.Value);
            solution = solution.AddProjectReference(projB.Id, new ProjectReference(projA.Id));

            var analyzer = new BestFriendAnalyzer();

            MetadataReference peRef;
            var refs = new[] {
                    RefFromType<object>(), RefFromType<Attribute>(),
                    MetadataReference.CreateFromFile(Assembly.Load("netstandard, Version=2.0.0.0").Location),
                    MetadataReference.CreateFromFile(Assembly.Load("System.Runtime, Version=0.0.0.0").Location)
                };
            using (var ms = new MemoryStream())
            {
                // We also test whether private protected can be accessed, so we need C# 7.2 at least.
                var parseOpts = new CSharpParseOptions(LanguageVersion.CSharp7_3);
                var tree = CSharpSyntaxTree.ParseText(SourceDeclaration.Value, parseOpts);
                var treeAttr = CSharpSyntaxTree.ParseText(SourceAttribute.Value, parseOpts);

                var compOpts = new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary);
                var innerComp = CSharpCompilation.Create(projA.Name, new[] { tree, treeAttr }, refs, compOpts);

                var emitResult = innerComp.Emit(ms);
                Assert.True(emitResult.Success, $"Compilation of {projA.Name} did not work. Diagnostics: {string.Join(" || ", emitResult.Diagnostics)}");

                var peImage = ms.ToArray().ToImmutableArray();
                peRef = MetadataReference.CreateFromImage(peImage);
            }

            var comp = projB.GetCompilationAsync().Result
                .WithReferences(refs.Append(peRef).ToArray());
            var compilationWithAnalyzers = comp.WithAnalyzers(ImmutableArray.Create((DiagnosticAnalyzer)analyzer));
            var allDiags = compilationWithAnalyzers.GetAnalyzerDiagnosticsAsync().Result;

            var projectTrees = new HashSet<SyntaxTree>(projB.Documents.Select(r => r.GetSyntaxTreeAsync().Result));
            var diags = allDiags
                .Where(d => d.Location == Location.None || d.Location.IsInMetadata || projectTrees.Contains(d.Location.SourceTree))
                .OrderBy(d => d.Location.SourceSpan.Start).ToArray();

            var diag = analyzer.SupportedDiagnostics[0];
            var expected = new DiagnosticResult[] {
                diag.CreateDiagnosticResult(10, 31, "A"),
                diag.CreateDiagnosticResult(11, 31, "A"),
                diag.CreateDiagnosticResult(11, 33, "My"),
                diag.CreateDiagnosticResult(14, 33, "Awhile"),
                diag.CreateDiagnosticResult(15, 33, "And"),
                diag.CreateDiagnosticResult(18, 13, "A"),
                diag.CreateDiagnosticResult(18, 25, "A"),
                diag.CreateDiagnosticResult(25, 13, "IA"),
                diag.CreateDiagnosticResult(25, 23, "IA"),
                diag.CreateDiagnosticResult(32, 38, ".ctor"),
                diag.CreateDiagnosticResult(38, 38, ".ctor"),
            };

            VerifyDiagnosticResults(diags, analyzer, expected);
        }
    }
}
