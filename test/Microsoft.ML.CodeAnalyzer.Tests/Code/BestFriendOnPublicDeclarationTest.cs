// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Reflection;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.ML.CodeAnalyzer.Tests.Helpers;
using Xunit;

namespace Microsoft.ML.InternalCodeAnalyzer.Tests
{
    public sealed class BestFriendOnPublicDeclarationTest : DiagnosticVerifier<BestFriendOnPublicDeclarationsAnalyzer>
    {
        private readonly Lazy<string> SourceAttribute = TestUtils.LazySource("BestFriendAttribute.cs");
        private readonly Lazy<string> SourceDeclaration = TestUtils.LazySource("BestFriendOnPublicDeclaration.cs");

        [Fact]
        public void BestFriendOnPublicDeclaration()
        {
            Solution solution = null;
            var projA = CreateProject("ProjectA", ref solution, SourceDeclaration.Value, SourceAttribute.Value);

            var analyzer = new BestFriendOnPublicDeclarationsAnalyzer();

            var refs = new List<MetadataReference> {
                RefFromType<object>(), RefFromType<Attribute>(),
                MetadataReference.CreateFromFile(Assembly.Load("netstandard, Version=2.0.0.0").Location),
                MetadataReference.CreateFromFile(Assembly.Load("System.Runtime, Version=0.0.0.0").Location)
            };

            var comp = projA.GetCompilationAsync().Result.WithReferences(refs.ToArray());
            var compilationWithAnalyzers = comp.WithAnalyzers(ImmutableArray.Create((DiagnosticAnalyzer)analyzer));
            var allDiags = compilationWithAnalyzers.GetAnalyzerDiagnosticsAsync().Result;

            var projectTrees = new HashSet<SyntaxTree>(projA.Documents.Select(r => r.GetSyntaxTreeAsync().Result));
            var diags = allDiags
                .Where(d => d.Location == Location.None || d.Location.IsInMetadata || projectTrees.Contains(d.Location.SourceTree))
                .OrderBy(d => d.Location.SourceSpan.Start).ToArray();

            var diag = analyzer.SupportedDiagnostics[0];
            var expected = new DiagnosticResult[] {
                diag.CreateDiagnosticResult(8, 6, "PublicClass"),
                diag.CreateDiagnosticResult(11, 10, "PublicField"),
                diag.CreateDiagnosticResult(14, 10, "PublicProperty"),
                diag.CreateDiagnosticResult(20, 10, "PublicMethod"),
                diag.CreateDiagnosticResult(26, 10, "PublicDelegate"),
                diag.CreateDiagnosticResult(29, 10, "PublicClass"),
                diag.CreateDiagnosticResult(35, 6, "PublicStruct"),
                diag.CreateDiagnosticResult(40, 6, "PublicEnum"),
                diag.CreateDiagnosticResult(47, 6, "PublicInterface"),
                diag.CreateDiagnosticResult(102, 10, "PublicMethod")
            };

            VerifyDiagnosticResults(diags, analyzer, expected);
        }
    }
}

