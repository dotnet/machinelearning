// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.CodeAnalyzer.Tests.Helpers;
using Xunit;

namespace Microsoft.ML.CodeAnalyzer.Tests
{
    public sealed class WhitespaceTest : DiagnosticVerifier<WhitespaceAnalyzer>
    {
        [Fact]
        public void Whitespace()
        {
            const string test = @"
using System; 

namespace TestNamespace
{
    class TypeName
    {  
        int foo = 5;

        int bar = 2;
  

        Console.WriteLine(foo + bar);

    }
}";
            var analyzer = GetCSharpDiagnosticAnalyzer();
            var diags = analyzer.SupportedDiagnostics;
            var multilineDiag = diags[0];
            var trailingDiag = diags[1];

            var expected = new DiagnosticResult[] {
                trailingDiag.CreateDiagnosticResult(1, 14),
                trailingDiag.CreateDiagnosticResult(6, 6),
                trailingDiag.CreateDiagnosticResult(10, 11),
                multilineDiag.CreateDiagnosticResult(11, 1),
            };

            VerifyCSharpDiagnostic(test, expected);
        }
    }
}
