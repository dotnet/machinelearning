// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.CodeAnalyzer.Tests.Helpers;
using Microsoft.ML.InternalCodeAnalyzer;
using Xunit;

namespace Microsoft.ML.InternalCodeAnalyzer.Tests
{
    public sealed class SingleVariableDeclarationTest : DiagnosticVerifier<SingleVariableDeclarationAnalyzer>
    {
        [Fact]
        public void SingleVariableDeclaration()
        {
            const string test = @"
namespace TestNamespace
{
    class TypeName
    {
        int a, b, c;
        int d;
        int e, f;

        public TypeName(int g, int h)
        {
            a = b = g;
            c = d = h;
            int i = 2;
            for (int j = 0, k = i; j < k; ++j)
            {
                int l = j, m = k;
            }
        }
    }
}";

            var analyzer = GetCSharpDiagnosticAnalyzer();
            var diag = analyzer.SupportedDiagnostics[0];

            var expected = new DiagnosticResult[] {
                diag.CreateDiagnosticResult(5, 9, "a', 'b', 'c"),
                diag.CreateDiagnosticResult(7, 9, "e', 'f"),
                diag.CreateDiagnosticResult(16, 17, "l', 'm"),
            };

            VerifyCSharpDiagnostic(test, expected);
        }
    }
}
