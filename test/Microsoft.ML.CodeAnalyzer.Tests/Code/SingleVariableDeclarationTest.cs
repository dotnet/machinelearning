// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Threading.Tasks;
using Microsoft.CodeAnalysis.Testing;
using Xunit;
using VerifyCS = Microsoft.ML.CodeAnalyzer.Tests.Helpers.CSharpCodeFixVerifier<
    Microsoft.ML.InternalCodeAnalyzer.SingleVariableDeclarationAnalyzer,
    Microsoft.CodeAnalysis.Testing.EmptyCodeFixProvider>;

namespace Microsoft.ML.InternalCodeAnalyzer.Tests
{
    public sealed class SingleVariableDeclarationTest
    {
        [Fact]
        public async Task SingleVariableDeclaration()
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

            var expected = new DiagnosticResult[] {
                VerifyCS.Diagnostic().WithLocation(6, 9).WithArguments("a', 'b', 'c"),
                VerifyCS.Diagnostic().WithLocation(8, 9).WithArguments("e', 'f"),
                VerifyCS.Diagnostic().WithLocation(17, 17).WithArguments("l', 'm"),
            };

            await VerifyCS.VerifyAnalyzerAsync(test, expected);
        }
    }
}
