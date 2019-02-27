// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Threading.Tasks;
using Microsoft.CodeAnalysis.Testing;
using Xunit;
using VerifyCS = Microsoft.ML.CodeAnalyzer.Tests.Helpers.CSharpCodeFixVerifier<
    Microsoft.ML.InternalCodeAnalyzer.ParameterVariableNameAnalyzer,
    Microsoft.CodeAnalysis.Testing.EmptyCodeFixProvider>;

namespace Microsoft.ML.InternalCodeAnalyzer.Tests
{
    public sealed class ParameterVariableNameTest
    {
        [Fact]
        public async Task ParameterVariableName()
        {
            const string test = @"
using System.Linq;
namespace TestNamespace
{
    public class A
    {
        public int Albatross, Buttermilk, Coffee;
        public A(int Unlimited, int POWER)
        {
            int foo = -POWER, Tada = Unlimited + POWER;
            string FORMAT = $""{Unlimited} + {POWER}"";
            int _coolSum = FORMAT.ToCharArray().Sum(CHAR => CHAR + POWER + Buttermilk++);
            Albatross = -_coolSum;
        }

        public static void B(string Hello, int goodbye, string HelloAgain)
        {
            int i_think_this_is_python = Hello.Length + goodbye + HelloAgain.Length;
        }
    }
}";

            const string param = "parameter";
            const string local = "local variable";

            var expected = new DiagnosticResult[] {
                VerifyCS.Diagnostic().WithLocation(8, 22).WithArguments("Unlimited", param),
                VerifyCS.Diagnostic().WithLocation(8, 37).WithArguments("POWER", param),
                VerifyCS.Diagnostic().WithLocation(10, 31).WithArguments("Tada", local),
                VerifyCS.Diagnostic().WithLocation(11, 20).WithArguments("FORMAT", local),
                VerifyCS.Diagnostic().WithLocation(12, 17).WithArguments("_coolSum", local),
                VerifyCS.Diagnostic().WithLocation(12, 53).WithArguments("CHAR", param),
                VerifyCS.Diagnostic().WithLocation(16, 37).WithArguments("Hello", param),
                VerifyCS.Diagnostic().WithLocation(16, 64).WithArguments("HelloAgain", param),
                VerifyCS.Diagnostic().WithLocation(18, 17).WithArguments("i_think_this_is_python", local),
            };

            await VerifyCS.VerifyAnalyzerAsync(test, expected);
        }
    }
}
