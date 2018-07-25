// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.CodeAnalyzer.Tests.Helpers;
using Xunit;

namespace Microsoft.ML.CodeAnalyzer.Tests
{
    public sealed class ParameterVariableNameTest : DiagnosticVerifier<ParameterVariableNameAnalyzer>
    {
        [Fact]
        public void ParameterVariableName()
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
            var analyzer = GetCSharpDiagnosticAnalyzer();
            var diag = analyzer.SupportedDiagnostics[0];

            const string param = "parameter";
            const string local = "local variable";

            var expected = new DiagnosticResult[] {
                diag.CreateDiagnosticResult(7, 22, "Unlimited", param),
                diag.CreateDiagnosticResult(7, 37, "POWER", param),
                diag.CreateDiagnosticResult(9, 31, "Tada", local),
                diag.CreateDiagnosticResult(10, 20, "FORMAT", local),
                diag.CreateDiagnosticResult(11, 17, "_coolSum", local),
                diag.CreateDiagnosticResult(11, 53, "CHAR", param),
                diag.CreateDiagnosticResult(15, 37, "Hello", param),
                diag.CreateDiagnosticResult(15, 64, "HelloAgain", param),
                diag.CreateDiagnosticResult(17, 17, "i_think_this_is_python", local),
            };

            VerifyCSharpDiagnostic(test, expected);
        }
    }
}
