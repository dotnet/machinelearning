// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.CodeAnalyzer.Tests.Helpers;
using Microsoft.ML.InternalCodeAnalyzer;
using Xunit;

namespace Microsoft.ML.CodeAnalyzer.Tests.Code
{
    public sealed class TypeParamNameTest : DiagnosticVerifier<TypeParamNameAnalyzer>
    {
        [Fact]
        public void TypeParamName()
        {
            const string test = @"
namespace TestNamespace
{
    interface IAlice<T1, hello> {}
    interface IBob<T> : IAlice<T, int> {}
    interface IChaz<Tom> : IAlice<IBob<Tom>, Tom> {}

    public class Foo<mytype>
    {
        public static void Bar<YourType, TArg>() {}
    }
}";
            var analyzer = GetCSharpDiagnosticAnalyzer();
            var diag = analyzer.SupportedDiagnostics[0];

            var expected = new DiagnosticResult[] {
                diag.CreateDiagnosticResult(3, 26, "hello"),
                diag.CreateDiagnosticResult(5, 21, "Tom"),
                diag.CreateDiagnosticResult(7, 22, "mytype"),
                diag.CreateDiagnosticResult(9, 32, "YourType"),
            };

            VerifyCSharpDiagnostic(test, expected);
        }
    }
}
