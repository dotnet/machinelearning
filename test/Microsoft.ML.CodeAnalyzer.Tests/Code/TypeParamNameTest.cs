// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Threading.Tasks;
using Microsoft.CodeAnalysis.Testing;
using Xunit;
using VerifyCS = Microsoft.ML.CodeAnalyzer.Tests.Helpers.CSharpCodeFixVerifier<
    Microsoft.ML.InternalCodeAnalyzer.TypeParamNameAnalyzer,
    Microsoft.CodeAnalysis.Testing.EmptyCodeFixProvider>;

namespace Microsoft.ML.InternalCodeAnalyzer.Tests
{
    public sealed class TypeParamNameTest
    {
        [Fact]
        public async Task TypeParamName()
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

            var expected = new DiagnosticResult[] {
                VerifyCS.Diagnostic().WithLocation(4, 26).WithArguments("hello"),
                VerifyCS.Diagnostic().WithLocation(6, 21).WithArguments("Tom"),
                VerifyCS.Diagnostic().WithLocation(8, 22).WithArguments("mytype"),
                VerifyCS.Diagnostic().WithLocation(10, 32).WithArguments("YourType"),
            };

            await VerifyCS.VerifyAnalyzerAsync(test, expected);
        }
    }
}
