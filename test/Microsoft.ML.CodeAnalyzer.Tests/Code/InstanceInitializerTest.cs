// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Threading.Tasks;
using Microsoft.CodeAnalysis.Testing;
using Xunit;
using VerifyCS = Microsoft.ML.CodeAnalyzer.Tests.Helpers.CSharpCodeFixVerifier<
    Microsoft.ML.InternalCodeAnalyzer.InstanceInitializerAnalyzer,
    Microsoft.CodeAnalysis.Testing.EmptyCodeFixProvider>;

namespace Microsoft.ML.InternalCodeAnalyzer.Tests
{
    public sealed class InstanceInitializerTest
    {
        [Fact]
        public async Task InstanceInitializer()
        {
            const string test = @"
namespace TestNamespace
{
    class TypeName
    {
        private int _foo = 5;
        private int _bar;
        private const int _bizz = 2;
        private static int _muck = 4;
        private readonly float _blorg = 3.0f;
        private string _fooBacking;
        public string Foo { get => _fooBacking; set => _fooBacking = value; }
        public string Bar { get; } = ""Hello"";
        public static string Bizz { get; } = ""Nice"";
    }
}";

            var expected = new DiagnosticResult[] {
                VerifyCS.Diagnostic().WithLocation(6, 21).WithArguments("_foo", "field"),
                VerifyCS.Diagnostic().WithLocation(10, 32).WithArguments("_blorg", "field"),
                VerifyCS.Diagnostic().WithLocation(13, 23).WithArguments("Bar", "property"),
            };

            await VerifyCS.VerifyAnalyzerAsync(test, expected);
        }
    }
}
