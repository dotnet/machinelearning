// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.CodeAnalyzer.Tests.Helpers;
using Xunit;

namespace Microsoft.ML.CodeAnalyzer.Tests.Code
{
    public sealed class InstanceInitializerTest : DiagnosticVerifier<InstanceInitializerAnalyzer>
    {
        [Fact]
        public void InstanceInitializer()
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
        public string Foo { get; set => _fooBacking = value; }
        public string Bar { get; } = ""Hello"";
        public static string Bizz { get; } = ""Nice"";
    }
}";

            var analyzer = GetCSharpDiagnosticAnalyzer();
            var diag = analyzer.SupportedDiagnostics[0];

            var expected = new DiagnosticResult[] {
                diag.CreateDiagnosticResult(5, 21, "_foo", "field"),
                diag.CreateDiagnosticResult(9, 32, "_blorg", "field"),
                diag.CreateDiagnosticResult(12, 23, "Bar", "property"),
            };

            VerifyCSharpDiagnostic(test, expected);
        }
    }
}
