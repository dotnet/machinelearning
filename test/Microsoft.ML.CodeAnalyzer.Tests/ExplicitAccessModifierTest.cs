// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.CodeAnalyzer.Tests.Helpers;
using Xunit;

namespace Microsoft.ML.CodeAnalyzer.Tests
{
    public sealed class ExplicitAccessModifierTest : DiagnosticVerifier<ExplicitAccessModifierAnalyzer>
    {
        [Fact]
        public void ExplicitAccessModifier()
        {
            var analyzer = GetCSharpDiagnosticAnalyzer();
            var diag = analyzer.SupportedDiagnostics[0];

            var expected = new DiagnosticResult[] {
                diag.CreateDiagnosticResult(5, 11, "TypeName"),
                diag.CreateDiagnosticResult(8, 13, "Foo"),
                diag.CreateDiagnosticResult(12, 27, "Hi"),
                diag.CreateDiagnosticResult(19, 9, "TypeName"),
                diag.CreateDiagnosticResult(30, 26, "A"),
                diag.CreateDiagnosticResult(32, 17, "C"),
                diag.CreateDiagnosticResult(33, 34, "_d"),
                diag.CreateDiagnosticResult(36, 40, "Yo"),
                diag.CreateDiagnosticResult(38, 80, "Dog"),
                diag.CreateDiagnosticResult(40, 14, "Enum1"),
                diag.CreateDiagnosticResult(46, 14, "Silly"),
                diag.CreateDiagnosticResult(51, 34, "-"),
                diag.CreateDiagnosticResult(59, 15, "ITest2"),
            };

            VerifyCSharpDiagnostic(TestSource, expected);
        }

        private static string _testSource;
        internal static string TestSource => TestUtils.EnsureSourceLoaded(ref _testSource, "ExplicitAccessBeforeFix.cs");
    }

    public sealed class ExplicitAccessModifierFixTest : CodeFixVerifier<ExplicitAccessModifierAnalyzer, ExplicitAccessModifierFixProvider>
    {
        [Fact]
        public void ExplicitAccessModifierFix()
        {
            VerifyCSharpFix("namespace Bubba { class Foo {}}", "namespace Bubba { internal class Foo {}}");
            VerifyCSharpFix(ExplicitAccessModifierTest.TestSource, ExpectedFix);
        }

        private static string _expectedFix;
        private static string ExpectedFix => TestUtils.EnsureSourceLoaded(ref _expectedFix, "ExplicitAccessAfterFix.cs");
    }
}