// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.CodeAnalyzer.Tests.Helpers;
using Xunit;

namespace Microsoft.ML.CodeAnalyzer.Tests
{
    public sealed class ModifierTest : DiagnosticVerifier<ModifierAnalyzer>
    {
        [Fact]
        public void ExplicitAccessModifier()
        {
            var analyzer = GetCSharpDiagnosticAnalyzer();
            var diag = analyzer.SupportedDiagnostics[0];
            var newDiag = analyzer.SupportedDiagnostics[1];

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
                diag.CreateDiagnosticResult(53, 13, "this"),
                diag.CreateDiagnosticResult(61, 15, "ITest2"),
                diag.CreateDiagnosticResult(76, 17, "Method0"),
                newDiag.CreateDiagnosticResult(77, 24, "Method1"),
                newDiag.CreateDiagnosticResult(79, 36, "Method3"),
                diag.CreateDiagnosticResult(84, 23, "AwesomeDelegate"),
                diag.CreateDiagnosticResult(85, 31, "Ev"),
                diag.CreateDiagnosticResult(86, 31, "EvProp"),
            };

            VerifyCSharpDiagnostic(TestSource, expected);
        }

        private static string _testSource;
        internal static string TestSource => TestUtils.EnsureSourceLoaded(ref _testSource, "ModifierBeforeFix.cs");
    }

    public sealed class ModifierFixTest : CodeFixVerifier<ModifierAnalyzer, ModifierFixProvider>
    {
        [Fact]
        public void ExplicitAccessModifierFix()
        {
            VerifyCSharpFix(
                "namespace Bubba { class Foo {}}",
                "namespace Bubba { internal class Foo {}}");
            VerifyCSharpFix(ModifierTest.TestSource, ExpectedFix);
        }

        [Fact]
        public void IndexerModifierFix()
        {
            VerifyCSharpFix(
                "namespace Bubba { public class Foo { int this[int i] => i + 2} }",
                "namespace Bubba { public class Foo { private int this[int i] => i + 2} }");
        }

        private static string _expectedFix;
        private static string ExpectedFix => TestUtils.EnsureSourceLoaded(ref _expectedFix, "ModifierAfterFix.cs");
    }
}