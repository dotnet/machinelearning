// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.CodeAnalyzer.Tests.Helpers;
using Microsoft.ML.InternalCodeAnalyzer;
using Xunit;

namespace Microsoft.ML.CodeAnalyzer.Tests.Code
{
    public sealed class NameTest : DiagnosticVerifier<NameAnalyzer>
    {
        [Fact]
        public void PrivateFieldName()
        {
            var analyzer = GetCSharpDiagnosticAnalyzer();
            var diag = analyzer.SupportedDiagnostics[0];

            var expected = new DiagnosticResult[] {
                diag.CreateDiagnosticResult(5, 21, "foo"),
                diag.CreateDiagnosticResult(7, 24, "_Bubba"),
                diag.CreateDiagnosticResult(8, 22, "_shouldParseHTML"),
                diag.CreateDiagnosticResult(11, 23, "BillyClub"),
                diag.CreateDiagnosticResult(13, 30, "bob"),
                diag.CreateDiagnosticResult(14, 30, "CHAZ"),
                diag.CreateDiagnosticResult(17, 21, "_liveFromNYC"),
                diag.CreateDiagnosticResult(19, 28, "nice"),
            };

            VerifyCSharpDiagnostic(PrivateTestSource, expected);
        }

        internal const string PrivateTestSource = @"
namespace TestNamespace
{
    class TypeName
    {
        private int foo;
        private int _bar;
        private string _Bubba;
        private bool _shouldParseHTML;
        private string _who2Call;
        float _burgers4babies;
        private float BillyClub;
        private const string Alice = ""Hello"";
        private const string bob = ""Hello"";
        private const string CHAZ = ""Hello"";
        private const string DEBora = ""Hello"";
        private const string _yuck = ""Hello"";
        private int _liveFromNYC;
        private int _liveFromNYCity;
        private static int nice;
    }
}";

        [Fact]
        public void MoreNameTests()
        {
            var analyzer = GetCSharpDiagnosticAnalyzer();
            var diagP = analyzer.SupportedDiagnostics[0];
            var diagG = analyzer.SupportedDiagnostics[1];

            var expected = new DiagnosticResult[] {
                diagG.CreateDiagnosticResult(6, 11, "CLASS"),
                diagG.CreateDiagnosticResult(8, 20, "alice"),
                diagG.CreateDiagnosticResult(9, 21, "_bob"),
                diagG.CreateDiagnosticResult(10, 22, "_chaz"),
                diagG.CreateDiagnosticResult(11, 30, "emily"),
                diagG.CreateDiagnosticResult(11, 37, "_francis"),
                diagG.CreateDiagnosticResult(16, 21, "this_is_not_python"),
                diagG.CreateDiagnosticResult(17, 21, "thisIsNotJava"),
                diagP.CreateDiagnosticResult(21, 30, "BadEvent"),
                diagG.CreateDiagnosticResult(22, 29, "bad_event"),
                diagG.CreateDiagnosticResult(25, 30, "_badEv"),
                diagG.CreateDiagnosticResult(27, 29, "one"),
                diagG.CreateDiagnosticResult(27, 39, "three"),
                diagG.CreateDiagnosticResult(28, 22, "enumb"),
                diagG.CreateDiagnosticResult(28, 35, "Two_Two"),
                diagG.CreateDiagnosticResult(28, 44, "_three"),
                diagG.CreateDiagnosticResult(30, 25, "_m2"),
                diagG.CreateDiagnosticResult(37, 12, "marco"),
                diagG.CreateDiagnosticResult(37, 31, "polo"),
            };

            VerifyCSharpDiagnostic(TestSource, expected);
        }

        internal const string TestSource = @"
using System;
namespace silly { }
namespace NotSilly { }
namespace foo.bar.Biz
{
    class CLASS
    {
        public int alice { get; }
        private int _bob { get; }
        internal int _chaz;
        internal int Debora, emily, _francis;
        int _george;

        CLASS() {  }

        public void this_is_not_python() { }
        public void thisIsNotJava() { }
        public void ThisIsCSharp() { }

        private event Action _goodEvent;
        private event Action BadEvent;
        public event Action bad_event;
        public event Action GoodEvent;
        private event Action GoodEv { add { } remove { } }
        private event Action _badEv { add { } remove { } }

        public enum EnumA { one, Two, three }
        private enum enumb { One, Two_Two, _three }

        protected float _m2;
        protected float M4;
    }

    class A { }
    class BeClass { }

    struct marco { public int polo; }
}";
        [Fact]
        public void ExternName()
        {
            var analyzer = GetCSharpDiagnosticAnalyzer();
            var diagP = analyzer.SupportedDiagnostics[0];
            var diagG = analyzer.SupportedDiagnostics[1];

            const string source = @"
using System;
using System.Runtime.InteropServices;

namespace TestNamespace
{
    class CLASS
    {

        [DllImport(""kernel32.dll"")]
        public static extern IntPtr who_run_bartertown(string libraryPath);

        public void masterBlaster() {}
    }
}
";

            var expected = new DiagnosticResult[] {
                diagG.CreateDiagnosticResult(6, 11, "CLASS"),
                diagG.CreateDiagnosticResult(12, 21, "masterBlaster"),
            };

            VerifyCSharpDiagnostic(source, expected);
        }
    }

    public sealed class NameFixTest : CodeFixVerifier<NameAnalyzer, NameFixProvider>
    {
        [Fact]
        public void NameFix()
        {
            VerifyCSharpFix(NameTest.TestSource, FixedTestSource);
        }

        private const string FixedTestSource = @"using System;
namespace silly { }
namespace NotSilly { }
namespace foo.bar.Biz
{
    class Class
    {
        public int Alice { get; }
        private int Bob { get; }
        internal int Chaz;
        internal int Debora, Emily, Francis;
        int _george;

        Class() {  }

        public void ThisIsNotPython() { }
        public void ThisIsNotJava() { }
        public void ThisIsCSharp() { }

        private event Action _goodEvent;
        private event Action _badEvent;
        public event Action BadEvent;
        public event Action GoodEvent;
        private event Action GoodEv { add { } remove { } }
        private event Action BadEv { add { } remove { } }

        public enum EnumA { One, Two, Three }
        private enum Enumb { One, TwoTwo, Three }

        protected float M2;
        protected float M4;
    }

    class A { }
    class BeClass { }

    struct Marco { public int Polo; }
}";
    }
}