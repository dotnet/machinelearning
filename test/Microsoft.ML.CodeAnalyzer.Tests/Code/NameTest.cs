// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Threading.Tasks;
using Microsoft.CodeAnalysis.Testing;
using Xunit;
using VerifyCS = Microsoft.ML.CodeAnalyzer.Tests.Helpers.CSharpCodeFixVerifier<
    Microsoft.ML.InternalCodeAnalyzer.NameAnalyzer,
    Microsoft.ML.InternalCodeAnalyzer.NameFixProvider>;

namespace Microsoft.ML.InternalCodeAnalyzer.Tests
{
    public sealed class NameTest
    {
        [Fact]
        public async Task PrivateFieldName()
        {
            var expected = new DiagnosticResult[] {
                VerifyCS.Diagnostic(NameAnalyzer.PrivateFieldName.Rule).WithLocation(6, 21).WithArguments("foo"),
                VerifyCS.Diagnostic(NameAnalyzer.PrivateFieldName.Rule).WithLocation(8, 24).WithArguments("_Bubba"),
                VerifyCS.Diagnostic(NameAnalyzer.PrivateFieldName.Rule).WithLocation(9, 22).WithArguments("_shouldParseHTML"),
                VerifyCS.Diagnostic(NameAnalyzer.PrivateFieldName.Rule).WithLocation(12, 23).WithArguments("BillyClub"),
                VerifyCS.Diagnostic(NameAnalyzer.PrivateFieldName.Rule).WithLocation(14, 30).WithArguments("bob"),
                VerifyCS.Diagnostic(NameAnalyzer.PrivateFieldName.Rule).WithLocation(15, 30).WithArguments("CHAZ"),
                VerifyCS.Diagnostic(NameAnalyzer.PrivateFieldName.Rule).WithLocation(18, 21).WithArguments("_liveFromNYC"),
                VerifyCS.Diagnostic(NameAnalyzer.PrivateFieldName.Rule).WithLocation(20, 28).WithArguments("nice"),
            };

            await VerifyCS.VerifyAnalyzerAsync(PrivateTestSource, expected);
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
        public async Task MoreNameTests()
        {
            var expected = new DiagnosticResult[] {
                VerifyCS.Diagnostic(NameAnalyzer.GeneralName.Rule).WithLocation(7, 11).WithArguments("CLASS"),
                VerifyCS.Diagnostic(NameAnalyzer.GeneralName.Rule).WithLocation(9, 20).WithArguments("alice"),
                VerifyCS.Diagnostic(NameAnalyzer.GeneralName.Rule).WithLocation(10, 21).WithArguments("_bob"),
                VerifyCS.Diagnostic(NameAnalyzer.GeneralName.Rule).WithLocation(11, 22).WithArguments("_chaz"),
                VerifyCS.Diagnostic(NameAnalyzer.GeneralName.Rule).WithLocation(12, 30).WithArguments("emily"),
                VerifyCS.Diagnostic(NameAnalyzer.GeneralName.Rule).WithLocation(12, 37).WithArguments("_francis"),
                VerifyCS.Diagnostic(NameAnalyzer.GeneralName.Rule).WithLocation(17, 21).WithArguments("this_is_not_python"),
                VerifyCS.Diagnostic(NameAnalyzer.GeneralName.Rule).WithLocation(18, 21).WithArguments("thisIsNotJava"),
                VerifyCS.Diagnostic(NameAnalyzer.PrivateFieldName.Rule).WithLocation(22, 30).WithArguments("BadEvent"),
                VerifyCS.Diagnostic(NameAnalyzer.GeneralName.Rule).WithLocation(23, 29).WithArguments("bad_event"),
                VerifyCS.Diagnostic(NameAnalyzer.GeneralName.Rule).WithLocation(26, 30).WithArguments("_badEv"),
                VerifyCS.Diagnostic(NameAnalyzer.GeneralName.Rule).WithLocation(28, 29).WithArguments("one"),
                VerifyCS.Diagnostic(NameAnalyzer.GeneralName.Rule).WithLocation(28, 39).WithArguments("three"),
                VerifyCS.Diagnostic(NameAnalyzer.GeneralName.Rule).WithLocation(29, 22).WithArguments("enumb"),
                VerifyCS.Diagnostic(NameAnalyzer.GeneralName.Rule).WithLocation(29, 35).WithArguments("Two_Two"),
                VerifyCS.Diagnostic(NameAnalyzer.GeneralName.Rule).WithLocation(29, 44).WithArguments("_three"),
                VerifyCS.Diagnostic(NameAnalyzer.GeneralName.Rule).WithLocation(31, 25).WithArguments("_m2"),
                VerifyCS.Diagnostic(NameAnalyzer.GeneralName.Rule).WithLocation(38, 12).WithArguments("marco"),
                VerifyCS.Diagnostic(NameAnalyzer.GeneralName.Rule).WithLocation(38, 31).WithArguments("polo"),
            };

            var test = new VerifyCS.Test
            {
                TestCode = TestSource,
                FixedCode = FixedTestSource,
                NumberOfFixAllIterations = 2,
            };

            test.ExpectedDiagnostics.AddRange(expected);
            await test.RunAsync();
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
        public async Task ExternName()
        {
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
                VerifyCS.Diagnostic(NameAnalyzer.GeneralName.Rule).WithLocation(7, 11).WithArguments("CLASS"),
                VerifyCS.Diagnostic(NameAnalyzer.GeneralName.Rule).WithLocation(13, 21).WithArguments("masterBlaster"),
            };

            await VerifyCS.VerifyAnalyzerAsync(source, expected);
        }

        private const string FixedTestSource = @"
using System;
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