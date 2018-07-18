// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using System.Reflection;
using System.Threading;
using Microsoft.ML.CodeAnalyzer.Tests.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.CodeAnalyzer.Tests
{
    public sealed class ContractsCheckTest : DiagnosticVerifier<ContractsCheckAnalyzer>
    {
        private static string _contractsSource;

        internal static string Source {
            get {
                if (_contractsSource == null)
                {
                    string contractsSource;
                    using (var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("ContractsCheckResource.cs"))
                    using (var reader = new StreamReader(stream))
                        contractsSource = reader.ReadToEnd();
                    Interlocked.CompareExchange(ref _contractsSource, contractsSource, null);
                }
                return _contractsSource;
            }
        }

        [Fact]
        public void ContractsCheck()
        {
            var analyzer = GetCSharpDiagnosticAnalyzer();
            var diagName = analyzer.SupportedDiagnostics[0];
            var diagExceptExp = analyzer.SupportedDiagnostics[1];
            var diagMsg = analyzer.SupportedDiagnostics[2];
            var diagDecode = analyzer.SupportedDiagnostics[3];

            const int basis = 10;
            var expected = new DiagnosticResult[] {
                diagName.CreateDiagnosticResult(basis + 8, 46, "CheckParam", "paramName", "\"p\""),
                diagName.CreateDiagnosticResult(basis + 9, 46, "CheckParam", "paramName", "nameof(p) + nameof(p)"),
                diagName.CreateDiagnosticResult(basis + 11, 28, "CheckValue", "paramName", "\"p\""),
                diagName.CreateDiagnosticResult(basis + 13, 39, "CheckUserArg", "name", "\"foo\""),
                diagExceptExp.CreateDiagnosticResult(basis + 15, 13, "Except"),
                diagExceptExp.CreateDiagnosticResult(basis + 16, 13, "ExceptParam"),
                diagName.CreateDiagnosticResult(basis + 22, 39, "ExceptParam", "paramName", "\"env\""),
                diagMsg.CreateDiagnosticResult(basis + 24, 29, "Check", "$\"Hello {foo} is cool\""),
                diagMsg.CreateDiagnosticResult(basis + 28, 29, "Check", "string.Format(\"Hello {0} is cool\", foo)"),
                diagMsg.CreateDiagnosticResult(basis + 32, 35, "Check", "\"Less fine: \" + env.GetType().Name"),
                diagName.CreateDiagnosticResult(basis + 34, 17, "CheckUserArg", "name", "\"p\""),
                diagDecode.CreateDiagnosticResult(basis + 39, 41, "CheckDecode", "\"This message is suspicious\""),
            };

            VerifyCSharpDiagnostic(Source, expected);
        }

        [Fact]
        public void ContractsCheckDecoy()
        {
            // Run a test with a "decoy" Contacts that has the same signature on the method,
            // except the namespace is distinct so it is a different type. We should not find
            // diagnostics on a class like this!
            const string decoySource = @"
using System;
namespace TestNamespace
{
    public static class Contracts

    {
        public static Exception ExceptParam(string paramName, string msg) => null;
    }

    public sealed class TypeName
    {
        public TypeName()
            => Contracts.ExceptParam(""myAwesomeParameter"", ""What a great thing"");
    }
}";
            VerifyCSharpDiagnostic(decoySource, new DiagnosticResult[0]);
        }
    }

    public sealed class ContractsCheckFixTest : CodeFixVerifier<ContractsCheckAnalyzer, ContractsCheckNameofFixProvider>
    {
        [Fact]
        public void ContractsCheckFix()
        {
            string s = ContractsCheckTest.Source;
            int min = ContractsCheckTest.Source.IndexOf("namespace TestNamespace");
            int lim = ContractsCheckTest.Source.IndexOf("// CUTOFF");
            //string prefix = ContractsCheckTest.Source.Substring(0, min);
            //string suffix = ContractsCheckTest.Source.Substring(lim);

            const string test = @"using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using System;
namespace Bubba {
public class Foo {
public Foo(int yuck) { if (false) Contracts.ExceptParam(""yuck""); }
public static void Bar(float tom, Arguments args) {
string str = ""hello"";
Contracts.CheckValue(str, ""str"");
Contracts.CheckUserArg(0 <= A.B.Foo && A.B.Foo < 10, ""Foo"", ""Should be in range [0,10)"");
Contracts.CheckUserArg(A.B.Bar.Length == 2, ""Bar"", ""Length must be exactly 2"");
Contracts.CheckUserArg(A.B.Bar.Length == 2, ""A"", ""Length must be exactly 2"");
if (false) throw Contracts.ExceptParam(""Bar"", $""Length should have been 2 but was {A.B.Bar.Length}"");
Func<int, bool> isFive = val => val == 5;
Contracts.CheckParam(!isFive(4),
    ""isFive"");
Contracts.CheckValue(typeof(X.Y.Z), ""Y"");
if (false) throw Contracts.ExceptParam(""tom"");
Contracts.CheckValue(str, ""noMatch"");
Contracts.CheckUserArg(str.Length == 2, ""chumble"", ""Whoa!"");
Contracts.CheckUserArg(str.Length == 2, ""sp"", ""Git along, little dogies, git along..."");
} }
public static class A { public static class B { public const int Foo = 5; public const string Bar = ""Yo""; } }
public static class X { public static class Y { public static class Z { } } }
public sealed class Arguments {
[Argument(ArgumentType.AtMostOnce, HelpText = ""Yakka foob mog."", ShortName = ""chum"")]
public int chumble;
[Argument(ArgumentType.AtMostOnce, HelpText = ""Grug pubbawup zink wattoom gazork."", ShortName = ""spu,sp"")]
public int spuzz; }
}";

            const string expected = @"using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using System;
namespace Bubba {
public class Foo {
public Foo(int yuck) { if (false) Contracts.ExceptParam(nameof(yuck)); }
public static void Bar(float tom, Arguments args) {
string str = ""hello"";
Contracts.CheckValue(str, nameof(str));
Contracts.CheckUserArg(0 <= A.B.Foo && A.B.Foo < 10, nameof(A.B.Foo), ""Should be in range [0,10)"");
Contracts.CheckUserArg(A.B.Bar.Length == 2, nameof(A.B.Bar), ""Length must be exactly 2"");
Contracts.CheckUserArg(A.B.Bar.Length == 2, nameof(A), ""Length must be exactly 2"");
if (false) throw Contracts.ExceptParam(nameof(A.B.Bar), $""Length should have been 2 but was {A.B.Bar.Length}"");
Func<int, bool> isFive = val => val == 5;
Contracts.CheckParam(!isFive(4),
    nameof(isFive));
Contracts.CheckValue(typeof(X.Y.Z), nameof(X.Y));
if (false) throw Contracts.ExceptParam(nameof(tom));
Contracts.CheckValue(str, ""noMatch"");
Contracts.CheckUserArg(str.Length == 2, nameof(args.chumble), ""Whoa!"");
Contracts.CheckUserArg(str.Length == 2, nameof(args.spuzz), ""Git along, little dogies, git along..."");
} }
public static class A { public static class B { public const int Foo = 5; public const string Bar = ""Yo""; } }
public static class X { public static class Y { public static class Z { } } }
public sealed class Arguments {
[Argument(ArgumentType.AtMostOnce, HelpText = ""Yakka foob mog."", ShortName = ""chum"")]
public int chumble;
[Argument(ArgumentType.AtMostOnce, HelpText = ""Grug pubbawup zink wattoom gazork."", ShortName = ""spu,sp"")]
public int spuzz; }
}";

            VerifyCSharpFix(test, expected);
        }
    }
}
