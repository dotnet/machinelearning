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
        internal static string Source => TestUtils.EnsureSourceLoaded(ref _contractsSource, "ContractsCheckResource.cs");

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
        private static string _preFix;
        private static string _postFix;

        [Fact]
        public void ContractsCheckFix()
        {
            string test = TestUtils.EnsureSourceLoaded(ref _preFix, "ContractsCheckBeforeFix.cs");
            string expected = TestUtils.EnsureSourceLoaded(ref _postFix, "ContractsCheckAfterFix.cs");

            VerifyCSharpFix(test, expected);
        }
    }
}
