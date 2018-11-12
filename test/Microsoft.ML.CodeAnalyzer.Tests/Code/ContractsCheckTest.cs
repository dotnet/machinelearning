// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.CodeAnalysis;
using Microsoft.ML.CodeAnalyzer.Tests.Helpers;
using System;
using System.Linq;
using Xunit;

namespace Microsoft.ML.InternalCodeAnalyzer.Tests
{
    public sealed class ContractsCheckTest : DiagnosticVerifier<ContractsCheckAnalyzer>
    {
        private readonly Lazy<string> Source = TestUtils.LazySource("ContractsCheckResource.cs");
        private readonly Lazy<string> SourceContracts = TestUtils.LazySource("Contracts.cs");
        private readonly Lazy<string> SourceFriend = TestUtils.LazySource("BestFriendAttribute.cs");

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

            VerifyCSharpDiagnostic(Source.Value + SourceContracts.Value + SourceFriend.Value, expected);
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
        private readonly Lazy<string> SourcePreFix = TestUtils.LazySource("ContractsCheckBeforeFix.cs");
        private readonly Lazy<string> SourcePostFix = TestUtils.LazySource("ContractsCheckAfterFix.cs");

        private readonly Lazy<string> SourceArgAttr = TestUtils.LazySource("ArgumentAttribute.cs");
        private readonly Lazy<string> SourceArgType = TestUtils.LazySource("ArgumentType.cs");
        private readonly Lazy<string> SourceBestAttr = TestUtils.LazySource("BestFriendAttribute.cs");
        private readonly Lazy<string> SourceDefArgAttr = TestUtils.LazySource("DefaultArgumentAttribute.cs");

        [Fact]
        public void ContractsCheckFix()
        {
            //VerifyCSharpFix(SourcePreFix.Value, SourcePostFix.Value);

            Solution solution = null;
            var proj = CreateProject(TestProjectName, ref solution, SourcePostFix.Value, SourceArgAttr.Value,
                SourceArgType.Value, SourceBestAttr.Value, SourceDefArgAttr.Value);
            var document = proj.Documents.First();
            var analyzer = GetCSharpDiagnosticAnalyzer();
            var comp = proj.GetCompilationAsync().Result;

            CycleAndVerifyFix(analyzer, GetCSharpCodeFixProvider(), SourcePostFix.Value, document);
        }
    }
}
