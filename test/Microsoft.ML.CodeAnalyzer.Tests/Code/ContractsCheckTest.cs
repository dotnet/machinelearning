// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Testing;
using Microsoft.ML.CodeAnalyzer.Tests.Helpers;
using Xunit;
using VerifyCS = Microsoft.ML.CodeAnalyzer.Tests.Helpers.CSharpCodeFixVerifier<
    Microsoft.ML.InternalCodeAnalyzer.ContractsCheckAnalyzer,
    Microsoft.ML.InternalCodeAnalyzer.ContractsCheckNameofFixProvider>;

namespace Microsoft.ML.InternalCodeAnalyzer.Tests
{
    public sealed class ContractsCheckTest
    {
        private readonly Lazy<string> Source = TestUtils.LazySource("ContractsCheckResource.cs");
        private readonly Lazy<string> SourceContracts = TestUtils.LazySource("Contracts.cs");
        private readonly Lazy<string> SourceFriend = TestUtils.LazySource("BestFriendAttribute.cs");

        [Fact]
        public async Task ContractsCheck()
        {
            const int basis = 11;
            var expected = new DiagnosticResult[] {
                new DiagnosticResult("CS0051", DiagnosticSeverity.Error).WithLocation(15, 16).WithMessage("Inconsistent accessibility: parameter type 'IHostEnvironment' is less accessible than method 'TypeName.TypeName(IHostEnvironment, float, int)'"),
                VerifyCS.Diagnostic(ContractsCheckAnalyzer.NameofDiagnostic.Rule).WithLocation(basis + 8, 46).WithArguments("CheckParam", "paramName", "\"p\""),
                VerifyCS.Diagnostic(ContractsCheckAnalyzer.NameofDiagnostic.Rule).WithLocation(basis + 9, 46).WithArguments("CheckParam", "paramName", "nameof(p) + nameof(p)"),
                VerifyCS.Diagnostic(ContractsCheckAnalyzer.NameofDiagnostic.Rule).WithLocation(basis + 11, 28).WithArguments("CheckValue", "paramName", "\"p\""),
                VerifyCS.Diagnostic(ContractsCheckAnalyzer.NameofDiagnostic.Rule).WithLocation(basis + 13, 39).WithArguments("CheckUserArg", "name", "\"foo\""),
                VerifyCS.Diagnostic(ContractsCheckAnalyzer.ExceptionDiagnostic.Rule).WithLocation(basis + 15, 13).WithArguments("Except"),
                VerifyCS.Diagnostic(ContractsCheckAnalyzer.ExceptionDiagnostic.Rule).WithLocation(basis + 16, 13).WithArguments("ExceptParam"),
                VerifyCS.Diagnostic(ContractsCheckAnalyzer.NameofDiagnostic.Rule).WithLocation(basis + 22, 39).WithArguments("ExceptParam", "paramName", "\"env\""),
                VerifyCS.Diagnostic(ContractsCheckAnalyzer.SimpleMessageDiagnostic.Rule).WithLocation(basis + 24, 29).WithArguments("Check", "$\"Hello {foo} is cool\""),
                VerifyCS.Diagnostic(ContractsCheckAnalyzer.SimpleMessageDiagnostic.Rule).WithLocation(basis + 28, 29).WithArguments("Check", "string.Format(\"Hello {0} is cool\", foo)"),
                VerifyCS.Diagnostic(ContractsCheckAnalyzer.SimpleMessageDiagnostic.Rule).WithLocation(basis + 32, 35).WithArguments("Check", "\"Less fine: \" + env.GetType().Name"),
                VerifyCS.Diagnostic(ContractsCheckAnalyzer.NameofDiagnostic.Rule).WithLocation(basis + 34, 17).WithArguments("CheckUserArg", "name", "\"p\""),
                VerifyCS.Diagnostic(ContractsCheckAnalyzer.DecodeMessageWithLoadContextDiagnostic.Rule).WithLocation(basis + 39, 41).WithArguments("CheckDecode", "\"This message is suspicious\""),
                new DiagnosticResult("CS0122", DiagnosticSeverity.Error).WithLocation("Test1.cs", 752, 24).WithMessage("'ICancelable' is inaccessible due to its protection level"),
                new DiagnosticResult("CS0122", DiagnosticSeverity.Error).WithLocation("Test1.cs", 752, 67).WithMessage("'ICancelable.IsCanceled' is inaccessible due to its protection level"),
            };

            var test = new VerifyCS.Test
            {
                LanguageVersion = LanguageVersion.CSharp7_2,
                TestState =
                {
                    Sources =
                    {
                        Source.Value,
                        SourceContracts.Value,
                        SourceFriend.Value,
                    },
                    AdditionalReferences = { AdditionalMetadataReferences.RefFromType<Memory<int>>() },
                }
            };

            test.ExpectedDiagnostics.AddRange(expected);
            await test.RunAsync();
        }

        [Fact]
        public async Task ContractsCheckDecoy()
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
            await VerifyCS.VerifyAnalyzerAsync(decoySource);
        }

        private readonly Lazy<string> SourcePreFix = TestUtils.LazySource("ContractsCheckBeforeFix.cs");
        private readonly Lazy<string> SourcePostFix = TestUtils.LazySource("ContractsCheckAfterFix.cs");

        private readonly Lazy<string> SourceArgAttr = TestUtils.LazySource("ArgumentAttribute.cs");
        private readonly Lazy<string> SourceArgType = TestUtils.LazySource("ArgumentType.cs");
        private readonly Lazy<string> SourceBestAttr = TestUtils.LazySource("BestFriendAttribute.cs");
        private readonly Lazy<string> SourceDefArgAttr = TestUtils.LazySource("DefaultArgumentAttribute.cs");

        [Fact]
        public async Task ContractsCheckFix()
        {
            var test = new VerifyCS.Test
            {
                LanguageVersion = LanguageVersion.CSharp7_2,
                TestState =
                {
                    Sources =
                    {
                        SourcePreFix.Value,
                        SourceContracts.Value,
                        SourceArgAttr.Value,
                        SourceArgType.Value,
                        SourceBestAttr.Value,
                        SourceDefArgAttr.Value,
                    },
                    ExpectedDiagnostics =
                    {
                        VerifyCS.Diagnostic(ContractsCheckAnalyzer.ExceptionDiagnostic.Rule).WithLocation(9, 43).WithArguments("ExceptParam"),
                        VerifyCS.Diagnostic(ContractsCheckAnalyzer.NameofDiagnostic.Rule).WithLocation(9, 65).WithArguments("ExceptParam", "paramName", "\"yuck\""),
                        VerifyCS.Diagnostic(ContractsCheckAnalyzer.NameofDiagnostic.Rule).WithLocation(13, 39).WithArguments("CheckValue", "paramName", "\"str\""),
                        VerifyCS.Diagnostic(ContractsCheckAnalyzer.NameofDiagnostic.Rule).WithLocation(14, 66).WithArguments("CheckUserArg", "name", "\"Foo\""),
                        VerifyCS.Diagnostic(ContractsCheckAnalyzer.NameofDiagnostic.Rule).WithLocation(15, 57).WithArguments("CheckUserArg", "name", "\"Bar\""),
                        VerifyCS.Diagnostic(ContractsCheckAnalyzer.NameofDiagnostic.Rule).WithLocation(16, 57).WithArguments("CheckUserArg", "name", "\"A\""),
                        VerifyCS.Diagnostic(ContractsCheckAnalyzer.NameofDiagnostic.Rule).WithLocation(17, 52).WithArguments("ExceptParam", "paramName", "\"Bar\""),
                        VerifyCS.Diagnostic(ContractsCheckAnalyzer.NameofDiagnostic.Rule).WithLocation(20, 17).WithArguments("CheckParam", "paramName", "\"isFive\""),
                        VerifyCS.Diagnostic(ContractsCheckAnalyzer.NameofDiagnostic.Rule).WithLocation(21, 49).WithArguments("CheckValue", "paramName", "\"Y\""),
                        VerifyCS.Diagnostic(ContractsCheckAnalyzer.NameofDiagnostic.Rule).WithLocation(22, 52).WithArguments("ExceptParam", "paramName", "\"tom\""),
                        VerifyCS.Diagnostic(ContractsCheckAnalyzer.NameofDiagnostic.Rule).WithLocation(23, 39).WithArguments("CheckValue", "paramName", "\"noMatch\""),
                        VerifyCS.Diagnostic(ContractsCheckAnalyzer.NameofDiagnostic.Rule).WithLocation(24, 53).WithArguments("CheckUserArg", "name", "\"chumble\""),
                        VerifyCS.Diagnostic(ContractsCheckAnalyzer.NameofDiagnostic.Rule).WithLocation(25, 53).WithArguments("CheckUserArg", "name", "\"sp\""),
                        new DiagnosticResult("CS0122", DiagnosticSeverity.Error).WithLocation("Test1.cs", 752, 24).WithMessage("'ICancelable' is inaccessible due to its protection level"),
                        new DiagnosticResult("CS0122", DiagnosticSeverity.Error).WithLocation("Test1.cs", 752, 67).WithMessage("'ICancelable.IsCanceled' is inaccessible due to its protection level"),
                        new DiagnosticResult("CS1503", DiagnosticSeverity.Error).WithLocation("Test1.cs", 753, 91).WithMessage("Argument 2: cannot convert from 'Microsoft.ML.Runtime.IHostEnvironment' to 'Microsoft.ML.Runtime.IExceptionContext'"),
                    },
                    AdditionalReferences = { AdditionalMetadataReferences.RefFromType<Memory<int>>() },
                },
                FixedState =
                {
                    Sources =
                    {
                        SourcePostFix.Value,
                        SourceContracts.Value,
                        SourceArgAttr.Value,
                        SourceArgType.Value,
                        SourceBestAttr.Value,
                        SourceDefArgAttr.Value,
                    },
                    ExpectedDiagnostics =
                    {
                        VerifyCS.Diagnostic(ContractsCheckAnalyzer.ExceptionDiagnostic.Rule).WithLocation(9, 43).WithArguments("ExceptParam"),
                        VerifyCS.Diagnostic(ContractsCheckAnalyzer.NameofDiagnostic.Rule).WithLocation(23, 39).WithArguments("CheckValue", "paramName", "\"noMatch\""),
                        new DiagnosticResult("CS0122", DiagnosticSeverity.Error).WithLocation("Test1.cs", 752, 24).WithMessage("'ICancelable' is inaccessible due to its protection level"),
                        new DiagnosticResult("CS0122", DiagnosticSeverity.Error).WithLocation("Test1.cs", 752, 67).WithMessage("'ICancelable.IsCanceled' is inaccessible due to its protection level"),
                        new DiagnosticResult("CS1503", DiagnosticSeverity.Error).WithLocation("Test1.cs", 753, 91).WithMessage("Argument 2: cannot convert from 'Microsoft.ML.Runtime.IHostEnvironment' to 'Microsoft.ML.Runtime.IExceptionContext'"),
                    },
                },
            };

            await test.RunAsync();
        }
    }
}
