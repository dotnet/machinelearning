// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Threading.Tasks;
using Microsoft.CodeAnalysis.Testing;
using Microsoft.ML.CodeAnalyzer.Tests.Helpers;
using Microsoft.ML.InternalCodeAnalyzer;
using Xunit;
using VerifyCS = Microsoft.ML.CodeAnalyzer.Tests.Helpers.CSharpCodeFixVerifier<
    Microsoft.ML.InternalCodeAnalyzer.LocalizableMessageAnalyzer,
    Microsoft.ML.InternalCodeAnalyzer.ContractsCheckNameofFixProvider>;

namespace Microsoft.ML.CodeAnalyzer.Tests
{
    public sealed class LocalizableMessageTest
    {
        [Fact]
        public async Task ReportsHardCodedExceptionMessages()
        {
            const string source = @"
using System;

internal static class Strings
{
    internal static string Message => ""message"";
    internal static string FormattedMessage => ""value: {0}"";
}

internal sealed class CustomException : Exception
{
    internal CustomException(string message) : base(message) { }
}

internal sealed class C
{
    private void M(string dynamicMessage)
    {
        _ = new InvalidOperationException({|#0:""literal""|});
        _ = new ArgumentException({|#1:$""value: {dynamicMessage}""|});
        _ = new Exception({|#2:""prefix "" + dynamicMessage|});
        _ = new Exception({|#3:string.Format(""value: {0}"", dynamicMessage)|});
        _ = new CustomException({|#4:""literal""|});
        _ = new Exception(Strings.Message);
        _ = new Exception(string.Format(Strings.FormattedMessage, dynamicMessage));
        _ = new Exception(dynamicMessage);
    }
}";

            var test = new VerifyCS.Test
            {
                TestCode = source,
            };

            test.ExpectedDiagnostics.Add(Diagnostic(0, "InvalidOperationException"));
            test.ExpectedDiagnostics.Add(Diagnostic(1, "ArgumentException"));
            test.ExpectedDiagnostics.Add(Diagnostic(2, "Exception"));
            test.ExpectedDiagnostics.Add(Diagnostic(3, "Exception"));
            test.ExpectedDiagnostics.Add(Diagnostic(4, "CustomException"));
            await test.RunAsync();
        }

        [Fact]
        public async Task ReportsHardCodedContractsMessages()
        {
            const string source = @"
namespace Microsoft.ML.Runtime
{
    internal static class Contracts
    {
        internal static void Check(bool condition, string msg) { }
    }
}

internal static class Strings
{
    internal static string Message => ""message"";
    internal static string FormattedMessage => ""value: {0}"";
}

internal sealed class C
{
    private void M(string dynamicMessage)
    {
        Microsoft.ML.Runtime.Contracts.Check(true, {|#0:""literal""|});
        Microsoft.ML.Runtime.Contracts.Check(true, Strings.Message);
        Microsoft.ML.Runtime.Contracts.Check(true, string.Format(Strings.FormattedMessage, dynamicMessage));
        Microsoft.ML.Runtime.Contracts.Check(true, nameof(dynamicMessage));
        Microsoft.ML.Runtime.Contracts.Check(true, dynamicMessage);
    }
}";

            var test = new VerifyCS.Test
            {
                TestCode = source,
            };

            test.ExpectedDiagnostics.Add(Diagnostic(0, "Check"));
            await test.RunAsync();
        }

        private static DiagnosticResult Diagnostic(int markupKey, string targetName) =>
            new DiagnosticResult(LocalizableMessageAnalyzer.HardCodedMessageDiagnostic.Rule)
                .WithLocation(markupKey)
                .WithArguments(targetName);
    }
}
