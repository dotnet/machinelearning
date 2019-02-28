// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis.Testing;
using Microsoft.ML.CodeAnalyzer.Tests.Helpers;
using Xunit;
using VerifyCS = Microsoft.ML.CodeAnalyzer.Tests.Helpers.CSharpCodeFixVerifier<
    Microsoft.ML.InternalCodeAnalyzer.BestFriendOnPublicDeclarationsAnalyzer,
    Microsoft.CodeAnalysis.Testing.EmptyCodeFixProvider>;

namespace Microsoft.ML.InternalCodeAnalyzer.Tests
{
    public sealed class BestFriendOnPublicDeclarationTest
    {
        private readonly Lazy<string> SourceAttribute = TestUtils.LazySource("BestFriendAttribute.cs");
        private readonly Lazy<string> SourceDeclaration = TestUtils.LazySource("BestFriendOnPublicDeclaration.cs");

        [Fact]
        public async Task BestFriendOnPublicDeclaration()
        {
            var expected = new DiagnosticResult[] {
                VerifyCS.Diagnostic().WithLocation(8, 6).WithArguments("PublicClass"),
                VerifyCS.Diagnostic().WithLocation(11, 10).WithArguments("PublicField"),
                VerifyCS.Diagnostic().WithLocation(14, 10).WithArguments("PublicProperty"),
                VerifyCS.Diagnostic().WithLocation(20, 10).WithArguments("PublicMethod"),
                VerifyCS.Diagnostic().WithLocation(26, 10).WithArguments("PublicDelegate"),
                VerifyCS.Diagnostic().WithLocation(29, 10).WithArguments("PublicClass"),
                VerifyCS.Diagnostic().WithLocation(35, 6).WithArguments("PublicStruct"),
                VerifyCS.Diagnostic().WithLocation(40, 6).WithArguments("PublicEnum"),
                VerifyCS.Diagnostic().WithLocation(47, 6).WithArguments("PublicInterface"),
                VerifyCS.Diagnostic().WithLocation(102, 10).WithArguments("PublicMethod"),
            };

            var test = new VerifyCS.Test
            {
                TestState =
                {
                    Sources =
                    {
                        SourceDeclaration.Value,
                        ("BestFriendAttribute.cs", SourceAttribute.Value),
                    },
                },
            };

            test.ExpectedDiagnostics.AddRange(expected);
            await test.RunAsync();
        }
    }
}
