// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.CodeAnalyzer.Tests.Helpers;
using Xunit;

namespace Microsoft.ML.CodeAnalyzer.Tests
{
    public sealed class InterfaceNameTest : DiagnosticVerifier<InterfaceNameAnalyzer>
    {
        [Fact]
        public void InterfaceName()
        {
            const string test = @"
namespace TestNamespace
{
    interface Alice {}
    interface Bob : Alice {}
    interface Chaz<T> : Alice {}
    interface IDebora {}
    interface Iernie {}
    interface _IFrancis {}
}";
            var analyzer = GetCSharpDiagnosticAnalyzer();
            var diag = analyzer.SupportedDiagnostics[0];

            var expected = new DiagnosticResult[] {
                diag.CreateDiagnosticResult(3, 15, "Alice"),
                diag.CreateDiagnosticResult(4, 15, "Bob"),
                diag.CreateDiagnosticResult(5, 15, "Chaz"),
                diag.CreateDiagnosticResult(7, 15, "Iernie"),
                diag.CreateDiagnosticResult(8, 15, "_IFrancis"),
            };

            VerifyCSharpDiagnostic(test, expected);
        }
    }
}
