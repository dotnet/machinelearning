// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.CodeAnalyzer.Tests.Helpers;
using Xunit;

namespace Microsoft.ML.Analyzer.Tests
{
    public sealed class TypeIsSchemaShapeTest : DiagnosticVerifier<TypeIsSchemaShapeAnalyzer>
    {
        private static string _srcResource;
        internal static string Source => TestUtils.EnsureSourceLoaded(ref _srcResource, "TypeIsSchemaShapeResource.cs");

        [Fact]
        public void ReturnTypeIsSchemaShape()
        {
            var analyzer = GetCSharpDiagnosticAnalyzer();
            var diag = analyzer.SupportedDiagnostics[0];

            string p(string i = "") => string.IsNullOrEmpty(i) ? "" : $" of item {i}";

            var expected = new DiagnosticResult[] {
                diag.CreateDiagnosticResult(23, 13, p()),
                diag.CreateDiagnosticResult(24, 13, p()),
                diag.CreateDiagnosticResult(25, 13, p()),
                diag.CreateDiagnosticResult(29, 13, p("c.Item2")),
                diag.CreateDiagnosticResult(39, 13, p("listen")),
            };

            VerifyCSharpDiagnostic(Source, expected);
        }

        private static string _srcResourceChained;
        internal static string SourceChained => TestUtils.EnsureSourceLoaded(
            ref _srcResourceChained, "TypeIsSchemaShapeResourceChained.cs");

        [Fact]
        public void ReturnTypeIsSchemaShapeChained()
        {
            // This is a somewhat more complex example, where instead of direct usage the user of the API is devising their own
            // function where the shape type is a generic type parameter. In this case, we would ideally like the analysis to get
            // chained out of their function.
            var analyzer = GetCSharpDiagnosticAnalyzer();
            var diag = analyzer.SupportedDiagnostics[0];
            var diagTp = analyzer.SupportedDiagnostics[1];

            string p(string i = "") => string.IsNullOrEmpty(i) ? "" : $" of item {i}";

            var expected = new DiagnosticResult[] {
                diagTp.CreateDiagnosticResult(17, 24, "T"),
                diagTp.CreateDiagnosticResult(40, 24, "T"),
                diag.CreateDiagnosticResult(54, 26, p("text")),
            };

            VerifyCSharpDiagnostic(SourceChained, expected);
        }
    }
}
