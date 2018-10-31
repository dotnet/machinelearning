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
            var diagC = analyzer.SupportedDiagnostics[2];

            string p(string i = "") => string.IsNullOrEmpty(i) ? "" : $" of item {i}";

            var expected = new DiagnosticResult[] {
                diag.CreateDiagnosticResult(23, 13, p()),
                diag.CreateDiagnosticResult(24, 13, p("bad")),
                diag.CreateDiagnosticResult(30, 13, p("c.Item2")),
                diag.CreateDiagnosticResult(40, 13, p("listen")),
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
                diagTp.CreateDiagnosticResult(18, 24, "T"),
                diagTp.CreateDiagnosticResult(41, 24, "T"),
                diag.CreateDiagnosticResult(55, 26, p("text")),
            };

            VerifyCSharpDiagnostic(SourceChained, expected);
        }

        private static string _srcResourceClass;
        internal static string SourceClass => TestUtils.EnsureSourceLoaded(
            ref _srcResourceClass, "TypeIsSchemaShapeClassResource.cs");

        [Fact]
        public void ReturnTypeIsSchemaShapeClass()
        {
            // This is a somewhat more complex example, where instead of direct usage the user of the API is devising their own
            // function where the shape type is a generic type parameter. In this case, we would ideally like the analysis to get
            // chained out of their function.
            var analyzer = GetCSharpDiagnosticAnalyzer();
            var diag = analyzer.SupportedDiagnostics[0];
            var diagTp = analyzer.SupportedDiagnostics[1];
            var diagC = analyzer.SupportedDiagnostics[2];
            var diagF = analyzer.SupportedDiagnostics[3];

            var diagPg = analyzer.SupportedDiagnostics[4];
            var diagPnas = analyzer.SupportedDiagnostics[5];
            var diagPs = analyzer.SupportedDiagnostics[6];
            var diagPc = analyzer.SupportedDiagnostics[7];

            var expected = new DiagnosticResult[] {
                diagF.CreateDiagnosticResult(32, 13, "Class4", "F1"),
                diagC.CreateDiagnosticResult(33, 13, "Class5"),
                diagC.CreateDiagnosticResult(34, 13, "Class6"),

                diagPnas.CreateDiagnosticResult(35, 13, "Class7", "F1"),
                diagPs.CreateDiagnosticResult(36, 13, "Class8", "F2"),
                diagPg.CreateDiagnosticResult(37, 13, "Class9", "F2"),
                diagPc.CreateDiagnosticResult(38, 13, "Class10"),
                diagPc.CreateDiagnosticResult(39, 13, "Class11"),
            };

            VerifyCSharpDiagnostic(SourceClass, expected);
        }
    }
}
