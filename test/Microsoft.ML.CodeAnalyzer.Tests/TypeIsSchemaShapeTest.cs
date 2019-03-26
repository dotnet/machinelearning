// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Testing;
using Microsoft.ML.CodeAnalyzer.Tests.Helpers;
using Xunit;
using VerifyCS = Microsoft.ML.CodeAnalyzer.Tests.Helpers.CSharpCodeFixVerifier<
    Microsoft.ML.Analyzer.TypeIsSchemaShapeAnalyzer,
    Microsoft.CodeAnalysis.Testing.EmptyCodeFixProvider>;

namespace Microsoft.ML.Analyzer.Tests
{
    public sealed class TypeIsSchemaShapeTest
    {
        private static string _srcResource;
        internal static string Source => TestUtils.EnsureSourceLoaded(ref _srcResource, "TypeIsSchemaShapeResource.cs");

        [Fact]
        public async Task ReturnTypeIsSchemaShape()
        {
            var expected = new DiagnosticResult[] {
                VerifyCS.Diagnostic(TypeIsSchemaShapeAnalyzer.ShapeDiagnostic.Rule).WithLocation(24, 13).WithArguments(""),
                VerifyCS.Diagnostic(TypeIsSchemaShapeAnalyzer.ShapeDiagnostic.Rule).WithLocation(25, 13).WithArguments(" of item bad"),
                VerifyCS.Diagnostic(TypeIsSchemaShapeAnalyzer.ShapeDiagnostic.Rule).WithLocation(31, 13).WithArguments(" of item c.Item2"),
                VerifyCS.Diagnostic(TypeIsSchemaShapeAnalyzer.ShapeDiagnostic.Rule).WithLocation(41, 13).WithArguments(" of item listen"),
            };

            var test = new VerifyCS.Test { TestCode = Source };
            test.ExpectedDiagnostics.AddRange(expected);
            test.Exclusions &= ~AnalysisExclusions.GeneratedCode;
            await test.RunAsync();
        }

        private static string _srcResourceChained;
        internal static string SourceChained => TestUtils.EnsureSourceLoaded(
            ref _srcResourceChained, "TypeIsSchemaShapeResourceChained.cs");

        [Fact]
        public async Task ReturnTypeIsSchemaShapeChained()
        {
            // This is a somewhat more complex example, where instead of direct usage the user of the API is devising their own
            // function where the shape type is a generic type parameter. In this case, we would ideally like the analysis to get
            // chained out of their function.
            var expected = new DiagnosticResult[] {
                VerifyCS.Diagnostic(TypeIsSchemaShapeAnalyzer.ShapeParameterDiagnostic.Rule).WithLocation(19, 24).WithArguments("T"),
                new DiagnosticResult("CS8205", DiagnosticSeverity.Error).WithLocation(22, 52).WithMessage("Attributes are not allowed on local function parameters or type parameters"),
                VerifyCS.Diagnostic(TypeIsSchemaShapeAnalyzer.ShapeParameterDiagnostic.Rule).WithLocation(42, 24).WithArguments("T"),
                VerifyCS.Diagnostic(TypeIsSchemaShapeAnalyzer.ShapeDiagnostic.Rule).WithLocation(56, 26).WithArguments(" of item text"),
            };

            var test = new VerifyCS.Test { TestCode = SourceChained };
            test.ExpectedDiagnostics.AddRange(expected);
            test.Exclusions &= ~AnalysisExclusions.GeneratedCode;
            await test.RunAsync();
        }

        private static string _srcResourceClass;
        internal static string SourceClass => TestUtils.EnsureSourceLoaded(
            ref _srcResourceClass, "TypeIsSchemaShapeClassResource.cs");

        [Fact]
        public async Task ReturnTypeIsSchemaShapeClass()
        {
            // This is a somewhat more complex example, where instead of direct usage the user of the API is devising their own
            // function where the shape type is a generic type parameter. In this case, we would ideally like the analysis to get
            // chained out of their function.
            var expected = new DiagnosticResult[] {
                VerifyCS.Diagnostic(TypeIsSchemaShapeAnalyzer.ShapeClassDiagnosticField.Rule).WithLocation(33, 13).WithArguments("Class4", "F1"),
                VerifyCS.Diagnostic(TypeIsSchemaShapeAnalyzer.ShapeClassDiagnosticConstructor.Rule).WithLocation(34, 13).WithArguments("Class5"),
                VerifyCS.Diagnostic(TypeIsSchemaShapeAnalyzer.ShapeClassDiagnosticConstructor.Rule).WithLocation(35, 13).WithArguments("Class6"),

                VerifyCS.Diagnostic(TypeIsSchemaShapeAnalyzer.ShapeClassDiagnosticNoArgsSettable.Rule).WithLocation(36, 13).WithArguments("Class7", "F1"),
                VerifyCS.Diagnostic(TypeIsSchemaShapeAnalyzer.ShapeClassDiagnosticArgsSettable.Rule).WithLocation(37, 13).WithArguments("Class8", "F2"),
                VerifyCS.Diagnostic(TypeIsSchemaShapeAnalyzer.ShapeClassDiagnosticGettable.Rule).WithLocation(38, 13).WithArguments("Class9", "F2"),
                VerifyCS.Diagnostic(TypeIsSchemaShapeAnalyzer.ShapeClassDiagnosticCorrespondence.Rule).WithLocation(39, 13).WithArguments("Class10"),
                VerifyCS.Diagnostic(TypeIsSchemaShapeAnalyzer.ShapeClassDiagnosticCorrespondence.Rule).WithLocation(40, 13).WithArguments("Class11"),

                new DiagnosticResult("CS0246", DiagnosticSeverity.Error).WithLocation(44, 71).WithMessage("The type or namespace name 'MissingClass' could not be found (are you missing a using directive or an assembly reference?)"),
            };

            var test = new VerifyCS.Test { TestCode = SourceClass };
            test.ExpectedDiagnostics.AddRange(expected);
            test.Exclusions &= ~AnalysisExclusions.GeneratedCode;
            await test.RunAsync();
        }
    }
}
