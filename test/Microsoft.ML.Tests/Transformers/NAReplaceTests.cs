// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class NAReplaceTests : TestDataPipeBase
    {
        private class TestClass
        {
            public float A;
            public double B;

            [VectorType(2)]
            public float[] C;

            [VectorType(2)]
            public double[] D;
        }

        private class TestOutputClass
        {
            public float A;

            [VectorType(2)]
            public float[] CA;

            [VectorType(2)]
            public float[] CB;

            public double B;

            [VectorType(2)]
            public double[] DA;

            [VectorType(2)]
            public double[] DB;
        }

        public NAReplaceTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void NAReplaceMode()
        {
            var data = new[]
            {
                new TestClass { A = 1f, B = 1d, C = new float[] { 1f, 10f }, D = new double[] { 1f, 10f } },
                new TestClass { A = 2f, B = 2d, C = new float[] { float.NaN, 9f }, D = new double[] { double.NaN, 9f } },
                new TestClass { A = float.NaN, B = double.NaN, C = new float[] { 2f, float.NaN }, D = new double[] { 2f, double.NaN } },
                new TestClass { A = 2f, B = 2f, C = new float[] { 3f, 9f}, D = new double[] { 3f, 9f} },
                new TestClass{ A = float.NaN, B = double.NaN, C = new float[] { 1f, float.NaN }, D = new double[] { 1f, double.NaN } },
            };

            var dataView = ML.Data.LoadFromEnumerable(data);
            var pipe = ML.Transforms.ReplaceMissingValues(
                new MissingValueReplacingEstimator.ColumnOptions("A", "A", MissingValueReplacingEstimator.ReplacementMode.Mode),
                new MissingValueReplacingEstimator.ColumnOptions("CA", "C", MissingValueReplacingEstimator.ReplacementMode.Mode, imputeBySlot: false),
                new MissingValueReplacingEstimator.ColumnOptions("CB", "C", MissingValueReplacingEstimator.ReplacementMode.Mode),
                new MissingValueReplacingEstimator.ColumnOptions("B", "B", MissingValueReplacingEstimator.ReplacementMode.Mode),
                new MissingValueReplacingEstimator.ColumnOptions("DA", "D", MissingValueReplacingEstimator.ReplacementMode.Mode, imputeBySlot: false),
                new MissingValueReplacingEstimator.ColumnOptions("DB", "D", MissingValueReplacingEstimator.ReplacementMode.Mode)
                );

            var transformedDataview = pipe.Fit(dataView).Transform(dataView);

            var expectedOutput = new TestOutputClass[]
            {
                new TestOutputClass{ A = 1, CA = new float[] { 1, 10 }, CB = new float[] { 1, 10 }, B = 1, DA = new double[] { 1, 10 }, DB = new double[] { 1, 10 } },
                new TestOutputClass{ A = 2, CA = new float[] { 9, 9 }, CB = new float[] { 1, 9 }, B = 2, DA = new double[] { 9, 9 }, DB = new double[] { 1, 9 } },
                new TestOutputClass{ A = 2, CA = new float[] { 2, 9 }, CB = new float[] { 2, 9 }, B = 2, DA = new double[] { 2, 9 }, DB = new double[] { 2, 9 } },
                new TestOutputClass{ A = 2, CA = new float[] { 3, 9 }, CB = new float[] { 3, 9 }, B = 2, DA = new double[] { 3, 9 }, DB = new double[] { 3, 9 } },
                new TestOutputClass{ A = 2, CA = new float[] { 1, 9 }, CB = new float[] { 1, 9 }, B = 2, DA = new double[] { 1, 9 }, DB = new double[] { 1, 9 } }
            };

            var expectedOutputDataview = ML.Data.LoadFromEnumerable(expectedOutput);
            // Compare all output results
            CompareResults("A", "A", expectedOutputDataview, transformedDataview);
            CompareResults("CA", "CA", expectedOutputDataview, transformedDataview);
            CompareResults("CB", "CB", expectedOutputDataview, transformedDataview);
            CompareResults("B", "B", expectedOutputDataview, transformedDataview);
            CompareResults("DA", "DA", expectedOutputDataview, transformedDataview);
            CompareResults("DB", "DB", expectedOutputDataview, transformedDataview);

            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void NAReplaceWorkout()
        {
            var data = new[] {
                new TestClass() { A = 1, B = 3, C= new float[2]{ 1, 2 } , D = new double[2]{ 3,4} },
                new TestClass() { A = float.NaN, B = double.NaN, C= new float[2]{ float.NaN, float.NaN } , D = new double[2]{ double.NaN,double.NaN}},
                new TestClass() { A = float.NegativeInfinity, B = double.NegativeInfinity,C= new float[2]{ float.NegativeInfinity, float.NegativeInfinity } , D = new double[2]{ double.NegativeInfinity, double.NegativeInfinity}},
                new TestClass() { A = float.PositiveInfinity, B = double.PositiveInfinity,C= new float[2]{ float.PositiveInfinity, float.PositiveInfinity, } , D = new double[2]{  double.PositiveInfinity, double.PositiveInfinity}},
                new TestClass() { A = 2, B = 1 ,C= new float[2]{ 3, 4 } , D = new double[2]{ 5,6}},
            };

            var dataView = ML.Data.LoadFromEnumerable(data);
            var pipe = ML.Transforms.ReplaceMissingValues(
                new MissingValueReplacingEstimator.ColumnOptions("NAA", "A", MissingValueReplacingEstimator.ReplacementMode.Mean),
                new MissingValueReplacingEstimator.ColumnOptions("NAB", "B", MissingValueReplacingEstimator.ReplacementMode.Mean),
                new MissingValueReplacingEstimator.ColumnOptions("NAC", "C", MissingValueReplacingEstimator.ReplacementMode.Mean),
                new MissingValueReplacingEstimator.ColumnOptions("NAD", "D", MissingValueReplacingEstimator.ReplacementMode.Mean));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void NAReplace()
        {
            string dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var data = ML.Data.LoadFromTextFile(dataPath, new[] {
                new TextLoader.Column("ScalarFloat", DataKind.Single, 1),
                new TextLoader.Column("ScalarDouble", DataKind.Double, 1),
                new TextLoader.Column("VectorFloat", DataKind.Single, 1, 4),
                new TextLoader.Column("VectorDouble", DataKind.Double, 1, 4)
            });

            var wrongCollection = new[] { new TestClass() { A = 1, B = 3, C = new float[2] { 1, 2 }, D = new double[2] { 3, 4 } } };
            var invalidData = ML.Data.LoadFromEnumerable(wrongCollection);

            var est = ML.Transforms.ReplaceMissingValues("A", "ScalarFloat", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Maximum)
                .Append(ML.Transforms.ReplaceMissingValues("B", "ScalarDouble", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(ML.Transforms.ReplaceMissingValues("C", "VectorFloat", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(ML.Transforms.ReplaceMissingValues("D", "VectorDouble", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Minimum))
                .Append(ML.Transforms.ReplaceMissingValues("E", "VectorDouble", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mode));

            TestEstimatorCore(est, data, invalidInput: invalidData);
            var outputPath = GetOutputPath("NAReplace", "featurized.tsv");
            var savedData = ML.Data.TakeRows(est.Fit(data).Transform(data), 4);
            var view = ML.Transforms.SelectColumns("A", "B", "C", "D", "E").Fit(savedData).Transform(savedData);
            using (var fs = File.Create(outputPath))
                ML.Data.SaveAsText(view, fs, headerRow: true, keepHidden: true);

            CheckEquality("NAReplace", "featurized.tsv");
            Done();
        }

        [Fact]
        public void TestCommandLine()
        {
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0}  xf=NAReplace{col=C:A} in=f:\2.txt" }), (int)0);
        }

        [Fact]
        public void TestOldSavingAndLoading()
        {
            var data = new[] {
                new TestClass() { A = 1,  B = 3, C= new float[2]{ 1, 2 } , D = new double[2]{ 3,4} },
                new TestClass() { A = float.NaN,  B = double.NaN, C= new float[2]{ float.NaN, float.NaN } , D = new double[2]{ double.NaN,double.NaN}},
                new TestClass() { A = float.NegativeInfinity, B = double.NegativeInfinity,C= new float[2]{ float.NegativeInfinity, float.NegativeInfinity } , D = new double[2]{ double.NegativeInfinity, double.NegativeInfinity}},
                new TestClass() { A = float.PositiveInfinity, B = double.PositiveInfinity,C= new float[2]{ float.PositiveInfinity, float.PositiveInfinity, } , D = new double[2]{  double.PositiveInfinity, double.PositiveInfinity}},
                new TestClass() { A = 2, B = 1 ,C= new float[2]{ 3, 4 } , D = new double[2]{ 5,6}},
            };

            var dataView = ML.Data.LoadFromEnumerable(data);
            var pipe = ML.Transforms.ReplaceMissingValues(
                new MissingValueReplacingEstimator.ColumnOptions("NAA", "A", MissingValueReplacingEstimator.ReplacementMode.Mean),
                new MissingValueReplacingEstimator.ColumnOptions("NAB", "B", MissingValueReplacingEstimator.ReplacementMode.Mean),
                new MissingValueReplacingEstimator.ColumnOptions("NAC", "C", MissingValueReplacingEstimator.ReplacementMode.Mean),
                new MissingValueReplacingEstimator.ColumnOptions("NAD", "D", MissingValueReplacingEstimator.ReplacementMode.Mean));

            var result = pipe.Fit(dataView).Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(Env, Env.Start("saving"), ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(Env, dataView, ms);
            }
        }
    }
}
