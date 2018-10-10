// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using System.IO;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class NAIndicatorTests : TestDataPipeBase
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

        public NAIndicatorTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void NAIndicatorWorkout()
        {
            var data = new[] {
                new TestClass() { A = 1, B = 3, C = new float[2]{ 1, 2 } , D = new double[2]{ 3,4} },
                new TestClass() { A = float.NaN, B = double.NaN, C = new float[2]{ float.NaN, float.NaN } , D = new double[2]{ double.NaN,double.NaN}},
                new TestClass() { A = float.NegativeInfinity, B = double.NegativeInfinity, C = new float[2]{ float.NegativeInfinity, float.NegativeInfinity } , D = new double[2]{ double.NegativeInfinity, double.NegativeInfinity}},
                new TestClass() { A = float.PositiveInfinity, B = double.PositiveInfinity, C = new float[2]{ float.PositiveInfinity, float.PositiveInfinity, } , D = new double[2]{  double.PositiveInfinity, double.PositiveInfinity}},
                new TestClass() { A = 2, B = 1, C = new float[2]{ 3, 4 } , D = new double[2]{ 5,6}},
            };

            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new NAIndicatorEstimator(Env,
                new NAIndicatorTransform.ColumnInfo("A", "NAA"),
                new NAIndicatorTransform.ColumnInfo("B", "NAB"),
                new NAIndicatorTransform.ColumnInfo("C", "NAC"),
                new NAIndicatorTransform.ColumnInfo("D", "NAD"));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void NAIndicatorStatic()
        {
            string dataPath = GetDataPath("breast-cancer.txt");
            var reader = TextLoader.CreateReader(Env, ctx => (
                ScalarFloat: ctx.LoadFloat(1),
                ScalarDouble: ctx.LoadDouble(1),
                VectorFloat: ctx.LoadFloat(1, 4),
                VectorDoulbe: ctx.LoadDouble(1, 4)
            ));

            var data = reader.Read(new MultiFileSource(dataPath));
            var wrongCollection = new[] { new TestClass() { A = 1, B = 3, C = new float[2] { 1, 2 }, D = new double[2] { 3, 4 } } };
            var invalidData = ComponentCreation.CreateDataView(Env, wrongCollection);

            var est = data.MakeNewEstimator().
                   Append(row => (
                   A: row.ScalarFloat.IsMissingValue(),
                   B: row.ScalarDouble.IsMissingValue(),
                   C: row.VectorFloat.IsMissingValue(),
                   D: row.VectorDoulbe.IsMissingValue()
                   ));

            TestEstimatorCore(est.AsDynamic, data.AsDynamic, invalidInput: invalidData);
            var outputPath = GetOutputPath("NAIndicator", "featurized.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                IDataView savedData = TakeFilter.Create(Env, est.Fit(data).Transform(data).AsDynamic, 4);
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("NAIndicator", "featurized.tsv");
            Done();
        }

        [Fact]
        public void TestCommandLine()
        {
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0}  xf=NAIndicator{col=B:A} in=f:\2.txt" }), (int)0);
        }

        [Fact]
        public void TestOldSavingAndLoading()
        {
            var data = new[] {
                new TestClass() { A = 1, B = 3, C = new float[2]{ 1, 2 } , D = new double[2]{ 3,4} },
                new TestClass() { A = float.NaN, B = double.NaN, C = new float[2]{ float.NaN, float.NaN } , D = new double[2]{ double.NaN,double.NaN}},
                new TestClass() { A = float.NegativeInfinity, B = double.NegativeInfinity, C = new float[2]{ float.NegativeInfinity, float.NegativeInfinity } , D = new double[2]{ double.NegativeInfinity, double.NegativeInfinity}},
                new TestClass() { A = float.PositiveInfinity,  B = double.PositiveInfinity, C = new float[2]{ float.PositiveInfinity, float.PositiveInfinity, } , D = new double[2]{  double.PositiveInfinity, double.PositiveInfinity}},
                new TestClass() { A = 2, B = 1 , C = new float[2]{ 3, 4 } , D = new double[2]{ 5,6}},
            };

            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new NAIndicatorEstimator(Env,
                new NAIndicatorTransform.ColumnInfo("A", "NAA"),
                new NAIndicatorTransform.ColumnInfo("B", "NAB"),
                new NAIndicatorTransform.ColumnInfo("C", "NAC"),
                new NAIndicatorTransform.ColumnInfo("D", "NAD"));

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
