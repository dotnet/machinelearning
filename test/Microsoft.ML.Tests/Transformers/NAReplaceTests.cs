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
    public class NAReplaceTests : TestDataPipeBase
    {
        private class TestClass
        {
            public float A;
            public string B;
            public double C;
            [VectorType(2)]
            public float[] D;
            [VectorType(2)]
            public double[] E;
        }

        public NAReplaceTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void NAReplaceWorkout()
        {
            var data = new[] {
                new TestClass() { A = 1, B = "A", C = 3, D= new float[2]{ 1, 2 } , E = new double[2]{ 3,4} },
                new TestClass() { A = float.NaN, B = null, C = double.NaN, D= new float[2]{ float.NaN, float.NaN } , E = new double[2]{ double.NaN,double.NaN}},
                new TestClass() { A = float.NegativeInfinity, B = null, C = double.NegativeInfinity,D= new float[2]{ float.NegativeInfinity, float.NegativeInfinity } , E = new double[2]{ double.NegativeInfinity, double.NegativeInfinity}},
                new TestClass() { A = float.PositiveInfinity, B = null, C = double.PositiveInfinity,D= new float[2]{ float.PositiveInfinity, float.PositiveInfinity, } , E = new double[2]{  double.PositiveInfinity, double.PositiveInfinity}},
                new TestClass() { A = 2, B = "B", C = 1 ,D= new float[2]{ 3, 4 } , E = new double[2]{ 5,6}},
            };

            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new NAReplaceEstimator(Env,
                new NAReplaceTransform.ColumnInfo("A", "NAA", NAReplaceTransform.ColumnInfo.ReplacementMode.Mean),
                new NAReplaceTransform.ColumnInfo("B", "NAB", NAReplaceTransform.ColumnInfo.ReplacementMode.DefaultValue),
                new NAReplaceTransform.ColumnInfo("C", "NAC", NAReplaceTransform.ColumnInfo.ReplacementMode.Mean),
                new NAReplaceTransform.ColumnInfo("D", "NAD", NAReplaceTransform.ColumnInfo.ReplacementMode.Mean),
                new NAReplaceTransform.ColumnInfo("E", "NAE", NAReplaceTransform.ColumnInfo.ReplacementMode.Mean));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void NAReplaceStatic()
        {
            string dataPath = GetDataPath("breast-cancer.txt");
            var reader = TextLoader.CreateReader(Env, ctx => (
                ScalarString: ctx.LoadText(1),
                ScalarFloat: ctx.LoadFloat(1),
                ScalarDouble: ctx.LoadDouble(1),
                VectorString: ctx.LoadText(1, 4),
                VectorFloat: ctx.LoadFloat(1, 4),
                VectorDoulbe: ctx.LoadDouble(1, 4)
            ));

            var data = reader.Read(new MultiFileSource(dataPath));
            var wrongCollection = new[] { new TestClass() { A = 1, B = "A", C = 3, D = new float[2] { 1, 2 }, E = new double[2] { 3, 4 } } };
            var invalidData = ComponentCreation.CreateDataView(Env, wrongCollection);

            var est = data.MakeNewEstimator().
                   Append(row => (
                   A: row.ScalarString.ReplaceWithMissingValues(),
                   B: row.ScalarFloat.ReplaceWithMissingValues(NAReplaceTransform.ColumnInfo.ReplacementMode.Maximum),
                   C: row.ScalarDouble.ReplaceWithMissingValues(NAReplaceTransform.ColumnInfo.ReplacementMode.Mean),
                   D: row.VectorString.ReplaceWithMissingValues(),
                   E: row.VectorFloat.ReplaceWithMissingValues(NAReplaceTransform.ColumnInfo.ReplacementMode.Mean),
                   F: row.VectorDoulbe.ReplaceWithMissingValues(NAReplaceTransform.ColumnInfo.ReplacementMode.Minimum)
                   ));

            TestEstimatorCore(est.AsDynamic, data.AsDynamic, invalidInput: invalidData);
            var outputPath = GetOutputPath("NAReplace", "featurized.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                IDataView savedData = TakeFilter.Create(Env, est.Fit(data).Transform(data).AsDynamic, 4);
                savedData = new ChooseColumnsTransform(Env, savedData, "A", "B", "C", "D", "E");
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

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
                new TestClass() { A = 1, B = "A", C = 3, D= new float[2]{ 1, 2 } , E = new double[2]{ 3,4} },
                new TestClass() { A = float.NaN, B = null, C = double.NaN, D= new float[2]{ float.NaN, float.NaN } , E = new double[2]{ double.NaN,double.NaN}},
                new TestClass() { A = float.NegativeInfinity, B = null, C = double.NegativeInfinity,D= new float[2]{ float.NegativeInfinity, float.NegativeInfinity } , E = new double[2]{ double.NegativeInfinity, double.NegativeInfinity}},
                new TestClass() { A = float.PositiveInfinity, B = null, C = double.PositiveInfinity,D= new float[2]{ float.PositiveInfinity, float.PositiveInfinity, } , E = new double[2]{  double.PositiveInfinity, double.PositiveInfinity}},
                new TestClass() { A = 2, B = "B", C = 1 ,D= new float[2]{ 3, 4 } , E = new double[2]{ 5,6}},
            };

            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new NAReplaceEstimator(Env,
                new NAReplaceTransform.ColumnInfo("A", "NAA", NAReplaceTransform.ColumnInfo.ReplacementMode.Mean),
                new NAReplaceTransform.ColumnInfo("B", "NAB", NAReplaceTransform.ColumnInfo.ReplacementMode.DefaultValue),
                new NAReplaceTransform.ColumnInfo("C", "NAC", NAReplaceTransform.ColumnInfo.ReplacementMode.Mean),
                new NAReplaceTransform.ColumnInfo("D", "NAD", NAReplaceTransform.ColumnInfo.ReplacementMode.Mean),
                new NAReplaceTransform.ColumnInfo("E", "NAE", NAReplaceTransform.ColumnInfo.ReplacementMode.Mean));

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
