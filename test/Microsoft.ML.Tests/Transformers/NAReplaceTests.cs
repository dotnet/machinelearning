﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.StaticPipe;
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

        public NAReplaceTests(ITestOutputHelper output) : base(output)
        {
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

            var dataView = ML.Data.ReadFromEnumerable(data);
            var pipe = new MissingValueReplacingEstimator(Env,
                new MissingValueReplacingTransformer.ColumnInfo("A", "NAA", MissingValueReplacingTransformer.ColumnInfo.ReplacementMode.Mean),
                new MissingValueReplacingTransformer.ColumnInfo("B", "NAB", MissingValueReplacingTransformer.ColumnInfo.ReplacementMode.Mean),
                new MissingValueReplacingTransformer.ColumnInfo("C", "NAC", MissingValueReplacingTransformer.ColumnInfo.ReplacementMode.Mean),
                new MissingValueReplacingTransformer.ColumnInfo("D", "NAD", MissingValueReplacingTransformer.ColumnInfo.ReplacementMode.Mean));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void NAReplaceStatic()
        {
            string dataPath = GetDataPath("breast-cancer.txt");
            var reader = TextLoaderStatic.CreateReader(Env, ctx => (
                ScalarFloat: ctx.LoadFloat(1),
                ScalarDouble: ctx.LoadDouble(1),
                VectorFloat: ctx.LoadFloat(1, 4),
                VectorDoulbe: ctx.LoadDouble(1, 4)
            ));

            var data = reader.Read(dataPath);
            var wrongCollection = new[] { new TestClass() { A = 1, B = 3, C = new float[2] { 1, 2 }, D = new double[2] { 3, 4 } } };
            var invalidData = ML.Data.ReadFromEnumerable(wrongCollection);

            var est = data.MakeNewEstimator().
                   Append(row => (
                   A: row.ScalarFloat.ReplaceNaNValues(MissingValueReplacingTransformer.ColumnInfo.ReplacementMode.Maximum),
                   B: row.ScalarDouble.ReplaceNaNValues(MissingValueReplacingTransformer.ColumnInfo.ReplacementMode.Mean),
                   C: row.VectorFloat.ReplaceNaNValues(MissingValueReplacingTransformer.ColumnInfo.ReplacementMode.Mean),
                   D: row.VectorDoulbe.ReplaceNaNValues(MissingValueReplacingTransformer.ColumnInfo.ReplacementMode.Minimum)
                   ));

            TestEstimatorCore(est.AsDynamic, data.AsDynamic, invalidInput: invalidData);
            var outputPath = GetOutputPath("NAReplace", "featurized.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                var savedData = TakeFilter.Create(Env, est.Fit(data).Transform(data).AsDynamic, 4);
                var view = ColumnSelectingTransformer.CreateKeep(Env, savedData, new[] { "A", "B", "C", "D" });
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, view, fs, keepHidden: true);
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
                new TestClass() { A = 1,  B = 3, C= new float[2]{ 1, 2 } , D = new double[2]{ 3,4} },
                new TestClass() { A = float.NaN,  B = double.NaN, C= new float[2]{ float.NaN, float.NaN } , D = new double[2]{ double.NaN,double.NaN}},
                new TestClass() { A = float.NegativeInfinity, B = double.NegativeInfinity,C= new float[2]{ float.NegativeInfinity, float.NegativeInfinity } , D = new double[2]{ double.NegativeInfinity, double.NegativeInfinity}},
                new TestClass() { A = float.PositiveInfinity, B = double.PositiveInfinity,C= new float[2]{ float.PositiveInfinity, float.PositiveInfinity, } , D = new double[2]{  double.PositiveInfinity, double.PositiveInfinity}},
                new TestClass() { A = 2, B = 1 ,C= new float[2]{ 3, 4 } , D = new double[2]{ 5,6}},
            };

            var dataView = ML.Data.ReadFromEnumerable(data);
            var pipe = new MissingValueReplacingEstimator(Env,
                new MissingValueReplacingTransformer.ColumnInfo("A", "NAA", MissingValueReplacingTransformer.ColumnInfo.ReplacementMode.Mean),
                new MissingValueReplacingTransformer.ColumnInfo("B", "NAB", MissingValueReplacingTransformer.ColumnInfo.ReplacementMode.Mean),
                new MissingValueReplacingTransformer.ColumnInfo("C", "NAC", MissingValueReplacingTransformer.ColumnInfo.ReplacementMode.Mean),
                new MissingValueReplacingTransformer.ColumnInfo("D", "NAD", MissingValueReplacingTransformer.ColumnInfo.ReplacementMode.Mean));

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
