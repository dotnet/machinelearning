// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.Runtime;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Tools;
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

            var dataView = ML.Data.LoadFromEnumerable(data);
            var pipe = ML.Transforms.IndicateMissingValues(new ColumnOptions[] { ("NAA", "A"), ("NAB", "B"), ("NAC", "C"), ("NAD", "D") });
            TestEstimatorCore(pipe, dataView);
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

            var dataView = ML.Data.LoadFromEnumerable(data);
            var pipe = ML.Transforms.IndicateMissingValues(new ColumnOptions[] { ("NAA", "A"), ("NAB", "B"), ("NAC", "C"), ("NAD", "D") });
            var result = pipe.Fit(dataView).Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(Env, Env.Start("saving"), ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(Env, dataView, ms);
            }
        }

        [Fact]
        public void NAIndicatorFileOutput()
        {
            string dataPath = GetDataPath("breast-cancer.txt");
            var reader = TextLoaderStatic.CreateLoader(ML, ctx => (
                ScalarFloat: ctx.LoadFloat(1),
                ScalarDouble: ctx.LoadDouble(1),
                VectorFloat: ctx.LoadFloat(1, 4),
                VectorDoulbe: ctx.LoadDouble(1, 4)
            ));

            var data = reader.Load(new MultiFileSource(dataPath)).AsDynamic;
            var wrongCollection = new[] { new TestClass() { A = 1, B = 3, C = new float[2] { 1, 2 }, D = new double[2] { 3, 4 } } };
            var invalidData = ML.Data.LoadFromEnumerable(wrongCollection);
            var est = ML.Transforms.IndicateMissingValues(new ColumnOptions[] 
            {
                ("A", "ScalarFloat"), ("B", "ScalarDouble"),
                ("C", "VectorFloat"), ("D", "VectorDoulbe")
            });

            TestEstimatorCore(est, data, invalidInput: invalidData);
            var outputPath = GetOutputPath("NAIndicator", "featurized.tsv");
            using (var ch = ((IHostEnvironment)ML).Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true });
                var savedData = ML.Data.TakeRows(est.Fit(data).Transform(data), 4);
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("NAIndicator", "featurized.tsv");
            Done();
        }

        [Fact]
        public void NAIndicatorMetadataTest()
        {
            var data = new[] {
                new TestClass() { A = 1, B = 3, C = new float[2]{ 1, 2 } , D = new double[2]{ 3,4} },
                new TestClass() { A = float.NaN, B = double.NaN, C = new float[2]{ float.NaN, float.NaN } , D = new double[2]{ double.NaN,double.NaN}},
                new TestClass() { A = float.NegativeInfinity, B = double.NegativeInfinity, C = new float[2]{ float.NegativeInfinity, float.NegativeInfinity } , D = new double[2]{ double.NegativeInfinity, double.NegativeInfinity}},
                new TestClass() { A = float.PositiveInfinity, B = double.PositiveInfinity, C = new float[2]{ float.PositiveInfinity, float.PositiveInfinity, } , D = new double[2]{  double.PositiveInfinity, double.PositiveInfinity}},
                new TestClass() { A = 2, B = 1, C = new float[2]{ 3, 4 } , D = new double[2]{ 5,6}},
            };

            var dataView = ML.Data.LoadFromEnumerable(data);
            var pipe = ML.Transforms.Categorical.OneHotEncoding("CatA", "A");
            var newpipe = pipe.Append(ML.Transforms.IndicateMissingValues(("NAA", "CatA")));
            var result = newpipe.Fit(dataView).Transform(dataView);
            Assert.True(result.Schema.TryGetColumnIndex("NAA", out var col));
            // Check that the column is normalized.
            Assert.True(result.Schema[col].IsNormalized());
            // Check that slot names metadata was correctly created.
            var value = new VBuffer<ReadOnlyMemory<char>>();
            result.Schema[col].GetSlotNames(ref value);
            Assert.True(value.Length == 4);
            var mem = new ReadOnlyMemory<char>();
            value.GetItemOrDefault(0, ref mem);
            Assert.True(mem.ToString() == "1");
            value.GetItemOrDefault(1, ref mem);
            Assert.True(mem.ToString() == "-Infinity");
            value.GetItemOrDefault(2, ref mem);
            Assert.True(mem.ToString() == "Infinity");
            value.GetItemOrDefault(3, ref mem);
            Assert.True(mem.ToString() == "2");
        }
    }
}
