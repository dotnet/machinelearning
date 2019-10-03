﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
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
            string dataPath = GetDataPath("breast-cancer.txt");
            var data = ML.Data.LoadFromTextFile(dataPath, new[] {
                new TextLoader.Column("ScalarFloat", DataKind.Single, 1),
                new TextLoader.Column("ScalarDouble", DataKind.Double, 1),
                new TextLoader.Column("VectorFloat", DataKind.Single, 1, 4),
                new TextLoader.Column("VectorDoulbe", DataKind.Double, 1, 4)
            });

            var wrongCollection = new[] { new TestClass() { A = 1, B = 3, C = new float[2] { 1, 2 }, D = new double[2] { 3, 4 } } };
            var invalidData = ML.Data.LoadFromEnumerable(wrongCollection);

            var est = ML.Transforms.ReplaceMissingValues("A", "ScalarFloat", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Maximum)
                .Append(ML.Transforms.ReplaceMissingValues("B", "ScalarDouble", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(ML.Transforms.ReplaceMissingValues("C", "VectorFloat", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(ML.Transforms.ReplaceMissingValues("D", "VectorDoulbe", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Minimum));

            TestEstimatorCore(est, data, invalidInput: invalidData);
            var outputPath = GetOutputPath("NAReplace", "featurized.tsv");
            var savedData = ML.Data.TakeRows(est.Fit(data).Transform(data), 4);
            var view = ML.Transforms.SelectColumns("A", "B", "C", "D").Fit(savedData).Transform(savedData);
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
