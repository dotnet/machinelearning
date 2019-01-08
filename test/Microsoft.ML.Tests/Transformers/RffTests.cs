// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Projections;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class RffTests : TestDataPipeBase
    {
        public RffTests(ITestOutputHelper output) : base(output)
        {
        }

        private class TestClass
        {
            [VectorType(100)]
            public float[] A;
        }

        private class TestClassBiggerSize
        {
            [VectorType(200)]
            public float[] A;
        }

        private class TestClassInvalidSchema
        {
            public int A;
        }

        [Fact]
        public void RffWorkout()
        {
            Random rand = new Random();
            var data = new[] {
                new TestClass() { A = Enumerable.Range(0, 100).Select(x => (float)rand.NextDouble()).ToArray() },
                new TestClass() { A = Enumerable.Range(0, 100).Select(x => (float)rand.NextDouble()).ToArray() }
            };
            var invalidData = ComponentCreation.CreateDataView(Env, new[] { new TestClassInvalidSchema { A = 1 }, new TestClassInvalidSchema { A = 1 } });
            var validFitInvalidData = ComponentCreation.CreateDataView(Env, new[] { new TestClassBiggerSize { A = new float[200] }, new TestClassBiggerSize { A = new float[200] } });
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var generator = new GaussianFourierSampler.Arguments();

            var pipe = new RandomFourierFeaturizingEstimator(Env, new[]{
                    new RandomFourierFeaturizingTransformer.ColumnInfo("A", "RffA", 5, false),
                    new RandomFourierFeaturizingTransformer.ColumnInfo("A", "RffB", 10, true, new LaplacianFourierSampler.Arguments())
                });

            TestEstimatorCore(pipe, dataView, invalidInput: invalidData, validForFitNotValidForTransformInput: validFitInvalidData);
            Done();
        }

        [Fact]
        public void RffStatic()
        {
            string dataPath = GetDataPath("breast-cancer.txt");
            var reader = TextLoaderStatic.CreateReader(Env, ctx => (
                VectorFloat: ctx.LoadFloat(1, 8),
                Label: ctx.LoadFloat(0)
            ));

            var data = reader.Read(dataPath);

            var est = data.MakeNewEstimator()
                .Append(row => (
                RffVectorFloat: row.VectorFloat.LowerVectorSizeWithRandomFourierTransformation(3, true), row.Label));

            TestEstimatorCore(est.AsDynamic, data.AsDynamic);

            var outputPath = GetOutputPath("Rff", "featurized.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                IDataView savedData = TakeFilter.Create(Env, est.Fit(data).Transform(data).AsDynamic, 4);
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }
            CheckEquality("Rff", "featurized.tsv");
            Done();
        }



        [Fact]
        public void TestCommandLine()
        {
            Assert.Equal(0, Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0-100} xf=Rff{col=B:A dim=4 useSin+ kernel=LaplacianRandom}  in=f:\2.txt" }));
        }

        [Fact]
        public void TestOldSavingAndLoading()
        {
            Random rand = new Random();
            var data = new[] {
                new TestClass() { A = Enumerable.Range(0, 100).Select(x => (float)rand.NextDouble()).ToArray() },
                new TestClass() { A = Enumerable.Range(0, 100).Select(x => (float)rand.NextDouble()).ToArray() }
            };
            var dataView = ComponentCreation.CreateDataView(Env, data);

            var est = new RandomFourierFeaturizingEstimator(Env, new[]{
                    new RandomFourierFeaturizingTransformer.ColumnInfo("A", "RffA", 5, false),
                    new RandomFourierFeaturizingTransformer.ColumnInfo("A", "RffB", 10, true,new LaplacianFourierSampler.Arguments())
                });
            var result = est.Fit(dataView).Transform(dataView);
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
