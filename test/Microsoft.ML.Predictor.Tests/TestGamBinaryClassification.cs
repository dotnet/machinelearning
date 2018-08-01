// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.FastTree;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Runtime.RunTests
{
    public class GamTrainerTests : BaseTestBaseline
    {
        private readonly ITestOutputHelper Logger;

        public GamTrainerTests(ITestOutputHelper logger)
            : base(logger)
        {
            Logger = logger;
        }

        [Fact]
        [TestCategory("ParallelFasttree")]
        public void TestBinaryClassificationGamTrainer()
        {
            using (var env = new TlcEnvironment())
            {
                var dataset = LoadDataset(env);

                var binaryTrainer = new BinaryClassificationGamTrainer(env, new BinaryClassificationGamTrainer.Arguments() { LearningRates = 0.1 });
                binaryTrainer.Train(dataset);
                var binaryPredictor = binaryTrainer.CreatePredictor();

                var regressionTrainer = new RegressionGamTrainer(env, new RegressionGamTrainer.Arguments());
                regressionTrainer.Train(dataset);
                var regressionPredictor = regressionTrainer.CreatePredictor();

                //// Compare the predictors
                //ComparePredictors(env, firstPredictor, secondPredictor, dataset);
            }
        }

        private RoleMappedData LoadDataset(TlcEnvironment env)
        {
            var dataPath = GetDataPath("breast-cancer.txt");
            var outRoot = @"..\Common\CheckPointTest";

            var modelOutPath = DeleteOutputPath(outRoot, "codegen-model.zip");
            var csOutPath = DeleteOutputPath(outRoot, "codegen-out.cs");

            // Pipeline

            var loader = new TextLoader(env,
                new TextLoader.Arguments()
                {
                    Column = new[]
                    {
                        new TextLoader.Column()
                        {
                            Name = "Label",
                            Source = new [] { new TextLoader.Range() { Min=0, Max=0} },
                            Type = DataKind.R4
                        },
                        new TextLoader.Column()
                        {
                            Name = "Features",
                            Source = new [] { new TextLoader.Range() { Min=1, Max=9} },
                            Type = DataKind.R4
                        }
                    }
                },
                new MultiFileSource(dataPath));

            // Specify the dataset
            return new RoleMappedData(loader, label: "Label", feature: "Features");
        }

        private void ComparePredictors(TlcEnvironment env, IPredictor firstPredictor, IPredictor secondPredictor, RoleMappedData dataset)
        {
            var firstModel = GetModel(env, firstPredictor, dataset);
            var firstPredictions = firstModel.Predict(GetTestData(), false).Select(p => p.Probability);

            var secondModel = GetModel(env, secondPredictor, dataset);
            var secondPredictions = secondModel.Predict(GetTestData(), false).Select(p => p.Probability);

            Assert.Equal(firstPredictions, secondPredictions);
            Logger.WriteLine("Prediction comparison passed!");
        }

        private BatchPredictionEngine<TestData, Prediction> GetModel(TlcEnvironment env, IPredictor predictor, RoleMappedData dataset)
        {
            var scorer = ScoreUtils.GetScorer(predictor, dataset, env, dataset.Schema);
            return env.CreateBatchPredictionEngine<TestData, Prediction>(scorer);
        }

        public class TestData
        {
            [Column(ordinal: "0", name: "Label")]
            public float Label;

            [Column(ordinal: "1-9")]
            [VectorType(9)]
            public float[] Features;
        }

        public class Prediction
        {
            [ColumnName("Probability")]
            public float Probability;
        }

        private IEnumerable<TestData> GetTestData()
        {
            return new[]
            {
                new TestData
                {
                    Features = new float[9] {0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,}
                },
                new TestData
                {
                    Features = new float[9] {0f, 1f, 0f, 0f, -100f, 1000f, 0f, 0f, 0f,}
                },
                new TestData
                {
                    Features = new float[9] {0f, 1f, 1f, 0f, 10f, 0f, 0f, -123f, 0f,}
                },
                new TestData
                {
                    Features = new float[9] {0f, 1f, 1f, 1f, 0f, 78f, 0f, 0f, 0f,}
                },
                new TestData
                {
                    Features = new float[9] {0f, 1f, 0.345f, 1f, 1f, 0f, 0f, 0f, 0f,}
                },
                new TestData
                {
                    Features = new float[9] {1f, 1f, 1f, 1f, 1f, 1f, 1e-4f, 1f, 2f,}
                }
            };
        }
    }
}
