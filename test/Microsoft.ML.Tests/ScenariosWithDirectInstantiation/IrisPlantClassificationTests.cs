// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Model;
using System;
using System.IO;
using Xunit;

namespace Microsoft.ML.Scenarios
{
    public partial class ScenariosTests
    {
        [Fact]
        public void TrainAndPredictIrisModelUsingDirectInstantiationTest()
        {
            string dataPath = GetDataPath("iris.txt");
            string testDataPath = dataPath;

            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                // Pipeline
                var loader = new TextLoader(env,
                    new TextLoader.Arguments()
                    {
                        HasHeader = false,
                        Column = new[] {
                            new TextLoader.Column()
                            {
                                Name = "Label",
                                Source = new [] { new TextLoader.Range() { Min = 0, Max = 0} },
                                Type = DataKind.R4
                            },
                            new TextLoader.Column()
                            {
                                Name = "SepalLength",
                                Source = new [] { new TextLoader.Range() { Min = 1, Max = 1} },
                                Type = DataKind.R4
                            },
                            new TextLoader.Column()
                            {
                                Name = "SepalWidth",
                                Source = new [] { new TextLoader.Range() { Min = 2, Max = 2} },
                                Type = DataKind.R4
                            },
                            new TextLoader.Column()
                            {
                                Name = "PetalLength",
                                Source = new [] { new TextLoader.Range() { Min = 3, Max = 3} },
                                Type = DataKind.R4
                            },
                            new TextLoader.Column()
                            {
                                Name = "PetalWidth",
                                Source = new [] { new TextLoader.Range() { Min = 4, Max = 4} },
                                Type = DataKind.R4
                            }
                        }
                    }, new MultiFileSource(dataPath));

                IDataTransform trans = new ConcatTransform(env, loader, "Features",
                    "SepalLength", "SepalWidth", "PetalLength", "PetalWidth");

                // Normalizer is not automatically added though the trainer has 'NormalizeFeatures' On/Auto
                trans = NormalizeTransform.CreateMinMaxNormalizer(env, trans, "Features");

                // Train
                var trainer = new SdcaMultiClassTrainer(env, new SdcaMultiClassTrainer.Arguments());

                // Explicity adding CacheDataView since caching is not working though trainer has 'Caching' On/Auto
                var cached = new CacheDataView(env, trans, prefetch: null);
                var trainRoles = new RoleMappedData(cached, label: "Label", feature: "Features");
                trainer.Train(trainRoles);

                // Get scorer and evaluate the predictions from test data
                var pred = trainer.CreatePredictor();
                IDataScorerTransform testDataScorer = GetScorer(env, trans, pred, testDataPath);
                var metrics = Evaluate(env, testDataScorer);
                CompareMatrics(metrics);

                // Create prediction engine and test predictions
                var model = env.CreatePredictionEngine<IrisData, IrisPrediction>(testDataScorer);
                ComparePredictions(model);

                // Get feature importance i.e. weight vector
                var summary = ((MulticlassLogisticRegressionPredictor)pred).GetSummaryInKeyValuePairs(trainRoles.Schema);
                Assert.Equal(7.757867, Convert.ToDouble(summary[0].Value), 5);
            }
        }

        private void ComparePredictions(PredictionEngine<IrisData, IrisPrediction> model)
        {
            IrisPrediction prediction = model.Predict(new IrisData()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f,
            });

            Assert.Equal(1, prediction.PredictedLabels[0], 2);
            Assert.Equal(0, prediction.PredictedLabels[1], 2);
            Assert.Equal(0, prediction.PredictedLabels[2], 2);

            prediction = model.Predict(new IrisData()
            {
                SepalLength = 3.1f,
                SepalWidth = 5.5f,
                PetalLength = 2.2f,
                PetalWidth = 6.4f,
            });

            Assert.Equal(0, prediction.PredictedLabels[0], 2);
            Assert.Equal(0, prediction.PredictedLabels[1], 2);
            Assert.Equal(1, prediction.PredictedLabels[2], 2);

            prediction = model.Predict(new IrisData()
            {
                SepalLength = 3.1f,
                SepalWidth = 2.5f,
                PetalLength = 1.2f,
                PetalWidth = 4.4f,
            });

            Assert.Equal(.2, prediction.PredictedLabels[0], 1);
            Assert.Equal(.8, prediction.PredictedLabels[1], 1);
            Assert.Equal(0, prediction.PredictedLabels[2], 2);
        }

        private void CompareMatrics(ClassificationMetrics metrics)
        {
            Assert.Equal(.98, metrics.AccuracyMacro);
            Assert.Equal(.98, metrics.AccuracyMicro, 2);
            Assert.Equal(.06, metrics.LogLoss, 2);
            Assert.InRange(metrics.LogLossReduction, 94, 96);
            Assert.Equal(1, metrics.TopKAccuracy);

            Assert.Equal(3, metrics.PerClassLogLoss.Length);
            Assert.Equal(0, metrics.PerClassLogLoss[0], 1);
            Assert.Equal(.1, metrics.PerClassLogLoss[1], 1);
            Assert.Equal(.1, metrics.PerClassLogLoss[2], 1);

            ConfusionMatrix matrix = metrics.ConfusionMatrix;
            Assert.Equal(3, matrix.Order);
            Assert.Equal(3, matrix.ClassNames.Count);
            Assert.Equal("0", matrix.ClassNames[0]);
            Assert.Equal("1", matrix.ClassNames[1]);
            Assert.Equal("2", matrix.ClassNames[2]);

            Assert.Equal(50, matrix[0, 0]);
            Assert.Equal(50, matrix["0", "0"]);
            Assert.Equal(0, matrix[0, 1]);
            Assert.Equal(0, matrix["0", "1"]);
            Assert.Equal(0, matrix[0, 2]);
            Assert.Equal(0, matrix["0", "2"]);

            Assert.Equal(0, matrix[1, 0]);
            Assert.Equal(0, matrix["1", "0"]);
            Assert.Equal(48, matrix[1, 1]);
            Assert.Equal(48, matrix["1", "1"]);
            Assert.Equal(2, matrix[1, 2]);
            Assert.Equal(2, matrix["1", "2"]);

            Assert.Equal(0, matrix[2, 0]);
            Assert.Equal(0, matrix["2", "0"]);
            Assert.Equal(1, matrix[2, 1]);
            Assert.Equal(1, matrix["2", "1"]);
            Assert.Equal(49, matrix[2, 2]);
            Assert.Equal(49, matrix["2", "2"]);
        }

        private ClassificationMetrics Evaluate(IHostEnvironment env, IDataView scoredData)
        {
            var dataEval = new RoleMappedData(scoredData, label: "Label", feature: "Features", opt: true);
            
            // Evaluate.
            // It does not work. It throws error "Failed to find 'Score' column" when Evaluate is called
            //var evaluator = new MultiClassClassifierEvaluator(env, new MultiClassClassifierEvaluator.Arguments() { OutputTopKAcc = 3 });

            var evaluator = new MultiClassMamlEvaluator(env, new MultiClassMamlEvaluator.Arguments() { OutputTopKAcc = 3 });
            var metricsDic = evaluator.Evaluate(dataEval);

            return ClassificationMetrics.FromMetrics(env, metricsDic["OverallMetrics"], metricsDic["ConfusionMatrix"])[0];
        }

        private IDataScorerTransform GetScorer(IHostEnvironment env, IDataView transforms, IPredictor pred, string testDataPath = null)
        {
            using (var ch = env.Start("Saving model"))
            using (var memoryStream = new MemoryStream())
            {
                var trainRoles = new RoleMappedData(transforms, label: "Label", feature: "Features");

                // Model cannot be saved with CacheDataView
                TrainUtils.SaveModel(env, ch, memoryStream, pred, trainRoles);
                memoryStream.Position = 0;
                using (var rep = RepositoryReader.Open(memoryStream, ch))
                {
                    IDataLoader testPipe = ModelFileUtils.LoadLoader(env, rep, new MultiFileSource(testDataPath), true);
                    RoleMappedData testRoles = new RoleMappedData(testPipe, label: "Label", feature: "Features");
                    return ScoreUtils.GetScorer(pred, testRoles, env, testRoles.Schema);
                }
            }
        }
    }
}
