// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.Internal.Calibration;
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

        /// <summary>
        /// Tests the BinaryClassificationGamTrainer to make sure it runs end-to-end
        /// and verifies that the training loss and validation metrics are what would
        /// be expected based on the output of the resulting model.
        /// Assumptions:
        ///  1. UnbalancedSets is false: The validation metric implemented here is for balanced sets.
        ///  2. The trainer does not prune: The training loss is then based on the 
        ///     subgraph output, and the test then computes off the full graph, testing
        ///     that the subgraph-to-graph conversion was done correctly. If pruning is used,
        ///     then the training loss is then based on the resulting graph, so the test will
        ///     not validate that the end graph was constructed correctly.
        /// </summary>
        [Fact]
        [TestCategory("GAM")]
        public void TestBinaryClassificationGamTrainer()
        {
            using (var env = new TlcEnvironment())
            {
                var trainFile = "binary_sine_logistic_10k.tsv";
                var validationFile = "binary_sine_logistic_10k_valid.tsv";

                var trainingSet = LoadDataset(env, trainFile);
                var validationSet = LoadDataset(env, validationFile);

                var context = new TrainContext(trainingSet: trainingSet, validationSet: validationSet);

                var numIterations = 100;
                var LearningRates = 0.002;

                var binaryTrainer = new BinaryClassificationGamTrainer(env,
                    new BinaryClassificationGamTrainer.Arguments() {
                        NumIterations = numIterations,
                        LearningRates = LearningRates
                    });
                var binaryPredictor = (CalibratedPredictor) binaryTrainer.Train(context);

                var gamPredictor = (BinaryClassGamPredictor)binaryPredictor.SubPredictor;

                GetSummary(gamPredictor, trainingSet.Schema);

                double trainLoss = ComputeLoss(env, binaryPredictor, trainingSet, trainFile, ComputeLogisticGradient);

                Assert.Equal(gamPredictor.TrainingSummary.TrainLoss, trainLoss, 4);

                double validationLoss = ComputeLoss(env, binaryPredictor, validationSet, validationFile,
                    ComputeBinaryLossRate, BinaryLossRateCorrection);

                Assert.Equal(gamPredictor.TrainingSummary.ValidMetric, validationLoss, 4);
            }
        }

        /// <summary>
        /// Tests the RegressionGamTrainer to make sure it runs end-to-end
        /// and verifies that the training loss and validation metrics are what would
        /// be expected based on the output of the resulting model.
        /// Assumptions:
        ///  1. The validation metric is L2 (here assumed to be the default). The validation
        ///     metric implemented here is L2.
        ///  2. The trainer does not prune: The training loss is then based on the 
        ///     subgraph output, and the test then computes off the full graph, testing
        ///     that the subgraph-to-graph conversion was done correctly. If pruning is used,
        ///     then the training loss is then based on the resulting graph, so the test will
        ///     not validate that the end graph was constructed correctly.
        /// </summary>
        [Fact]
        [TestCategory("GAM")]
        public void TestRegressionGamTrainer()
        {
            using (var env = new TlcEnvironment())
            {
                // Tuning parameters for testing
                var numIterations = 100;
                var LearningRates = 0.002;

                // The datasets to use here
                var trainFile = "regression_sine_identity_10k.tsv";
                var validationFile = "regression_sine_identity_10k_valid.tsv";

                var trainingSet = LoadDataset(env, trainFile);
                var validationSet = LoadDataset(env, validationFile);
                var context = new TrainContext(trainingSet: trainingSet, validationSet: validationSet);

                var regressionTrainer = new RegressionGamTrainer(env, 
                    new RegressionGamTrainer.Arguments()
                    {
                        NumIterations = numIterations,
                        LearningRates = LearningRates
                    });
                var regressionPredictor = regressionTrainer.Train(context);

                GetSummary(regressionPredictor, trainingSet.Schema);

                double trainLoss = ComputeLoss(env, regressionPredictor, trainingSet, trainFile, ComputeRegressionGradient);

                Assert.Equal(regressionPredictor.TrainingSummary.TrainLoss, trainLoss, 4);

                double validationLoss = ComputeLoss(env, regressionPredictor, validationSet, validationFile, 
                    ComputeL2Loss, L2LossCorrection);

                Assert.Equal(regressionPredictor.TrainingSummary.ValidMetric, validationLoss, 4);
            }
        }

        /// <summary>
        /// Compute the loss function over a dataset
        /// </summary>
        /// <param name="env">TLC Environment</param>
        /// <param name="predictor">The predictor to compute the loss function with</param>
        /// <param name="dataset">The daatset used to create the predictor</param>
        /// <param name="fileName">The location of the data file to score with in the data directory</param>
        /// <param name="lossFunction">The loss function to use on each point</param>
        /// <param name="correction">The correction to apply on the sum of per-instance loss</param>
        /// <returns></returns>
        private double ComputeLoss(TlcEnvironment env, IPredictor predictor,
            RoleMappedData dataset, string fileName,
            Func<float, TestData, double> lossFunction, Func<double, int, double> correction = null)
        {
            var testData = LoadDataAsObjects(GetDataPath(fileName));
            var scores = GetModel(env, predictor, dataset)
                            .Predict(testData, false)
                            .Select(p => p.Score);

            var gradients = scores.Zip(testData, (score, row) => lossFunction(score, row));

            var total = gradients.Sum();

            if (correction != null)
                total = correction(total, testData.Count());

            return total;
        }


        private double ComputeLogisticGradient(float score, TestData row)
        {
            double sigmoidParameter = 1.0;
            int label = row.Label == 1 ? 1 : -1;
            double recip = 1;
            double response = 2.0 * label * sigmoidParameter / (1.0 + Math.Exp(2.0 * label * sigmoidParameter * score));
            double absResponse = Math.Abs(response);
            double pLambda = response * recip;

            return pLambda;
        }

        private double ComputeBinaryLossRate(float score, TestData row)
        {
            double sigmoidParameter = 1.0;
            int label = row.Label == 1 ? 1 : -1;
            double loss = Math.Log(1.0 + Math.Exp(-2.0 * sigmoidParameter * label * score));

            return loss;
        }
        
        private double BinaryLossRateCorrection(double total, int length)
        {
            return total / length;
        }

        private double ComputeRegressionGradient(float score, TestData row)
        {
            return row.Label - score;
        }

        private double ComputeL2Loss(float score, TestData row)
        {
            return (row.Label - score)*(row.Label - score);
        }

        private double L2LossCorrection(double total, int length)
        {
            return Math.Sqrt(total / length);
        }

        private void GetSummary(GamPredictorBase gamPredictor, RoleMappedSchema schema)
        {
            using (StringWriter writer = new StringWriter())
            {
                gamPredictor.SaveSummary(writer, schema);
                Logger.WriteLine("Summary {0}", writer.ToString());
            }
        }

        /// <summary>
        /// Load a dataset with fixed features.
        /// Note: Possible to extend to variable-length features and generalize
        /// </summary>
        /// <param name="env">TLC Environment</param>
        /// <param name="name">The name of the file in the data directory</param>
        /// <returns></returns>
        private RoleMappedData LoadDataset(TlcEnvironment env, string name)
        {
            var dataPath = GetDataPath(name);

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
                            Source = new [] { new TextLoader.Range() { Min=1, Max=3} },
                            Type = DataKind.R4
                        }
                    }
                },
                new MultiFileSource(dataPath));
            
            return new RoleMappedData(loader, label: "Label", feature: "Features");
        }

        /// <summary>
        /// Cretae a model from a predictor and a dataset
        /// </summary>
        /// <param name="env">TLC Environment</param>
        /// <param name="predictor">The predictor that was built on the dataset</param>
        /// <param name="dataset">The dataset used to build the predictor</param>
        /// <returns></returns>
        private BatchPredictionEngine<TestData, Prediction> GetModel(TlcEnvironment env, IPredictor predictor, RoleMappedData dataset)
        {
            var scorer = ScoreUtils.GetScorer(predictor, dataset, env, dataset.Schema);
            return env.CreateBatchPredictionEngine<TestData, Prediction>(scorer);
        }

        /// <summary>
        /// The test data, of fixed feature size
        /// </summary>
        private class TestData
        {
            [Column(ordinal: "0", name: "Label")]
            public float Label;

            [Column(ordinal: "1")]
            [VectorType(3)]
            public float[] Features;
        }

        public class Prediction
        {
            //[ColumnName("Probability")]
            //public float Probability;

            [ColumnName("Score")]
            public float Score;
        }

        /// <summary>
        /// Loader to bring the data back as a typed IEnumerable
        /// </summary>
        /// <param name="filePath">The location of the data on the local disk</param>
        /// <returns></returns>
        private IEnumerable<TestData> LoadDataAsObjects(string filePath)
        {
            StreamReader reader = new StreamReader(filePath);
            string inputLine = "";
            var dataset = new List<TestData>();
            while ((inputLine = reader.ReadLine()) != null)
            {
                string[] inputArray = inputLine.Split(new char[] { '\t' });
                if (inputArray.Count() > 0)
                {
                    yield return new TestData
                    {
                        Label = float.Parse(inputArray[0]),
                        Features = inputArray.Skip(1).Select(x => float.Parse(x)).ToArray<float>()
                    };
                }
            }
        }
    }
}
