// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.IntegrationTests.Datasets;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.Trainers;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.IntegrationTests
{
    public class PredictionScenarios : IntegrationTestBaseClass
    {
        public PredictionScenarios(ITestOutputHelper output) : base(output)
        {
        }

        class Prediction
        {
            public float Score { get; set; }
            public bool PredictedLabel { get; set; }
        }
        /// <summary>
        /// Reconfigurable predictions: The following should be possible: A user trains a binary classifier,
        /// and through the test evaluator gets a PR curve, the based on the PR curve picks a new threshold
        /// and configures the scorer (or more precisely instantiates a new scorer over the same model parameters)
        /// with some threshold derived from that.
        /// </summary>
        [Fact]
        public void ReconfigurablePrediction()
        {
            var mlContext = new MLContext(seed: 1);

            var options = new TextLoader.Options
            {
                HasHeader = TestDatasets.Sentiment.fileHasHeader,
                Separators = new[] { TestDatasets.Sentiment.fileSeparator }
            };

            var data = mlContext.Data.LoadFromTextFile<TweetSentiment>(TestCommon.GetDataPath(DataDir, TestDatasets.Sentiment.trainFilename),
                options);

            // Create a training pipeline.
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "SentimentText")
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
                    new LbfgsLogisticRegressionBinaryTrainer.Options { NumberOfThreads = 1 }));

            // Train the model.
            var model = pipeline.Fit(data);
            var engine = mlContext.Model.CreatePredictionEngine<TweetSentiment, Prediction>(model);
            var pr = engine.Predict(new TweetSentiment() { SentimentText = "Good Bad job" });
            // Score is 0.64 so predicted label is true.
            Assert.True(pr.PredictedLabel);
            Assert.True(pr.Score > 0);
            var transformers = new List<ITransformer>();
            foreach (var transform in model)
            {
                if (transform != model.LastTransformer)
                    transformers.Add(transform);
            }
            transformers.Add(mlContext.BinaryClassification.ChangeModelThreshold(model.LastTransformer, 0.7f));
            var newModel = new TransformerChain<BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>>(transformers.ToArray());
            var newEngine = mlContext.Model.CreatePredictionEngine<TweetSentiment, Prediction>(newModel);
            pr = newEngine.Predict(new TweetSentiment() { SentimentText = "Good Bad job" });
            // Score is still 0.64 but since threshold is no longer 0 but 0.7 predicted label now is false.

            Assert.False(pr.PredictedLabel);
            Assert.False(pr.Score > 0.7);
        }

        [Fact]
        public void ReconfigurablePredictionNoPipeline()
        {
            var mlContext = new MLContext(seed: 1);
            var data = mlContext.Data.LoadFromEnumerable(TypeTestData.GenerateDataset());
            var pipeline = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
                     new Trainers.LbfgsLogisticRegressionBinaryTrainer.Options { NumberOfThreads = 1 });
            var model = pipeline.Fit(data);
            var newModel = mlContext.BinaryClassification.ChangeModelThreshold(model, -2.0f);
            var rnd = new Random(1);
            var randomDataPoint = TypeTestData.GetRandomInstance(rnd);
            var engine = mlContext.Model.CreatePredictionEngine<TypeTestData, Prediction>(model);
            var pr = engine.Predict(randomDataPoint);
            // Score is -1.38 so predicted label is false.
            Assert.False(pr.PredictedLabel);
            Assert.True(pr.Score <= 0);
            var newEngine = mlContext.Model.CreatePredictionEngine<TypeTestData, Prediction>(newModel);
            pr = newEngine.Predict(randomDataPoint);
            // Score is still -1.38 but since threshold is no longer 0 but -2 predicted label now is true.
            Assert.True(pr.PredictedLabel);
            Assert.True(pr.Score <= 0);
        }

        [Fact]
        public void PredictionEngineModelDisposal()
        {
            var mlContext = new MLContext(seed: 1);
            var data = mlContext.Data.LoadFromEnumerable(TypeTestData.GenerateDataset());
            var pipeline = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
                     new Trainers.LbfgsLogisticRegressionBinaryTrainer.Options { NumberOfThreads = 1 });
            var model = pipeline.Fit(data);

            var engine = mlContext.Model.CreatePredictionEngine<TypeTestData, Prediction>(model, new PredictionEngineOptions());

            // Dispose of prediction engine, should dispose of model
            engine.Dispose();

            // Get disposed flag using reflection
            var bfIsDisposed = BindingFlags.Instance | BindingFlags.NonPublic;
            var field = model.GetType().BaseType.BaseType.GetField("_disposed", bfIsDisposed);

            // Make sure the model is actually disposed
            Assert.True((bool)field.GetValue(model));

            // Make a new model/prediction engine. Set the options so prediction engine doesn't dispose
            model = pipeline.Fit(data);

            var options = new PredictionEngineOptions()
            {
                OwnsTransformer = false
            };

            engine = mlContext.Model.CreatePredictionEngine<TypeTestData, Prediction>(model, options);

            // Dispose of prediction engine, shouldn't dispose of model
            engine.Dispose();

            // Make sure model is not disposed of.
            Assert.False((bool)field.GetValue(model));

            // Dispose of the model for test cleanliness
            model.Dispose();
        }
    }
}
