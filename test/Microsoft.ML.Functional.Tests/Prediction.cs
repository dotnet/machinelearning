// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Functional.Tests.Datasets;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Functional.Tests
{
    public class PredictionScenarios : BaseTestClass
    {
        public PredictionScenarios(ITestOutputHelper output) : base(output)
        {
        }

        class Answer
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

            var data = mlContext.Data.LoadFromTextFile<TweetSentiment>(GetDataPath(TestDatasets.Sentiment.trainFilename),
                hasHeader: TestDatasets.Sentiment.fileHasHeader,
                separatorChar: TestDatasets.Sentiment.fileSeparator);

            // Create a training pipeline.
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "SentimentText")
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.BinaryClassification.Trainers.LogisticRegression(
                    new Trainers.LogisticRegressionBinaryClassificationTrainer.Options { NumberOfThreads = 1 }));

            // Train the model.
            var model = pipeline.Fit(data);
            var engine = model.CreatePredictionEngine<TweetSentiment, Answer>(mlContext);
            var pr = engine.Predict(new TweetSentiment() { SentimentText = "Good Bad job" });
            // Score is 0.64 so predicted label is true.
            Assert.True(pr.PredictedLabel);
            Assert.True(pr.Score > 0);
            var newModel = mlContext.BinaryClassification.ChangeModelThreshold(model, 0.7f);
            var newEngine = newModel.CreatePredictionEngine<TweetSentiment, Answer>(mlContext);
            pr = newEngine.Predict(new TweetSentiment() { SentimentText = "Good Bad job" });
            // Score is still 0.64 but since threshold is no longer 0 but 0.7 predicted label now is false.

            Assert.False(pr.PredictedLabel);
            Assert.False(pr.Score > 0.7);
        }
    }
}
