﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms.Text;
using System.Linq;
using Xunit;
using Microsoft.ML.Runtime.Internal.Internallearn;

namespace Microsoft.ML.Scenarios
{
#pragma warning disable 612
    public partial class ScenariosTests
    {
        [Fact]
        public void TrainAndPredictSentimentModelWithDirectionInstantiationTest()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var testDataPath = GetDataPath(SentimentTestPath);

            var env = new MLContext(seed: 1, conc: 1);
            // Pipeline
            var loader = env.Data.ReadFromTextFile(dataPath,
                columns: new[]
                {
                    new TextLoader.Column("Label", DataKind.Num, 0),
                    new TextLoader.Column("SentimentText", DataKind.Text, 1)
                },
                hasHeader: true
            );

            var trans = TextFeaturizingEstimator.Create(env, new TextFeaturizingEstimator.Arguments()
            {
                Column = new TextFeaturizingEstimator.Column
                {
                    Name = "Features",
                    Source = new[] { "SentimentText" }
                },
                OutputTokens = true,
                KeepPunctuations = false,
                UsePredefinedStopWordRemover = true,
                VectorNormalizer = TextFeaturizingEstimator.TextNormKind.L2,
                CharFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments() { NgramLength = 3, AllLengths = false },
                WordFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments() { NgramLength = 2, AllLengths = true },
            },
            loader);

            // Train
            var trainer = new FastTreeBinaryClassificationTrainer(env, DefaultColumnNames.Label, DefaultColumnNames.Features,
                numLeaves: 5, numTrees: 5, minDatapointsInLeaves: 2);

            var trainRoles = new RoleMappedData(trans, label: "Label", feature: "Features");
            var pred = trainer.Train(trainRoles);

            // Get scorer and evaluate the predictions from test data
            IDataScorerTransform testDataScorer = GetScorer(env, trans, pred, testDataPath);
            var metrics = EvaluateBinary(env, testDataScorer);
            ValidateBinaryMetrics(metrics);

            // Create prediction engine and test predictions
            var model = env.CreateBatchPredictionEngine<SentimentData, SentimentPrediction>(testDataScorer);
            var sentiments = GetTestData();
            var predictions = model.Predict(sentiments, false);
            Assert.Equal(2, predictions.Count());
            Assert.True(predictions.ElementAt(0).Sentiment);
            Assert.True(predictions.ElementAt(1).Sentiment);

            // Get feature importance based on feature gain during training
            var summary = ((ICanGetSummaryInKeyValuePairs)pred).GetSummaryInKeyValuePairs(trainRoles.Schema);
            Assert.Equal(1.0, (double)summary[0].Value, 1);
        }

        [Fact]
        public void TrainAndPredictSentimentModelWithDirectionInstantiationTestWithWordEmbedding()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var testDataPath = GetDataPath(SentimentTestPath);

            var env = new MLContext(seed: 1, conc: 1);
            // Pipeline
            var loader = env.Data.ReadFromTextFile(dataPath,
                columns: new[]
                {
                    new TextLoader.Column("Label", DataKind.Num, 0),
                    new TextLoader.Column("SentimentText", DataKind.Text, 1)
                },
                hasHeader: true 
            );

            var text = TextFeaturizingEstimator.Create(env, new TextFeaturizingEstimator.Arguments()
            {
                Column = new TextFeaturizingEstimator.Column
                {
                    Name = "WordEmbeddings",
                    Source = new[] { "SentimentText" }
                },
                OutputTokens = true,
                KeepPunctuations = false,
                UsePredefinedStopWordRemover = true,
                VectorNormalizer = TextFeaturizingEstimator.TextNormKind.None,
                CharFeatureExtractor = null,
                WordFeatureExtractor = null,
            },
            loader);

            var trans = WordEmbeddingsExtractingTransformer.Create(env, new WordEmbeddingsExtractingTransformer.Arguments()
            {
                Column = new WordEmbeddingsExtractingTransformer.Column[1]
                {
                        new WordEmbeddingsExtractingTransformer.Column
                        {
                            Name = "Features",
                            Source = "WordEmbeddings_TransformedText"
                        }
                },
                ModelKind = WordEmbeddingsExtractingTransformer.PretrainedModelKind.Sswe,
            }, text);
            // Train
            var trainer = new FastTreeBinaryClassificationTrainer(env, DefaultColumnNames.Label, DefaultColumnNames.Features, numLeaves: 5, numTrees: 5, minDatapointsInLeaves: 2);

            var trainRoles = new RoleMappedData(trans, label: "Label", feature: "Features");
            var pred = trainer.Train(trainRoles);
            // Get scorer and evaluate the predictions from test data
            IDataScorerTransform testDataScorer = GetScorer(env, trans, pred, testDataPath);
            var metrics = EvaluateBinary(env, testDataScorer);

            // SSWE is a simple word embedding model + we train on a really small dataset, so metrics are not great.
            Assert.Equal(.6667, metrics.Accuracy, 4);
            Assert.Equal(.71, metrics.Auc, 1);
            Assert.Equal(.58, metrics.Auprc, 2);
            // Create prediction engine and test predictions
            var model = env.CreateBatchPredictionEngine<SentimentData, SentimentPrediction>(testDataScorer);
            var sentiments = GetTestData();
            var predictions = model.Predict(sentiments, false);
            Assert.Equal(2, predictions.Count());
            Assert.True(predictions.ElementAt(0).Sentiment);
            Assert.True(predictions.ElementAt(1).Sentiment);

            // Get feature importance based on feature gain during training
            var summary = ((ICanGetSummaryInKeyValuePairs)pred).GetSummaryInKeyValuePairs(trainRoles.Schema);
            Assert.Equal(1.0, (double)summary[0].Value, 1);
        }

        private Microsoft.ML.Legacy.Models.BinaryClassificationMetrics EvaluateBinary(IHostEnvironment env, IDataView scoredData)
        {
            var dataEval = new RoleMappedData(scoredData, label: "Label", feature: "Features", opt: true);

            // Evaluate.
            // It does not work. It throws error "Failed to find 'Score' column" when Evaluate is called
            //var evaluator = new BinaryClassifierEvaluator(env, new BinaryClassifierEvaluator.Arguments());

            var evaluator = new BinaryClassifierMamlEvaluator(env, new BinaryClassifierMamlEvaluator.Arguments());
            var metricsDic = evaluator.Evaluate(dataEval);

            return Microsoft.ML.Legacy.Models.BinaryClassificationMetrics
                    .FromMetrics(env, metricsDic["OverallMetrics"], metricsDic["ConfusionMatrix"])[0];
        }
    }
#pragma warning restore 612
}
