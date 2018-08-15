// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Xunit;

namespace Microsoft.ML.Scenarios
{
    public partial class ScenariosTests
    {
        [Fact]
        public void TrainAndPredictSentimentModelWithDirectionInstantiationTest()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var testDataPath = GetDataPath(SentimentTestPath);

            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                // Pipeline
                var loader = new TextLoader(env,
                new TextLoader.Arguments()
                {
                    Separator = "tab",
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoader.Column("Label", DataKind.Num, 0),
                        new TextLoader.Column("SentimentText", DataKind.Text, 1)
                    }
                }, new MultiFileSource(dataPath));

                var trans = TextTransform.Create(env, new TextTransform.Arguments()
                {
                    Column = new TextTransform.Column
                    {
                        Name = "Features",
                        Source = new[] { "SentimentText" }
                    },
                    KeepDiacritics = false,
                    KeepPunctuations = false,
                    TextCase = Runtime.TextAnalytics.TextNormalizerTransform.CaseNormalizationMode.Lower,
                    OutputTokens = true,
                    StopWordsRemover = new Runtime.TextAnalytics.PredefinedStopWordsRemoverFactory(),
                    VectorNormalizer = TextTransform.TextNormKind.L2,
                    CharFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments() { NgramLength = 3, AllLengths = false },
                    WordFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments() { NgramLength = 2, AllLengths = true },
                },
                loader);

                // Train
                var trainer = new FastTreeBinaryClassificationTrainer(env, new FastTreeBinaryClassificationTrainer.Arguments()
                {
                    NumLeaves = 5,
                    NumTrees = 5,
                    MinDocumentsInLeafs = 2
                });

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
                Assert.True(predictions.ElementAt(0).Sentiment.IsTrue);
                Assert.True(predictions.ElementAt(1).Sentiment.IsTrue);

                // Get feature importance based on feature gain during training
                var summary = ((FeatureWeightsCalibratedPredictor)pred).GetSummaryInKeyValuePairs(trainRoles.Schema);
                Assert.Equal(1.0, (double)summary[0].Value, 1);
            }
        }

        [Fact]
        public void TrainAndPredictSentimentModelWithDirectionInstantiationTestWithWordEmbedding()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var testDataPath = GetDataPath(SentimentTestPath);

            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                // Pipeline
                var loader = new TextLoader(env,
                new TextLoader.Arguments()
                {
                    Separator = "tab",
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoader.Column("Label", DataKind.Num, 0),
                        new TextLoader.Column("SentimentText", DataKind.Text, 1)
                    }
                }, new MultiFileSource(dataPath));

                var text = TextTransform.Create(env, new TextTransform.Arguments()
                {
                    Column = new TextTransform.Column
                    {
                        Name = "WordEmbeddings",
                        Source = new[] { "SentimentText" }
                    },
                    KeepDiacritics = false,
                    KeepPunctuations = false,
                    TextCase = Runtime.TextAnalytics.TextNormalizerTransform.CaseNormalizationMode.Lower,
                    OutputTokens = true,
                    StopWordsRemover = new Runtime.TextAnalytics.PredefinedStopWordsRemoverFactory(),
                    VectorNormalizer = TextTransform.TextNormKind.None,
                    CharFeatureExtractor = null,
                    WordFeatureExtractor = null,
                },
                loader);

                var trans = new WordEmbeddingsTransform(env, new WordEmbeddingsTransform.Arguments()
                {
                    Column = new WordEmbeddingsTransform.Column[1]
                    {
                        new WordEmbeddingsTransform.Column
                        {
                            Name = "Features",
                            Source = "WordEmbeddings_TransformedText"
                        }
                    },
                    ModelKind = WordEmbeddingsTransform.PretrainedModelKind.Sswe,
                }, text);
                // Train
                var trainer = new FastTreeBinaryClassificationTrainer(env, new FastTreeBinaryClassificationTrainer.Arguments()
                {
                    NumLeaves = 5,
                    NumTrees = 5,
                    MinDocumentsInLeafs = 2
                });

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
                Assert.True(predictions.ElementAt(0).Sentiment.IsTrue);
                Assert.True(predictions.ElementAt(1).Sentiment.IsTrue);

                // Get feature importance based on feature gain during training
                var summary = ((FeatureWeightsCalibratedPredictor)pred).GetSummaryInKeyValuePairs(trainRoles.Schema);
                Assert.Equal(1.0, (double)summary[0].Value, 1);
            }
        }

        private BinaryClassificationMetrics EvaluateBinary(IHostEnvironment env, IDataView scoredData)
        {
            var dataEval = new RoleMappedData(scoredData, label: "Label", feature: "Features", opt: true);

            // Evaluate.
            // It does not work. It throws error "Failed to find 'Score' column" when Evaluate is called
            //var evaluator = new BinaryClassifierEvaluator(env, new BinaryClassifierEvaluator.Arguments());

            var evaluator = new BinaryClassifierMamlEvaluator(env, new BinaryClassifierMamlEvaluator.Arguments());
            var metricsDic = evaluator.Evaluate(dataEval);

            return BinaryClassificationMetrics.FromMetrics(env, metricsDic["OverallMetrics"], metricsDic["ConfusionMatrix"])[0];
        }
    }
}
