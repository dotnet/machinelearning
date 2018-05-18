// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace Microsoft.ML.Scenarios
{
    public partial class ScenariosTests
    {
        public const string SentimentDataPath = "wikipedia-detox-250-line-data.tsv";
        public const string SentimentTestPath = "wikipedia-detox-250-line-test.tsv";

        [Fact]
        public void TrainAndPredictSentimentModelTest()
        {
            string dataPath = GetDataPath(SentimentDataPath);
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader<SentimentData>(dataPath, useHeader: true, separator: "tab"));
            pipeline.Add(new TextFeaturizer("Features", "SentimentText")
            {
                KeepDiacritics = false,
                KeepPunctuations = false,
                TextCase = TextNormalizerTransformCaseNormalizationMode.Lower,
                OutputTokens = true,
                StopWordsRemover = new PredefinedStopWordsRemover(),
                VectorNormalizer = TextTransformTextNormKind.L2,
                CharFeatureExtractor = new NGramNgramExtractor() { NgramLength = 3, AllLengths = false },
                WordFeatureExtractor = new NGramNgramExtractor() { NgramLength = 2, AllLengths = true }
            });

            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });
            //pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });
            BinaryCrossValidator bcv = new BinaryCrossValidator();
            bcv.CrossValidate<SentimentData, SentimentPrediction>(pipeline);
            PredictionModel<SentimentData, SentimentPrediction> model = pipeline.Train<SentimentData, SentimentPrediction>();

            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "Please refrain from adding nonsense to Wikipedia."
                },
                new SentimentData
                {
                    SentimentText = "He is a CHEATER, and the article should say that."
                }
            };

            IEnumerable<SentimentPrediction> predictions = model.Predict(sentiments);

            Assert.Equal(2, predictions.Count());
            Assert.False(predictions.ElementAt(0).Sentiment);
            Assert.True(predictions.ElementAt(1).Sentiment);

            string testDataPath = GetDataPath(SentimentTestPath);
            var testData = new TextLoader<SentimentData>(testDataPath, useHeader: true, separator: "tab");

            var evaluator = new BinaryClassificationEvaluator();
            BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);

            Assert.Equal(.5556, metrics.Accuracy, 4);
            Assert.Equal(.8, metrics.Auc, 1);
            Assert.Equal(.87, metrics.Auprc, 2);
            Assert.Equal(1, metrics.Entropy, 3);
            Assert.Equal(.6923, metrics.F1Score, 4);
            Assert.Equal(.969, metrics.LogLoss, 3);
            Assert.Equal(3.083, metrics.LogLossReduction, 3);
            Assert.Equal(1, metrics.NegativePrecision, 3);
            Assert.Equal(.111, metrics.NegativeRecall, 3);
            Assert.Equal(.529, metrics.PositivePrecision, 3);
            Assert.Equal(1, metrics.PositiveRecall);

            ConfusionMatrix matrix = metrics.ConfusionMatrix;
            Assert.Equal(2, matrix.Order);
            Assert.Equal(2, matrix.ClassNames.Count);
            Assert.Equal("positive", matrix.ClassNames[0]);
            Assert.Equal("negative", matrix.ClassNames[1]);

            Assert.Equal(9, matrix[0, 0]);
            Assert.Equal(9, matrix["positive", "positive"]);
            Assert.Equal(0, matrix[0, 1]);
            Assert.Equal(0, matrix["positive", "negative"]);

            Assert.Equal(8, matrix[1, 0]);
            Assert.Equal(8, matrix["negative", "positive"]);
            Assert.Equal(1, matrix[1, 1]);
            Assert.Equal(1, matrix["negative", "negative"]);
        }

        public class SentimentData
        {
            [Column(ordinal: "0", name: "Label")]
            public float Sentiment;
            [Column(ordinal: "1")]
            public string SentimentText;
        }

        public class SentimentPrediction
        {
            [ColumnName("PredictedLabel")]
            public bool Sentiment;
        }
    }
}

