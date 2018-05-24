// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
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

            pipeline.Add(new Data.TextLoader(dataPath)
            {
                Arguments = new TextLoaderArguments
                {
                    Separator = new[] { '\t' },
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoaderColumn()
                        {
                            Name = "Label",
                            Source = new [] { new TextLoaderRange(0) },
                            Type = Runtime.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "SentimentText",
                            Source = new [] { new TextLoaderRange(1) },
                            Type = Runtime.Data.DataKind.Text
                        }
                    }
                }
            });
            
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
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });
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
            Assert.True(predictions.ElementAt(0).Sentiment.IsFalse);
            Assert.True(predictions.ElementAt(1).Sentiment.IsTrue);

            string testDataPath = GetDataPath(SentimentTestPath);
            var testData = new Data.TextLoader(testDataPath)
            {
                Arguments = new TextLoaderArguments
                {
                    Separator = new[] { '\t' },
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoaderColumn()
                        {
                            Name = "Label",
                            Source = new [] { new TextLoaderRange(0) },
                            Type = Runtime.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "SentimentText",
                            Source = new [] { new TextLoaderRange(1) },
                            Type = Runtime.Data.DataKind.Text
                        }
                    }
                }
            };
            var evaluator = new BinaryClassificationEvaluator();
            BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData).FirstOrDefault();

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

            var cv = new CrossValidator().CrossValidate<SentimentData, SentimentPrediction>(pipeline);

            Assert.Equal(2, cv.PredictorModels.Count());
            Assert.Null(cv.ClassificationMetrics);
            Assert.Null(cv.RegressionMetrics);
            Assert.NotNull(cv.BinaryClassificationMetrics);
            Assert.Equal(2, cv.BinaryClassificationMetrics.Count());

            metrics = cv.BinaryClassificationMetrics[0];
            Assert.Equal(0.53030303030303028, metrics.Accuracy, 4);
            Assert.Equal(0.52854072128015284, metrics.Auc, 1);
            Assert.Equal(0.62464073827546951, metrics.Auprc, 2);
            Assert.Equal(0, metrics.Entropy, 3);
            Assert.Equal(0.65934065934065933, metrics.F1Score, 4);
            Assert.Equal(1.0098658732948276, metrics.LogLoss, 3);
            Assert.Equal(-3.9138397565662424, metrics.LogLossReduction, 3);
            Assert.Equal(0.34482758620689657, metrics.NegativePrecision, 3);
            Assert.Equal(0.18867924528301888, metrics.NegativeRecall, 3);
            Assert.Equal(0.58252427184466016, metrics.PositivePrecision, 3);
            Assert.Equal(0.759493670886076, metrics.PositiveRecall);

            matrix = metrics.ConfusionMatrix;
            Assert.Equal(2, matrix.Order);
            Assert.Equal(2, matrix.ClassNames.Count);
            Assert.Equal("positive", matrix.ClassNames[0]);
            Assert.Equal("negative", matrix.ClassNames[1]);

            Assert.Equal(60, matrix[0, 0]);
            Assert.Equal(60, matrix["positive", "positive"]);
            Assert.Equal(19, matrix[0, 1]);
            Assert.Equal(19, matrix["positive", "negative"]);

            Assert.Equal(43, matrix[1, 0]);
            Assert.Equal(43, matrix["negative", "positive"]);
            Assert.Equal(10, matrix[1, 1]);
            Assert.Equal(10, matrix["negative", "negative"]);

            metrics = cv.BinaryClassificationMetrics[1];
            Assert.Equal(0.61016949152542377, metrics.Accuracy, 4);
            Assert.Equal(0.57067307692307689, metrics.Auc, 1);
            Assert.Equal(0.71632480611861549, metrics.Auprc, 2);
            Assert.Equal(0, metrics.Entropy, 3);
            Assert.Equal(0.71951219512195119, metrics.F1Score, 4);
            Assert.Equal(0.94405231894454111, metrics.LogLoss, 3);
            Assert.Equal(-2.1876127616628396, metrics.LogLossReduction, 3);
            Assert.Equal(0.40625, metrics.NegativePrecision, 3);
            Assert.Equal(0.325, metrics.NegativeRecall, 3);
            Assert.Equal(0.686046511627907, metrics.PositivePrecision, 3);
            Assert.Equal(0.75641025641025639, metrics.PositiveRecall);

            matrix = metrics.ConfusionMatrix;
            Assert.Equal(2, matrix.Order);
            Assert.Equal(2, matrix.ClassNames.Count);
            Assert.Equal("positive", matrix.ClassNames[0]);
            Assert.Equal("negative", matrix.ClassNames[1]);

            Assert.Equal(59, matrix[0, 0]);
            Assert.Equal(59, matrix["positive", "positive"]);
            Assert.Equal(19, matrix[0, 1]);
            Assert.Equal(19, matrix["positive", "negative"]);

            Assert.Equal(27, matrix[1, 0]);
            Assert.Equal(27, matrix["negative", "positive"]);
            Assert.Equal(13, matrix[1, 1]);
            Assert.Equal(13, matrix["negative", "negative"]);

            predictions = cv.PredictorModels[0].Predict(sentiments);
            Assert.Equal(2, predictions.Count());
            Assert.True(predictions.ElementAt(0).Sentiment.IsTrue);
            Assert.True(predictions.ElementAt(1).Sentiment.IsTrue);

            predictions = cv.PredictorModels[1].Predict(sentiments);
            Assert.Equal(2, predictions.Count());
            Assert.True(predictions.ElementAt(0).Sentiment.IsTrue);
            Assert.True(predictions.ElementAt(1).Sentiment.IsTrue);
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
            public DvBool Sentiment;
        }
    }
}

