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
            var pipeline = PreparePipeline();
            var model = pipeline.Train<SentimentData, SentimentPrediction>();
            var testData = PrepareTextLoaderTestData();
            var evaluator = new BinaryClassificationEvaluator();
            BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);
            ValidateExamples(model);
            ValidateBinaryMetrics(metrics);
        }

        [Fact]
        public void TrainTestPredictSentimentModelTest()
        {
            var pipeline = PreparePipeline();
            PredictionModel<SentimentData, SentimentPrediction> model = pipeline.Train<SentimentData, SentimentPrediction>();
            var testData = PrepareTextLoaderTestData();
            var tt = new TrainTestEvaluator().TrainTestEvaluate<SentimentData, SentimentPrediction>(pipeline, testData);

            Assert.Null(tt.ClassificationMetrics);
            Assert.Null(tt.RegressionMetrics);
            Assert.NotNull(tt.BinaryClassificationMetrics);
            Assert.NotNull(tt.PredictorModels);
            ValidateExamples(tt.PredictorModels);
            ValidateBinaryMetrics(tt.BinaryClassificationMetrics);
        }

        [Fact]
        public void CrossValidateSentimentModelTest()
        {
            var pipeline = PreparePipeline();

            var cv = new CrossValidator().CrossValidate<SentimentData, SentimentPrediction>(pipeline);

            //First two items are average and std. deviation of metrics from the folds.
            Assert.Equal(2, cv.PredictorModels.Count());
            Assert.Null(cv.ClassificationMetrics);
            Assert.Null(cv.RegressionMetrics);
            Assert.NotNull(cv.BinaryClassificationMetrics);
            Assert.Equal(4, cv.BinaryClassificationMetrics.Count());

            //Avergae of all folds.
            var metrics = cv.BinaryClassificationMetrics[0];
            Assert.Equal(0.57023626091422708, metrics.Accuracy, 4);
            Assert.Equal(0.54960689910161487, metrics.Auc, 1);
            Assert.Equal(0.67048277219704255, metrics.Auprc, 2);
            Assert.Equal(0, metrics.Entropy, 3);
            Assert.Equal(0.68942642723130532, metrics.F1Score, 4);
            Assert.Equal(0.97695909611968434, metrics.LogLoss, 3);
            Assert.Equal(-3.050726259114541, metrics.LogLossReduction, 3);
            Assert.Equal(0.37553879310344829, metrics.NegativePrecision, 3);
            Assert.Equal(0.25683962264150945, metrics.NegativeRecall, 3);
            Assert.Equal(0.63428539173628362, metrics.PositivePrecision, 3);
            Assert.Equal(0.75795196364816619, metrics.PositiveRecall);
            Assert.Null(metrics.ConfusionMatrix);

            //Std. Deviation.
            metrics = cv.BinaryClassificationMetrics[1];
            Assert.Equal(0.039933230611196011, metrics.Accuracy, 4);
            Assert.Equal(0.021066177821462407, metrics.Auc, 1);
            Assert.Equal(0.045842033921572725, metrics.Auprc, 2);
            Assert.Equal(0, metrics.Entropy, 3);
            Assert.Equal(0.030085767890644915, metrics.F1Score, 4);
            Assert.Equal(0.032906777175141941, metrics.LogLoss, 3);
            Assert.Equal(0.86311349745170118, metrics.LogLossReduction, 3);
            Assert.Equal(0.030711206896551647, metrics.NegativePrecision, 3);
            Assert.Equal(0.068160377358490579, metrics.NegativeRecall, 3);
            Assert.Equal(0.051761119891622735, metrics.PositivePrecision, 3);
            Assert.Equal(0.0015417072379052127, metrics.PositiveRecall);
            Assert.Null(metrics.ConfusionMatrix);

            //Fold 1.
            metrics = cv.BinaryClassificationMetrics[2];
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

            var matrix = metrics.ConfusionMatrix;
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

            //Fold 2.
            metrics = cv.BinaryClassificationMetrics[3];
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

            var sentiments = GetTestData();
            var predictions = cv.PredictorModels[0].Predict(sentiments);
            Assert.Equal(2, predictions.Count());
            Assert.True(predictions.ElementAt(0).Sentiment.IsTrue);
            Assert.True(predictions.ElementAt(1).Sentiment.IsTrue);

            predictions = cv.PredictorModels[1].Predict(sentiments);
            Assert.Equal(2, predictions.Count());
            Assert.True(predictions.ElementAt(0).Sentiment.IsTrue);
            Assert.True(predictions.ElementAt(1).Sentiment.IsTrue);
        }

        private void ValidateBinaryMetrics(BinaryClassificationMetrics metrics)
        {
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

            var matrix = metrics.ConfusionMatrix;
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

        private LearningPipeline PreparePipeline()
        {
            var dataPath = GetDataPath(SentimentDataPath);
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
            return pipeline;
        }

        private void ValidateExamples(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            var sentiments = GetTestData();
            var predictions = model.Predict(sentiments);
            Assert.Equal(2, predictions.Count());
            Assert.True(predictions.ElementAt(0).Sentiment.IsFalse);
            Assert.True(predictions.ElementAt(1).Sentiment.IsTrue);
        }

        private Data.TextLoader PrepareTextLoaderTestData()
        {
            var testDataPath = GetDataPath(SentimentTestPath);
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
            return testData;
        }

        private IEnumerable<SentimentData> GetTestData()
        {
            return new[]
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

