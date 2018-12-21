// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Models;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace Microsoft.ML.Scenarios
{
#pragma warning disable 612, 618
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
            var metrics = evaluator.Evaluate(model, testData);
            ValidateExamples(model);
            ValidateBinaryMetrics(metrics);
        }

        [Fact]
        public void TrainAndPredictSymSGDSentimentModelTest()
        {
            var pipeline = PreparePipelineSymSGD();
            var model = pipeline.Train<SentimentData, SentimentPrediction>();
            var testData = PrepareTextLoaderTestData();
            var evaluator = new BinaryClassificationEvaluator();
            var metrics = evaluator.Evaluate(model, testData);
            ValidateExamplesSymSGD(model);
            ValidateBinaryMetricsSymSGD(metrics);
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
        public void TrainAndPredictLightGBMSentimentModelTest()
        {
            var pipeline = PreparePipelineLightGBM();
            var model = pipeline.Train<SentimentData, SentimentPrediction>();
            var testData = PrepareTextLoaderTestData();
            var evaluator = new BinaryClassificationEvaluator();
            var metrics = evaluator.Evaluate(model, testData);
            ValidateExamplesLightGBM(model);
            ValidateBinaryMetricsLightGBM(metrics);
        }

        private void ValidateBinaryMetricsSymSGD(Microsoft.ML.Legacy.Models.BinaryClassificationMetrics metrics)
        {

            Assert.Equal(.8889, metrics.Accuracy, 4);
            Assert.Equal(1, metrics.Auc, 1);
            Assert.Equal(0.96, metrics.Auprc, 2);
            Assert.Equal(1, metrics.Entropy, 3);
            Assert.Equal(.9, metrics.F1Score, 4);
            Assert.Equal(.97, metrics.LogLoss, 3);
            Assert.Equal(3.030, metrics.LogLossReduction, 3);
            Assert.Equal(1, metrics.NegativePrecision, 3);
            Assert.Equal(.778, metrics.NegativeRecall, 3);
            Assert.Equal(.818, metrics.PositivePrecision, 3);
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

            Assert.Equal(2, matrix[1, 0]);
            Assert.Equal(2, matrix["negative", "positive"]);
            Assert.Equal(7, matrix[1, 1]);
            Assert.Equal(7, matrix["negative", "negative"]);

        }

        private void ValidateBinaryMetricsLightGBM(Microsoft.ML.Legacy.Models.BinaryClassificationMetrics metrics)
        {

            Assert.Equal(0.61111111111111116, metrics.Accuracy, 4);
            Assert.Equal(0.83950617283950613, metrics.Auc, 1);
            Assert.Equal(0.88324268324268318, metrics.Auprc, 2);
            Assert.Equal(1, metrics.Entropy, 3);
            Assert.Equal(.72, metrics.F1Score, 4);
            Assert.Equal(0.96456100297125325, metrics.LogLoss, 4);
            Assert.Equal(3.5438997028746755, metrics.LogLossReduction, 4);
            Assert.Equal(1, metrics.NegativePrecision, 3);
            Assert.Equal(0.22222222222222221, metrics.NegativeRecall, 3);
            Assert.Equal(0.5625, metrics.PositivePrecision, 3);
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

            Assert.Equal(7, matrix[1, 0]);
            Assert.Equal(7, matrix["negative", "positive"]);
            Assert.Equal(2, matrix[1, 1]);
            Assert.Equal(2, matrix["negative", "negative"]);

        }

        private void ValidateBinaryMetrics(Microsoft.ML.Legacy.Models.BinaryClassificationMetrics metrics)
        {

            Assert.Equal(0.6111, metrics.Accuracy, 4);
            Assert.Equal(0.6667, metrics.Auc, 4);
            Assert.Equal(0.8621, metrics.Auprc, 4);
            Assert.Equal(1, metrics.Entropy, 3);
            Assert.Equal(0.72, metrics.F1Score, 2);
            Assert.Equal(0.9689, metrics.LogLoss, 4);
            Assert.Equal(3.1122, metrics.LogLossReduction, 4);
            Assert.Equal(1, metrics.NegativePrecision, 1);
            Assert.Equal(0.2222, metrics.NegativeRecall, 4);
            Assert.Equal(0.5625, metrics.PositivePrecision, 4);
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

            Assert.Equal(7, matrix[1, 0]);
            Assert.Equal(7, matrix["negative", "positive"]);
            Assert.Equal(2, matrix[1, 1]);
            Assert.Equal(2, matrix["negative", "negative"]);
        }

        private Legacy.LearningPipeline PreparePipeline()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var pipeline = new LearningPipeline();

            pipeline.Add(new Legacy.Data.TextLoader(dataPath)
            {
                Arguments = new Legacy.Data.TextLoaderArguments
                {
                    Separator = new[] { '\t' },
                    HasHeader = true,
                    Column = new[]
                    {
                        new Legacy.Data.TextLoaderColumn()
                        {
                            Name = "Label",
                            Source = new [] { new Legacy.Data.TextLoaderRange(0) },
                            Type = Legacy.Data.DataKind.Num
                        },

                        new Legacy.Data.TextLoaderColumn()
                        {
                            Name = "SentimentText",
                            Source = new [] { new Legacy.Data.TextLoaderRange(1) },
                            Type = Legacy.Data.DataKind.Text
                        }
                    }
                }
            });

            pipeline.Add(new TextFeaturizer("Features", "SentimentText")
            {
                KeepPunctuations = false,
                OutputTokens = true,
                UsePredefinedStopWordRemover = true,
                VectorNormalizer = TextFeaturizingEstimatorTextNormKind.L2,
                CharFeatureExtractor = new NGramNgramExtractor() { NgramLength = 3, AllLengths = false },
                WordFeatureExtractor = new NGramNgramExtractor() { NgramLength = 2, AllLengths = true }
            });


            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });

            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });
            return pipeline;
        }

        private LearningPipeline PreparePipelineLightGBM()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var pipeline = new LearningPipeline();

            pipeline.Add(new Legacy.Data.TextLoader(dataPath)
            {
                Arguments = new Legacy.Data.TextLoaderArguments
                {
                    Separator = new[] { '\t' },
                    HasHeader = true,
                    Column = new[]
                    {
                        new Legacy.Data.TextLoaderColumn()
                        {
                            Name = "Label",
                            Source = new [] { new Legacy.Data.TextLoaderRange(0) },
                            Type = Legacy.Data.DataKind.Num
                        },

                        new Legacy.Data.TextLoaderColumn()
                        {
                            Name = "SentimentText",
                            Source = new [] { new Legacy.Data.TextLoaderRange(1) },
                            Type = Legacy.Data.DataKind.Text
                        }
                    }
                }
            });

            pipeline.Add(new TextFeaturizer("Features", "SentimentText")
            {
                KeepPunctuations = false,
                OutputTokens = true,
                UsePredefinedStopWordRemover = true,
                VectorNormalizer = TextFeaturizingEstimatorTextNormKind.L2,
                CharFeatureExtractor = new NGramNgramExtractor() { NgramLength = 3, AllLengths = false },
                WordFeatureExtractor = new NGramNgramExtractor() { NgramLength = 2, AllLengths = true }
            });


            pipeline.Add(new LightGbmBinaryClassifier() { NumLeaves = 5, NumBoostRound = 5, MinDataPerLeaf = 2 });

            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });
            return pipeline;
        }

        private LearningPipeline PreparePipelineSymSGD()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var pipeline = new LearningPipeline();

            pipeline.Add(new Legacy.Data.TextLoader(dataPath)
            {
                Arguments = new Legacy.Data.TextLoaderArguments
                {
                    Separator = new[] { '\t' },
                    HasHeader = true,
                    Column = new[]
                    {
                        new Legacy.Data.TextLoaderColumn()
                        {
                            Name = "Label",
                            Source = new [] { new Legacy.Data.TextLoaderRange(0) },
                            Type = Legacy.Data.DataKind.Num
                        },

                        new Legacy.Data.TextLoaderColumn()
                        {
                            Name = "SentimentText",
                            Source = new [] { new Legacy.Data.TextLoaderRange(1) },
                            Type = Legacy.Data.DataKind.Text
                        }
                    }
                }
            });

            pipeline.Add(new TextFeaturizer("Features", "SentimentText")
            {
                KeepPunctuations = false,
                OutputTokens = true,
                UsePredefinedStopWordRemover = true,
                VectorNormalizer = TextFeaturizingEstimatorTextNormKind.L2,
                CharFeatureExtractor = new NGramNgramExtractor() { NgramLength = 3, AllLengths = false },
                WordFeatureExtractor = new NGramNgramExtractor() { NgramLength = 2, AllLengths = true }
            });

            pipeline.Add(new SymSgdBinaryClassifier() { NumberOfThreads = 1 });

            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });
            return pipeline;
        }

        private void ValidateExamples(PredictionModel<SentimentData, SentimentPrediction> model, bool useLightGBM = false)
        {
            var sentiments = GetTestData();
            var predictions = model.Predict(sentiments);
            Assert.Equal(2, predictions.Count());

            Assert.True(predictions.ElementAt(0).Sentiment);
            Assert.True(predictions.ElementAt(1).Sentiment);

        }

        private void ValidateExamplesLightGBM(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            var sentiments = GetTestData();
            var predictions = model.Predict(sentiments);
            Assert.Equal(2, predictions.Count());

            Assert.True(predictions.ElementAt(0).Sentiment);
            Assert.True(predictions.ElementAt(1).Sentiment);
        }

        private void ValidateExamplesSymSGD(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            var sentiments = GetTestData();
            var predictions = model.Predict(sentiments);
            Assert.Equal(2, predictions.Count());

            Assert.False(predictions.ElementAt(0).Sentiment);
            Assert.True(predictions.ElementAt(1).Sentiment);
        }

        private Legacy.Data.TextLoader PrepareTextLoaderTestData()
        {
            var testDataPath = GetDataPath(SentimentTestPath);
            var testData = new Legacy.Data.TextLoader(testDataPath)
            {
                Arguments = new Legacy.Data.TextLoaderArguments
                {
                    Separator = new[] { '\t' },
                    HasHeader = true,
                    Column = new[]
                    {
                        new Legacy.Data.TextLoaderColumn()
                        {
                            Name = "Label",
                            Source = new [] { new Legacy.Data.TextLoaderRange(0) },
                            Type = Legacy.Data.DataKind.Num
                        },

                        new Legacy.Data.TextLoaderColumn()
                        {
                            Name = "SentimentText",
                            Source = new [] { new Legacy.Data.TextLoaderRange(1) },
                            Type = Legacy.Data.DataKind.Text
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
            [LoadColumn(0), ColumnName("Label")]
            public float Sentiment;
            [LoadColumn(1)]
            public string SentimentText;
        }

        public class SentimentPrediction
        {
            [ColumnName("PredictedLabel")]
            public bool Sentiment;
        }
    }
#pragma warning restore 612, 618
}

