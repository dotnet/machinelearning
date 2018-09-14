// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Models;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
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

        [Fact]
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

        [Fact]
        public void TrainTestPredictSentimentModelTest()
        {
            var pipeline = PreparePipeline();
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
            Assert.Equal(0.603235747303544, metrics.Accuracy, 4);
            Assert.Equal(0.58811318075483943, metrics.Auc, 4);
            Assert.Equal(0.70302385499183984, metrics.Auprc, 4);
            Assert.Equal(0, metrics.Entropy, 3);
            Assert.Equal(0.71751777634130576, metrics.F1Score, 4);
            Assert.Equal(0.95263103280238037, metrics.LogLoss, 4);
            Assert.Equal(-0.39971801589876232, metrics.LogLossReduction, 4);
            Assert.Equal(0.43965517241379309, metrics.NegativePrecision, 4);
            Assert.Equal(0.26627358490566039, metrics.NegativeRecall, 4);
            Assert.Equal(0.64937737441958632, metrics.PositivePrecision, 4);
            Assert.Equal(0.8027426160337553, metrics.PositiveRecall);
            Assert.Null(metrics.ConfusionMatrix);

            //Std. Deviation.
            metrics = cv.BinaryClassificationMetrics[1];
            Assert.Equal(0.057781201848998764, metrics.Accuracy, 4);
            Assert.Equal(0.04249579360413544, metrics.Auc, 4);
            Assert.Equal(0.086083866074815427, metrics.Auprc, 4);
            Assert.Equal(0, metrics.Entropy, 3);
            Assert.Equal(0.04718810601163604, metrics.F1Score, 4);
            Assert.Equal(0.063839715206238851, metrics.LogLoss, 4);
            Assert.Equal(4.1937544629633878, metrics.LogLossReduction, 4);
            Assert.Equal(0.060344827586206781, metrics.NegativePrecision, 4);
            Assert.Equal(0.058726415094339748, metrics.NegativeRecall, 4);
            Assert.Equal(0.057144364710848418, metrics.PositivePrecision, 4);
            Assert.Equal(0.030590717299577637, metrics.PositiveRecall);
            Assert.Null(metrics.ConfusionMatrix);

            //Fold 1.
            metrics = cv.BinaryClassificationMetrics[2];
            Assert.Equal(0.54545454545454541, metrics.Accuracy, 4);
            Assert.Equal(0.54561738715070451, metrics.Auc, 4);
            Assert.Equal(0.61693998891702417, metrics.Auprc, 4);
            Assert.Equal(0, metrics.Entropy, 3);
            Assert.Equal(0.67032967032967028, metrics.F1Score, 4);
            Assert.Equal(1.0164707480086188, metrics.LogLoss, 4);
            Assert.Equal(-4.59347247886215, metrics.LogLossReduction, 4);
            Assert.Equal(0.37931034482758619, metrics.NegativePrecision, 4);
            Assert.Equal(0.20754716981132076, metrics.NegativeRecall, 4);
            Assert.Equal(0.59223300970873782, metrics.PositivePrecision, 4);
            Assert.Equal(0.77215189873417722, metrics.PositiveRecall);

            var matrix = metrics.ConfusionMatrix;
            Assert.Equal(2, matrix.Order);
            Assert.Equal(2, matrix.ClassNames.Count);
            Assert.Equal("positive", matrix.ClassNames[0]);
            Assert.Equal("negative", matrix.ClassNames[1]);

            Assert.Equal(61, matrix[0, 0]);
            Assert.Equal(61, matrix["positive", "positive"]);
            Assert.Equal(18, matrix[0, 1]);
            Assert.Equal(18, matrix["positive", "negative"]);

            Assert.Equal(42, matrix[1, 0]);
            Assert.Equal(42, matrix["negative", "positive"]);
            Assert.Equal(11, matrix[1, 1]);
            Assert.Equal(11, matrix["negative", "negative"]);

            //Fold 2.
            metrics = cv.BinaryClassificationMetrics[3];
            Assert.Equal(0.66101694915254239, metrics.Accuracy, 4);
            Assert.Equal(0.63060897435897434, metrics.Auc, 4);
            Assert.Equal(0.7891077210666555, metrics.Auprc, 4);
            Assert.Equal(0, metrics.Entropy, 3);
            Assert.Equal(0.76470588235294124, metrics.F1Score, 4);
            Assert.Equal(0.88879131759614194, metrics.LogLoss, 4);
            Assert.Equal(3.7940364470646255, metrics.LogLossReduction, 4);
            Assert.Equal(0.5, metrics.NegativePrecision, 3);
            Assert.Equal(0.325, metrics.NegativeRecall, 3);
            Assert.Equal(0.70652173913043481, metrics.PositivePrecision, 4);
            Assert.Equal(0.83333333333333337, metrics.PositiveRecall);

            matrix = metrics.ConfusionMatrix;
            Assert.Equal(2, matrix.Order);
            Assert.Equal(2, matrix.ClassNames.Count);
            Assert.Equal("positive", matrix.ClassNames[0]);
            Assert.Equal("negative", matrix.ClassNames[1]);

            Assert.Equal(65, matrix[0, 0]);
            Assert.Equal(65, matrix["positive", "positive"]);
            Assert.Equal(13, matrix[0, 1]);
            Assert.Equal(13, matrix["positive", "negative"]);

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

        private void ValidateBinaryMetricsSymSGD(BinaryClassificationMetrics metrics)
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

        private void ValidateBinaryMetricsLightGBM(BinaryClassificationMetrics metrics)
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

        private void ValidateBinaryMetrics(BinaryClassificationMetrics metrics)
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
                KeepDiacritics = false,
                KeepPunctuations = false,
                TextCase = TextNormalizerTransformCaseNormalizationMode.Lower,
                OutputTokens = true,
                StopWordsRemover = new PredefinedStopWordsRemover(),
                VectorNormalizer = TextTransformTextNormKind.L2,
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
                KeepDiacritics = false,
                KeepPunctuations = false,
                TextCase = TextNormalizerTransformCaseNormalizationMode.Lower,
                OutputTokens = true,
                StopWordsRemover = new PredefinedStopWordsRemover(),
                VectorNormalizer = TextTransformTextNormKind.L2,
                CharFeatureExtractor = new NGramNgramExtractor() { NgramLength = 3, AllLengths = false },
                WordFeatureExtractor = new NGramNgramExtractor() { NgramLength = 2, AllLengths = true }
            });


            pipeline.Add(new SymSgdBinaryClassifier() { NumberOfThreads = 1});

            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });
            return pipeline;
        }

        private void ValidateExamples(PredictionModel<SentimentData, SentimentPrediction> model, bool useLightGBM = false)
        {
            var sentiments = GetTestData();
            var predictions = model.Predict(sentiments);
            Assert.Equal(2, predictions.Count());

            Assert.True(predictions.ElementAt(0).Sentiment.IsTrue);
            Assert.True(predictions.ElementAt(1).Sentiment.IsTrue);

        }

        private void ValidateExamplesLightGBM(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            var sentiments = GetTestData();
            var predictions = model.Predict(sentiments);
            Assert.Equal(2, predictions.Count());

            Assert.True(predictions.ElementAt(0).Sentiment.IsTrue);
            Assert.True(predictions.ElementAt(1).Sentiment.IsTrue);
        }

        private void ValidateExamplesSymSGD(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            var sentiments = GetTestData();
            var predictions = model.Predict(sentiments);
            Assert.Equal(2, predictions.Count());

            Assert.True(predictions.ElementAt(0).Sentiment.IsFalse);
            Assert.True(predictions.ElementAt(1).Sentiment.IsTrue);
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

