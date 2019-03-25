// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Functional.Tests.Datasets;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Functional.Tests
{
    public class Evaluation : BaseTestClass
    {
        public Evaluation(ITestOutputHelper output): base(output)
        {
        }

        /// <summary>
        /// Train and Evaluate: Anomaly Detection.
        /// </summary>
        [Fact]
        public void TrainAndEvaluateAnomalyDetection()
        {
            var mlContext = new MLContext(seed: 1);

            var trainData = MnistOneClass.GetTextLoader(mlContext,
                    TestDatasets.mnistOneClass.fileHasHeader, TestDatasets.mnistOneClass.fileSeparator)
                .Load(GetDataPath(TestDatasets.mnistOneClass.trainFilename));
            var testData = MnistOneClass.GetTextLoader(mlContext,
                    TestDatasets.mnistOneClass.fileHasHeader, TestDatasets.mnistOneClass.fileSeparator)
                .Load(GetDataPath(TestDatasets.mnistOneClass.testFilename));

            // Create a training pipeline.
            var pipeline = mlContext.AnomalyDetection.Trainers.RandomizedPca();

            // Train the model.
            var model = pipeline.Fit(trainData);

            // Evaluate the model.
            //  TODO #2464: Using the train dataset will cause NaN metrics to be returned.
            var scoredTest = model.Transform(testData);
            var metrics = mlContext.AnomalyDetection.Evaluate(scoredTest);

            // Check that the metrics returned are valid.
            Common.AssertMetrics(metrics);
        }

        /// <summary>
        /// Train and Evaluate: Binary Classification with no calibration.
        /// </summary>
        [Fact]
        public void TrainAndEvaluateBinaryClassification()
        {
            var mlContext = new MLContext(seed: 1);

            var data = mlContext.Data.LoadFromTextFile<TweetSentiment>(GetDataPath(TestDatasets.Sentiment.trainFilename),
                hasHeader: TestDatasets.Sentiment.fileHasHeader,
                separatorChar: TestDatasets.Sentiment.fileSeparator);

            // Create a training pipeline.
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "SentimentText")
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.BinaryClassification.Trainers.SdcaNonCalibrated(
                    new SdcaNonCalibratedBinaryTrainer.Options { NumberOfThreads = 1 }));

            // Train the model.
            var model = pipeline.Fit(data);

            // Evaluate the model.
            var scoredData = model.Transform(data);
            var metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(scoredData);

            // Check that the metrics returned are valid.
            Common.AssertMetrics(metrics);
        }

        /// <summary>
        /// Train and Evaluate: Binary Classification with a calibrated predictor.
        /// </summary>
        [Fact]
        public void TrainAndEvaluateBinaryClassificationWithCalibration()
        {
            var mlContext = new MLContext(seed: 1);

            var data = mlContext.Data.LoadFromTextFile<TweetSentiment>(GetDataPath(TestDatasets.Sentiment.trainFilename),
                hasHeader: TestDatasets.Sentiment.fileHasHeader,
                separatorChar: TestDatasets.Sentiment.fileSeparator);

            // Create a training pipeline.
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "SentimentText")
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.BinaryClassification.Trainers.LogisticRegression(
                    new LogisticRegressionBinaryTrainer.Options { NumberOfThreads = 1 }));

            // Train the model.
            var model = pipeline.Fit(data);

            // Evaluate the model.
            var scoredData = model.Transform(data);
            var metrics = mlContext.BinaryClassification.Evaluate(scoredData);

            // Check that the metrics returned are valid.
            Common.AssertMetrics(metrics);
        }

        /// <summary>
        /// Train and Evaluate: Clustering.
        /// </summary>
        [Fact]
        public void TrainAndEvaluateClustering()
        {
            var mlContext = new MLContext(seed: 1);

            var data = mlContext.Data.LoadFromTextFile<Iris>(GetDataPath(TestDatasets.iris.trainFilename),
                hasHeader: TestDatasets.iris.fileHasHeader,
                separatorChar: TestDatasets.iris.fileSeparator);

            // Create a training pipeline.
            var pipeline = mlContext.Transforms.Concatenate("Features", Iris.Features)
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.Clustering.Trainers.KMeans(new KMeansTrainer.Options { NumberOfThreads = 1 }));

            // Train the model.
            var model = pipeline.Fit(data);

            // Evaluate the model.
            var scoredData = model.Transform(data);
            var metrics = mlContext.Clustering.Evaluate(scoredData);

            // Check that the metrics returned are valid.
            Common.AssertMetrics(metrics);
        }

        /// <summary>
        /// Train and Evaluate: Multiclass Classification.
        /// </summary>
        [Fact]
        public void TrainAndEvaluateMulticlassClassification()
        {
            var mlContext = new MLContext(seed: 1);

            var data = mlContext.Data.LoadFromTextFile<Iris>(GetDataPath(TestDatasets.iris.trainFilename),
                hasHeader: TestDatasets.iris.fileHasHeader,
                separatorChar: TestDatasets.iris.fileSeparator);

            // Create a training pipeline.
            var pipeline = mlContext.Transforms.Concatenate("Features", Iris.Features)
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.MulticlassClassification.Trainers.SdcaCalibrated(
                    new SdcaCalibratedMulticlassTrainer.Options { NumberOfThreads = 1}));

            // Train the model.
            var model = pipeline.Fit(data);

            // Evaluate the model.
            var scoredData = model.Transform(data);
            var metrics = mlContext.MulticlassClassification.Evaluate(scoredData);

            // Check that the metrics returned are valid.
            Common.AssertMetrics(metrics);
        }

        /// <summary>
        /// Train and Evaluate: Ranking.
        /// </summary>
        [Fact]
        public void TrainAndEvaluateRanking()
        {
            var mlContext = new MLContext(seed: 1);

            var data = Iris.LoadAsRankingProblem(mlContext,
                GetDataPath(TestDatasets.iris.trainFilename),
                hasHeader: TestDatasets.iris.fileHasHeader,
                separatorChar: TestDatasets.iris.fileSeparator);

            // Create a training pipeline.
            var pipeline = mlContext.Transforms.Concatenate("Features", Iris.Features)
                .Append(mlContext.Ranking.Trainers.FastTree(new FastTreeRankingTrainer.Options { NumberOfThreads = 1 }));

            // Train the model.
            var model = pipeline.Fit(data);

            // Evaluate the model.
            var scoredData = model.Transform(data);
            var metrics = mlContext.Ranking.Evaluate(scoredData, labelColumnName: "Label", rowGroupColumnName: "GroupId");

            // Check that the metrics returned are valid.
            Common.AssertMetrics(metrics);
        }

        /// <summary>
        /// Train and Evaluate: Recommendation.
        /// </summary>
        [MatrixFactorizationFact]
        public void TrainAndEvaluateRecommendation()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset.
            var data = TrivialMatrixFactorization.LoadAndFeaturizeFromTextFile(
                mlContext,
                GetDataPath(TestDatasets.trivialMatrixFactorization.trainFilename),
                TestDatasets.trivialMatrixFactorization.fileHasHeader,
                TestDatasets.trivialMatrixFactorization.fileSeparator);

            // Create a pipeline to train on the sentiment data.
            var pipeline = mlContext.Recommendation().Trainers.MatrixFactorization(
                new MatrixFactorizationTrainer.Options{
                    MatrixColumnIndexColumnName = "MatrixColumnIndex",
                    MatrixRowIndexColumnName = "MatrixRowIndex",
                    LabelColumnName = "Label",
                    NumberOfIterations = 3,
                    NumberOfThreads = 1,
                    ApproximationRank = 4,
                });

            // Train the model.
            var model = pipeline.Fit(data);

            // Evaluate the model.
            var scoredData = model.Transform(data);
            var metrics = mlContext.Recommendation().Evaluate(scoredData);

            // Check that the metrics returned are valid.
            Common.AssertMetrics(metrics);
        }

        /// <summary>
        /// Train and Evaluate: Regression.
        /// </summary>
        [Fact]
        public void TrainAndEvaluateRegression()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(GetDataPath(TestDatasets.housing.trainFilename), hasHeader: true);
            // Create a pipeline to train on the housing data.
            var pipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .Append(mlContext.Regression.Trainers.FastForest(new FastForestRegressionTrainer.Options { NumberOfThreads = 1 }));

            // Train the model.
            var model = pipeline.Fit(data);

            // Evaluate the model.
            var scoredData = model.Transform(data);
            var metrics = mlContext.Regression.Evaluate(scoredData);

            // Check that the metrics returned are valid.
            Common.AssertMetrics(metrics);
        }

        /// <summary>
        /// Evaluate With Precision-Recall Curves.
        /// </summary>
        /// <remarks>
        /// This is currently not possible using the APIs.
        /// </remarks>
        [Fact]
        public void TrainAndEvaluateWithPrecisionRecallCurves()
        {
            var mlContext = new MLContext(seed: 1);

            var data = mlContext.Data.LoadFromTextFile<TweetSentiment>(GetDataPath(TestDatasets.Sentiment.trainFilename),
                hasHeader: TestDatasets.Sentiment.fileHasHeader,
                separatorChar: TestDatasets.Sentiment.fileSeparator);

            // Create a training pipeline.
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "SentimentText")
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.BinaryClassification.Trainers.LogisticRegression(
                    new LogisticRegressionBinaryTrainer.Options { NumberOfThreads = 1 }));

            // Train the model.
            var model = pipeline.Fit(data);

            // Evaluate the model.
            var scoredData = model.Transform(data);
            var metrics = mlContext.BinaryClassification.Evaluate(scoredData);

            Common.AssertMetrics(metrics);

            // This scenario is not possible with the current set of APIs.
            // There could be two ways imaginable:
            //  1. Getting a list of (P,R) from the Evaluator (as it calculates most of the information already).
            //     Not currently possible.
            //  2. Manually setting the classifier threshold and calling evaluate many times:
            //     Not currently possible: Todo #2465: Allow the setting of threshold and thresholdColumn for scoring.
            // Technically, this scenario is possible using custom mappers like so:
            //  1. Get a list of all unique probability scores.
            //     e.g. By reading the IDataView as an IEnumerable, and keeping a hash of known probabilities up to some precision.
            //  2. For each value of probability:
            //     a. Write a custom mapper to produce PredictedLabel at that probability threshold.
            //     b. Calculate Precision and Recall with these labels.
            //     c. Append the Precision and Recall to an IList.
        }
    }
}