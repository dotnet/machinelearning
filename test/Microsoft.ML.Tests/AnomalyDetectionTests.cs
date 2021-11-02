// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFrameworkCommon;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class AnomalyDetectionTests : TestDataPipeBase
    {
        public AnomalyDetectionTests(ITestOutputHelper output) : base(output)
        {
        }

        /// <summary>
        /// RandomizedPcaTrainer test.
        /// </summary>
        [Fact]
        public void RandomizedPcaTrainerBaselineTest()
        {
            var trainPath = GetDataPath(TestDatasets.mnistOneClass.trainFilename);
            var testPath = GetDataPath(TestDatasets.mnistOneClass.testFilename);

            var transformedData = DetectAnomalyInMnistOneClass(trainPath, testPath);

            // Evaluate
            var metrics = ML.AnomalyDetection.Evaluate(transformedData, falsePositiveCount: 5);

            Assert.Equal(0.98667, metrics.AreaUnderRocCurve, 5);
            Assert.Equal(0.90000, metrics.DetectionRateAtFalsePositiveCount, 5);
        }

        /// <summary>
        /// Test anomaly detection when the test data has no anomalies.
        /// </summary>
        [Fact]
        public void NoAnomalyTest()
        {
            var trainPath = GetDataPath(TestDatasets.mnistOneClass.trainFilename);

            var transformedData = DetectAnomalyInMnistOneClass(trainPath, trainPath);

            Assert.Throws<ArgumentOutOfRangeException>(() => ML.AnomalyDetection.Evaluate(transformedData));
        }

        [Fact]
        public static void RandomizedPcaInMemory()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging,
            // as a catalog of available operations and as the source of randomness.
            // Setting the seed to a fixed number in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Create an anomaly detector. Its underlying algorithm is randomized PCA.
            var trainer1 = mlContext.AnomalyDetection.Trainers.RandomizedPca(featureColumnName: nameof(DataPoint.Features), rank: 1, ensureZeroMean: false);

            // Test the first detector.
            ExecutePipelineWithGivenRandomizedPcaTrainer(mlContext, trainer1);

            // Object required in the creation of another detector.
            var options = new Trainers.RandomizedPcaTrainer.Options()
            {
                FeatureColumnName = nameof(DataPoint.Features),
                Rank = 1,
                EnsureZeroMean = false,
                Seed = 10
            };

            // Create anther anomaly detector. Its underlying algorithm is randomized PCA.
            var trainer2 = mlContext.AnomalyDetection.Trainers.RandomizedPca(options);

            // Test the second detector.
            ExecutePipelineWithGivenRandomizedPcaTrainer(mlContext, trainer2);
        }

        [Fact]
        public static void RandomizedPcaChangeThreshold()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging,
            // as a catalog of available operations and as the source of randomness.
            // Setting the seed to a fixed number in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Create an anomaly detector. Its underlying algorithm is randomized PCA.
            var trainer1 = mlContext.AnomalyDetection.Trainers.RandomizedPca(featureColumnName: nameof(DataPoint.Features), rank: 1, ensureZeroMean: false);

            // Test the first detector.
            ExecuteRandomizedPcaTrainerChangeThreshold(mlContext, trainer1);

            // Object required in the creation of another detector.
            var options = new Trainers.RandomizedPcaTrainer.Options()
            {
                FeatureColumnName = nameof(DataPoint.Features),
                Rank = 1,
                EnsureZeroMean = false,
                Seed = 10
            };

            // Create anther anomaly detector. Its underlying algorithm is randomized PCA.
            var trainer2 = mlContext.AnomalyDetection.Trainers.RandomizedPca(options);

            // Test the second detector.
            ExecuteRandomizedPcaTrainerChangeThreshold(mlContext, trainer2);
        }

        /// <summary>
        /// Example with 3 feature values used in <see cref="ExecutePipelineWithGivenRandomizedPcaTrainer"/>.
        /// </summary>
        private class DataPoint
        {
            [VectorType(3)]
            public float[] Features { get; set; }
        }

        /// <summary>
        /// Class used to capture prediction of <see cref="DataPoint"/> in <see cref="ExecutePipelineWithGivenRandomizedPcaTrainer"/>.
        /// </summary>
        private class Result
        {
            // Outlier gets false while inlier has true.
            public bool PredictedLabel { get; set; }
            // Outlier gets smaller score.
            public float Score { get; set; }
        }

        /// <summary>
        /// Help function used to execute trainers defined in <see cref="RandomizedPcaInMemory"/>.
        /// </summary>
        private static void ExecutePipelineWithGivenRandomizedPcaTrainer(MLContext mlContext, Trainers.RandomizedPcaTrainer trainer)
        {
            var samples = new List<DataPoint>()
            {
                new DataPoint(){ Features = new float[3] {0, 2, 1} },
                new DataPoint(){ Features = new float[3] {0, 2, 3} },
                new DataPoint(){ Features = new float[3] {0, 2, 4} },
                new DataPoint(){ Features = new float[3] {0, 2, 1} },
                new DataPoint(){ Features = new float[3] {0, 2, 2} },
                new DataPoint(){ Features = new float[3] {0, 2, 3} },
                new DataPoint(){ Features = new float[3] {0, 2, 4} },
                new DataPoint(){ Features = new float[3] {1, 0, 0} }
            };

            // Convert the List<DataPoint> to IDataView, a consumable format to ML.NET functions.
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // Train the anomaly detector.
            var model = trainer.Fit(data);

            // Apply the trained model on the training data.
            var transformed = model.Transform(data);

            // Read ML.NET predictions into IEnumerable<Result>.
            var results = mlContext.Data.CreateEnumerable<Result>(transformed, reuseRowObject: false).ToList();

            // First 5 examples are inliers.
            for (int i = 0; i < 7; ++i)
            {
                // Inlier should be predicted as false.
                Assert.False(results[i].PredictedLabel);
                // Higher score means closer to inlier.
                Assert.InRange(results[i].Score, 0, 0.5);
            }

            // Last example is outlier. Note that outlier should be predicted as true.
            Assert.True(results[7].PredictedLabel);
            Assert.InRange(results[7].Score, 0.5, 1);
        }


        /// <summary>
        /// Help function used to execute trainers defined in <see cref="RandomizedPcaInMemory"/>.
        /// </summary>
        private static void ExecuteRandomizedPcaTrainerChangeThreshold(MLContext mlContext, Trainers.RandomizedPcaTrainer trainer)
        {
            var samples = new List<DataPoint>()
            {
                new DataPoint(){ Features = new float[3] {0, 2, 1} },
                new DataPoint(){ Features = new float[3] {0, 2, 3} },
                new DataPoint(){ Features = new float[3] {0, 2, 4} },
                new DataPoint(){ Features = new float[3] {0, 2, 1} },
                new DataPoint(){ Features = new float[3] {0, 2, 2} },
                new DataPoint(){ Features = new float[3] {0, 2, 3} },
                new DataPoint(){ Features = new float[3] {0, 2, 4} },
                new DataPoint(){ Features = new float[3] {1, 0, 0} }
            };

            // Convert the List<DataPoint> to IDataView, a consumble format to ML.NET functions.
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // Train the anomaly detector.
            var model = trainer.Fit(data);

            var transformer = mlContext.AnomalyDetection.ChangeModelThreshold(model, 0.3f);

            // Apply the trained model on the training data.
            var transformed = transformer.Transform(data);

            // Read ML.NET predictions into IEnumerable<Result>.
            var results = mlContext.Data.CreateEnumerable<Result>(transformed, reuseRowObject: false).ToList();

            // Outlier should be predicted as true.
            Assert.True(results[0].PredictedLabel);
            Assert.InRange(results[0].Score, 0.3, 1);
            // Inlier should be predicted as false.
            Assert.False(results[1].PredictedLabel);
            Assert.InRange(results[1].Score, 0, 0.3);
            // Inlier should be predicted as false.
            Assert.False(results[2].PredictedLabel);
            Assert.InRange(results[2].Score, 0, 0.3);
            // Outlier should be predicted as true.
            Assert.True(results[3].PredictedLabel);
            Assert.InRange(results[3].Score, 0.3, 1);

            // Inlier should be predicted as false.
            Assert.False(results[4].PredictedLabel);
            Assert.InRange(results[4].Score, 0, 0.3);

            // Inlier should be predicted as false.
            Assert.False(results[5].PredictedLabel);
            Assert.InRange(results[5].Score, 0, 0.3);
            // Inlier should be predicted as false.
            Assert.False(results[6].PredictedLabel);
            Assert.InRange(results[6].Score, 0, 0.3);

            // Outlier should be predicted as true.
            Assert.True(results[7].PredictedLabel);
            Assert.InRange(results[7].Score, 0.3, 1);
        }

        private IDataView DetectAnomalyInMnistOneClass(string trainPath, string testPath)
        {
            var loader = ML.Data.CreateTextLoader(new[]
            {
                new TextLoader.Column(DefaultColumnNames.Label, DataKind.Single, 0),
                new TextLoader.Column(DefaultColumnNames.Features, DataKind.Single, 1, 784)
            },
            allowSparse: true);

            var trainData = loader.Load(trainPath);
            var testData = loader.Load(testPath);

            var trainer = ML.AnomalyDetection.Trainers.RandomizedPca();

            var model = trainer.Fit(trainData);
            return model.Transform(testData);
        }

        /// <summary>
        /// Check that when PCA created invalid eigenvectors with NaNs a readable exception message is thrown.
        /// </summary>
        [Fact]

        public void PcaTrainerInvalidEigenvectorsException()
        {
            var mlContext = new MLContext(seed: 0);

            var trainer = mlContext.AnomalyDetection.Trainers.RandomizedPca(
                featureColumnName: nameof(DataPoint.Features), rank: 3);

            var samples = new List<DataPoint>()
            {
                new DataPoint(){ Features = new float[3] {1, 0, 2} },
                new DataPoint(){ Features = new float[3] {2, 0, 4} },
                new DataPoint(){ Features = new float[3] {4, 0, 8} },
                new DataPoint(){ Features = new float[3] {8, 0, 16} }
            };

            var data = mlContext.Data.LoadFromEnumerable(samples);

            bool exceptionThrown = false;
            try
            {
                // Since we provided a dataset where all rows are linearly dependent,
                // the PCA algorithm will likely fail when extracting 3 eigenvectors
                // and produce eigenvectors with NaN.
                var model = trainer.Fit(data);
            }
            catch (ArgumentOutOfRangeException ex)
            {
                exceptionThrown = true;
                Assert.Contains("The learnt eigenvectors contained NaN values", ex.Message);
            }

            Assert.True(exceptionThrown);
        }
    }
}
