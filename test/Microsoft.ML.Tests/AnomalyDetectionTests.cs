// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
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
            var metrics = ML.AnomalyDetection.Evaluate(transformedData, k: 5);

            Assert.Equal(0.98667, metrics.AreaUnderRocCurve, 5);
            Assert.Equal(0.90000, metrics.DetectionRateAtKFalsePositives, 5);
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
                EnsureZeroMean = false
            };

            // Create anther anomaly detector. Its underlying algorithm is randomized PCA.
            var trainer2 = mlContext.AnomalyDetection.Trainers.RandomizedPca(options);

            // Test the second detector.
            ExecutePipelineWithGivenRandomizedPcaTrainer(mlContext, trainer2);
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
                new DataPoint(){ Features= new float[3] {1, 0, 0} },
                new DataPoint(){ Features= new float[3] {0, 2, 1} },
                new DataPoint(){ Features= new float[3] {1, 2, 3} },
                new DataPoint(){ Features= new float[3] {0, 1, 0} },
                new DataPoint(){ Features= new float[3] {0, 2, 1} },
                new DataPoint(){ Features= new float[3] {-100, 50, -100} }
            };

            // Convert the List<DataPoint> to IDataView, a consumble format to ML.NET functions.
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // Train the anomaly detector.
            var model = trainer.Fit(data);

            // Apply the trained model on the training data.
            var transformed = model.Transform(data);

            // Read ML.NET predictions into IEnumerable<Result>.
            var results = mlContext.Data.CreateEnumerable<Result>(transformed, reuseRowObject: false).ToList();

            // First 5 examples are inliers.
            for (int i = 0; i < 5; ++i)
            {
                // Inlier should be predicted as true.
                Assert.True(results[i].PredictedLabel);
                // Higher score means closer to inlier.
                Assert.InRange(results[i].Score, 0.3, 1);
            }

            // Last example is outlier. Note that outlier should be predicted as false.
            Assert.False(results[5].PredictedLabel);
            Assert.InRange(results[5].Score, 0, 0.3);
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
    }
}
