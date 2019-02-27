// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.Data.DataView;
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

            Assert.Equal(0.98667, metrics.Auc, 5);
            Assert.Equal(0.90000, metrics.DrAtK, 5);
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
