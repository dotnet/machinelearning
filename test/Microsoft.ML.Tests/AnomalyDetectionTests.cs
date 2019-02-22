// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
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
        /// RandomizedPcaTrainer test 
        /// </summary>
        [Fact]
        public void RandomizedPcaTrainerBaselineTest()
        {
            var reader = new TextLoader(Env, new TextLoader.Options()
            {
                HasHeader = false,
                Separator = "\t",
                Columns = new[]
                {
                    new TextLoader.Column("Label", DataKind.R4, 0),
                    new TextLoader.Column(DefaultColumnNames.Features, DataKind.R4, new [] { new TextLoader.Range(1, 784) })
                }
            });

            var trainData = reader.Read(GetDataPath(TestDatasets.mnistOneClass.trainFilename));
            var testData = reader.Read(GetDataPath(TestDatasets.mnistOneClass.testFilename));

            var pipeline = ML.AnomalyDetection.Trainers.RandomizedPca();

            var transformer = pipeline.Fit(trainData);
            var transformedData = transformer.Transform(testData);

            // Evaluate
            var metrics = ML.AnomalyDetection.Evaluate(transformedData, k: 5);

            Assert.Equal(0.98286, metrics.Auc, 5);
            Assert.Equal(0.90000, metrics.DrAtK, 5);
        }

        /// <summary>
        /// Test anomaly detection when the test data has no anomalies
        /// </summary>
        [Fact]
        public void NoAnomalyTest()
        {
            var reader = new TextLoader(Env, new TextLoader.Options()
            {
                HasHeader = false,
                Separator = "\t",
                Columns = new[]
                {
                    new TextLoader.Column("Label", DataKind.R4, 0),
                    new TextLoader.Column(DefaultColumnNames.Features, DataKind.R4, new [] { new TextLoader.Range(1, 784) })
                }
            });

            var trainData = reader.Read(GetDataPath(TestDatasets.mnistOneClass.trainFilename));

            var pipeline = ML.AnomalyDetection.Trainers.RandomizedPca();

            var transformer = pipeline.Fit(trainData);
            var transformedData = transformer.Transform(trainData);

            Assert.Throws<ArgumentOutOfRangeException>(() => ML.AnomalyDetection.Evaluate(transformedData));
        }
    }
}
