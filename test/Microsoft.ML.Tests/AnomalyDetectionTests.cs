// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.ImageAnalytics;
using Microsoft.ML.Model;
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
            var mlContext = new MLContext(seed: 1, conc: 1);
            string featureColumn = "NumericFeatures";

            var reader = new TextLoader(Env, new TextLoader.Options()
            {
                HasHeader = true,
                Separator = "\t",
                Columns = new[]
                {
                    new TextLoader.Column("Label", DataKind.R4, 0),
                    new TextLoader.Column(featureColumn, DataKind.R4, new [] { new TextLoader.Range(1, 784) })
                }
            });

            var trainData = reader.Read(GetDataPath(TestDatasets.mnistOneClass.trainFilename));
            var testData = reader.Read(GetDataPath(TestDatasets.mnistOneClass.testFilename));

            var pipeline = ML.AnomalyDetection.Trainers.RandomizedPca(featureColumn);

            var transformer = pipeline.Fit(trainData);
            var transformedData = transformer.Transform(testData);

            // Evaluate
            var metrics = ML.AnomalyDetection.Evaluate(transformedData, k: 5);

            Assert.Equal(0.98269, metrics.Auc, 5);
            Assert.Equal(0.90000, metrics.DrAtK, 5);
        }
    }
}
