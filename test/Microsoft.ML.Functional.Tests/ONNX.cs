// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Functional.Tests.Datasets;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.Trainers;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Functional.Tests
{
    public class ONNX : BaseTestClass
    {
        public ONNX(ITestOutputHelper output) : base(output)
        {
        }

        /// <summary>
        /// ONNX: I can save a model to ONNX and reload it and use it in a pipeline.
        /// </summary>
        [OnnxFactAttribute]
        public void SaveOnnxModelLoadAndScoreSDCA()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset.
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(GetDataPath(TestDatasets.housing.trainFilename), hasHeader: true);

            // Create a pipeline to train on the housing data.
            var pipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .Append(mlContext.Transforms.Normalize("Features"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.Regression.Trainers.Sdca(
                    new SdcaRegressionTrainer.Options { NumberOfThreads = 1, MaximumNumberOfIterations = 10 }));

            // Fit the pipeline.
            var model = pipeline.Fit(data);

            // Serialize the pipeline to a file.
            var modelFileName = "model.onnx";
            var modelPath = DeleteOutputPath(modelFileName);
            using (var file = File.Create(modelPath))
                mlContext.Model.ConvertToOnnx(model, data, file);

            // Load the model as a transform.
            var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(modelPath);
            var onnxModel = onnxEstimator.Fit(data);

            // Create prediction engine and test predictions.
            var originalPredictionEngine = model.CreatePredictionEngine<HousingRegression, ScoreColumn>(mlContext);
            var onnxPredictionEngine = onnxModel.CreatePredictionEngine<HousingRegression, ScoreColumn>(mlContext);

            // Take a handful of examples out of the dataset and compute predictions.
            var dataEnumerator = mlContext.Data.CreateEnumerable<HousingRegression>(mlContext.Data.TakeRows(data, 5), false);
            foreach (var row in dataEnumerator)
            {
                var originalPrediction = originalPredictionEngine.Predict(row);
                var onnxPrediction = onnxPredictionEngine.Predict(row);
                // Check that the predictions are identical.
                Assert.Equal(originalPrediction.Score, onnxPrediction.Score);
            }
        }

        private class ScoreColumn
        {
            public float Score { get; set; }
        }
    }
}
