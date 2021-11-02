// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.IntegrationTests.Datasets;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.TestFrameworkCommon.Attributes;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.IntegrationTests
{
    public class ONNX : IntegrationTestBaseClass
    {
        // These two members are meant to be changed
        // Only when manually testing the Onnx GPU nuggets
        private const bool _fallbackToCpu = true;
        private static int? _gpuDeviceId = null;

        public ONNX(ITestOutputHelper output) : base(output)
        {
        }

        /// <summary>
        /// ONNX: Models can be serialized to ONNX, deserialized back to ML.NET, and used a pipeline.
        /// </summary>
        [OnnxFactAttribute]
        public void SaveOnnxModelLoadAndScoreFastTree()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset.
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(TestCommon.GetDataPath(DataDir, TestDatasets.housing.trainFilename), hasHeader: true);

            // Create a pipeline to train on the housing data.
            var pipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.Regression.Trainers.FastTree(
                    new FastTreeRegressionTrainer.Options { NumberOfThreads = 1, NumberOfTrees = 10 }));

            // Fit the pipeline.
            var model = pipeline.Fit(data);

            // Serialize the pipeline to a file.
            var modelFileName = "SaveOnnxLoadAndScoreFastTreeModel.onnx";
            var modelPath = TestCommon.DeleteOutputPath(OutDir, modelFileName);
            using (var file = File.Create(modelPath))
                mlContext.Model.ConvertToOnnx(model, data, file);

            // Load the model as a transform.
            // ONNX uses tensors and will return an output of a tensor with the dimension of [1,1] for a single float.
            // Therefore the VectorScoreColumn class (which contains a float [] field called Score) is used for the return
            // type on the Prediction engine.
            // See #2980 and #2981 for more information.
            var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(modelPath, gpuDeviceId: _gpuDeviceId, fallbackToCpu: _fallbackToCpu);
            var onnxModel = onnxEstimator.Fit(data);

            // Create prediction engine and test predictions.
            var originalPredictionEngine = mlContext.Model.CreatePredictionEngine<HousingRegression, ScoreColumn>(model);
            var onnxPredictionEngine = mlContext.Model.CreatePredictionEngine<HousingRegression, VectorScoreColumn>(onnxModel);

            // Take a handful of examples out of the dataset and compute predictions.
            var dataEnumerator = mlContext.Data.CreateEnumerable<HousingRegression>(mlContext.Data.TakeRows(data, 5), false);
            foreach (var row in dataEnumerator)
            {
                var originalPrediction = originalPredictionEngine.Predict(row);
                var onnxPrediction = onnxPredictionEngine.Predict(row);
                // Check that the predictions are identical.
                Assert.Equal(originalPrediction.Score, onnxPrediction.Score[0], precision: 4);
            }
        }

        /// <summary>
        /// ONNX: Models can be serialized to ONNX, deserialized back to ML.NET, and used a pipeline.
        /// </summary>
        [OnnxFactAttribute]
        public void SaveOnnxModelLoadAndScoreKMeans()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset.
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(TestCommon.GetDataPath(DataDir, TestDatasets.housing.trainFilename), hasHeader: true);

            // Create a pipeline to train on the housing data.
            var pipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.Clustering.Trainers.KMeans(
                    new KMeansTrainer.Options { NumberOfThreads = 1, MaximumNumberOfIterations = 10 }));

            // Fit the pipeline.
            var model = pipeline.Fit(data);

            // Serialize the pipeline to a file.
            var modelFileName = "SaveOnnxLoadAndScoreKMeansModel.onnx";
            var modelPath = TestCommon.DeleteOutputPath(OutDir, modelFileName);
            using (var file = File.Create(modelPath))
                mlContext.Model.ConvertToOnnx(model, data, file);

            // Load the model as a transform.
            var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(modelPath, gpuDeviceId: _gpuDeviceId, fallbackToCpu: _fallbackToCpu);
            var onnxModel = onnxEstimator.Fit(data);

            // TODO #2980: ONNX outputs don't match the outputs of the model, so we must hand-correct this for now.
            // TODO #2981: ONNX models cannot be fit as part of a pipeline, so we must use a workaround like this.
            var onnxWorkaroundPipeline = onnxModel.Append(
                mlContext.Transforms.CopyColumns("Score", "Score").Fit(onnxModel.Transform(data)));

            // Create prediction engine and test predictions.
            var originalPredictionEngine = mlContext.Model.CreatePredictionEngine<HousingRegression, VectorScoreColumn>(model);
            // TODO #2982: ONNX produces vector types and not the original output type.
            var onnxPredictionEngine = mlContext.Model.CreatePredictionEngine<HousingRegression, VectorScoreColumn>(onnxWorkaroundPipeline);

            // Take a handful of examples out of the dataset and compute predictions.
            var dataEnumerator = mlContext.Data.CreateEnumerable<HousingRegression>(mlContext.Data.TakeRows(data, 5), false);
            foreach (var row in dataEnumerator)
            {
                var originalPrediction = originalPredictionEngine.Predict(row);
                var onnxPrediction = onnxPredictionEngine.Predict(row);
                // Check that the predictions are identical.
                Common.AssertEqual(originalPrediction.Score, onnxPrediction.Score, precision: 4);
            }
        }

        /// <summary>
        /// ONNX: Models can be serialized to ONNX, deserialized back to ML.NET, and used a pipeline.
        /// </summary>
        [OnnxFactAttribute]
        public void SaveOnnxModelLoadAndScoreSDCA()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset.
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(TestCommon.GetDataPath(DataDir, TestDatasets.housing.trainFilename), hasHeader: true);

            // Create a pipeline to train on the housing data.
            var pipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.Regression.Trainers.Sdca(
                    new SdcaRegressionTrainer.Options { NumberOfThreads = 1, MaximumNumberOfIterations = 10 }));

            // Fit the pipeline.
            var model = pipeline.Fit(data);

            // Serialize the pipeline to a file.
            var modelFileName = "SaveOnnxLoadAndScoreSdcaModel.onnx";
            var modelPath = TestCommon.DeleteOutputPath(OutDir, modelFileName);
            using (var file = File.Create(modelPath))
                mlContext.Model.ConvertToOnnx(model, data, file);

            // Load the model as a transform.
            var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(modelPath, gpuDeviceId: _gpuDeviceId, fallbackToCpu: _fallbackToCpu);
            var onnxModel = onnxEstimator.Fit(data);

            // Create prediction engine and test predictions.
            var originalPredictionEngine = mlContext.Model.CreatePredictionEngine<HousingRegression, ScoreColumn>(model);
            // TODO #2982: ONNX produces vector types and not the original output type.
            var onnxPredictionEngine = mlContext.Model.CreatePredictionEngine<HousingRegression, VectorScoreColumn>(onnxModel);

            // Take a handful of examples out of the dataset and compute predictions.
            var dataEnumerator = mlContext.Data.CreateEnumerable<HousingRegression>(mlContext.Data.TakeRows(data, 5), false);
            foreach (var row in dataEnumerator)
            {
                var originalPrediction = originalPredictionEngine.Predict(row);
                var onnxPrediction = onnxPredictionEngine.Predict(row);
                // Check that the predictions are identical.
                Assert.Equal(originalPrediction.Score, onnxPrediction.Score[0], precision: 4);
            }
        }
    }
}
