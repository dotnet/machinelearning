// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;
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
    public class ONNX : BaseTestClass
    {
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
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(GetDataPath(TestDatasets.housing.trainFilename), hasHeader: true);

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
            var modelPath = DeleteOutputPath(modelFileName);
            using (var file = File.Create(modelPath))
                mlContext.Model.ConvertToOnnx(model, data, file);

            // Load the model as a transform.
            // Note that when saving an ML.NET model as an ONNX model, the column types and column names will
            // change. The name changes as ONNX doesn't not allow the same name for an input and output within the ONNX model.
            // Therefore names maintained but have a number appended to the end of the name. In this case, Score0 is the output
            // of the ONNX model. We are renaming Score0 to Score using Copy Columns.
            // ONNX also uses tensors and will return an output of a tensor with the dimension of [1,1] for a single float.
            // Therefore the VectorScoreColumn class (which contains a float [] field called Score) is used for the return
            // type on the Prediction engine.
            // See #2980 and #2981 for more information.
            var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(modelPath)
                .Append(mlContext.Transforms.CopyColumns("Score", "Score0"));
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
                Assert.Equal(originalPrediction.Score, onnxPrediction.Score[0], precision: 4); // Note the low-precision equality!
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
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(GetDataPath(TestDatasets.housing.trainFilename), hasHeader: true);

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
            var modelPath = DeleteOutputPath(modelFileName);
            using (var file = File.Create(modelPath))
                mlContext.Model.ConvertToOnnx(model, data, file);

            // Load the model as a transform.
            var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(modelPath);
            var onnxModel = onnxEstimator.Fit(data);

            // TODO #2980: ONNX outputs don't match the outputs of the model, so we must hand-correct this for now.
            // TODO #2981: ONNX models cannot be fit as part of a pipeline, so we must use a workaround like this.
            var onnxWorkaroundPipeline = onnxModel.Append(
                mlContext.Transforms.CopyColumns("Score", "Score0").Fit(onnxModel.Transform(data)));

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
                Common.AssertEqual(originalPrediction.Score, onnxPrediction.Score, precision: 4); // Note the low precision!
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
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(GetDataPath(TestDatasets.housing.trainFilename), hasHeader: true);

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
            var modelPath = DeleteOutputPath(modelFileName);
            using (var file = File.Create(modelPath))
                mlContext.Model.ConvertToOnnx(model, data, file);

            // Load the model as a transform.
            var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(modelPath);
            var onnxModel = onnxEstimator.Fit(data);

            // TODO #2980: ONNX outputs don't match the outputs of the model, so we must hand-correct this for now.
            // TODO #2981: ONNX models cannot be fit as part of a pipeline, so we must use a workaround like this.
            var onnxWorkaroundPipeline = onnxModel.Append(
                mlContext.Transforms.CopyColumns("Score", "Score0").Fit(onnxModel.Transform(data)));

            // Create prediction engine and test predictions.
            var originalPredictionEngine = mlContext.Model.CreatePredictionEngine<HousingRegression, ScoreColumn>(model);
            // TODO #2982: ONNX produces vector types and not the original output type.
            var onnxPredictionEngine = mlContext.Model.CreatePredictionEngine<HousingRegression, VectorScoreColumn>(onnxWorkaroundPipeline);

            // Take a handful of examples out of the dataset and compute predictions.
            var dataEnumerator = mlContext.Data.CreateEnumerable<HousingRegression>(mlContext.Data.TakeRows(data, 5), false);
            foreach (var row in dataEnumerator)
            {
                var originalPrediction = originalPredictionEngine.Predict(row);
                var onnxPrediction = onnxPredictionEngine.Predict(row);
                // Check that the predictions are identical.
                Assert.Equal(originalPrediction.Score, onnxPrediction.Score[0], precision: 4); // Note the low-precision equality!
            }
        }
    }
}
