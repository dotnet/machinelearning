// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using System.IO.Compression;
using System.Linq;
using Microsoft.ML.Functional.Tests.Datasets;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Functional.Tests
{
    public class ModelFiles : BaseTestClass
    {
        public ModelFiles(ITestOutputHelper output) : base(output)
        {
        }

        /// <summary>
        /// Model Files: The (minimum) nuget version can be found in the model file.
        /// </summary>
        [Fact]
        public void DetermineNugetVersionFromModel()
        {
            var modelFile = GetDataPath($"backcompat{Path.DirectorySeparatorChar}keep-model.zip");
            var versionFileName = @"TrainingInfo\Version.txt"; // Must use '\' for cross-platform testing.
            using (ZipArchive archive = ZipFile.OpenRead(modelFile))
            {
                // The version of the entire model is kept in the version file.
                var versionPath = archive.Entries.First(x => x.FullName == versionFileName);
                Assert.NotNull(versionPath);
                using (var stream = versionPath.Open())
                using (var reader = new StreamReader(stream))
                {
                    // The only line in the file is the version of the model.
                    var line = reader.ReadLine();
                    Assert.Equal(@"1.0.0.0", line);
                }
            }
        }

        /// <summary>
        /// Model Files: Supported model classes can be saved as ONNX files.
        /// </summary>
        [Fact]
        public void SaveModelAsOnnx()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset.
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(GetDataPath(TestDatasets.housing.trainFilename), hasHeader: true);

            // Create a pipeline to train on the housing data.
            var pipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .Append(mlContext.Regression.Trainers.Sdca(
                    new SdcaRegressionTrainer.Options { NumberOfThreads = 1 }));

            // Fit the pipeline.
            var model = pipeline.Fit(data);

            // Save as Onnx
            var modelPath = DeleteOutputPath("SaveModelAsOnnx.onnx");
            using (var file = File.Create(modelPath))
                mlContext.Model.ConvertToOnnx(model, data, file);
        }

        /// <summary>
        /// Model Files: Save a model, including all transforms, then load and make predictions.
        /// </summary>
        /// <remarks>
        /// Serves two scenarios:
        ///  1. I can train a model and save it to a file, including transforms.
        ///  2. Training and prediction happen in different processes (or even different machines). 
        ///     The actual test will not run in different processes, but will simulate the idea that the 
        ///     "communication pipe" is just a serialized model of some form.
        /// </remarks>
        [Fact]
        public void FitPipelineSaveModelAndPredict()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset.
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(GetDataPath(TestDatasets.housing.trainFilename), hasHeader: true);

            // Create a pipeline to train on the housing data.
            var pipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .Append(mlContext.Regression.Trainers.FastTree(
                    new FastTreeRegressionTrainer.Options { NumberOfThreads = 1, NumberOfTrees = 10 }));

            // Fit the pipeline.
            var model = pipeline.Fit(data);

            var modelPath = DeleteOutputPath("fitPipelineSaveModelAndPredict.zip");
            // Save model to a file.
            using (var file = File.Create(modelPath))
                mlContext.Model.Save(model, file);

            // Load model from a file.
            ITransformer serializedModel;
            using (var file = File.OpenRead(modelPath))
                serializedModel = mlContext.Model.Load(file);

            // Create prediction engine and test predictions.
            var originalPredictionEngine = model.CreatePredictionEngine<HousingRegression, ScoreColumn>(mlContext);
            var serializedPredictionEngine = serializedModel.CreatePredictionEngine<HousingRegression, ScoreColumn>(mlContext);
            
            // Take a handful of examples out of the dataset and compute predictions.
            var dataEnumerator = mlContext.Data.CreateEnumerable<HousingRegression>(mlContext.Data.TakeRows(data, 5), false);
            foreach (var row in dataEnumerator)
            {
                var originalPrediction = originalPredictionEngine.Predict(row);
                var serializedPrediction = serializedPredictionEngine.Predict(row);
                // Check that the predictions are identical.
                Assert.Equal(originalPrediction.Score, serializedPrediction.Score);
            }
        }
    }
}