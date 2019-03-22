// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using System.IO.Compression;
using System.Linq;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
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
                mlContext.Model.Save(model, data.Schema, file);

            // Load model from a file.
            ITransformer serializedModel;
            using (var file = File.OpenRead(modelPath))
                serializedModel = mlContext.Model.Load(file, out var serializedSchema);

            // Create prediction engine and test predictions.
            var originalPredictionEngine = mlContext.Model.CreatePredictionEngine<HousingRegression, ScoreColumn>(model);
            var serializedPredictionEngine = mlContext.Model.CreatePredictionEngine<HousingRegression, ScoreColumn>(serializedModel);
            
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

        [Fact]
        public void LoadModelAndExtractPredictor()
        {
            var mlContext = new MLContext(seed: 1);

            // Load the dataset.
            var file = new MultiFileSource(GetDataPath(TestDatasets.adult.trainFilename));
            var loader = mlContext.Data.CreateTextLoader<Adult>(hasHeader: true, dataSample: file);
            var data = mlContext.Data.LoadFromTextFile<Adult>(GetDataPath(TestDatasets.adult.trainFilename),
                hasHeader: TestDatasets.adult.fileHasHeader,
                separatorChar: TestDatasets.adult.fileSeparator);

            // Pipeline.
            var trainerPipeline = mlContext.Transforms.Concatenate("Features", Adult.NumericalFeatures)
                .Append(mlContext.BinaryClassification.Trainers.LogisticRegression());
            // Define the same pipeline starting with the loader.
            var loaderAndTrainerPipeline = loader.Append(mlContext.Transforms.Concatenate("Features", Adult.NumericalFeatures))
                .Append(mlContext.BinaryClassification.Trainers.LogisticRegression());

            // Fit the pipelines to the dataset.
            var transformerModel = trainerPipeline.Fit(data);
            var compositeLoaderModel = loaderAndTrainerPipeline.Fit(file);

            // Serialize the models to a stream.
            // Save a transformer model with an input schema.
            string modelAndSchemaPath = DeleteOutputPath(FullTestName + "-model-schema.zip");
            mlContext.Model.Save(transformerModel, data.Schema, modelAndSchemaPath);
            // Save a loader model without an input schema.
            string compositeLoaderModelPath = DeleteOutputPath(FullTestName + "-composite-model.zip");
            mlContext.Model.Save(compositeLoaderModel, compositeLoaderModelPath);
            // Save a transformer model, specifying the loader.
            string loaderAndTransformerModelPath = DeleteOutputPath(FullTestName + "-loader-transformer.zip");
            mlContext.Model.Save(loader, transformerModel, loaderAndTransformerModelPath);

            // Load the serialized models back in.
            ITransformer serializedTransformerModel;
            IDataLoader<IMultiStreamSource> serializedCompositeLoader;
            ITransformer serializedCompositeLoaderWithSchema;
            ITransformer serializedCompositeLoaderWithLoader;
            IDataLoader<IMultiStreamSource> serializedLoaderAndTransformerModel;
            ITransformer serializedLoaderAndTransformerModelWithSchema;
            ITransformer serializedLoaderAndTransformerModelWithLoader;
            // Load the transformer model.
            using (var fs = File.OpenRead(modelAndSchemaPath))
                serializedTransformerModel = mlContext.Model.Load(fs, out var loadedSchema);
            using (var fs = File.OpenRead(compositeLoaderModelPath))
            {
                // This model can be loaded either as a composite data loader,
                // a transformer model + an input schema, or a transformer model + a data loader.
                serializedCompositeLoader = mlContext.Model.Load(fs);
                serializedCompositeLoaderWithLoader = mlContext.Model.LoadWithDataLoader(fs, out IDataLoader<IMultiStreamSource> serializedLoader);
                serializedCompositeLoaderWithSchema = mlContext.Model.Load(fs, out var schema);
                Common.AssertEqual(loader.GetOutputSchema(), schema);
            }
            using (var fs = File.OpenRead(loaderAndTransformerModelPath))
            {
                // This model can be loaded either as a composite data loader,
                // a transformer model + an input schema, or a transformer model + a data loader.
                serializedLoaderAndTransformerModel = mlContext.Model.Load(fs);
                serializedLoaderAndTransformerModelWithSchema = mlContext.Model.Load(fs, out var schema);
                Common.AssertEqual(data.Schema, schema);
                serializedLoaderAndTransformerModelWithLoader = mlContext.Model.LoadWithDataLoader(fs, out IDataLoader<IMultiStreamSource> serializedLoader);
            }

            // Validate that the models contain the expected estimator.
            var gam = ((serializedTransformerModel as ISingleFeaturePredictionTransformer<object>).Model
                as CalibratedModelParametersBase).SubModel
                as GamBinaryModelParameters;
            Assert.NotNull(gam);

            gam = (((serializedCompositeLoader as CompositeDataLoader<IMultiStreamSource, ITransformer>).Transformer.LastTransformer
                as ISingleFeaturePredictionTransformer<object>).Model
                as CalibratedModelParametersBase).SubModel
                as GamBinaryModelParameters;
            Assert.NotNull(gam);

            gam = (((serializedLoaderAndTransformerModelWithLoader as TransformerChain<ITransformer>).LastTransformer
                as ISingleFeaturePredictionTransformer<object>).Model
                as CalibratedModelParametersBase).SubModel
                as GamBinaryModelParameters;
            Assert.NotNull(gam);
        }
    }
}