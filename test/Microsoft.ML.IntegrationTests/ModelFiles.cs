// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.IntegrationTests.Datasets;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.IntegrationTests
{
    public partial class ModelFiles : IntegrationTestBaseClass
    {
        public ModelFiles(ITestOutputHelper output) : base(output)
        {
        }

        private class InputData
        {
            [LoadColumn(0)]
            public bool Label { get; set; }
            [LoadColumn(9, 14)]
            [VectorType(6)]
            public float[] Features { get; set; }
        }

        /// <summary>
        /// Model Files: The (minimum) nuget version can be found in the model file.
        /// </summary>
        [Fact]
        public void DetermineNugetVersionFromModel()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset.
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(TestCommon.GetDataPath(DataDir, TestDatasets.housing.trainFilename), hasHeader: true);

            // Create a pipeline to train on the housing data.
            var pipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .Append(mlContext.Regression.Trainers.FastTree(
                    new FastTreeRegressionTrainer.Options { NumberOfThreads = 1, NumberOfTrees = 10 }));

            // Fit the pipeline.
            var model = pipeline.Fit(data);

            // Save model to a file.
            var modelPath = TestCommon.DeleteOutputPath(OutDir, "determineNugetVersionFromModel.zip");
            mlContext.Model.Save(model, data.Schema, modelPath);

            // Check that the version can be extracted from the model.
            var versionFileName = @"TrainingInfo" + Path.DirectorySeparatorChar + "Version.txt";
            using (ZipArchive archive = ZipFile.OpenRead(modelPath))
            {
                // The version of the entire model is kept in the version file.
                var versionPath = archive.Entries.First(x => x.FullName == versionFileName);
                Assert.NotNull(versionPath);
                using (var stream = versionPath.Open())
                using (var reader = new StreamReader(stream))
                {
                    // The only line in the file is the version of the model.
                    var line = reader.ReadLine();
                    Assert.Matches(new Regex(@"(\d+)\.(\d+)\.(\d+)(-[dev|ci|preview\.(\d+)\.(\d+)\.(\d+)]){0,1}"), line);
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
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(TestCommon.GetDataPath(DataDir, TestDatasets.housing.trainFilename), hasHeader: true);

            // Create a pipeline to train on the housing data.
            var pipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .Append(mlContext.Regression.Trainers.FastTree(
                    new FastTreeRegressionTrainer.Options { NumberOfThreads = 1, NumberOfTrees = 10 }));

            // Fit the pipeline.
            var model = pipeline.Fit(data);

            var modelPath = TestCommon.DeleteOutputPath(OutDir, "fitPipelineSaveModelAndPredict.zip");
            // Save model to a file.
            mlContext.Model.Save(model, data.Schema, modelPath);

            // Load model from a file.
            ITransformer serializedModel;
            using (var file = File.OpenRead(modelPath))
            {
                serializedModel = mlContext.Model.Load(file, out var serializedSchema);
                TestCommon.CheckSameSchemas(data.Schema, serializedSchema);
            }

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
            var mlContext = new MLContext(1);

            var file = new MultiFileSource(TestCommon.GetDataPath(DataDir, TestDatasets.adult.trainFilename));
            var loader = mlContext.Data.CreateTextLoader<InputData>(hasHeader: true, dataSample: file);
            var data = loader.Load(file);

            // Pipeline.
            var pipeline = mlContext.BinaryClassification.Trainers.Gam();
            // Define the same pipeline starting with the loader.
            var pipeline1 = loader.Append(mlContext.BinaryClassification.Trainers.Gam());

            // Train.
            var transformerModel = pipeline.Fit(data);
            var compositeLoaderModel = pipeline1.Fit(file);

            // Save and reload the "same" model with some differences in structure.

            // In this case we are saving the transformer model, but *not* the loader, just the schema from that loader.
            string modelAndSchemaPath = TestCommon.GetOutputPath(OutDir, FullTestName + "-model-schema.zip");
            mlContext.Model.Save(transformerModel, data.Schema, modelAndSchemaPath);

            // In this case we have combined the loader with the transformer model to form a "composite" loader, and are just
            // saving that one loader to this file.
            string compositeLoaderModelPath = TestCommon.GetOutputPath(OutDir, FullTestName + "-composite-model.zip");
            mlContext.Model.Save(null, compositeLoaderModel, compositeLoaderModelPath);

            // In this case we are saving the transformer model, as well as the associated data loader.
            string loaderAndTransformerModelPath = TestCommon.GetOutputPath(OutDir, FullTestName + "-loader-transformer.zip");
            mlContext.Model.Save(transformerModel, loader, loaderAndTransformerModelPath);

            ITransformer loadedTransformerModel;
            IDataLoader<IMultiStreamSource> loadedCompositeLoader;
            ITransformer loadedTransformerModel1;
            using (var fs = File.OpenRead(modelAndSchemaPath))
                loadedTransformerModel = mlContext.Model.Load(fs, out var loadedSchema);
            using (var fs = File.OpenRead(compositeLoaderModelPath))
            {
                // This model can be loaded either as a composite data loader,
                // a transformer model + an input schema, or a transformer model + a data loader.
                var t = mlContext.Model.LoadWithDataLoader(fs, out loadedCompositeLoader);
                // This is a bit strange, as it seems to test that it can reload from the same
                // stream twice opened only once, which as far as I know is not really a requirement
                // of the design or API, but we are nonetheless testing it. If this winds up failing,
                // I'm not sure we should really insist on this as a design requirement.
                var t1 = mlContext.Model.Load(fs, out var s);

                TestCommon.CheckSameSchemas(loadedCompositeLoader.GetOutputSchema(), s);
                // We combined the GAM with the loader, so the remaining chain should just be empty.
                Assert.Empty(Assert.IsType<TransformerChain<ITransformer>>(t));
                Assert.Empty(Assert.IsType<TransformerChain<ITransformer>>(t1));
            }
            using (var fs = File.OpenRead(loaderAndTransformerModelPath))
            {
                // This model can be loaded either as a composite data loader,
                // a transformer model + an input schema, or a transformer model + a data loader.
                var t = mlContext.Model.Load(fs, out var s);
                TestCommon.CheckSameSchemas(loader.GetOutputSchema(), s);

                loadedTransformerModel1 = mlContext.Model.LoadWithDataLoader(fs, out var l);
            }

            static void AssertIsGam(ITransformer trans)
            {
                Assert.IsType<GamBinaryModelParameters>(
                    Assert.IsAssignableFrom<CalibratedModelParametersBase>(
                        Assert.IsAssignableFrom<ISingleFeaturePredictionTransformer<object>>(trans).Model).SubModel);
            }

            // In the case of the directly used transformer model, the thing we loaded should be itself the result from fitting GAM.
            AssertIsGam(loadedTransformerModel);

            // This is quite similar, the fact that we omitted saving the loader and saved the input schema to the model itself.
            AssertIsGam(loadedTransformerModel1);

            // If we had combined the transformer with the loader, and then saved *that*, then the resulting loaded "model"
            // will be empty (as tested above), but the loader itself with a composite loader containing the result from
            // fitting GAM as the sole item in its transformer chain.
            var fromComposite = Assert.Single(Assert.IsType<TransformerChain<ITransformer>>(
                Assert.IsType<CompositeDataLoader<IMultiStreamSource, ITransformer>>(loadedCompositeLoader).Transformer));
            AssertIsGam(fromComposite);
        }

        public class ModelInput
        {
#pragma warning disable SA1401
            public string[] CategoricalFeatures;
            public float[] NumericalFeatures;
#pragma warning restore SA1401
            public float Label;
        }

        public class ModelOutput
        {
#pragma warning disable SA1401
            public float[] Score;
#pragma warning restore SA1401
        }


        [Fact]
        public void LoadModelWithOptionalColumnTransform()
        {
            SchemaDefinition inputSchemaDefinition = SchemaDefinition.Create(typeof(ModelInput));
            inputSchemaDefinition[nameof(ModelInput.CategoricalFeatures)].ColumnType = new VectorDataViewType(TextDataViewType.Instance, 5);
            inputSchemaDefinition[nameof(ModelInput.NumericalFeatures)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, 3);

            var mlContext = new MLContext(1);
            ITransformer trainedModel;
            DataViewSchema dataViewSchema;
            trainedModel = mlContext.Model.Load(TestCommon.GetDataPath(DataDir, "backcompat", "modelwithoptionalcolumntransform.zip"), out dataViewSchema);

            var modelInput = new ModelInput()
            {
                CategoricalFeatures = new[] { "ABC", "ABC", "ABC", "ABC", "ABC" },
                NumericalFeatures = new float[] { 1, 1, 1 },
                Label = 1
            };

            // test create prediction engine with user defined schema
            var model = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel, inputSchemaDefinition: inputSchemaDefinition);
            var prediction = model.Predict(modelInput);

            // test create prediction engine with schema loaded from model
            var model2 = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel, inputSchema: dataViewSchema);
            var prediction2 = model2.Predict(modelInput);

            Assert.Equal(1, prediction.Score[0]);
            Assert.Equal(1, prediction2.Score[0]);
        }

        [Fact]
        public void SaveAndLoadModelWithLoader()
        {
            var mlContext = new MLContext(1);

            var file = new MultiFileSource(TestCommon.GetDataPath(DataDir, TestDatasets.adult.trainFilename));
            var loader = mlContext.Data.CreateTextLoader<InputData>(hasHeader: true, dataSample: file);
            var data = loader.Load(file);

            // Pipeline.
            var pipeline = mlContext.BinaryClassification.Trainers.Gam();

            // Train.
            var model = pipeline.Fit(data);

            // Save and reload.
            string modelPath = TestCommon.GetOutputPath(OutDir, FullTestName + "-model.zip");
            mlContext.Model.Save(model, loader, modelPath);

            IDataLoader<IMultiStreamSource> loadedLoader;
            ITransformer loadedModelWithoutLoader;
            ITransformer loadedModelWithLoader;
            DataViewSchema loadedSchema;
            using (var fs = File.OpenRead(modelPath))
            {
                loadedModelWithLoader = mlContext.Model.LoadWithDataLoader(fs, out loadedLoader);
                Assert.IsAssignableFrom<ISingleFeaturePredictionTransformer<object>>(loadedModelWithLoader);
                loadedModelWithoutLoader = mlContext.Model.Load(fs, out loadedSchema);
                Assert.IsAssignableFrom<ISingleFeaturePredictionTransformer<object>>(loadedModelWithoutLoader);

                TestCommon.CheckSameSchemas(loadedLoader.GetOutputSchema(), loadedSchema);
            }

            // When using a novel data source other than one derived from the loader, we will not have
            // the slot names.
            data = mlContext.Data.LoadFromEnumerable(new[] { new InputData() });
            data = loadedModelWithoutLoader.Transform(data);
            Assert.False(data.Schema["Features"].HasSlotNames());
            // When we plumb the loaded schema through the transformer though, we should have slot names.
            var noLoaderTransformedSchema = loadedModelWithoutLoader.GetOutputSchema(loadedSchema);
            Assert.True(noLoaderTransformedSchema["Features"].HasSlotNames());

            data = loadedLoader.Load(file);
            Assert.True(data.Schema["Features"].HasSlotNames());
            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            data.Schema["Features"].GetSlotNames(ref slotNames);
            var ageIndex = FindIndex(slotNames.GetValues(), "age");
            var singleFeaturePredictionTransformer = loadedModelWithLoader as ISingleFeaturePredictionTransformer<object>;
            Assert.NotNull(singleFeaturePredictionTransformer);
            var calibratedModelParameters = singleFeaturePredictionTransformer.Model as CalibratedModelParametersBase;
            Assert.NotNull(calibratedModelParameters);
            var gamModel = calibratedModelParameters.SubModel as GamBinaryModelParameters;
            Assert.NotNull(gamModel);
            var ageBinUpperBounds = gamModel.GetBinUpperBounds(ageIndex);
            var ageBinEffects = gamModel.GetBinEffects(ageIndex);
        }

        [Fact]
        public void LoadSchemaAndCreateNewData()
        {
            var mlContext = new MLContext(1);

            var file = new MultiFileSource(TestCommon.GetDataPath(DataDir, TestDatasets.adult.trainFilename));
            var loader = mlContext.Data.CreateTextLoader<InputData>(hasHeader: true, dataSample: file);
            var data = loader.Load(file);

            // Pipeline.
            var pipeline = mlContext.Transforms.NormalizeMinMax("Features");

            // Train.
            var model = pipeline.Fit(data);

            // Save and reload.
            string modelPath = TestCommon.GetOutputPath(OutDir, FullTestName + "-model.zip");
            mlContext.Model.Save(model, loader, modelPath);

            ITransformer loadedModel;
            DataViewSchema loadedSchema;
            using (var fs = File.OpenRead(modelPath))
                loadedModel = mlContext.Model.Load(fs, out loadedSchema);

            // Without using the schema from the model we lose the slot names.
            data = mlContext.Data.LoadFromEnumerable(new[] { new InputData() });
            data = loadedModel.Transform(data);
            Assert.True(!data.Schema["Features"].HasSlotNames());

            data = mlContext.Data.LoadFromEnumerable(new[] { new InputData() }, loadedSchema);
            Assert.True(data.Schema["Features"].HasSlotNames());
        }

        [Fact]
        public void SaveTextLoaderAndLoad()
        {
            var mlContext = new MLContext(1);

            var file = new MultiFileSource(TestCommon.GetDataPath(DataDir, TestDatasets.adult.trainFilename));
            var loader = mlContext.Data.CreateTextLoader<InputData>(hasHeader: true, dataSample: file);

            string modelPath = TestCommon.GetOutputPath(OutDir, FullTestName + "-model.zip");
            mlContext.Model.Save(null, loader, modelPath);

            Load(mlContext, modelPath, out var loadedWithSchema, out var loadedSchema,
                out var loadedWithLoader, out var loadedLoaderWithTransformer);
            Assert.True(loadedWithSchema is TransformerChain<ITransformer>);
            Assert.False((loadedWithSchema as TransformerChain<ITransformer>).Any());
            Assert.True(loadedSchema.Count == 2 &&
                loadedSchema.GetColumnOrNull("Label") != null
                && loadedSchema.GetColumnOrNull("Features") != null
                && loadedSchema["Features"].HasSlotNames());
            Assert.True(loadedWithLoader is TransformerChain<ITransformer>);
            Assert.False((loadedWithLoader as TransformerChain<ITransformer>).Any());
            Assert.True(loadedLoaderWithTransformer is TextLoader);
            var schema = loadedLoaderWithTransformer.GetOutputSchema();
            Assert.True(schema.Count == 2 &&
                schema.GetColumnOrNull("Label") != null
                && schema.GetColumnOrNull("Features") != null
                && schema["Features"].HasSlotNames());
        }

        [Fact]
        public void SaveCompositeLoaderAndLoad()
        {
            var mlContext = new MLContext(1);

            var file = new MultiFileSource(TestCommon.GetDataPath(DataDir, TestDatasets.adult.trainFilename));
            var loader = mlContext.Data.CreateTextLoader<InputData>(hasHeader: true, dataSample: file);
            var composite = loader.Append(mlContext.Transforms.NormalizeMinMax("Features"));
            var loaderWithEmbeddedModel = composite.Fit(file);

            string modelPath = TestCommon.GetOutputPath(OutDir, FullTestName + "-model.zip");
            mlContext.Model.Save(null, loaderWithEmbeddedModel, modelPath);

            Load(mlContext, modelPath, out var loadedWithSchema, out var loadedSchema,
                out var loadedWithLoader, out var loadedLoaderWithTransformer);
            // Because we saved the transform model as part of the composite loader, with no transforms,
            // the transform that should be loaded should be an empty transformer chain, since the "model,"
            // such as it is, has been combined with the loader.
            Assert.Empty(Assert.IsType<TransformerChain<ITransformer>>(loadedWithSchema));
            Assert.Empty(Assert.IsType<TransformerChain<ITransformer>>(loadedWithLoader));

            var expectedSchema = loaderWithEmbeddedModel.GetOutputSchema();
            Assert.True(expectedSchema.Count == 3);
            Assert.NotNull(expectedSchema.GetColumnOrNull("Label"));
            Assert.NotNull(expectedSchema.GetColumnOrNull("Features"));
            Assert.True(expectedSchema["Features"].HasSlotNames());

            TestCommon.CheckSameSchemas(loaderWithEmbeddedModel.GetOutputSchema(), loadedSchema);
            var schemaFromLoadedLoader = loadedLoaderWithTransformer.GetOutputSchema();
            TestCommon.CheckSameSchemas(loaderWithEmbeddedModel.GetOutputSchema(), schemaFromLoadedLoader);

            // The type of the loader itself should be a composite data loader, and its single transformer
            // should be the normalizing transformer.
            var compositeLoader = Assert.IsType<CompositeDataLoader<IMultiStreamSource, ITransformer>>(loadedLoaderWithTransformer);
            var chainFromLoader = compositeLoader.Transformer;
            Assert.IsType<NormalizingTransformer>(Assert.Single(compositeLoader.Transformer));
        }

        [Fact]
        public void SaveLoaderAndTransformerAndLoad()
        {
            var mlContext = new MLContext(1);

            var file = new MultiFileSource(TestCommon.GetDataPath(DataDir, TestDatasets.adult.trainFilename));
            var loader = mlContext.Data.CreateTextLoader<InputData>(hasHeader: true, dataSample: file);
            var estimator = mlContext.Transforms.NormalizeMinMax("Features");
            var data = loader.Load(file);
            var model = estimator.Fit(data);

            // First get the input schema.
            var expectedInputSchema = loader.GetOutputSchema();
            Assert.Equal(2, expectedInputSchema.Count);
            Assert.NotNull(expectedInputSchema.GetColumnOrNull("Label"));
            Assert.NotNull(expectedInputSchema.GetColumnOrNull("Features"));
            Assert.True(expectedInputSchema["Features"].HasSlotNames());

            string modelPath = TestCommon.GetOutputPath(OutDir, FullTestName + "-model.zip");
            mlContext.Model.Save(model, loader, modelPath);

            // Reload the loader and schema.
            Load(mlContext, modelPath, out var loadedWithSchema, out var loadedInputSchema,
                out var loadedWithLoader, out var loadedLoaderWithTransformer);
            Assert.IsType<NormalizingTransformer>(loadedWithSchema);
            Assert.IsType<NormalizingTransformer>(loadedWithLoader);
            Assert.IsType<TextLoader>(loadedLoaderWithTransformer);

            TestCommon.CheckSameSchemas(expectedInputSchema, loadedInputSchema);
            var reloadedLoaderInputSchema = loadedLoaderWithTransformer.GetOutputSchema();
            TestCommon.CheckSameSchemas(expectedInputSchema, reloadedLoaderInputSchema);
        }

        [Fact]
        public void SaveTransformerAndSchemaAndLoad()
        {
            var mlContext = new MLContext(1);

            var file = new MultiFileSource(TestCommon.GetDataPath(DataDir, TestDatasets.adult.trainFilename));
            var loader = mlContext.Data.CreateTextLoader<InputData>(hasHeader: true, dataSample: file);
            var estimator = mlContext.Transforms.NormalizeMinMax("Features");
            var model = estimator.Fit(loader.Load(file));

            string modelPath = TestCommon.GetOutputPath(OutDir, FullTestName + "-model.zip");
            mlContext.Model.Save(model, loader.GetOutputSchema(), modelPath);

            Load(mlContext, modelPath, out var loadedWithSchema, out var loadedSchema,
                out var loadedWithLoader, out var loadedLoaderWithTransformer);
            Assert.True(loadedWithSchema is NormalizingTransformer);
            Assert.True(loadedSchema.Count == 2 &&
                loadedSchema.GetColumnOrNull("Label") != null
                && loadedSchema.GetColumnOrNull("Features") != null
                && loadedSchema["Features"].HasSlotNames());
            Assert.Null(loadedWithLoader);
            Assert.Null(loadedLoaderWithTransformer);
        }

        private void Load(MLContext mlContext, string filename, out ITransformer loadedWithSchema, out DataViewSchema loadedSchema,
            out ITransformer loadedWithLoader, out IDataLoader<IMultiStreamSource> loadedLoaderWithTransformer)
        {
            using (var fs = File.OpenRead(filename))
            {
                loadedWithSchema = mlContext.Model.Load(fs, out loadedSchema);
                try
                {
                    loadedWithLoader = mlContext.Model.LoadWithDataLoader(fs, out loadedLoaderWithTransformer);
                }
                catch (Exception)
                {
                    loadedWithLoader = null;
                    loadedLoaderWithTransformer = null;
                }
            }
        }

        private int FindIndex(ReadOnlySpan<ReadOnlyMemory<char>> values, string slotName)
        {
            int index = 0;
            foreach (var value in values)
            {
                if (value.Span.SequenceEqual(slotName.AsSpan()))
                    return index;
                index++;
            }
            return -1;
        }
    }
}
