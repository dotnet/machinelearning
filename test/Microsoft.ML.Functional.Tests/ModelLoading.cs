﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Functional.Tests
{
    public partial class ModelLoadingTests : TestDataPipeBase
    {
        public ModelLoadingTests(ITestOutputHelper output) : base(output)
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

        [Fact]
        public void LoadModelAndExtractPredictor()
        {
            var file = new MultiFileSource(GetDataPath(TestDatasets.adult.trainFilename));
            var loader = ML.Data.CreateTextLoader<InputData>(hasHeader: true, dataSample: file);
            var data = loader.Load(file);

            // Pipeline.
            var pipeline = ML.BinaryClassification.Trainers.Gam();
            // Define the same pipeline starting with the loader.
            var pipeline1 = loader.Append(ML.BinaryClassification.Trainers.Gam());

            // Train.
            var transformerModel = pipeline.Fit(data);
            var compositeLoaderModel = pipeline1.Fit(file);

            // Save and reload the "same" model with some differences in structure.

            // In this case we are saving the transformer model, but *not* the loader, just the schema from that loader.
            string modelAndSchemaPath = GetOutputPath(FullTestName + "-model-schema.zip");
            ML.Model.Save(transformerModel, data.Schema, modelAndSchemaPath);

            // In this case we have combined the loader with the transformer model to form a "composite" loader, and are just
            // saving that one loader to this file.
            string compositeLoaderModelPath = GetOutputPath(FullTestName + "-composite-model.zip");
            ML.Model.Save(null, compositeLoaderModel, compositeLoaderModelPath);

            // In this case we are saving the transformer model, as well as the associated data loader.
            string loaderAndTransformerModelPath = GetOutputPath(FullTestName + "-loader-transformer.zip");
            ML.Model.Save(transformerModel, loader, loaderAndTransformerModelPath);

            ITransformer loadedTransformerModel;
            IDataLoader<IMultiStreamSource> loadedCompositeLoader;
            ITransformer loadedTransformerModel1;
            using (var fs = File.OpenRead(modelAndSchemaPath))
                loadedTransformerModel = ML.Model.Load(fs, out var loadedSchema);
            using (var fs = File.OpenRead(compositeLoaderModelPath))
            {
                // This model can be loaded either as a composite data loader,
                // a transformer model + an input schema, or a transformer model + a data loader.
                var t = ML.Model.LoadWithDataLoader(fs, out loadedCompositeLoader);
                // This is a bit strange, as it seems to test that it can reload from the same
                // stream twice opened only once, which as far as I know is not really a requirement
                // of the design or API, but we are nonetheless testing it. If this winds up failing,
                // I'm not sure we should really insist on this as a design requirement.
                var t1 = ML.Model.Load(fs, out var s);

                CheckSameSchemas(loadedCompositeLoader.GetOutputSchema(), s);
                // We combined the GAM with the loader, so the remaining chain should just be empty.
                Assert.Empty(Assert.IsType<TransformerChain<ITransformer>>(t));
                Assert.Empty(Assert.IsType<TransformerChain<ITransformer>>(t1));
            }
            using (var fs = File.OpenRead(loaderAndTransformerModelPath))
            {
                // This model can be loaded either as a composite data loader,
                // a transformer model + an input schema, or a transformer model + a data loader.
                var t = ML.Model.Load(fs, out var s);
                CheckSameSchemas(loader.GetOutputSchema(), s);

                loadedTransformerModel1 = ML.Model.LoadWithDataLoader(fs, out var l);
            }

            void AssertIsGam(ITransformer trans)
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

            Done();
        }

        [Fact]
        public void SaveAndLoadModelWithLoader()
        {
            var file = new MultiFileSource(GetDataPath(TestDatasets.adult.trainFilename));
            var loader = ML.Data.CreateTextLoader<InputData>(hasHeader: true, dataSample: file);
            var data = loader.Load(file);

            // Pipeline.
            var pipeline = ML.BinaryClassification.Trainers.Gam();

            // Train.
            var model = pipeline.Fit(data);

            // Save and reload.
            string modelPath = GetOutputPath(FullTestName + "-model.zip");
            ML.Model.Save(model, loader, modelPath);

            IDataLoader<IMultiStreamSource> loadedLoader;
            ITransformer loadedModelWithoutLoader;
            ITransformer loadedModelWithLoader;
            DataViewSchema loadedSchema;
            using (var fs = File.OpenRead(modelPath))
            {
                loadedModelWithLoader = ML.Model.LoadWithDataLoader(fs, out loadedLoader);
                Assert.IsAssignableFrom<ISingleFeaturePredictionTransformer<object>>(loadedModelWithLoader);
                loadedModelWithoutLoader = ML.Model.Load(fs, out loadedSchema);
                Assert.IsAssignableFrom<ISingleFeaturePredictionTransformer<object>>(loadedModelWithoutLoader);

                CheckSameSchemas(loadedLoader.GetOutputSchema(), loadedSchema);
            }

            // When using a novel data source other than one derived from the loader, we will not have
            // the slot names.
            data = ML.Data.LoadFromEnumerable(new[] { new InputData() });
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
            var file = new MultiFileSource(GetDataPath(TestDatasets.adult.trainFilename));
            var loader = ML.Data.CreateTextLoader<InputData>(hasHeader: true, dataSample: file);
            var data = loader.Load(file);

            // Pipeline.
            var pipeline = ML.Transforms.Normalize("Features");

            // Train.
            var model = pipeline.Fit(data);

            // Save and reload.
            string modelPath = GetOutputPath(FullTestName + "-model.zip");
            ML.Model.Save(model, loader, modelPath);

            ITransformer loadedModel;
            DataViewSchema loadedSchema;
            using (var fs = File.OpenRead(modelPath))
                loadedModel = ML.Model.Load(fs, out loadedSchema);

            // Without using the schema from the model we lose the slot names.
            data = ML.Data.LoadFromEnumerable(new[] { new InputData() });
            data = loadedModel.Transform(data);
            Assert.True(!data.Schema["Features"].HasSlotNames());

            data = ML.Data.LoadFromEnumerable(new[] { new InputData() }, loadedSchema);
            Assert.True(data.Schema["Features"].HasSlotNames());
        }

        [Fact]
        public void SaveTextLoaderAndLoad()
        {
            var file = new MultiFileSource(GetDataPath(TestDatasets.adult.trainFilename));
            var loader = ML.Data.CreateTextLoader<InputData>(hasHeader: true, dataSample: file);

            string modelPath = GetOutputPath(FullTestName + "-model.zip");
            ML.Model.Save(null, loader, modelPath);

            Load(modelPath, out var loadedWithSchema, out var loadedSchema,
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
            var file = new MultiFileSource(GetDataPath(TestDatasets.adult.trainFilename));
            var loader = ML.Data.CreateTextLoader<InputData>(hasHeader: true, dataSample: file);
            var composite = loader.Append(ML.Transforms.Normalize("Features"));
            var loaderWithEmbeddedModel = composite.Fit(file);

            string modelPath = GetOutputPath(FullTestName + "-model.zip");
            ML.Model.Save(null, loaderWithEmbeddedModel, modelPath);

            Load(modelPath, out var loadedWithSchema, out var loadedSchema,
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

            CheckSameSchemas(loaderWithEmbeddedModel.GetOutputSchema(), loadedSchema);
            var schemaFromLoadedLoader = loadedLoaderWithTransformer.GetOutputSchema();
            CheckSameSchemas(loaderWithEmbeddedModel.GetOutputSchema(), schemaFromLoadedLoader);

            // The type of the loader itself should be a composite data loader, and its single transformer
            // should be the normalizing transformer.
            var compositeLoader = Assert.IsType<CompositeDataLoader<IMultiStreamSource, ITransformer>>(loadedLoaderWithTransformer);
            var chainFromLoader = compositeLoader.Transformer;
            Assert.IsType<NormalizingTransformer>(Assert.Single(compositeLoader.Transformer));

            Done();
        }

        [Fact]
        public void SaveLoaderAndTransformerAndLoad()
        {
            var file = new MultiFileSource(GetDataPath(TestDatasets.adult.trainFilename));
            var loader = ML.Data.CreateTextLoader<InputData>(hasHeader: true, dataSample: file);
            var estimator = ML.Transforms.Normalize("Features");
            var data = loader.Load(file);
            var model = estimator.Fit(data);

            // First get the input schema.
            var expectedInputSchema = loader.GetOutputSchema();
            Assert.Equal(2, expectedInputSchema.Count);
            Assert.NotNull(expectedInputSchema.GetColumnOrNull("Label"));
            Assert.NotNull(expectedInputSchema.GetColumnOrNull("Features"));
            Assert.True(expectedInputSchema["Features"].HasSlotNames());

            string modelPath = GetOutputPath(FullTestName + "-model.zip");
            ML.Model.Save(model, loader, modelPath);

            // Reload the loader and schema.
            Load(modelPath, out var loadedWithSchema, out var loadedInputSchema,
                out var loadedWithLoader, out var loadedLoaderWithTransformer);
            Assert.IsType<NormalizingTransformer>(loadedWithSchema);
            Assert.IsType<NormalizingTransformer>(loadedWithLoader);
            Assert.IsType<TextLoader>(loadedLoaderWithTransformer);

            CheckSameSchemas(expectedInputSchema, loadedInputSchema);
            var reloadedLoaderInputSchema = loadedLoaderWithTransformer.GetOutputSchema();
            CheckSameSchemas(expectedInputSchema, reloadedLoaderInputSchema);

            Done();
        }

        [Fact]
        public void SaveTransformerAndSchemaAndLoad()
        {
            var file = new MultiFileSource(GetDataPath(TestDatasets.adult.trainFilename));
            var loader = ML.Data.CreateTextLoader<InputData>(hasHeader: true, dataSample: file);
            var estimator = ML.Transforms.Normalize("Features");
            var model = estimator.Fit(loader.Load(file));

            string modelPath = GetOutputPath(FullTestName + "-model.zip");
            ML.Model.Save(model, loader.GetOutputSchema(), modelPath);

            Load(modelPath, out var loadedWithSchema, out var loadedSchema,
                out var loadedWithLoader, out var loadedLoaderWithTransformer);
            Assert.True(loadedWithSchema is NormalizingTransformer);
            Assert.True(loadedSchema.Count == 2 &&
                loadedSchema.GetColumnOrNull("Label") != null
                && loadedSchema.GetColumnOrNull("Features") != null
                && loadedSchema["Features"].HasSlotNames());
            Assert.Null(loadedWithLoader);
            Assert.Null(loadedLoaderWithTransformer);
        }

        private void Load(string filename, out ITransformer loadedWithSchema, out DataViewSchema loadedSchema,
            out ITransformer loadedWithLoader, out IDataLoader<IMultiStreamSource> loadedLoaderWithTransformer)
        {
            using (var fs = File.OpenRead(filename))
            {
                loadedWithSchema = ML.Model.Load(fs, out loadedSchema);
                try
                {
                    loadedWithLoader = ML.Model.LoadWithDataLoader(fs, out loadedLoaderWithTransformer);
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
