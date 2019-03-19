// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Functional.Tests
{
    public partial class ModelLoadingTests : BaseTestClass
    {
        private MLContext _ml;

        public ModelLoadingTests(ITestOutputHelper output) : base(output)
        {
        }

        protected override void Initialize()
        {
            base.Initialize();

            _ml = new MLContext(42);
            _ml.AddStandardComponents();
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
            var loader = _ml.Data.CreateTextLoader<InputData>(hasHeader: true, dataSample: file);
            var data = loader.Load(file);

            // Pipeline.
            var pipeline = _ml.BinaryClassification.Trainers.Gam();
            // Define the same pipeline starting with the loader.
            var pipeline1 = loader.Append(_ml.BinaryClassification.Trainers.Gam());
            
            // Train.
            var transformerModel = pipeline.Fit(data);
            var compositeLoaderModel = pipeline1.Fit(file);

            // Save and reload.
            string modelAndSchemaPath = GetOutputPath(FullTestName + "-model-schema.zip");
            _ml.Model.Save(transformerModel, data.Schema, modelAndSchemaPath);
            string compositeLoaderModelPath = GetOutputPath(FullTestName + "-composite-model.zip");
            _ml.Model.Save(compositeLoaderModel, compositeLoaderModelPath);
            string loaderAndTransformerModelPath = GetOutputPath(FullTestName + "-loader-transformer.zip");
            _ml.Model.Save(loader, transformerModel, loaderAndTransformerModelPath);

            ITransformer loadedTransformerModel;
            IDataLoader<IMultiStreamSource> loadedCompositeLoader;
            ITransformer loadedTransformerModel1;
            using (var fs = File.OpenRead(modelAndSchemaPath))
                loadedTransformerModel = _ml.Model.Load(fs, out var loadedSchema);
            using (var fs = File.OpenRead(compositeLoaderModelPath))
            {
                // This model can be loaded either as a composite data loader,
                // a transformer model + an input schema, or a transformer model + a data loader.
                var t = _ml.Model.LoadWithDataLoader(fs, out IDataLoader<IMultiStreamSource> l);
                var t1 = _ml.Model.Load(fs, out var s);
                loadedCompositeLoader = _ml.Model.Load(fs);
            }
            using (var fs = File.OpenRead(loaderAndTransformerModelPath))
            {
                // This model can be loaded either as a composite data loader,
                // a transformer model + an input schema, or a transformer model + a data loader.
                var t = _ml.Model.Load(fs, out var s);
                var c = _ml.Model.Load(fs);
                loadedTransformerModel1 = _ml.Model.LoadWithDataLoader(fs, out IDataLoader<IMultiStreamSource> l);
            }

            var gam = ((loadedTransformerModel as ISingleFeaturePredictionTransformer<object>).Model
                as CalibratedModelParametersBase).SubModel
                as GamBinaryModelParameters;
            Assert.NotNull(gam);

            gam = (((loadedCompositeLoader as CompositeDataLoader<IMultiStreamSource, ITransformer>).Transformer.LastTransformer
                as ISingleFeaturePredictionTransformer<object>).Model
                as CalibratedModelParametersBase).SubModel
                as GamBinaryModelParameters;
            Assert.NotNull(gam);

            gam = (((loadedTransformerModel1 as TransformerChain<ITransformer>).LastTransformer
                as ISingleFeaturePredictionTransformer<object>).Model
                as CalibratedModelParametersBase).SubModel
                as GamBinaryModelParameters;
            Assert.NotNull(gam);
        }

        [Fact]
        public void SaveAndLoadModelWithLoader()
        {
            var file = new MultiFileSource(GetDataPath(TestDatasets.adult.trainFilename));
            var loader = _ml.Data.CreateTextLoader<InputData>(hasHeader: true, dataSample: file);
            var data = loader.Load(file);

            // Pipeline.
            var pipeline = _ml.BinaryClassification.Trainers.Gam();

            // Train.
            var model = pipeline.Fit(data);

            // Save and reload.
            string modelPath = GetOutputPath(FullTestName + "-model.zip");
            _ml.Model.Save(loader, model, modelPath);

            IDataLoader<IMultiStreamSource> loadedModel;
            ITransformer loadedModelWithoutLoader;
            DataViewSchema loadedSchema;
            using (var fs = File.OpenRead(modelPath))
            {
                loadedModel = _ml.Model.Load(fs);
                loadedModelWithoutLoader = _ml.Model.Load(fs, out loadedSchema);
            }

            // Without deserializing the loader from the model we lose the slot names.
            data = _ml.Data.LoadFromEnumerable(new[] { new InputData() });
            data = loadedModelWithoutLoader.Transform(data);
            Assert.True(!data.Schema["Features"].HasSlotNames());

            data = loadedModel.Load(file);
            Assert.True(data.Schema["Features"].HasSlotNames());
            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            data.Schema["Features"].GetSlotNames(ref slotNames);
            var ageIndex = FindIndex(slotNames.GetValues(), "age");
            var transformer = (loadedModel as CompositeDataLoader<IMultiStreamSource, ITransformer>).Transformer.LastTransformer;
            var singleFeaturePredictionTransformer = transformer as ISingleFeaturePredictionTransformer<object>;
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
            var loader = _ml.Data.CreateTextLoader<InputData>(hasHeader: true, dataSample: file);
            var data = loader.Load(file);

            // Pipeline.
            var pipeline = _ml.Transforms.Normalize("Features");

            // Train.
            var model = pipeline.Fit(data);

            // Save and reload.
            string modelPath = GetOutputPath(FullTestName + "-model.zip");
            _ml.Model.Save(loader, model, modelPath);

            ITransformer loadedModel;
            DataViewSchema loadedSchema;
            using (var fs = File.OpenRead(modelPath))
                loadedModel = _ml.Model.Load(fs, out loadedSchema);

            // Without using the schema from the model we lose the slot names.
            data = _ml.Data.LoadFromEnumerable(new[] { new InputData() });
            data = loadedModel.Transform(data);
            Assert.True(!data.Schema["Features"].HasSlotNames());

            data = _ml.Data.LoadFromEnumerable(new[] { new InputData() }, loadedSchema);
            Assert.True(data.Schema["Features"].HasSlotNames());
        }

        [Fact]
        public void SaveTextLoaderAndLoad()
        {
            var file = new MultiFileSource(GetDataPath(TestDatasets.adult.trainFilename));
            var loader = _ml.Data.CreateTextLoader<InputData>(hasHeader: true, dataSample: file);

            string modelPath = GetOutputPath(FullTestName + "-model.zip");
            _ml.Model.Save(loader, modelPath);

            Load(modelPath, out var loadedWithSchema, out var loadedSchema, out var loadedLoader,
                out var loadedWithLoader, out var loadedLoaderWithTransformer);
            Assert.True(loadedWithSchema is TransformerChain<ITransformer>);
            Assert.False((loadedWithSchema as TransformerChain<ITransformer>).Any());
            Assert.True(loadedSchema.Count == 2 &&
                loadedSchema.GetColumnOrNull("Label") != null
                && loadedSchema.GetColumnOrNull("Features") != null
                && loadedSchema["Features"].HasSlotNames());
            Assert.True(loadedLoader is TextLoader);
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
            var loader = _ml.Data.CreateTextLoader<InputData>(hasHeader: true, dataSample: file);
            var composite = loader.Append(_ml.Transforms.Normalize("Features"));
            var model = composite.Fit(file);

            string modelPath = GetOutputPath(FullTestName + "-model.zip");
            _ml.Model.Save(model, modelPath);

            Load(modelPath, out var loadedWithSchema, out var loadedSchema, out var loadedLoader,
                out var loadedWithLoader, out var loadedLoaderWithTransformer);
            Assert.True(loadedWithSchema is TransformerChain<ITransformer>);
            Assert.True((loadedWithSchema as TransformerChain<ITransformer>).Count() == 1);
            Assert.True(loadedSchema.Count == 2 &&
                loadedSchema.GetColumnOrNull("Label") != null
                && loadedSchema.GetColumnOrNull("Features") != null
                && loadedSchema["Features"].HasSlotNames());
            Assert.True(loadedLoader is CompositeDataLoader<IMultiStreamSource, ITransformer>);
            Assert.True(loadedWithLoader is TransformerChain<ITransformer>);
            Assert.True((loadedWithLoader as TransformerChain<ITransformer>).Count() == 1);
            Assert.True(loadedLoaderWithTransformer is TextLoader);
            var schema = loadedLoaderWithTransformer.GetOutputSchema();
            Assert.True(schema.Count == 2 &&
                schema.GetColumnOrNull("Label") != null
                && schema.GetColumnOrNull("Features") != null
                && schema["Features"].HasSlotNames());
        }

        [Fact]
        public void SaveLoaderAndTransformerAndLoad()
        {
            var file = new MultiFileSource(GetDataPath(TestDatasets.adult.trainFilename));
            var loader = _ml.Data.CreateTextLoader<InputData>(hasHeader: true, dataSample: file);
            var estimator = _ml.Transforms.Normalize("Features");
            var model = estimator.Fit(loader.Load(file));

            string modelPath = GetOutputPath(FullTestName + "-model.zip");
            _ml.Model.Save(loader, model, modelPath);

            Load(modelPath, out var loadedWithSchema, out var loadedSchema, out var loadedLoader,
                out var loadedWithLoader, out var loadedLoaderWithTransformer);
            Assert.True(loadedWithSchema is TransformerChain<ITransformer>);
            Assert.True((loadedWithSchema as TransformerChain<ITransformer>).Count() == 1);
            Assert.True(loadedSchema.Count == 2 &&
                loadedSchema.GetColumnOrNull("Label") != null
                && loadedSchema.GetColumnOrNull("Features") != null
                && loadedSchema["Features"].HasSlotNames());
            Assert.True(loadedLoader is CompositeDataLoader<IMultiStreamSource, ITransformer>);
            Assert.True(loadedWithLoader is TransformerChain<ITransformer>);
            Assert.True((loadedWithLoader as TransformerChain<ITransformer>).Count() == 1);
            Assert.True(loadedLoaderWithTransformer is TextLoader);
            var schema = loadedLoaderWithTransformer.GetOutputSchema();
            Assert.True(schema.Count == 2 &&
                schema.GetColumnOrNull("Label") != null
                && schema.GetColumnOrNull("Features") != null
                && schema["Features"].HasSlotNames());
        }

        [Fact]
        public void SaveTransformerAndSchemaAndLoad()
        {
            var file = new MultiFileSource(GetDataPath(TestDatasets.adult.trainFilename));
            var loader = _ml.Data.CreateTextLoader<InputData>(hasHeader: true, dataSample: file);
            var estimator = _ml.Transforms.Normalize("Features");
            var model = estimator.Fit(loader.Load(file));

            string modelPath = GetOutputPath(FullTestName + "-model.zip");
            _ml.Model.Save(model, loader.GetOutputSchema(), modelPath);

            Load(modelPath, out var loadedWithSchema, out var loadedSchema, out var loadedLoader,
                out var loadedWithLoader, out var loadedLoaderWithTransformer);
            Assert.True(loadedWithSchema is NormalizingTransformer);
            Assert.True(loadedSchema.Count == 2 &&
                loadedSchema.GetColumnOrNull("Label") != null
                && loadedSchema.GetColumnOrNull("Features") != null
                && loadedSchema["Features"].HasSlotNames());
            Assert.Null(loadedLoader);
            Assert.Null(loadedWithLoader);
            Assert.Null(loadedLoaderWithTransformer);
        }

        private void Load(string filename, out ITransformer loadedWithSchema, out DataViewSchema loadedSchema,
            out IDataLoader<IMultiStreamSource> loadedLoader, out ITransformer loadedWithLoader,
            out IDataLoader<IMultiStreamSource> loadedLoaderWithTransformer)
        {
            using (var fs = File.OpenRead(filename))
            {
                try
                {
                    loadedLoader = _ml.Model.Load(fs);
                }
                catch (Exception)
                {
                    loadedLoader = null;
                }
                loadedWithSchema = _ml.Model.Load(fs, out loadedSchema);
                try
                {
                    loadedWithLoader = _ml.Model.LoadWithDataLoader(fs, out loadedLoaderWithTransformer);
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
