// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Trainers.FastTree;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Functional.Tests
{
    public partial class ModelLoadingTests : BaseTestClass
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
            var ml = new MLContext(seed: 1);
            var file = new MultiFileSource(GetDataPath(TestDatasets.adult.trainFilename));
            var loader = ml.Data.CreateTextLoader<InputData>(hasHeader: true, dataSample: file);
            var data = loader.Load(file);

            // Pipeline.
            var pipeline = ml.BinaryClassification.Trainers.GeneralizedAdditiveModels();
            // Define the same pipeline starting with the loader.
            var pipeline1 = loader.Append(ml.BinaryClassification.Trainers.GeneralizedAdditiveModels());
            
            // Train.
            var transformerModel = pipeline.Fit(data);
            var compositeLoaderModel = pipeline1.Fit(file);

            // Save and reload.
            string modelAndSchemaPath = GetOutputPath(FullTestName + "-model-schema.zip");
            using (var fs = File.Create(modelAndSchemaPath))
                ml.Model.Save(data.Schema, transformerModel, fs);
            string compositeLoaderModelPath = GetOutputPath(FullTestName + "-composite-model.zip");
            using (var fs = File.Create(compositeLoaderModelPath))
                ml.Model.Save(compositeLoaderModel, fs);
            string loaderAndTransformerModelPath = GetOutputPath(FullTestName + "-loader-transformer.zip");
            using (var fs = File.Create(loaderAndTransformerModelPath))
                ml.Model.Save(loader, transformerModel, fs);

            ITransformer loadedTransformerModel;
            IDataLoader<IMultiStreamSource> loadedCompositeLoader;
            ITransformer loadedTransformerModel1;
            using (var fs = File.OpenRead(modelAndSchemaPath))
                loadedTransformerModel = ml.Model.Load(fs, out DataViewSchema loadedSchema);
            using (var fs = File.OpenRead(compositeLoaderModelPath))
            {
                // This model can be loaded either as a composite data loader,
                // a transformer model + an input schema, or a transformer model + a data loader.
                var t = ml.Model.Load(fs, out IDataLoader<IMultiStreamSource> l);
                var t1 = ml.Model.Load(fs, out DataViewSchema s);
                loadedCompositeLoader = ml.Model.Load(fs);
            }
            using (var fs = File.OpenRead(loaderAndTransformerModelPath))
            {
                // This model can be loaded either as a composite data loader,
                // a transformer model + an input schema, or a transformer model + a data loader.
                var t = ml.Model.Load(fs, out DataViewSchema s);
                var c = ml.Model.Load(fs);
                loadedTransformerModel1 = ml.Model.Load(fs, out IDataLoader<IMultiStreamSource> l);
            }

            var gam = ((loadedTransformerModel as ISingleFeaturePredictionTransformer<object>).Model
                as CalibratedModelParametersBase).SubModel
                as BinaryClassificationGamModelParameters;
            Assert.NotNull(gam);

            gam = (((loadedCompositeLoader as CompositeDataLoader<IMultiStreamSource, ITransformer>).Transformer.LastTransformer
                as ISingleFeaturePredictionTransformer<object>).Model
                as CalibratedModelParametersBase).SubModel
                as BinaryClassificationGamModelParameters;
            Assert.NotNull(gam);

            gam = (((loadedTransformerModel1 as TransformerChain<ITransformer>).LastTransformer
                as ISingleFeaturePredictionTransformer<object>).Model
                as CalibratedModelParametersBase).SubModel
                as BinaryClassificationGamModelParameters;
            Assert.NotNull(gam);
        }

        [Fact]
        public void SaveAndLoadModelWithLoader()
        {
            var ml = new MLContext(seed: 1);
            var file = new MultiFileSource(GetDataPath(TestDatasets.adult.trainFilename));
            var loader = ml.Data.CreateTextLoader<InputData>(hasHeader: true, dataSample: file);
            var data = loader.Load(file);

            // Pipeline.
            var pipeline = ml.BinaryClassification.Trainers.GeneralizedAdditiveModels();

            // Train.
            var model = pipeline.Fit(data);

            // Save and reload.
            string modelPath = GetOutputPath(FullTestName + "-model.zip");
            using (var fs = File.Create(modelPath))
                ml.Model.Save(loader, model, fs);

            IDataLoader<IMultiStreamSource> loadedModel;
            ITransformer loadedModelWithoutLoader;
            DataViewSchema loadedSchema;
            using (var fs = File.OpenRead(modelPath))
            {
                loadedModel = ml.Model.Load(fs);
                loadedModelWithoutLoader = ml.Model.Load(fs, out loadedSchema);
            }

            // Without deserializing the loader from the model we lose the slot names.
            data = ml.Data.LoadFromEnumerable(new[] { new InputData() });
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
            var gamModel = calibratedModelParameters.SubModel as BinaryClassificationGamModelParameters;
            Assert.NotNull(gamModel);
            var ageBinUpperBounds = gamModel.GetBinUpperBounds(ageIndex);
            var ageBinEffects = gamModel.GetBinEffects(ageIndex);
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
