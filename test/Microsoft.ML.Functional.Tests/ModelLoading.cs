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
            var model = pipeline.Fit(data);
            var model1 = pipeline1.Fit(file);

            // Save and reload.
            string modelPath = GetOutputPath(FullTestName + "-model.zip");
            using (var fs = File.Create(modelPath))
                ml.Model.Save(data.Schema, model, fs);
            string modelPath1 = GetOutputPath(FullTestName + "-model1.zip");
            using (var fs = File.Create(modelPath1))
                ml.Model.Save(model1, fs);

            ITransformer loadedModel;
            IDataLoader<IMultiStreamSource> loadedModel1;
            using (var fs = File.OpenRead(modelPath))
                loadedModel = ml.Model.Load(fs, out var loadedSchema);
            using (var fs = File.OpenRead(modelPath1))
                loadedModel1 = ml.Model.Load(fs);

            var gam = ((loadedModel as ISingleFeaturePredictionTransformer<object>).Model
                as CalibratedModelParametersBase).SubModel
                as BinaryClassificationGamModelParameters;
            Assert.NotNull(gam);

            gam = (((loadedModel1 as CompositeDataLoader<IMultiStreamSource, ITransformer>).Transformer.LastTransformer
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
