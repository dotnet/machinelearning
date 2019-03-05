// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Trainers.FastTree;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        private class InputData
        {
            [LoadColumn(0)]
            public float Label { get; set; }
            [LoadColumn(9, 14)]
            [VectorType(6)]
            public float[] Features { get; set; }
        }

        [Fact]
        public void LoadModelAndExtractPredictor()
        {
            var ml = new MLContext(seed: 1, conc: 1);
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
                ml.Model.Save(model, fs);

            ITransformer loadedModel;
            using (var fs = File.OpenRead(modelPath))
                loadedModel = ml.Model.Load(fs);

            var gam = (((loadedModel as TransformerChain<ITransformer>).LastTransformer
                as BinaryPredictionTransformer<object>).Model
                as CalibratedModelParametersBase<object, ICalibrator>).SubModel
                as BinaryClassificationGamModelParameters;
            Assert.NotNull(gam);
        }

        [Fact]
        public void SaveAndLoadModelWithLoader()
        {
            var ml = new MLContext(seed: 1, conc: 1);
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
            using (var fs = File.OpenRead(modelPath))
            {
                loadedModel = ml.Model.LoadAsCompositeDataLoader(fs);
                loadedModelWithoutLoader = ml.Model.Load(fs);
            }

            // Without deserializing the loader from the model we lose the slot names.
            data = ml.Data.LoadFromEnumerable(new[] { new InputData() });
            data = loadedModelWithoutLoader.Transform(data);
            Assert.Null(data.Schema["Features"].Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.SlotNames));

            data = loadedModel.Load(file);
            Assert.True(data.Schema["Features"].HasSlotNames(data.Schema["Features"].Type.GetValueCount()));
            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            data.Schema["Features"].GetSlotNames(ref slotNames);
            var ageIndex = FindIndex(slotNames.GetValues(), "age");
            var transformer = (loadedModel as CompositeDataLoader<IMultiStreamSource, ITransformer>).Transformer.LastTransformer;
            var gamModel = ((transformer as BinaryPredictionTransformer<object>).Model
                as CalibratedModelParametersBase<object, ICalibrator>).SubModel
                as BinaryClassificationGamModelParameters;
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
