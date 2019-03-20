// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Trainers;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Multiclass predictions produce single PredictedLabel column and array of scores.
        /// This examples shows how to map score value to original label.
        /// In case if you don't apply KeyToValue estimator on top of predictor label we won't convert
        /// key value to original label value. This example also shows how to convert key value to original label.
        /// </summary>
        [Fact]
        void PredictAndMetadata()
        {
            var dataPath = GetDataPath(TestDatasets.irisData.trainFilename);
            var ml = new MLContext();

            var data = ml.Data.LoadFromTextFile<IrisData>(dataPath, separatorChar: ',');

            var pipeline = ml.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(ml.Transforms.Conversion.MapValueToKey("Label"), TransformerScope.TrainTest)
                .Append(ml.MulticlassClassification.Trainers.SdcaCalibrated(
                    new SdcaCalibratedMulticlassTrainer.Options { MaximumNumberOfIterations = 100, Shuffle = true, NumberOfThreads = 1, }));

            var model = pipeline.Fit(data).GetModelFor(TransformerScope.Scoring);
            var engine = ml.Model.CreatePredictionEngine<IrisDataNoLabel, IrisPredictionNotCasted>(model);

            var testLoader = ml.Data.LoadFromTextFile(dataPath, TestDatasets.irisData.GetLoaderColumns(), separatorChar: ',', hasHeader: true);
            var testData = ml.Data.CreateEnumerable<IrisData>(testLoader, false);
            
            // During prediction we will get Score column with 3 float values.
            // We need to find way to map each score to original label.
            // In order to do what we need to get SlotNames from Score column.
            // Slot names on top of Score column represent original labels for i-th value in Score array.
            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            engine.OutputSchema[nameof(IrisPrediction.Score)].GetSlotNames(ref slotNames);
            // Since we apply MapValueToKey estimator with default parameters, key values
            // depends on order of occurence in data file. Which is "Iris-setosa", "Iris-versicolor", "Iris-virginica"
            // So if we have Score column equal to [0.2, 0.3, 0.5] that's mean what score for
            // Iris-setosa is 0.2
            // Iris-versicolor is 0.3
            // Iris-virginica is 0.5.
            Assert.True(slotNames.GetItemOrDefault(0).ToString() == "Iris-setosa");
            Assert.True(slotNames.GetItemOrDefault(1).ToString() == "Iris-versicolor");
            Assert.True(slotNames.GetItemOrDefault(2).ToString() == "Iris-virginica");

            // Let's look how we can convert key value for PredictedLabel to original labels.
            // We need to read KeyValues for "PredictedLabel" column.
            VBuffer<ReadOnlyMemory<char>> keys = default;
            engine.OutputSchema[nameof(IrisPrediction.PredictedLabel)].GetKeyValues(ref keys);
            foreach (var input in testData.Take(20))
            {
                var prediction = engine.Predict(input);
                // Predicted label is key type which internal representation starts from 1.
                // (0 reserved for NaN value) so in order to cast key to index in key metadata we need to distract 1 from it.
                var deciphieredLabel = keys.GetItemOrDefault((int)prediction.PredictedLabel - 1).ToString();
                Assert.True(deciphieredLabel == input.Label);
            }
        }
    }
}
