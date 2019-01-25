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
        /// </summary>
        [Fact]
        void PredictAndMetadata()
        {
            var dataPath = GetDataPath(TestDatasets.irisData.trainFilename);
            var ml = new MLContext();

            var data = ml.Data.ReadFromTextFile<IrisData>(dataPath, separatorChar: ',');

            var pipeline = ml.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(ml.Transforms.Conversion.MapValueToKey("Label"), TransformerScope.TrainTest)
                .Append(ml.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(
                    new SdcaMultiClassTrainer.Options { MaxIterations = 100, Shuffle = true, NumThreads = 1, }));

            var model = pipeline.Fit(data).GetModelFor(TransformerScope.Scoring);
            var engine = model.CreatePredictionEngine<IrisDataNoLabel, IrisPredictionNotCasted>(ml);

            var testLoader = ml.Data.ReadFromTextFile(dataPath, TestDatasets.irisData.GetLoaderColumns(), hasHeader: true, separatorChar: ',');
            var testData = ml.CreateEnumerable<IrisData>(testLoader, false);
            // Slot names on top of Score column represent original labels for i-th value in Score array.
            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            engine.OutputSchema[nameof(IrisPrediction.Score)].GetSlotNames(ref slotNames);
            // Key names represent original values for PredictedLabel column.
            VBuffer<ReadOnlyMemory<char>> keys = default;
            engine.OutputSchema[nameof(IrisPrediction.PredictedLabel)].GetKeyValues(ref keys);

            Assert.True(slotNames.GetItemOrDefault(0).ToString() == "Iris-setosa");
            Assert.True(slotNames.GetItemOrDefault(1).ToString() == "Iris-versicolor");
            Assert.True(slotNames.GetItemOrDefault(2).ToString() == "Iris-virginica");

            foreach (var input in testData.Take(120))
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
