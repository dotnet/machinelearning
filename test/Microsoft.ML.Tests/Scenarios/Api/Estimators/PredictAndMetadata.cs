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
            VBuffer<ReadOnlyMemory<char>> keys = default;
            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            engine.OutputSchema[nameof(IrisPrediction.Score)].GetSlotNames(ref slotNames);
            engine.OutputSchema[nameof(IrisPrediction.PredictedLabel)].GetKeyValues(ref keys);

            Assert.True(keys.GetItemOrDefault(0).ToString() == "iris-setosa");
            Assert.True(keys.GetItemOrDefault(1).ToString() == "iris-versicolor");
            Assert.True(keys.GetItemOrDefault(2).ToString() == "iris-virginica");

            foreach (var input in testData.Take(20))
            {
                var prediction = engine.Predict(input);
                var deciphieredLabel = keys.GetItemOrDefault((int)prediction.PredictedLabel).ToString();
                Assert.True(deciphieredLabel == input.Label);
            }
        }
    }
}
