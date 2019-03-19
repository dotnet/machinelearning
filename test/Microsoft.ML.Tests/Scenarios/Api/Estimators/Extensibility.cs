// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Extensibility: We can't possibly write every conceivable transform and should not try.
        /// It should somehow be possible for a user to inject custom code to, say, transform data.
        /// This might have a much steeper learning curve than the other usages (which merely involve
        /// usage of already established components), but should still be possible.
        /// </summary>
        [Fact]
        void Extensibility()
        {
            var dataPath = GetDataPath(TestDatasets.irisData.trainFilename);

            var ml = new MLContext();
            var data = ml.Data.CreateTextLoader(TestDatasets.irisData.GetLoaderColumns(), separatorChar: ',')
                .Load(dataPath);

            Action<IrisData, IrisData> action = (i, j) =>
            {
                j.Label = i.Label;
                j.PetalLength = i.SepalLength > 3 ? i.PetalLength : i.SepalLength;
                j.PetalWidth = i.PetalWidth;
                j.SepalLength = i.SepalLength;
                j.SepalWidth = i.SepalWidth;
            };
            var pipeline = new ColumnConcatenatingEstimator (ml, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(new CustomMappingEstimator<IrisData, IrisData>(ml, action, null), TransformerScope.TrainTest)
                .Append(new ValueToKeyMappingEstimator(ml, "Label"), TransformerScope.TrainTest)
                .Append(ml.MulticlassClassification.Trainers.SdcaCalibrated(
                    new SdcaCalibratedMulticlassTrainer.Options { MaximumNumberOfIterations = 100, Shuffle = true, NumberOfThreads = 1 }))
                .Append(new KeyToValueMappingEstimator(ml, "PredictedLabel"));

            var model = pipeline.Fit(data).GetModelFor(TransformerScope.Scoring);
            var engine = ml.Model.CreatePredictionEngine<IrisDataNoLabel, IrisPrediction>(model);

            var testLoader = ml.Data.LoadFromTextFile(dataPath, TestDatasets.irisData.GetLoaderColumns(), separatorChar: ',');
            var testData = ml.Data.CreateEnumerable<IrisData>(testLoader, false);
            foreach (var input in testData.Take(20))
            {
                var prediction = engine.Predict(input);
                Assert.True(prediction.PredictedLabel == input.Label);
            }
        }
    }
}
