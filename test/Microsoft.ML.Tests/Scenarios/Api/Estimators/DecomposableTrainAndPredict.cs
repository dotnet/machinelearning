// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.RunTests;
using System.Linq;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Decomposable train and predict: Train on Iris multiclass problem, which will require
        /// a transform on labels. Be able to reconstitute the pipeline for a prediction only task,
        /// which will essentially "drop" the transform over labels, while retaining the property
        /// that the predicted label for this has a key-type, the probability outputs for the classes
        /// have the class labels as slot names, etc. This should be do-able without ugly compromises like,
        /// say, injecting a dummy label.
        /// </summary>
        [Fact]
        void New_DecomposableTrainAndPredict()
        {
            var dataPath = GetDataPath(TestDatasets.irisData.trainFilename);
            var ml = new MLContext();

            var data = ml.Data.CreateTextReader(TestDatasets.irisData.GetLoaderColumns(), separatorChar: ',')
                    .Read(dataPath);

            var pipeline = ml.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(ml.Transforms.Conversion.MapValueToKey("Label"), TransformerScope.TrainTest)
                .Append(ml.MulticlassClassification.Trainers.StochasticDualCoordinateAscent("Label", "Features",advancedSettings: s => { s.MaxIterations = 100; s.Shuffle = true; s.NumThreads = 1; }))
                .Append(ml.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(data).GetModelFor(TransformerScope.Scoring);
            var engine = model.CreatePredictionEngine<IrisDataNoLabel, IrisPrediction>(ml);

            var testLoader = ml.Data.ReadFromTextFile(dataPath, TestDatasets.irisData.GetLoaderColumns(), hasHeader: true, separatorChar: ',');
            var testData = testLoader.AsEnumerable<IrisData>(ml, false);
            foreach (var input in testData.Take(20))
            {
                var prediction = engine.Predict(input);
                Assert.True(prediction.PredictedLabel == input.Label);
            }
        }
    }
}
