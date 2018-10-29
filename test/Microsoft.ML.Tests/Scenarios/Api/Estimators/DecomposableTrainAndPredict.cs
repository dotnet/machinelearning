// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Categorical;
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

            var data = ml.Data.TextReader(MakeIrisTextLoaderArgs())
                    .Read(dataPath);

            var pipeline = new ColumnConcatenatingEstimator (ml, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(new ValueToKeyMappingEstimator(ml, "Label"), TransformerScope.TrainTest)
                .Append(ml.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(advancedSettings: s => { s.MaxIterations = 100; s.Shuffle = true; s.NumThreads = 1; }))
                .Append(new KeyToValueEstimator(ml, "PredictedLabel"));

            var model = pipeline.Fit(data).GetModelFor(TransformerScope.Scoring);
            var engine = model.MakePredictionFunction<IrisDataNoLabel, IrisPrediction>(ml);

            var testLoader = TextLoader.ReadFile(ml, MakeIrisTextLoaderArgs(), new MultiFileSource(dataPath));
            var testData = testLoader.AsEnumerable<IrisData>(ml, false);
            foreach (var input in testData.Take(20))
            {
                var prediction = engine.Predict(input);
                Assert.True(prediction.PredictedLabel == input.Label);
            }
        }
    }
}
