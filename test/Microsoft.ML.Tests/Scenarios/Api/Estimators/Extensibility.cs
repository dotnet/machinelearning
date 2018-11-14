﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Transforms.Conversions;
using System;
using System.Linq;
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
        void New_Extensibility()
        {
            var dataPath = GetDataPath(TestDatasets.irisData.trainFilename);

            var ml = new MLContext();
            var data = ml.Data.TextReader(MakeIrisTextLoaderArgs())
                .Read(dataPath);

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
                .Append(ml.MulticlassClassification.Trainers.StochasticDualCoordinateAscent("Label", "Features", advancedSettings: (s) => { s.MaxIterations = 100; s.Shuffle = true; s.NumThreads = 1; }))
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
