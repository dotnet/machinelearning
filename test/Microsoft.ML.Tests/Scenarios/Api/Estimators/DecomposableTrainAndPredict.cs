// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
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
            var dataPath = GetDataPath(IrisDataPath);
            using (var env = new TlcEnvironment())
            {
                var data = new TextLoader(env, MakeIrisTextLoaderArgs())
                    .Read(new MultiFileSource(dataPath));

                var pipeline = new ConcatEstimator(env, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                    .Append(new TermEstimator(env, "Label"), TransformerScope.TrainTest)
                    .Append(new SdcaMultiClassTrainer(env, new SdcaMultiClassTrainer.Arguments { MaxIterations = 100, Shuffle = true, NumThreads = 1 }, "Features", "Label"))
                    .Append(new KeyToValueEstimator(env, "PredictedLabel"));

                var model = pipeline.Fit(data).GetModelFor(TransformerScope.Scoring);
                var engine = model.MakePredictionFunction<IrisDataNoLabel, IrisPrediction>(env);

                var testLoader = TextLoader.ReadFile(env, MakeIrisTextLoaderArgs(), new MultiFileSource(dataPath));
                var testData = testLoader.AsEnumerable<IrisData>(env, false);
                foreach (var input in testData.Take(20))
                {
                    var prediction = engine.Predict(input);
                    Assert.True(prediction.PredictedLabel == input.Label);
                }
            }
        }
    }
}
