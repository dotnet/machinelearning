// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.RunTests;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Train with initial predictor: Similar to the simple train scenario, but also accept a pre-trained initial model.
        /// The scenario might be one of the online linear learners that can take advantage of this, for example, averaged perceptron.
        /// </summary>
        [Fact]
        public void TrainWithInitialPredictor()
        {

            using (var env = new LocalEnvironment(seed: 1, conc: 1))
            {
                // Pipeline
                var loader = TextLoader.ReadFile(env, MakeSentimentTextLoaderArgs(), new MultiFileSource(GetDataPath(TestDatasets.Sentiment.trainFilename)));

                var trans = TextTransform.Create(env, MakeSentimentTextTransformArgs(), loader);
                var trainData = trans;

                var cachedTrain = new CacheDataView(env, trainData, prefetch: null);
                // Train the first predictor.
                var trainer = new StochasticDualCoordinateAscent(env, new StochasticDualCoordinateAscent.Arguments
                {
                    NumThreads = 1
                });
                var trainRoles = new RoleMappedData(cachedTrain, label: "Label", feature: "Features");
                var predictor = trainer.Train(new Runtime.TrainContext(trainRoles));

                // Train the second predictor on the same data.
                var secondTrainer = new AveragedPerceptronTrainer(env, "Label", "Features");
                var finalPredictor = secondTrainer.Train(new TrainContext(trainRoles, initialPredictor: predictor));
            }
        }
    }
}
