// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.RunTests;
using System.Linq;
using System.Threading.Tasks;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Multi-threaded prediction. A twist on "Simple train and predict", where we account that
        /// multiple threads may want predictions at the same time. Because we deliberately do not
        /// reallocate internal memory buffers on every single prediction, the PredictionEngine
        /// (or its estimator/transformer based successor) is, like most stateful .NET objects,
        /// fundamentally not thread safe. This is deliberate and as designed. However, some mechanism
        /// to enable multi-threaded scenarios (e.g., a web server servicing requests) should be possible
        /// and performant in the new API.
        /// </summary>
        [Fact]
        void MultithreadedPrediction()
        {
            using (var env = new LocalEnvironment(seed: 1, conc: 1))
            {
                // Pipeline
                var loader = TextLoader.ReadFile(env, MakeSentimentTextLoaderArgs(), new MultiFileSource(GetDataPath(TestDatasets.Sentiment.trainFilename)));

                var trans = TextTransform.Create(env, MakeSentimentTextTransformArgs(), loader);

                // Train
                var trainer = new StochasticDualCoordinateAscent(env, new StochasticDualCoordinateAscent.Arguments
                {
                    NumThreads = 1
                });

                var cached = new CacheDataView(env, trans, prefetch: null);
                var trainRoles = new RoleMappedData(cached, label: "Label", feature: "Features");
                var predictor = trainer.Train(new Runtime.TrainContext(trainRoles));

                var scoreRoles = new RoleMappedData(trans, label: "Label", feature: "Features");
                IDataScorerTransform scorer = ScoreUtils.GetScorer(predictor, scoreRoles, env, trainRoles.Schema);

                // Create prediction engine and test predictions.
                var model = env.CreatePredictionEngine<SentimentData, SentimentPrediction>(scorer);

                // Take a couple examples out of the test data and run predictions on top.
                var testLoader = TextLoader.ReadFile(env, MakeSentimentTextLoaderArgs(), new MultiFileSource(GetDataPath(TestDatasets.Sentiment.testFilename)));
                var testData = testLoader.AsEnumerable<SentimentData>(env, false);

                Parallel.ForEach(testData, (input) =>
                {
                    lock (model)
                    {
                        var prediction = model.Predict(input);
                    }
                });
            }
        }
    }
}
