// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
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
        void New_MultithreadedPrediction()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var testDataPath = GetDataPath(SentimentTestPath);

            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                var reader = new TextLoader(env, MakeSentimentTextLoaderArgs());
                var data = reader.Read(new MultiFileSource(dataPath));

                // Pipeline.
                var pipeline = new MyTextTransform(env, MakeSentimentTextTransformArgs())
                    .Append(new MySdca(env, new LinearClassificationTrainer.Arguments { NumThreads = 1 }, "Features", "Label"));

                // Train.
                var model = pipeline.Fit(data);

                // Create prediction engine and test predictions.
                var engine = new MyPredictionEngine<SentimentData, SentimentPrediction>(env, model);

                // Take a couple examples out of the test data and run predictions on top.
                var testData = reader.Read(new MultiFileSource(GetDataPath(SentimentTestPath)))
                    .AsEnumerable<SentimentData>(env, false);

                Parallel.ForEach(testData, (input) =>
                {
                    lock (engine)
                    {
                        var prediction = engine.Predict(input);
                    }
                });
            }
        }
    }
}
