// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
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
        /// to enable multi-threaded scenarios (for example, a web server servicing requests) should be possible
        /// and performant in the new API.
        /// </summary>
        [Fact]
        void New_MultithreadedPrediction()
        {
            var ml = new MLContext(seed: 1, conc: 1);
            var data = ml.Data.ReadFromTextFile<SentimentData>(GetDataPath(TestDatasets.Sentiment.trainFilename), hasHeader: true);

            // Pipeline.
            var pipeline = ml.Transforms.Text.FeaturizeText("SentimentText", "Features")
                .AppendCacheCheckpoint(ml)
                .Append(ml.BinaryClassification.Trainers.StochasticDualCoordinateAscent("Label", "Features", advancedSettings: s => s.NumThreads = 1));

            // Train.
            var model = pipeline.Fit(data);

            // Create prediction engine and test predictions.
            var engine = model.CreatePredictionEngine<SentimentData, SentimentPrediction>(ml);

            // Take a couple examples out of the test data and run predictions on top.
            var testData = ml.Data.ReadFromTextFile<SentimentData>(GetDataPath(TestDatasets.Sentiment.testFilename), hasHeader: true)
                .AsEnumerable<SentimentData>(ml, false);

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
