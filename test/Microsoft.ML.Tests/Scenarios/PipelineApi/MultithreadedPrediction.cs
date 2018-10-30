// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;
using System.Collections.Generic;
using System.Threading.Tasks;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.PipelineApi
{
    public partial class PipelineApiScenarioTests
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
        void MultithreadedPrediction()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var testDataPath = GetDataPath(SentimentDataPath);
            var pipeline = new Legacy.LearningPipeline();

            var loader = new TextLoader(dataPath).CreateFrom<SentimentData>();
            loader.Arguments.HasHeader = true;
            pipeline.Add(loader);

            pipeline.Add(MakeSentimentTextTransform());

            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });

            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });
            var model = pipeline.Train<SentimentData, SentimentPrediction>();
            var collection = new List<SentimentData>();
            int numExamples = 100;
            for (int i = 0; i < numExamples; i++)
                collection.Add(new SentimentData() { SentimentText = "Let's predict this one!" });

            Parallel.ForEach(collection, (input) =>
            {
                // We need this lock because model itself is stateful object, and probably not thread safe.
                // See comment on top of test.
                lock (model)
                {
                    var prediction = model.Predict(input);
                }
            });
        }
    }
}
