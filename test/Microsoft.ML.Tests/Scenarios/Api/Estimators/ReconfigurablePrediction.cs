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
        /// Reconfigurable predictions: The following should be possible: A user trains a binary classifier,
        /// and through the test evaluator gets a PR curve, the based on the PR curve picks a new threshold
        /// and configures the scorer (or more precisely instantiates a new scorer over the same predictor)
        /// with some threshold derived from that.
        /// </summary>
        [Fact]
        public void New_ReconfigurablePrediction()
        {
            var ml = new MLContext(seed: 1, conc: 1);
            var dataReader = ml.Data.TextReader(MakeSentimentTextLoaderArgs());

            var data = dataReader.Read(GetDataPath(TestDatasets.Sentiment.trainFilename));
            var testData = dataReader.Read(GetDataPath(TestDatasets.Sentiment.testFilename));

            // Pipeline.
            var pipeline = ml.Transform.Text.FeaturizeText("SentimentText", "Features")
                .Fit(data);

            var trainer = ml.BinaryClassification.Trainers.StochasticDualCoordinateAscent(advancedSettings: (s) => s.NumThreads = 1);
            var trainData = pipeline.Transform(data);
            var model = trainer.Fit(trainData);

            var scoredTest = model.Transform(pipeline.Transform(testData));
            var metrics = ml.BinaryClassification.Evaluate(scoredTest);

            var newModel = new BinaryPredictionTransformer<IPredictorProducing<float>>(ml, model.Model, trainData.Schema, model.FeatureColumn, threshold: 0.01f, thresholdColumn: DefaultColumnNames.Probability);
            var newScoredTest = newModel.Transform(pipeline.Transform(testData));
            var newMetrics = ml.BinaryClassification.Evaluate(scoredTest);
        }
    }
}
