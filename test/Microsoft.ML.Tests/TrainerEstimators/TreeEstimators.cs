// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Xunit;
using System.Linq;
using Microsoft.ML.Runtime.FastTree;
using Xunit.Abstractions;
using Microsoft.ML.Runtime.RunTests;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TreeEstimators : TestDataPipeBase
    {

        public TreeEstimators(ITestOutputHelper output) : base(output)
        {
        }


        /// <summary>
        /// FastTreeBinaryClassification TrainerEstimator test 
        /// </summary>
        [Fact]
        public void FastTreeBinaryEstimator()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var testDataPath = GetDataPath(SentimentTestPath);

            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                var reader = new TextLoader(env, MakeSentimentTextLoaderArgs());
                var data = reader.Read(new MultiFileSource(dataPath));

                // Pipeline.
                var pipeline = new TextTransform(env, "SentimentText", "Features")
                  .Append(new FastTreeBinaryClassificationTrainer(env, "Label", "Features", advancedSettings: s => { s.NumTrees = 10 }));

                // Train.
                var model = pipeline.Fit(data);

                // Create prediction engine and test predictions.
                var engine = model.MakePredictionFunction<SentimentData, SentimentPrediction>(env);

                // Take a couple examples out of the test data and run predictions on top.
                var testData = reader.Read(new MultiFileSource(GetDataPath(SentimentTestPath)))
                    .AsEnumerable<SentimentData>(env, false);
                foreach (var input in testData.Take(5))
                {
                    var prediction = engine.Predict(input);
                    // Verify that predictions match and scores are separated from zero.
                    Assert.Equal(input.Sentiment, prediction.Sentiment);
                    Assert.True(input.Sentiment && prediction.Score > 1 || !input.Sentiment && prediction.Score < -1);
                }
            }
        }
        private static TextLoader.Arguments MakeSentimentTextLoaderArgs()
        {
            return new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.BL, 0),
                    new TextLoader.Column("SentimentText", DataKind.Text, 1)
                }
            };
        }
    }
}
