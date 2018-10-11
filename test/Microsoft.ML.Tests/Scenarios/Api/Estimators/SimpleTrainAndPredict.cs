// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Xunit;
using System.Linq;
using Microsoft.ML.Runtime.RunTests;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Start with a dataset in a text file. Run text featurization on text values. 
        /// Train a linear model over that. (I am thinking sentiment classification.) 
        /// Out of the result, produce some structure over which you can get predictions programmatically 
        /// (e.g., the prediction does not happen over a file as it did during training).
        /// </summary>
        [Fact]
        public void New_SimpleTrainAndPredict()
        {
            using (var env = new LocalEnvironment(seed: 1, conc: 1))
            {
                var reader = new TextLoader(env, MakeSentimentTextLoaderArgs());
                var data = reader.Read(new MultiFileSource(GetDataPath(TestDatasets.Sentiment.trainFilename)));
                // Pipeline.
                var pipeline = new TextTransform(env, "SentimentText", "Features")
                    .Append(new LinearClassificationTrainer(env, "Features", "Label", advancedSettings: (s) => s.NumThreads = 1));

                // Train.
                var model = pipeline.Fit(data);

                // Create prediction engine and test predictions.
                var engine = model.MakePredictionFunction<SentimentData, SentimentPrediction>(env);

                // Take a couple examples out of the test data and run predictions on top.
                var testData = reader.Read(new MultiFileSource(GetDataPath(TestDatasets.Sentiment.testFilename)))
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
    }
}
