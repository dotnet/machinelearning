// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
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
        /// Train, save/load model, predict: 
        /// Serve the scenario where training and prediction happen in different processes (or even different machines). 
        /// The actual test will not run in different processes, but will simulate the idea that the 
        /// "communication pipe" is just a serialized model of some form.
        /// </summary>
        [Fact]
        public void New_TrainSaveModelAndPredict()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var testDataPath = GetDataPath(SentimentTestPath);

            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                // Pipeline.
                var pipeline = new MyTextLoader(env, MakeSentimentTextLoaderArgs())
                    .Append(new MyTextTransform(env, MakeSentimentTextTransformArgs()))
                    .Append(new LinearClassificationTrainer(env, new LinearClassificationTrainer.Arguments { NumThreads = 1 }, "Features", "Label"));

                // Train.
                var model = pipeline.Fit(new MultiFileSource(dataPath));

                ITransformer loadedModel;
                using (var file = env.CreateTempFile())
                {
                    // Save model. 
                    using (var fs = file.CreateWriteStream())
                        model.Transformer.SaveTo(env, fs);

                    // Load model.
                    loadedModel = TransformerChain.LoadFrom(env, file.OpenReadStream());
                }

                // Create prediction engine and test predictions.
                var engine = new MyPredictionEngine<SentimentData, SentimentPrediction>(env, loadedModel);

                // Take a couple examples out of the test data and run predictions on top.
                var testData = model.Reader.Read(new MultiFileSource(GetDataPath(SentimentTestPath)))
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
