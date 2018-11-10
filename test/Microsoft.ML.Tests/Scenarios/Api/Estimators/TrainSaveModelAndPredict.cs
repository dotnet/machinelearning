// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.RunTests;
using System.IO;
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
            var ml = new MLContext(seed: 1, conc: 1);
            var reader = ml.Data.TextReader(MakeSentimentTextLoaderArgs());
            var data = reader.Read(GetDataPath(TestDatasets.Sentiment.trainFilename));

            // Pipeline.
            var pipeline = ml.Transforms.Text.FeaturizeText("SentimentText", "Features")
                .Append(ml.BinaryClassification.Trainers.StochasticDualCoordinateAscent("Label", "Features", advancedSettings: s => s.NumThreads = 1));

            // Train.
            var model = pipeline.Fit(data);

            var modelPath = GetOutputPath("temp.zip");
            // Save model. 
            using (var file = File.Create(modelPath))
                model.SaveTo(ml, file);

            // Load model.
            ITransformer loadedModel;
            using (var file = File.OpenRead(modelPath))
                loadedModel = TransformerChain.LoadFrom(ml, file);

            // Create prediction engine and test predictions.
            var engine = loadedModel.MakePredictionFunction<SentimentData, SentimentPrediction>(ml);

            // Take a couple examples out of the test data and run predictions on top.
            var testData = reader.Read(GetDataPath(TestDatasets.Sentiment.testFilename))
                .AsEnumerable<SentimentData>(ml, false);
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
