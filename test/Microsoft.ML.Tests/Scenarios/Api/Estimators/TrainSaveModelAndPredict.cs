// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using System.Linq;
using Microsoft.ML.RunTests;
using Microsoft.ML.Trainers;
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
        public void TrainSaveModelAndPredict()
        {
            var ml = new MLContext(seed: 1);
            var data = ml.Data.LoadFromTextFile<SentimentData>(GetDataPath(TestDatasets.Sentiment.trainFilename), hasHeader: true);

            // Pipeline.
            var pipeline = ml.Transforms.Text.FeaturizeText("Features", "SentimentText")
                .AppendCacheCheckpoint(ml)
                .Append(ml.BinaryClassification.Trainers.SdcaNonCalibrated(
                    new SdcaNonCalibratedBinaryTrainer.Options { NumberOfThreads = 1 }));

            // Train.
            var model = pipeline.Fit(data);

            var modelPath = GetOutputPath("temp.zip");
            // Save model. 
            ml.Model.Save(model, data.Schema, modelPath);

            // Load model.
            ITransformer loadedModel;
            DataViewSchema inputSchema;
            using (var file = File.OpenRead(modelPath))
                loadedModel = ml.Model.Load(file, out inputSchema);

            // Create prediction engine and test predictions.
            var engine = ml.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(loadedModel, inputSchema);

            // Take a couple examples out of the test data and run predictions on top.
            var testData = ml.Data.CreateEnumerable<SentimentData>(
                ml.Data.LoadFromTextFile<SentimentData>(GetDataPath(TestDatasets.Sentiment.testFilename), hasHeader: true), false);
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
