﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.SymSgd;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Start with a dataset in a text file. Run text featurization on text values. 
        /// Train a linear model over that. (I am thinking sentiment classification.) 
        /// Out of the result, produce some structure over which you can get predictions programmatically 
        /// (for example, the prediction does not happen over a file as it did during training).
        /// </summary>
        [Fact]
        public void SimpleTrainAndPredict()
        {
            var ml = new MLContext(seed: 1, conc: 1);
            var data = ml.Data.ReadFromTextFile<SentimentData>(GetDataPath(TestDatasets.Sentiment.trainFilename), hasHeader: true);

            // Pipeline.
            var pipeline = ml.Transforms.Text.FeaturizeText("SentimentText", "Features")
                .AppendCacheCheckpoint(ml)
                .Append(ml.BinaryClassification.Trainers.StochasticDualCoordinateAscent(
                    new SdcaBinaryTrainer.Options { NumThreads = 1 }));

            // Train.
            var model = pipeline.Fit(data);

            // Create prediction engine and test predictions.
            var engine = model.CreatePredictionEngine<SentimentData, SentimentPrediction>(ml);

            // Take a couple examples out of the test data and run predictions on top.
            var testData = ml.CreateEnumerable<SentimentData>(
                ml.Data.ReadFromTextFile<SentimentData>(GetDataPath(TestDatasets.Sentiment.testFilename), hasHeader: true), false);
            foreach (var input in testData.Take(5))
            {
                var prediction = engine.Predict(input);
                // Verify that predictions match and scores are separated from zero.
                Assert.Equal(input.Sentiment, prediction.Sentiment);
                Assert.True(input.Sentiment && prediction.Score > 1 || !input.Sentiment && prediction.Score < -1);
            }
        }

        /// <summary>
        /// Start with a dataset in a text file. Run text featurization on text values. 
        /// Train a linear model over that. (I am thinking sentiment classification.) 
        /// Out of the result, produce some structure over which you can get predictions programmatically 
        /// (for example, the prediction does not happen over a file as it did during training).
        /// Uses Symbolic SDCA Trainer.
        /// </summary>
        [Fact]
        public void SimpleTrainAndPredictSymSGD()
        {
            var ml = new MLContext(seed: 1, conc: 1);
            var data = ml.Data.ReadFromTextFile<SentimentData>(GetDataPath(TestDatasets.Sentiment.trainFilename), hasHeader: true);

            // Pipeline.
            var pipeline = ml.Transforms.Text.FeaturizeText("SentimentText", "Features")
                .AppendCacheCheckpoint(ml)
                .Append(ml.BinaryClassification.Trainers.SymbolicStochasticGradientDescent(new SymSgdClassificationTrainer.Options
                {
                    NumberOfThreads = 1
                }));

            // Train.
            var model = pipeline.Fit(data);

            // Create prediction engine and test predictions.
            var engine = model.CreatePredictionEngine<SentimentData, SentimentPrediction>(ml);

            // Take a couple examples out of the test data and run predictions on top.
            var testData = ml.CreateEnumerable<SentimentData>(
                ml.Data.ReadFromTextFile<SentimentData>(GetDataPath(TestDatasets.Sentiment.testFilename), hasHeader: true), false);
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
