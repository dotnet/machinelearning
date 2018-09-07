// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.PipelineApi
{
    public partial class PipelineApiScenarioTests
    {
        /// <summary>
        /// Train, save/load model, predict: 
        /// Serve the scenario where training and prediction happen in different processes (or even different machines). 
        /// The actual test will not run in different processes, but will simulate the idea that the 
        /// "communication pipe" is just a serialized model of some form.
        /// </summary>
        [Fact]
        public async void TrainSaveModelAndPredict()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var testDataPath = GetDataPath(SentimentDataPath);
            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader(dataPath).CreateFrom<SentimentData>());
            pipeline.Add(MakeSentimentTextTransform());
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            var model = pipeline.Train<SentimentData, SentimentPrediction>();
            var modelName = "trainSaveAndPredictdModel.zip";
            DeleteOutputPath(modelName);
            await model.WriteAsync(modelName);
            var loadedModel = await PredictionModel.ReadAsync<SentimentData, SentimentPrediction>(modelName);
            var singlePrediction = loadedModel.Predict(new SentimentData() { SentimentText = "Not big fan of this." });
            Assert.True(singlePrediction.Sentiment);

        }
    }
}
