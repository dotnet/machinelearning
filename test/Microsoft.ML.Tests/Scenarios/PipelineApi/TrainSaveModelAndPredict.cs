// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Scenarios;
using Microsoft.ML.Transforms.TensorFlow;
using System.IO;
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
            var pipeline = new Legacy.LearningPipeline();

            pipeline.Add(new TextLoader(dataPath).CreateFrom<SentimentData>());
            pipeline.Add(MakeSentimentTextTransform());
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            var model = pipeline.Train<SentimentData, SentimentPrediction>();
            var modelName = "trainSaveAndPredictdModel.zip";
            DeleteOutputPath(modelName);
            await model.WriteAsync(modelName);
            var loadedModel = await Legacy.PredictionModel.ReadAsync<SentimentData, SentimentPrediction>(modelName);
            var singlePrediction = loadedModel.Predict(new SentimentData() { SentimentText = "Not big fan of this." });
            Assert.True(singlePrediction.Sentiment);

        }

        [Fact]
        public async void TensorFlowTransformTrainSaveModelAndPredict()
        {
            var imageHeight = 32;
            var imageWidth = 32;
            var model_location = "cifar_saved_model";
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);

            var pipeline = new Legacy.LearningPipeline(seed: 1);
            pipeline.Add(new TextLoader(dataFile).CreateFrom<CifarData>(useHeader: false));
            pipeline.Add(new ImageLoader(("ImagePath", "ImageReal"))
            {
                ImageFolder = imageFolder
            });

            pipeline.Add(new ImageResizer(("ImageReal", "ImageCropped"))
            {
                ImageHeight = imageHeight,
                ImageWidth = imageWidth,
                Resizing = ImageResizerTransformResizingKind.IsoCrop
            });

            pipeline.Add(new ImagePixelExtractor(("ImageCropped", "Input"))
            {
                UseAlpha = false,
                InterleaveArgb = true
            });

            pipeline.Add(new TensorFlowScorer()
            {
                Model = model_location,
                InputColumns = new[] { "Input" },
                OutputColumns = new[] { "Output" },
            });

            pipeline.Add(new ColumnConcatenator(outputColumn: "Features", "Output"));
            pipeline.Add(new TextToKeyConverter("Label"));
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

             TensorFlowUtils.Initialize();
            var model = pipeline.Train<CifarData, CifarPrediction>();

            var modelName = "tf_TrainSaveAndPredictdModel.zip";
            DeleteOutputPath(modelName);
            await model.WriteAsync(modelName);

            var loadedModel = await Legacy.PredictionModel.ReadAsync<CifarData, CifarPrediction>(modelName);

            string[] scoreLabels;
            loadedModel.TryGetScoreLabelNames(out scoreLabels);

            Assert.NotNull(scoreLabels);
            Assert.Equal(3, scoreLabels.Length);
            Assert.Equal("banana", scoreLabels[0]);
            Assert.Equal("hotdog", scoreLabels[1]);
            Assert.Equal("tomato", scoreLabels[2]);

            CifarPrediction prediction = loadedModel.Predict(new CifarData()
            {
                ImagePath = GetDataPath("images/banana.jpg")
            });
            Assert.Equal(1, prediction.PredictedLabels[0], 2);
            Assert.Equal(0, prediction.PredictedLabels[1], 2);
            Assert.Equal(0, prediction.PredictedLabels[2], 2);

            prediction = loadedModel.Predict(new CifarData()
            {
                ImagePath = GetDataPath("images/hotdog.jpg")
            });
            Assert.Equal(0, prediction.PredictedLabels[0], 2);
            Assert.Equal(1, prediction.PredictedLabels[1], 2);
            Assert.Equal(0, prediction.PredictedLabels[2], 2);
        }
    }
}
