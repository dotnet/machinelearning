// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Transforms.TensorFlow;
using System.IO;
using Xunit;

namespace Microsoft.ML.Scenarios
{
    public partial class ScenariosTests
    {
        [Fact]
        public void TensorFlowTransformCifarLearningPipelineTest()
        {
            var imageHeight = 32;
            var imageWidth = 32;
            var model_location = "cifar_model/frozen_model.pb";
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);

            var pipeline = new Legacy.LearningPipeline(seed: 1);
            pipeline.Add(new Microsoft.ML.Legacy.Data.TextLoader(dataFile).CreateFrom<CifarData>(useHeader: false));
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
                OutputColumns = new[] { "Output" }
            });

            pipeline.Add(new ColumnConcatenator(outputColumn: "Features", "Output"));
            pipeline.Add(new TextToKeyConverter("Label"));
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            var model = pipeline.Train<CifarData, CifarPrediction>();
            string[] scoreLabels;
            model.TryGetScoreLabelNames(out scoreLabels);

            Assert.NotNull(scoreLabels);
            Assert.Equal(3, scoreLabels.Length);
            Assert.Equal("banana", scoreLabels[0]);
            Assert.Equal("hotdog", scoreLabels[1]);
            Assert.Equal("tomato", scoreLabels[2]);

            CifarPrediction prediction = model.Predict(new CifarData()
            {
                ImagePath = GetDataPath("images/banana.jpg")
            });
            Assert.Equal(1, prediction.PredictedLabels[0], 2);
            Assert.Equal(0, prediction.PredictedLabels[1], 2);
            Assert.Equal(0, prediction.PredictedLabels[2], 2);

            prediction = model.Predict(new CifarData()
            {
                ImagePath = GetDataPath("images/hotdog.jpg")
            });
            Assert.Equal(0, prediction.PredictedLabels[0], 2);
            Assert.Equal(1, prediction.PredictedLabels[1], 2);
            Assert.Equal(0, prediction.PredictedLabels[2], 2);
        }

        [Fact(Skip = "Model files are not available yet")]
        public void TensorFlowTransformInceptionPipelineTest()
        {
            var model_location = @"C:\models\TensorFlow\tensorflow_inception_graph.pb";
            var dataFile = @"C:\Data\tags.tsv";
            var imagesFolder = @"C:\Data\images";

            const int imageHeight = 224;
            const int imageWidth = 224;

            const string inputTensorName = "input";
            const string outputTensorName = "softmax2_pre_activation";

            const float mean = 117;

            var pipeline = new Legacy.LearningPipeline();
            pipeline.Add(new Legacy.Data.TextLoader(dataFile).CreateFrom<ImageNetData>(useHeader: false));
            pipeline.Add(new ImageLoader(("ImagePath", "ImageReal"))
            {
                ImageFolder = imagesFolder
            });

            pipeline.Add(new ImageResizer(("ImageReal", "ImageCropped"))
            {
                ImageHeight = imageHeight,
                ImageWidth = imageWidth,
                Resizing = ImageResizerTransformResizingKind.IsoCrop
            });

            pipeline.Add(new ImagePixelExtractor(("ImageCropped", inputTensorName))
            {
                UseAlpha = false,
                InterleaveArgb = true,
                Convert = true,
                Offset = mean,
                Scale = 1
            });

            pipeline.Add(new TensorFlowScorer()
            {
                Model = model_location,
                InputColumns = new[] { inputTensorName },
                OutputColumns = new[] { outputTensorName }
            });

            pipeline.Add(new ColumnConcatenator(outputColumn: "Features", inputColumns: outputTensorName));
            pipeline.Add(new TextToKeyConverter("Label"));
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            var model = pipeline.Train<ImageNetData, ImageNetPrediction>();
            string[] scoreLabels;
            model.TryGetScoreLabelNames(out scoreLabels);

            //Test Scoring

            ImageNetPrediction prediction = model.Predict(new ImageNetData()
            {
                ImagePath = @"C:\Data\images\violin.jpg"
            });
        }
    }

    public class CifarData
    {
        [Column("0")]
        public string ImagePath;

        [Column("1")]
        public string Label;
    }

    public class CifarPrediction
    {
        [ColumnName("Score")]
        public float[] PredictedLabels;
    }

    public class ImageNetData
    {
        [Column("0")]
        public string ImagePath;

        [Column("1")]
        public string Label;
    }

    public class ImageNetPrediction
    {
        [ColumnName("Score")]
        public float[] PredictedLabels;
    }
}
