// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
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
            var dataTrain = GetDataPath("images/images.tsv");
            var dataTest = GetDataPath("images/images_test.tsv");

            var pipeline = new LearningPipeline(seed: 1);

            pipeline.Add(new TextLoader(dataTrain).CreateFrom<CifarData>(useHeader: false));
            pipeline.Add(new ImageLoader(("ImagePath", "ImageReal"))
            {
                ImageFolder = Path.GetDirectoryName(dataTrain)
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
                ModelFile = model_location,
                InputColumns = new[] { "Input" },
                OutputColumn = "Output"
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
}
