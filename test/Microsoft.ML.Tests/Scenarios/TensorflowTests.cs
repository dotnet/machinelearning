// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.Transforms;
using Xunit;

namespace Microsoft.ML.Scenarios
{
    public partial class ScenariosTests
    {
        [TensorFlowFact]
        public void TensorFlowTransforCifarEndToEndTest()
        {
            var imageHeight = 32;
            var imageWidth = 32;
            var model_location = "cifar_model/frozen_model.pb";
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);

            var mlContext = new MLContext(seed: 1);
            var data = TextLoader.Create(mlContext, new TextLoader.Options()
            {
                Columns = new[]
                    {
                        new TextLoader.Column("ImagePath", DataKind.String, 0),
                        new TextLoader.Column("Label", DataKind.String, 1),
                    }
            }, new MultiFileSource(dataFile));

            var pipeEstimator = new ImageLoadingEstimator(mlContext, imageFolder, ("ImageReal", "ImagePath"))
                    .Append(new ImageResizingEstimator(mlContext, "ImageCropped", imageHeight, imageWidth, "ImageReal"))
                    .Append(new ImagePixelExtractingEstimator(mlContext, "Input", "ImageCropped", interleavePixelColors: true))
                    .Append(mlContext.Model.LoadTensorFlowModel(model_location).ScoreTensorFlowModel("Output", "Input"))
                    .Append(new ColumnConcatenatingEstimator(mlContext, "Features", "Output"))
                    .Append(new ValueToKeyMappingEstimator(mlContext, "Label"))
                    .AppendCacheCheckpoint(mlContext)
                    .Append(mlContext.MulticlassClassification.Trainers.SdcaCalibrated());


            var transformer = pipeEstimator.Fit(data);
            var predictions = transformer.Transform(data);

            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);
            Assert.Equal(1, metrics.MicroAccuracy, 2);

            var predictFunction = mlContext.Model.CreatePredictionEngine<CifarData, CifarPrediction>(transformer);
            var prediction = predictFunction.Predict(new CifarData()
            {
                ImagePath = GetDataPath("images/banana.jpg")
            });
            Assert.Equal(0, prediction.PredictedScores[0], 2);
            Assert.Equal(1, prediction.PredictedScores[1], 2);
            Assert.Equal(0, prediction.PredictedScores[2], 2);

            prediction = predictFunction.Predict(new CifarData()
            {
                ImagePath = GetDataPath("images/hotdog.jpg")
            });
            Assert.Equal(0, prediction.PredictedScores[0], 2);
            Assert.Equal(0, prediction.PredictedScores[1], 2);
            Assert.Equal(1, prediction.PredictedScores[2], 2);
        }
    }

    public class CifarData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;
    }

    public class CifarPrediction
    {
        [ColumnName("Score")]
        public float[] PredictedScores;
    }

    public class ImageNetData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;
    }

    public class ImageNetPrediction
    {
        [ColumnName("Score")]
        public float[] PredictedLabels;
    }
}
