// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Conversions;
using System;
using System.IO;
using Xunit;

namespace Microsoft.ML.Scenarios
{
    public partial class ScenariosTests
    {
        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // TensorFlow is 64-bit only
        public void TensorFlowTransforCifarEndToEndTest()
        {
            var imageHeight = 32;
            var imageWidth = 32;
            var model_location = "cifar_model/frozen_model.pb";
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);

            var mlContext = new MLContext(seed: 1, conc: 1);
            var data = TextLoader.Create(mlContext, new TextLoader.Arguments()
            {
                Column = new[]
                    {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Label", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));

            var pipeEstimator = new ImageLoadingEstimator(mlContext, imageFolder, ("ImagePath", "ImageReal"))
                    .Append(new ImageResizingEstimator(mlContext, "ImageReal", "ImageCropped", imageHeight, imageWidth))
                    .Append(new ImagePixelExtractingEstimator(mlContext, "ImageCropped", "Input", interleave: true))
                    .Append(new TensorFlowEstimator(mlContext, model_location, new[] { "Input" }, new[] { "Output" }))
                    .Append(new ColumnConcatenatingEstimator(mlContext, "Features", "Output"))
                    .Append(new ValueToKeyMappingEstimator(mlContext, "Label"))
                    .Append(new SdcaMultiClassTrainer(mlContext));


            var transformer = pipeEstimator.Fit(data);
            var predictions = transformer.Transform(data);

            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);
            Assert.Equal(1, metrics.AccuracyMicro, 2);

            var predictFunction = transformer.MakePredictionFunction<CifarData, CifarPrediction>(mlContext);
            var prediction = predictFunction.Predict(new CifarData()
            {
                ImagePath = GetDataPath("images/banana.jpg")
            });
            Assert.Equal(1, prediction.PredictedScores[0], 2);
            Assert.Equal(0, prediction.PredictedScores[1], 2);
            Assert.Equal(0, prediction.PredictedScores[2], 2);

            prediction = predictFunction.Predict(new CifarData()
            {
                ImagePath = GetDataPath("images/hotdog.jpg")
            });
            Assert.Equal(0, prediction.PredictedScores[0], 2);
            Assert.Equal(1, prediction.PredictedScores[1], 2);
            Assert.Equal(0, prediction.PredictedScores[2], 2);
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
        public float[] PredictedScores;
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
