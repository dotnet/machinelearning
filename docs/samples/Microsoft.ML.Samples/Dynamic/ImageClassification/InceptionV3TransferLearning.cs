using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Samples.Dynamic
{
    public static class InceptionV3TransferLearning
    {
        /// <summary>
        /// Example use of Image classification API in a ML.NET pipeline.
        /// </summary>
        public static void Example()
        {
            var mlContext = new MLContext(seed: 1);

            var imagesDataFile = Path.GetDirectoryName(
                Microsoft.ML.SamplesUtils.DatasetUtils.DownloadImages());

            var data = mlContext.Data.LoadFromEnumerable(
                ImageNetData.LoadImagesFromDirectory(imagesDataFile, 4));

            data = mlContext.Data.ShuffleRows(data, 5);
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.LoadImages("ImageObject", null,
                    "ImagePath"))
                .Append(mlContext.Transforms.ResizeImages("Image",
                    inputColumnName: "ImageObject", imageWidth: 299,
                    imageHeight: 299))
                .Append(mlContext.Transforms.ExtractPixels("Image",
                    interleavePixelColors: true))
                .Append(mlContext.Model.ImageClassification("Image",
                    "Label", arch: DnnEstimator.Architecture.InceptionV3, epoch: 4,
                    batchSize: 4));

            var trainedModel = pipeline.Fit(data);
            var predicted = trainedModel.Transform(data);
            var metrics = mlContext.MulticlassClassification.Evaluate(predicted);

            Console.WriteLine($"Micro-accuracy: {metrics.MicroAccuracy}," +
                $"macro-accuracy = {metrics.MacroAccuracy}");

            // Create prediction function and test prediction
            var predictFunction = mlContext.Model
                .CreatePredictionEngine<ImageNetData, ImagePrediction>(trainedModel);

            var prediction = predictFunction
                .Predict(ImageNetData.LoadImagesFromDirectory(imagesDataFile, 4)
                .First());

            Console.WriteLine($"Scores : [{string.Join(",", prediction.Score)}], " +
                $"Predicted Label : {prediction.PredictedLabel}");

        }
    }

    public class ImageNetData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;

        public static IEnumerable<ImageNetData> LoadImagesFromDirectory(
            string folder, int repeat = 1, bool useFolderNameasLabel = false)
        {
            var files = Directory.GetFiles(folder, "*",
                searchOption: SearchOption.AllDirectories);

            foreach (var file in files)
            {
                if (Path.GetExtension(file) != ".jpg")
                    continue;

                var label = Path.GetFileName(file);
                if (useFolderNameasLabel)
                    label = Directory.GetParent(file).Name;
                else
                {
                    for (int index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);
                            break;
                        }
                    }
                }

                for (int index = 0; index < repeat; index++)
                    yield return new ImageNetData() {
                        ImagePath = file,Label = label };
            }
        }
    }

    public class ImagePrediction
    {
        [ColumnName("Score")]
        public float[] Score;

        [ColumnName("PredictedLabel")]
        public Int64 PredictedLabel;
    }
}
