using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.Transforms;
using Xunit;
using static Microsoft.ML.DataOperationsCatalog;

namespace Microsoft.ML.Dnn.Tests
{
    public class ImageClassification
    {
        [TensorFlowFact]
        public void TensorFlowImageClassification()
        {
            // On Ubuntu the results seem to vary quite a bit but they can probably be 
            // controlled by training more epochs, however that will slow the 
            // build down.
            if (!(RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ||
                (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))))
            {
                return;
            }

            string assetsRelativePath = @"assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);
            string imagesDownloadFolderPath = Path.Combine(assetsPath, "inputs",
                "images");

            //Download the image set and unzip
            string finalImagesFolderName = DownloadImageSet(
                imagesDownloadFolderPath);

            string fullImagesetFolderPath = Path.Combine(
                imagesDownloadFolderPath, finalImagesFolderName);

            MLContext mlContext = new MLContext(seed: 1);

            //Load all the original images info
            IEnumerable<ImageData> images = LoadImagesFromDirectory(
                folder: fullImagesetFolderPath, useFolderNameasLabel: true);

            IDataView shuffledFullImagesDataset = mlContext.Data.ShuffleRows(
                mlContext.Data.LoadFromEnumerable(images), seed: 1);

            shuffledFullImagesDataset = mlContext.Transforms.Conversion
                .MapValueToKey("Label")
                .Fit(shuffledFullImagesDataset)
                .Transform(shuffledFullImagesDataset);

            // Split the data 90:10 into train and test sets, train and evaluate.
            TrainTestData trainTestData = mlContext.Data.TrainTestSplit(
                shuffledFullImagesDataset, testFraction: 0.2, seed: 1);

            IDataView trainDataset = trainTestData.TrainSet;
            IDataView testDataset = trainTestData.TestSet;

            var pipeline = mlContext.Model.ImageClassification(
                "ImagePath", "Label",
                arch: ImageClassificationEstimator.Architecture.ResnetV2101,
                epoch: 5,
                batchSize: 5,
                learningRate: 0.01f,
                testOnTrainSet: false);

            var trainedModel = pipeline.Fit(trainDataset);

            mlContext.Model.Save(trainedModel, shuffledFullImagesDataset.Schema,
                "model.zip");

            ITransformer loadedModel;
            DataViewSchema schema;
            using (var file = File.OpenRead("model.zip"))
                loadedModel = mlContext.Model.Load(file, out schema);

            IDataView predictions = trainedModel.Transform(testDataset);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            Assert.InRange(metrics.MicroAccuracy, 0.75, 1);
            Assert.InRange(metrics.MacroAccuracy, 0.75, 1);
        }

        public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder,
            bool useFolderNameasLabel = true)
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

                yield return new ImageData()
                {
                    ImagePath = file,
                    Label = label
                };

            }
        }

        public static string DownloadImageSet(string imagesDownloadFolder)
        {
            string fileName = "flower_photos_tiny_set_for_unit_tests.zip";
            string url = $"https://mlnetfilestorage.file.core.windows.net/imagesets" +
                $"/flower_images/flower_photos_tiny_set_for_unit_tests.zip?st=2019" +
                $"-08-29T00%3A07%3A21Z&se=2030-08-30T00%3A07%3A00Z&sp=rl&sv=2018" +
                $"-03-28&sr=f&sig=N8HbLziTcT61kstprNLmn%2BDC0JoMrNwo6yRWb3hLLag%3D";

            Download(url, imagesDownloadFolder, fileName);
            UnZip(Path.Combine(imagesDownloadFolder, fileName), imagesDownloadFolder);

            return Path.GetFileNameWithoutExtension(fileName);
        }

        private static bool Download(string url, string destDir, string destFileName)
        {
            if (destFileName == null)
                destFileName = url.Split(Path.DirectorySeparatorChar).Last();

            Directory.CreateDirectory(destDir);

            string relativeFilePath = Path.Combine(destDir, destFileName);

            if (File.Exists(relativeFilePath))
            {
                Console.WriteLine($"{relativeFilePath} already exists.");
                return false;
            }

            var wc = new WebClient();
            Console.WriteLine($"Downloading {relativeFilePath}");
            var download = Task.Run(() => wc.DownloadFile(url, relativeFilePath));
            while (!download.IsCompleted)
            {
                Thread.Sleep(1000);
                Console.Write(".");
            }
            Console.WriteLine("");
            Console.WriteLine($"Downloaded {relativeFilePath}");

            return true;
        }

        private static void UnZip(String gzArchiveName, String destFolder)
        {
            var flag = gzArchiveName.Split(Path.DirectorySeparatorChar)
                .Last()
                .Split('.')
                .First() + ".bin";

            if (File.Exists(Path.Combine(destFolder, flag))) return;

            Console.WriteLine($"Extracting.");
            var task = Task.Run(() =>
            {
                ZipFile.ExtractToDirectory(gzArchiveName, destFolder);
            });

            while (!task.IsCompleted)
            {
                Thread.Sleep(200);
                Console.Write(".");
            }

            File.Create(Path.Combine(destFolder, flag));
            Console.WriteLine("");
            Console.WriteLine("Extracting is completed.");
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(
                ImageClassification).Assembly.Location);

            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        public class ImageData
        {
            [LoadColumn(0)]
            public string ImagePath;

            [LoadColumn(1)]
            public string Label;
        }

        public class ImagePrediction
        {
            [ColumnName("Score")]
            public float[] Score;

            [ColumnName("PredictedLabel")]
            public UInt32 PredictedLabel;
        }
    }
}
