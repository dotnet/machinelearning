﻿
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using static Microsoft.ML.DataOperationsCatalog;

namespace Samples.Dynamic
{
    public class ResnetV2101TransferLearningTrainTestSplit
    {
        public static void Example()
        {
            string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);

            var outputMlNetModelFilePath = Path.Combine(assetsPath, "outputs",
                "imageClassifier.zip");

            string imagesDownloadFolderPath = Path.Combine(assetsPath, "inputs",
                "images");

            //Download the image set and unzip
            string finalImagesFolderName = DownloadImageSet(
                imagesDownloadFolderPath);
            string fullImagesetFolderPath = Path.Combine(
                imagesDownloadFolderPath, finalImagesFolderName);

            try
            {

                MLContext mlContext = new MLContext(seed: 1);

                //Load all the original images info
                IEnumerable<ImageData> images = LoadImagesFromDirectory(
                    folder: fullImagesetFolderPath, useFolderNameAsLabel: true);

                IDataView shuffledFullImagesDataset = mlContext.Data.ShuffleRows(
                    mlContext.Data.LoadFromEnumerable(images));

                shuffledFullImagesDataset = mlContext.Transforms.Conversion
                        .MapValueToKey("Label")
                    .Append(mlContext.Transforms.LoadImages("Image",
                                fullImagesetFolderPath, false, "ImagePath"))
                    .Fit(shuffledFullImagesDataset)
                    .Transform(shuffledFullImagesDataset);

                // Split the data 90:10 into train and test sets, train and
                // evaluate.
                TrainTestData trainTestData = mlContext.Data.TrainTestSplit(
                    shuffledFullImagesDataset, testFraction: 0.1, seed: 1);

                IDataView trainDataset = trainTestData.TrainSet;
                IDataView testDataset = trainTestData.TestSet;

                var options = new ImageClassificationEstimator.Options()
                { 
                    FeaturesColumnName = "Image",
                    LabelColumnName = "Label",
                    // Just by changing/selecting InceptionV3/MobilenetV2 here instead of 
                    // ResnetV2101 you can try a different architecture/
                    // pre-trained model. 
                    Arch = ImageClassificationEstimator.Architecture.ResnetV2101,
                    Epoch = 50,
                    BatchSize = 10,
                    LearningRate = 0.01f,
                    MetricsCallback = (metrics) => Console.WriteLine(metrics),
                    ValidationSet = testDataset,
                    DisableEarlyStopping = true
                };

                var pipeline = mlContext.Model.ImageClassification(options)
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                        outputColumnName: "PredictedLabel", 
                        inputColumnName: "PredictedLabel"));


                Console.WriteLine("*** Training the image classification model " +
                    "with DNN Transfer Learning on top of the selected " +
                    "pre-trained model/architecture ***");

                // Measuring training time
                var watch = System.Diagnostics.Stopwatch.StartNew();

                var trainedModel = pipeline.Fit(trainDataset);

                watch.Stop();
                long elapsedMs = watch.ElapsedMilliseconds;

                Console.WriteLine("Training with transfer learning took: " +
                    (elapsedMs / 1000).ToString() + " seconds");

                mlContext.Model.Save(trainedModel, shuffledFullImagesDataset.Schema,
                    "model.zip");

                ITransformer loadedModel;
                DataViewSchema schema;
                using (var file = File.OpenRead("model.zip"))
                    loadedModel = mlContext.Model.Load(file, out schema);

                EvaluateModel(mlContext, testDataset, loadedModel);

                watch = System.Diagnostics.Stopwatch.StartNew();

                // Predict image class using an in-memory image.
                TrySinglePrediction(fullImagesetFolderPath, mlContext, loadedModel);

                watch.Stop();
                elapsedMs = watch.ElapsedMilliseconds;

                Console.WriteLine("Prediction engine took: " +
                    (elapsedMs / 1000).ToString() + " seconds");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            Console.WriteLine("Press any key to finish");
            Console.ReadKey();
        }

        private static void TrySinglePrediction(string imagesForPredictions,
            MLContext mlContext, ITransformer trainedModel)
        {
            // Create prediction function to try one prediction
            var predictionEngine = mlContext.Model
                .CreatePredictionEngine<InMemoryImageData, ImagePrediction>(trainedModel);

            IEnumerable<InMemoryImageData> testImages = LoadInMemoryImagesFromDirectory(
                imagesForPredictions, false);

            InMemoryImageData imageToPredict = new InMemoryImageData
            {
                Image = testImages.First().Image
            };

            var prediction = predictionEngine.Predict(imageToPredict);

            Console.WriteLine($"Scores : [{string.Join(",", prediction.Score)}], " +
                $"Predicted Label : {prediction.PredictedLabel}");
        }


        private static void EvaluateModel(MLContext mlContext,
            IDataView testDataset, ITransformer trainedModel)
        {
            Console.WriteLine("Making bulk predictions and evaluating model's " +
                "quality...");

            // Measuring time
            var watch2 = System.Diagnostics.Stopwatch.StartNew();

            IDataView predictions = trainedModel.Transform(testDataset);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine($"Micro-accuracy: {metrics.MicroAccuracy}," +
                              $"macro-accuracy = {metrics.MacroAccuracy}");

            watch2.Stop();
            long elapsed2Ms = watch2.ElapsedMilliseconds;

            Console.WriteLine("Predicting and Evaluation took: " +
                (elapsed2Ms / 1000).ToString() + " seconds");
        }

        public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder,
            bool useFolderNameAsLabel = true)
        {
            var files = Directory.GetFiles(folder, "*",
                searchOption: SearchOption.AllDirectories);
            foreach (var file in files)
            {
                if (Path.GetExtension(file) != ".jpg")
                    continue;

                var label = Path.GetFileName(file);
                if (useFolderNameAsLabel)
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

        public static IEnumerable<InMemoryImageData> 
            LoadInMemoryImagesFromDirectory(string folder, 
                bool useFolderNameAsLabel = true)
        {
            var files = Directory.GetFiles(folder, "*",
                searchOption: SearchOption.AllDirectories);
            foreach (var file in files)
            {
                if (Path.GetExtension(file) != ".jpg")
                    continue;

                var label = Path.GetFileName(file);
                if (useFolderNameAsLabel)
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

                yield return new InMemoryImageData()
                {
                    Image = File.ReadAllBytes(file),
                    Label = label
                };

            }
        }

        public static string DownloadImageSet(string imagesDownloadFolder)
        {
            // get a set of images to teach the network about the new classes

            //SINGLE SMALL FLOWERS IMAGESET (200 files)
            string fileName = "flower_photos_small_set.zip";
            string url = $"https://mlnetfilestorage.file.core.windows.net/" +
                $"imagesets/flower_images/flower_photos_small_set.zip?st=2019-08-" +
                $"07T21%3A27%3A44Z&se=2030-08-08T21%3A27%3A00Z&sp=rl&sv=2018-03-" +
                $"28&sr=f&sig=SZ0UBX47pXD0F1rmrOM%2BfcwbPVob8hlgFtIlN89micM%3D";

            Download(url, imagesDownloadFolder, fileName);
            UnZip(Path.Combine(imagesDownloadFolder, fileName), imagesDownloadFolder);

            return Path.GetFileNameWithoutExtension(fileName);
        }

        public static bool Download(string url, string destDir, string destFileName)
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

        public static void UnZip(String gzArchiveName, String destFolder)
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
                ResnetV2101TransferLearningTrainTestSplit).Assembly.Location);

            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        public class InMemoryImageData
        {
            [LoadColumn(0)]
            public byte[] Image;

            [LoadColumn(1)]
            public string Label;
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
            public string PredictedLabel;
        }
    }
}
