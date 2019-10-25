
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
using Microsoft.ML.Dnn;
using Microsoft.ML.Transforms;

namespace Samples.Dynamic
{
    public class LearningRateSchedulingCifarResnetTransferLearning
    {
        public static void Example()
        {
            string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);

            var outputMlNetModelFilePath = Path.Combine(assetsPath, "outputs",
                "imageClassifier.zip");

            string imagesDownloadFolderPath = Path.Combine(assetsPath, "inputs",
                "images");

            // Download Cifar Dataset. 
            string finalImagesFolderName = DownloadImageSet(
                   imagesDownloadFolderPath);
            string finalImagesFolderNameTrain = "cifar\\train";
            string fullImagesetFolderPathTrain = Path.Combine(
                imagesDownloadFolderPath, finalImagesFolderNameTrain);

            string finalImagesFolderNameTest = "cifar\\test";
            string fullImagesetFolderPathTest = Path.Combine(
                imagesDownloadFolderPath, finalImagesFolderNameTest);

            try
            {

                MLContext mlContext = new MLContext(seed: 1);

                //Load all the original train images info
                IEnumerable<ImageData> train_images = LoadImagesFromDirectory(
                    folder: fullImagesetFolderPathTrain, useFolderNameAsLabel: true);
                IDataView trainDataset = mlContext.Data.LoadFromEnumerable(train_images);
                trainDataset = mlContext.Transforms.Conversion
                        .MapValueToKey("Label")
                    .Append(mlContext.Transforms.LoadRawImageBytes("Image",
                                fullImagesetFolderPathTrain, "ImagePath"))
                    .Fit(trainDataset)
                    .Transform(trainDataset);

                //Load all the original test images info
                IEnumerable<ImageData> test_images = LoadImagesFromDirectory(
                    folder: fullImagesetFolderPathTest, useFolderNameAsLabel: true);
                IDataView testDataset = mlContext.Data.LoadFromEnumerable(test_images);
                testDataset = mlContext.Transforms.Conversion
                        .MapValueToKey("Label")
                    .Append(mlContext.Transforms.LoadRawImageBytes("Image",
                                fullImagesetFolderPathTest, "ImagePath"))
                    .Fit(testDataset)
                    .Transform(testDataset);

                var options = new ImageClassificationTrainer.Options()
                {
                    FeatureColumnName = "Image",
                    LabelColumnName = "Label",
                    // Just by changing/selecting InceptionV3/MobilenetV2 here instead of 
                    // ResnetV2101 you can try a different architecture/
                    // pre-trained model. 
                    Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                    Epoch = 182,
                    BatchSize = 128,
                    LearningRate = 0.01f,
                    MetricsCallback = (metrics) => Console.WriteLine(metrics),
                    ValidationSet = testDataset,
                    ReuseValidationSetBottleneckCachedValues = false,
                    ReuseTrainSetBottleneckCachedValues = false,
                    // Use linear scaling rule and Learning rate decay as an option
                    // This is known to do well for Cifar dataset and Resnet models
                    // You can also try other types of Learning rate scheduling methods
                    // available in LearningRateScheduler.cs  
                    LearningRateScheduler = new LsrDecay()
                };

                var pipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(options)
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

                mlContext.Model.Save(trainedModel, testDataset.Schema,
                    "model.zip");

                ITransformer loadedModel;
                DataViewSchema schema;
                using (var file = File.OpenRead("model.zip"))
                    loadedModel = mlContext.Model.Load(file, out schema);

                EvaluateModel(mlContext, testDataset, loadedModel);

                watch = System.Diagnostics.Stopwatch.StartNew();

                // Predict image class using an in-memory image.
                TrySinglePrediction(fullImagesetFolderPathTest, mlContext, loadedModel);

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
                if (Path.GetExtension(file) != ".jpg" &&
                    Path.GetExtension(file) != ".JPEG" &&
                    Path.GetExtension(file) != ".png")
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
                if (Path.GetExtension(file) != ".jpg" &&
                    Path.GetExtension(file) != ".JPEG" &&
                    Path.GetExtension(file) != ".png")
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
            // CIFAR dataset ( 50000 train images and 10000 test images )
            string fileName = "cifar10.zip";
            string url = $"https://tlcresources.blob.core.windows.net/datasets/cifar10.zip";

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
