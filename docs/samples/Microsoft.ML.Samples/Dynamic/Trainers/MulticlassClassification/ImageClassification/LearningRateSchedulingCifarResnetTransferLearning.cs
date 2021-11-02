
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
using Microsoft.ML.Trainers;
using Microsoft.ML.Vision;

namespace Samples.Dynamic
{
    public class LearningRateSchedulingCifarResnetTransferLearning
    {
        public static void Example()
        {
            // Set the path for input images.
            string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);

            string imagesDownloadFolderPath = Path.Combine(assetsPath, "inputs",
                "images");

            // Download Cifar Dataset and set train and test dataset
            // paths.
            string finalImagesFolderName = DownloadImageSet(
                   imagesDownloadFolderPath);
            string finalImagesFolderNameTrain = "cifar\\train";
            string fullImagesetFolderPathTrain = Path.Combine(
                imagesDownloadFolderPath, finalImagesFolderNameTrain);

            string finalImagesFolderNameTest = "cifar\\test";
            string fullImagesetFolderPathTest = Path.Combine(
                imagesDownloadFolderPath, finalImagesFolderNameTest);

            MLContext mlContext = new MLContext(seed: 1);

            //Load all the original train images info.
            IEnumerable<ImageData> train_images = LoadImagesFromDirectory(
                folder: fullImagesetFolderPathTrain, useFolderNameAsLabel: true);

            IDataView trainDataset = mlContext.Data.
                LoadFromEnumerable(train_images);

            // Apply transforms to the input dataset:
            // MapValueToKey : map 'string' type labels to keys
            // LoadImages : load raw images to "Image" column
            trainDataset = mlContext.Transforms.Conversion
                    .MapValueToKey("Label", keyOrdinality: Microsoft.ML.Transforms
                    .ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(mlContext.Transforms.LoadRawImageBytes("Image",
                            fullImagesetFolderPathTrain, "ImagePath"))
                .Fit(trainDataset)
                .Transform(trainDataset);

            // Load all the original test images info and apply
            // the same transforms as above.
            IEnumerable<ImageData> test_images = LoadImagesFromDirectory(
                folder: fullImagesetFolderPathTest, useFolderNameAsLabel: true);

            IDataView testDataset = mlContext.Data.
                LoadFromEnumerable(test_images);

            testDataset = mlContext.Transforms.Conversion
                    .MapValueToKey("Label", keyOrdinality: Microsoft.ML.Transforms
                    .ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(mlContext.Transforms.LoadRawImageBytes("Image",
                            fullImagesetFolderPathTest, "ImagePath"))
                .Fit(testDataset)
                .Transform(testDataset);

            // Set the options for ImageClassification.
            var options = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "Label",
                // Just by changing/selecting InceptionV3/MobilenetV2 
                // here instead of 
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
                // You can also try other types of Learning rate scheduling 
                // methods available in LearningRateScheduler.cs  
                LearningRateScheduler = new LsrDecay()
            };

            // Create the ImageClassification pipeline.
            var pipeline = mlContext.MulticlassClassification.Trainers.
                ImageClassification(options)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                    outputColumnName: "PredictedLabel",
                    inputColumnName: "PredictedLabel"));


            Console.WriteLine("*** Training the image classification model " +
                "with DNN Transfer Learning on top of the selected " +
                "pre-trained model/architecture ***");

            // Train the model.
            // This involves calculating the bottleneck values, and then
            // training the final layer. Sample output is: 
            // Phase: Bottleneck Computation, Dataset used: Train, Image Index:   1
            // Phase: Bottleneck Computation, Dataset used: Train, Image Index:   2
            // ...
            // Phase: Training, Dataset used: Train, Batch Processed Count:  18, Learning Rate: 0.01 Epoch: 0, Accuracy: 0.9166667,Cross-Entropy:  0.4866541
            // ...
            var trainedModel = pipeline.Fit(trainDataset);

            Console.WriteLine("Training with transfer learning finished.");

            // Save the trained model.
            mlContext.Model.Save(trainedModel, testDataset.Schema,
                "model.zip");

            // Load the trained and saved model for prediction.
            ITransformer loadedModel;
            DataViewSchema schema;
            using (var file = File.OpenRead("model.zip"))
                loadedModel = mlContext.Model.Load(file, out schema);

            // Evaluate the model on the test dataset.
            // Sample output:
            // Making bulk predictions and evaluating model's quality...
            // Micro - accuracy: ...,macro - accuracy = ...
            EvaluateModel(mlContext, testDataset, loadedModel);

            // Predict image class using a single in-memory image.
            TrySinglePrediction(fullImagesetFolderPathTest, mlContext,
                loadedModel);

            Console.WriteLine("Prediction on a single image finished.");

            Console.WriteLine("Press any key to finish");
            Console.ReadKey();
        }

        // Predict on a single image.
        private static void TrySinglePrediction(string imagesForPredictions,
            MLContext mlContext, ITransformer trainedModel)
        {
            // Create prediction function to try one prediction.
            var predictionEngine = mlContext.Model
                .CreatePredictionEngine<InMemoryImageData,
                ImagePrediction>(trainedModel);

            // Load test images.
            IEnumerable<InMemoryImageData> testImages =
                LoadInMemoryImagesFromDirectory(imagesForPredictions, false);

            // Create an in-memory image object from the first image in the test data.
            InMemoryImageData imageToPredict = new InMemoryImageData
            {
                Image = testImages.First().Image
            };

            // Predict on the single image.
            var prediction = predictionEngine.Predict(imageToPredict);

            Console.WriteLine($"Scores : [{string.Join(",", prediction.Score)}], " +
                $"Predicted Label : {prediction.PredictedLabel}");
        }

        // Evaluate the trained model on the passed test dataset.
        private static void EvaluateModel(MLContext mlContext,
            IDataView testDataset, ITransformer trainedModel)
        {
            Console.WriteLine("Making bulk predictions and evaluating model's " +
                "quality...");

            // Evaluate the model on the test data and get the evaluation metrics.
            IDataView predictions = trainedModel.Transform(testDataset);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine($"Micro-accuracy: {metrics.MicroAccuracy}," +
                              $"macro-accuracy = {metrics.MacroAccuracy}");

            Console.WriteLine("Predicting and Evaluation complete.");
        }

        //Load the Image Data from input directory.
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

        // Load In memory raw images from directory.
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

        // Download and unzip the image dataset.
        public static string DownloadImageSet(string imagesDownloadFolder)
        {
            // get a set of images to teach the network about the new classes
            // CIFAR dataset ( 50000 train images and 10000 test images )
            string fileName = "cifar10.zip";
            string url = $"https://aka.ms/mlnet-resources/" +
                "datasets/cifar10.zip";

            Download(url, imagesDownloadFolder, fileName);
            UnZip(Path.Combine(imagesDownloadFolder, fileName),
                imagesDownloadFolder);

            return Path.GetFileNameWithoutExtension(fileName);
        }

        // Download file to destination directory from input URL.
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

        // Unzip the file to destination folder.
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

        // Get absolute path from relative path.
        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(
                LearningRateSchedulingCifarResnetTransferLearning).Assembly.Location);

            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        // InMemoryImageData class holding the raw image byte array and label.
        public class InMemoryImageData
        {
            [LoadColumn(0)]
            public byte[] Image;

            [LoadColumn(1)]
            public string Label;
        }

        // ImageData class holding the imagepath and label.
        public class ImageData
        {
            [LoadColumn(0)]
            public string ImagePath;

            [LoadColumn(1)]
            public string Label;
        }

        // ImagePrediction class holding the score and predicted label metrics.
        public class ImagePrediction
        {
            [ColumnName("Score")]
            public float[] Score;

            [ColumnName("PredictedLabel")]
            public string PredictedLabel;
        }
    }
}
