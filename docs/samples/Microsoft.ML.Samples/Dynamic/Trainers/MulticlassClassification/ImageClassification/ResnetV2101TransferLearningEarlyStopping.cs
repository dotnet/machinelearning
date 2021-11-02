
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
using Microsoft.ML.Vision;
using static Microsoft.ML.DataOperationsCatalog;

namespace Samples.Dynamic
{
    public class ResnetV2101TransferLearningEarlyStopping
    {
        public static void Example()
        {
            // Set the path for input images.
            string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);

            string imagesDownloadFolderPath = Path.Combine(assetsPath, "inputs",
                "images");

            //Download the image set and unzip, set the path to image folder.
            string finalImagesFolderName = DownloadImageSet(
                imagesDownloadFolderPath);
            string fullImagesetFolderPath = Path.Combine(
                imagesDownloadFolderPath, finalImagesFolderName);

            MLContext mlContext = new MLContext(seed: 1);

            // Load all the original images info.
            IEnumerable<ImageData> images = LoadImagesFromDirectory(
                folder: fullImagesetFolderPath, useFolderNameAsLabel: true);

            // Shuffle images.
            IDataView shuffledFullImagesDataset = mlContext.Data.ShuffleRows(
                mlContext.Data.LoadFromEnumerable(images));

            // Apply transforms to the input dataset:
            // MapValueToKey : map 'string' type labels to keys
            // LoadImages : load raw images to "Image" column
            shuffledFullImagesDataset = mlContext.Transforms.Conversion
                    .MapValueToKey("Label", keyOrdinality: Microsoft.ML.Transforms
                    .ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(mlContext.Transforms.LoadRawImageBytes("Image",
                            fullImagesetFolderPath, "ImagePath"))
                .Fit(shuffledFullImagesDataset)
                .Transform(shuffledFullImagesDataset);

            // Split the data 90:10 into train and test sets.
            TrainTestData trainTestData = mlContext.Data.TrainTestSplit(
                shuffledFullImagesDataset, testFraction: 0.1, seed: 1);

            IDataView trainDataset = trainTestData.TrainSet;
            IDataView testDataset = trainTestData.TestSet;

            // Set the options for ImageClassification.
            var options = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "Label",
                // Just by changing/selecting InceptionV3/MobilenetV2/ResnetV250
                // here instead of ResnetV2101 you can try a different 
                // architecture/pre-trained model. 
                Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                BatchSize = 10,
                LearningRate = 0.01f,
                // Early Stopping allows for not having to set the number of 
                // epochs for training and monitor the change in metrics to 
                // decide when to stop training.
                // Set EarlyStopping criteria parameters where:
                // minDelta sets the minimum improvement in metric so as to not 
                //      consider stopping early
                // patience sets the number of epochs to wait to see if there is 
                //      any improvement before stopping.
                // metric: the metric to monitor
                EarlyStoppingCriteria = new ImageClassificationTrainer.EarlyStopping(
                    minDelta: 0.001f, patience: 20,
                    metric: ImageClassificationTrainer.EarlyStoppingMetric.Loss),
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                ValidationSet = testDataset
            };

            // Create the ImageClassification pipeline.
            var pipeline = mlContext.Transforms.LoadRawImageBytes(
                "Image", fullImagesetFolderPath, "ImagePath")
                .Append(mlContext.MulticlassClassification.Trainers.
                    ImageClassification(options));

            Console.WriteLine("*** Training the image classification model " +
                "with DNN Transfer Learning on top of the selected " +
                "pre-trained model/architecture ***");

            // Train the model.
            // This involves calculating the bottleneck values, and then
            // training the final layer. Sample output is: 
            // Phase: Bottleneck Computation, Dataset used: Train, Image Index: 1
            // Phase: Bottleneck Computation, Dataset used: Train, Image Index: 2
            // ...
            // Phase: Training, Dataset used:      Train, Batch Processed Count:  18, Learning Rate:       0.01 Epoch:   0, Accuracy:  0.9388888, Cross-Entropy:  0.4604503
            // ...
            // Phase: Training, Dataset used:      Train, Batch Processed Count:  18, Learning Rate: 0.005729948 Epoch:  19, Accuracy:          1, Cross-Entropy: 0.05458016
            // Phase: Training, Dataset used: Validation, Batch Processed Count:   3, Epoch:  19, Accuracy:   0.852381
            // We see that the training stops when the metric stops improving.
            var trainedModel = pipeline.Fit(trainDataset);

            Console.WriteLine("Training with transfer learning finished.");

            // Save the trained model.
            mlContext.Model.Save(trainedModel, shuffledFullImagesDataset.Schema,
                "model.zip");

            // Load the trained and saved model for prediction.
            ITransformer loadedModel;
            DataViewSchema schema;
            using (var file = File.OpenRead("model.zip"))
                loadedModel = mlContext.Model.Load(file, out schema);

            // Evaluate the model on the test dataset.
            // Sample output:
            // Making bulk predictions and evaluating model's quality...
            // Micro-accuracy: 0.851851851851852,macro-accuracy = 0.85
            EvaluateModel(mlContext, testDataset, loadedModel);

            VBuffer<ReadOnlyMemory<char>> keys = default;
            loadedModel.GetOutputSchema(schema)["Label"].GetKeyValues(ref keys);

            // Predict on a single image class using an in-memory image.
            // Sample output:
            // ImageFile : [100080576_f52e8ee070_n.jpg], Scores : [0.8987003,0.00917082,0.0003364562,0.08622094,0.005571446], Predicted Label : dandelion
            TrySinglePrediction(fullImagesetFolderPath, mlContext, loadedModel,
                keys.DenseValues().ToArray());

            Console.WriteLine("Prediction on a single image finished.");

            Console.WriteLine("Press any key to finish");
            Console.ReadKey();
        }

        // Predict on a single image.
        private static void TrySinglePrediction(string imagesForPredictions,
            MLContext mlContext, ITransformer trainedModel,
            ReadOnlyMemory<char>[] originalLabels)
        {
            // Create prediction function to try one prediction.
            var predictionEngine = mlContext.Model
                .CreatePredictionEngine<ImageData, ImagePrediction>(trainedModel);

            // Load test images
            IEnumerable<ImageData> testImages = LoadImagesFromDirectory(
                imagesForPredictions, false);

            // Create an in-memory image object from the first image in the test data.
            ImageData imageToPredict = new ImageData
            {
                ImagePath = testImages.First().ImagePath
            };

            // Predict on the single image.
            var prediction = predictionEngine.Predict(imageToPredict);
            var index = prediction.PredictedLabel;

            Console.WriteLine($"ImageFile : " +
                $"[{Path.GetFileName(imageToPredict.ImagePath)}], " +
                $"Scores : [{string.Join(",", prediction.Score)}], " +
                $"Predicted Label : {originalLabels[index]}");
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

        // Download and unzip the image dataset.
        public static string DownloadImageSet(string imagesDownloadFolder)
        {
            // get a set of images to teach the network about the new classes

            //SINGLE SMALL FLOWERS IMAGESET (200 files)
            string fileName = "flower_photos_small_set.zip";
            string url = $"https://aka.ms/mlnet-resources/datasets/flower_photos_small_set.zip";

            Download(url, imagesDownloadFolder, fileName);
            UnZip(Path.Combine(imagesDownloadFolder, fileName), imagesDownloadFolder);

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
                ResnetV2101TransferLearningEarlyStopping).Assembly.Location);

            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
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
            public UInt32 PredictedLabel;
        }
    }
}
