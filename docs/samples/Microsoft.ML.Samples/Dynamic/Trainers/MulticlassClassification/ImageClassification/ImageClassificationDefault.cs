
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
using static Microsoft.ML.DataOperationsCatalog;

namespace Samples.Dynamic
{
    public class ImageClassificationDefault
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
            mlContext.Log += MlContext_Log;

            // Load all the original images info
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

            // Create the ImageClassification pipeline by just passing the 
            // input feature and label column name. 
            var pipeline = mlContext.MulticlassClassification.Trainers
                    .ImageClassification(featureColumnName: "Image")
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                    outputColumnName: "PredictedLabel",
                    inputColumnName: "PredictedLabel"));

            Console.WriteLine("*** Training the image classification model " +
                "with DNN Transfer Learning on top of the selected " +
                "pre-trained model/architecture ***");

            // Train the model.
            // This involves calculating the bottleneck values, and then
            // training the final layerSample output is: 
            // [Source=ImageClassificationTrainer; ImageClassificationTrainer, Kind=Trace] Phase: Bottleneck Computation, Dataset used:      Train, Image Index:   1
            // [Source=ImageClassificationTrainer; ImageClassificationTrainer, Kind=Trace] Phase: Bottleneck Computation, Dataset used:      Train, Image Index:   2
            // ...
            // [Source=ImageClassificationTrainer; ImageClassificationTrainer, Kind=Trace] Phase: Training, Dataset used:      Train, Batch Processed Count:  18, Learning Rate:       0.01 Epoch:   0, Accuracy:        0.9, Cross-Entropy:  0.481340
            // ...
            // [Source=ImageClassificationTrainer; ImageClassificationTrainer, Kind=Trace] Phase: Training, Dataset used:      Train, Batch Processed Count:  18, Learning Rate: 0.004759203 Epoch:  25, Accuracy:          1, Cross-Entropy: 0.04848097
            // [Source=ImageClassificationTrainer; ImageClassificationTrainer, Kind=Trace] Phase: Training, Dataset used:      Train, Batch Processed Count:  18, Learning Rate: 0.004473651 Epoch:  26, Accuracy:          1, Cross-Entropy: 0.04930306
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
            // Micro-accuracy: 0.925925925925926,macro-accuracy = 0.933333333333333
            EvaluateModel(mlContext, testDataset, loadedModel);

            // Predict on a single image class using an in-memory image.
            // Sample output:
            // Scores :  [0.8657553,0.006911285,1.46484E-05,0.1266835,0.0006352618], Predicted Label : daisy
            TrySinglePrediction(fullImagesetFolderPath, mlContext, loadedModel);

            Console.WriteLine("Prediction on a single image finished.");

            Console.WriteLine("Press any key to finish");
            Console.ReadKey();
        }

        private static void MlContext_Log(object sender, LoggingEventArgs e)
        {
            if (e.Message.StartsWith("[Source=ImageClassificationTrainer;"))
            {
                Console.WriteLine(e.Message);
            }
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

        // Load In memory raw images from directory.
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
            ZipFile.ExtractToDirectory(gzArchiveName, destFolder);

            File.Create(Path.Combine(destFolder, flag));
            Console.WriteLine("");
            Console.WriteLine("Extracting is completed.");
        }

        // Get absolute path from relative path.
        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(
                ImageClassificationDefault).Assembly.Location);

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
