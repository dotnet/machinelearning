// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.IO.Compression;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using BenchmarkDotNet.Attributes;
using static Microsoft.ML.DataOperationsCatalog;
using System.Net.Http;
using Microsoft.ML.Dnn;

namespace Microsoft.ML.Benchmarks
{
    [Config(typeof(TrainConfig))]
    public class ImageClassificationBench
    {
        private string assetsPath;
        private MLContext mlContext;
        private IDataView trainDataset;
        private IDataView testDataset;


        [GlobalSetup]
        public void SetupData()
        {
            mlContext = new MLContext(seed: 1);
            /*
             * Running in benchmarks causes to create a new temporary dir for each run
             * However this dir is deleted while still running, as such need to get one
             * level up to prevent issues with saving data.
             */
            string assetsRelativePath = @"../../../../assets";
            assetsPath = GetAbsolutePath(assetsRelativePath);

            var outputMlNetModelFilePath = Path.Combine(assetsPath, "outputs",
                "imageClassifier.zip");


            string imagesDownloadFolderPath = Path.Combine(assetsPath, "inputs",
                "images");

            //Download the image set and unzip
            string finalImagesFolderName = DownloadImageSet(
                imagesDownloadFolderPath);
            string fullImagesetFolderPath = Path.Combine(
                imagesDownloadFolderPath, finalImagesFolderName);

            //Load all the original images info
            IEnumerable<ImageData> images = LoadImagesFromDirectory(
                folder: fullImagesetFolderPath, useFolderNameAsLabel: true);

            IDataView shuffledFullImagesDataset = mlContext.Data.ShuffleRows(
                mlContext.Data.LoadFromEnumerable(images));

            shuffledFullImagesDataset = mlContext.Transforms.Conversion
                    .MapValueToKey("Label")
                .Append(mlContext.Transforms.LoadRawImageBytes("Image",
                            fullImagesetFolderPath, "ImagePath"))
                .Fit(shuffledFullImagesDataset)
                .Transform(shuffledFullImagesDataset);

            // Split the data 90:10 into train and test sets, train and
            // evaluate.
            TrainTestData trainTestData = mlContext.Data.TrainTestSplit(
                shuffledFullImagesDataset, testFraction: 0.1, seed: 1);

            trainDataset = trainTestData.TrainSet;
            testDataset = trainTestData.TestSet;

        }

        [Benchmark]
        public TransformerChain<KeyToValueMappingTransformer> TrainResnetV250()
        {
            var options = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "Label",
                Arch = ImageClassificationTrainer.Architecture.ResnetV250,
                Epoch = 50,
                BatchSize = 10,
                LearningRate = 0.01f,
                EarlyStoppingCriteria = new ImageClassificationTrainer.EarlyStopping(minDelta: 0.001f, patience: 20, metric: ImageClassificationTrainer.EarlyStoppingMetric.Loss),
                ValidationSet = testDataset,
                ModelSavePath = assetsPath
            };
            var pipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(options)
            .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                outputColumnName: "PredictedLabel",
                inputColumnName: "PredictedLabel"));

            return pipeline.Fit(trainDataset);
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

            string relativeFilePath = Path.Combine(destDir, destFileName);


            using (HttpClient client = new HttpClient())
            {
                if (File.Exists(relativeFilePath))
                {
                    var headerResponse = client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead).Result;
                    var totalSizeInBytes = headerResponse.Content.Headers.ContentLength;
                    var currentSize = new FileInfo(relativeFilePath).Length;

                    //If current file size is not equal to expected file size, re-download file
                    if (currentSize != totalSizeInBytes)
                    {
                        File.Delete(relativeFilePath);
                        var response = client.GetAsync(url).Result;
                        using FileStream fileStream = new FileStream(relativeFilePath, FileMode.Create, FileAccess.Write, FileShare.None);
                        using Stream contentStream = response.Content.ReadAsStreamAsync().Result;
                        contentStream.CopyTo(fileStream);
                    }
                }
                else
                {
                    Directory.CreateDirectory(destDir);
                    var response = client.GetAsync(url).Result;
                    using FileStream fileStream = new FileStream(relativeFilePath, FileMode.Create, FileAccess.Write, FileShare.None);
                    using Stream contentStream = response.Content.ReadAsStreamAsync().Result;
                    contentStream.CopyTo(fileStream);
                }
            }
            return true;
        }


        public static void UnZip(String gzArchiveName, String destFolder)
        {
            var flag = gzArchiveName.Split(Path.DirectorySeparatorChar)
                .Last()
                .Split('.')
                .First() + ".bin";

            if (File.Exists(Path.Combine(destFolder, flag))) return;

            ZipFile.ExtractToDirectory(gzArchiveName, destFolder);

            File.Create(Path.Combine(destFolder, flag));
            Console.WriteLine("");
            Console.WriteLine("Extracting is completed.");
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(
                ImageClassificationBench).Assembly.Location);

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

    }
    public static class HttpContentExtensions
    {
        public static Task ReadAsFileAsync(this HttpContent content, string filename, bool overwrite)
        {
            string pathname = Path.GetFullPath(filename);
            if (!overwrite && File.Exists(filename))
            {
                throw new InvalidOperationException(string.Format("File {0} already exists.", pathname));
            }

            FileStream fileStream = null;
            try
            {
                fileStream = new FileStream(pathname, FileMode.Create, FileAccess.Write, FileShare.None);
                return content.CopyToAsync(fileStream).ContinueWith(
                    (copyTask) =>
                    {
                        fileStream.Close();
                    });
            }
            catch
            {
                if (fileStream != null)
                {
                    fileStream.Close();
                }

                throw;
            }
        }
    }
}
