// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Threading;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML.Test
{
    internal static class DatasetUtil
    {
        public const string UciAdultLabel = DefaultColumnNames.Label;
        public const string TrivialMulticlassDatasetLabel = "Target";
        public const string MlNetGeneratedRegressionLabel = "target";
        public const int IrisDatasetLabelColIndex = 0;

        public static string TrivialMulticlassDatasetPath = Path.Combine("TestData", "TrivialMulticlassDataset.txt");

        private static IDataView _uciAdultDataView;

        public static IDataView GetUciAdultDataView()
        {
            if(_uciAdultDataView == null)
            {
                var context = new MLContext();
                var uciAdultDataFile = DownloadUciAdultDataset();
                var columnInferenceResult = context.Auto().InferColumns(uciAdultDataFile, UciAdultLabel);
                var textLoader = context.Data.CreateTextLoader(columnInferenceResult.TextLoaderOptions);
                _uciAdultDataView = textLoader.Load(uciAdultDataFile);
            }
            return _uciAdultDataView;
        }

        // downloads the UCI Adult dataset from the ML.Net repo
        public static string DownloadUciAdultDataset() =>
            DownloadIfNotExists("https://raw.githubusercontent.com/dotnet/machinelearning/f0e639af5ffdc839aae8e65d19b5a9a1f0db634a/test/data/adult.tiny.with-schema.txt", "uciadult.dataset");

        public static string DownloadMlNetGeneratedRegressionDataset() =>
            DownloadIfNotExists("https://raw.githubusercontent.com/dotnet/machinelearning/e78971ea6fd736038b4c355b840e5cbabae8cb55/test/data/generated_regression_dataset.csv", "mlnet_generated_regression.dataset");

        public static string DownloadIrisDataset() =>
            DownloadIfNotExists("https://raw.githubusercontent.com/dotnet/machinelearning/54596ac/test/data/iris.txt", "iris.dataset");

        private static string DownloadIfNotExists(string baseGitPath, string dataFile)
        {
            foreach (var nextIteration in Enumerable.Range(0, 10))
            {
                // if file doesn't already exist, download it
                if (!File.Exists(dataFile))
                {
                    var tempFile = Path.GetTempFileName();

                    try
                    {
                        using (var client = new WebClient())
                        {
                            client.DownloadFile(new Uri($"{baseGitPath}"), tempFile);

                            if (!File.Exists(dataFile))
                            {
                                File.Copy(tempFile, dataFile);
                                File.Delete(tempFile);
                            }
                        }
                    }
                    catch(Exception)
                    {
                    }
                }

                if (File.Exists(dataFile) && (new FileInfo(dataFile).Length > 0))
                {
                    return dataFile;
                }

                Thread.Sleep(300);
            }

            throw new Exception($"Failed to download test file {dataFile}.");
        }

        public static string GetFlowersDataset()
        {
            const string datasetName = @"flowers";
            string assetsRelativePath = @"assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);
            string imagesDownloadFolderPath = Path.Combine(assetsPath, "inputs",
                "images");

            //Download the image set and unzip
            string finalImagesFolderName = DownloadImageSet(
                imagesDownloadFolderPath);

            string fullImagesetFolderPath = Path.Combine(
                imagesDownloadFolderPath, finalImagesFolderName);

            var images = LoadImagesFromDirectory(folder: fullImagesetFolderPath);

            using (StreamWriter file = new StreamWriter(datasetName))
            {
                file.WriteLine("Label,ImagePath");
                foreach (var image in images)
                    file.WriteLine(image.Label + "," + image.ImagePath);
            }

            return datasetName;
        }

        public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder)
        {
            var files = Directory.GetFiles(folder, "*",
                searchOption: SearchOption.AllDirectories);
            foreach (var file in files)
            {
                var extension = Path.GetExtension(file).ToLower();
                if (extension != ".jpg" &&
                    extension != ".jpeg" &&
                    extension != ".png" &&
                    extension != ".gif"
                )
                    continue;

                var label = Path.GetFileName(file);
                label = Directory.GetParent(file).Name;
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

        private static void Download(string url, string destDir, string destFileName)
        {
            if (destFileName == null)
                destFileName = Path.GetFileName(new Uri(url).AbsolutePath); ;

            Directory.CreateDirectory(destDir);

            string relativeFilePath = Path.Combine(destDir, destFileName);

            if (File.Exists(relativeFilePath))
                return;

            new WebClient().DownloadFile(url, relativeFilePath);
            return;
        }

        private static void UnZip(String gzArchiveName, String destFolder)
        {
            var flag = gzArchiveName.Split(Path.DirectorySeparatorChar)
                .Last()
                .Split('.')
                .First() + ".bin";

            if (File.Exists(Path.Combine(destFolder, flag)))
                return;

            ZipFile.ExtractToDirectory(gzArchiveName, destFolder);
            File.Create(Path.Combine(destFolder, flag));
        }

        public static string GetAbsolutePath(string relativePath) =>
            Path.Combine(new FileInfo(typeof(
                DatasetUtil).Assembly.Location).Directory.FullName, relativePath);

        public class ImageData
        {
            [LoadColumn(0)]
            public string ImagePath;

            [LoadColumn(1)]
            public string Label;
        }
    }
}
