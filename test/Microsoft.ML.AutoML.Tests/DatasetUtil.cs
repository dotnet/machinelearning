// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using Microsoft.ML.Data;
using Microsoft.ML.TestFrameworkCommon;

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

        public static string GetUciAdultDataset() => GetDataPath("adult.tiny.with-schema.txt");

        public static string GetMlNetGeneratedRegressionDataset() => GetDataPath("generated_regression_dataset.csv");

        public static string GetIrisDataset() => GetDataPath("iris.txt");

        public static string GetDataPath(string fileName)
        {
            return Path.Combine(TestCommon.GetRepoRoot(), "test", "data", fileName);
        }

        public static IDataView GetUciAdultDataView()
        {
            if (_uciAdultDataView == null)
            {
                var context = new MLContext(1);
                var uciAdultDataFile = GetUciAdultDataset();
                var columnInferenceResult = context.Auto().InferColumns(uciAdultDataFile, UciAdultLabel);
                var textLoader = context.Data.CreateTextLoader(columnInferenceResult.TextLoaderOptions);
                _uciAdultDataView = textLoader.Load(uciAdultDataFile);
            }
            return _uciAdultDataView;
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
            /*
             * This is only needed as Linux can produce files in a different 
             * order than other OSes. As this is a test case we want to maintain
             * consistent accuracy across all OSes, so we sort to remove this discrepancy.
             */
            Array.Sort(files);
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
            string url = $"https://aka.ms/mlnet-resources/datasets/flower_photos_tiny_set_for_unit_test.zip";

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
