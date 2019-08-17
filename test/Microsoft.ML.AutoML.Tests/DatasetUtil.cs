// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
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
    }
}
