// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Net;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Auto;

namespace mlnet.Test
{
    internal static class DatasetUtil
    {
        public const string UciAdultLabel = DefaultColumnNames.Label;
        public const string TrivialDatasetLabel = DefaultColumnNames.Label;
        public const string MlNetGeneratedRegressionLabel = "target";
        public const int IrisDatasetLabelColIndex = 0;

        private static IDataView _uciAdultDataView;

        public static IDataView GetUciAdultDataView()
        {
            if (_uciAdultDataView == null)
            {
                var context = new MLContext();
                var uciAdultDataFile = DownloadUciAdultDataset();
                var columnInferenceResult = context.Auto().InferColumns(uciAdultDataFile, UciAdultLabel);
                var textLoader = context.Data.CreateTextLoader(columnInferenceResult.TextLoaderArgs);
                _uciAdultDataView = textLoader.Read(uciAdultDataFile);
            }
            return _uciAdultDataView;
        }

        // downloads the UCI Adult dataset from the ML.Net repo
        public static string DownloadUciAdultDataset() =>
            DownloadIfNotExists("https://raw.githubusercontent.com/dotnet/machinelearning/f0e639af5ffdc839aae8e65d19b5a9a1f0db634a/test/data/adult.tiny.with-schema.txt", "uciadult.dataset");

        public static string DownloadTrivialDataset() =>
            DownloadIfNotExists("https://raw.githubusercontent.com/dotnet/machinelearning/eae76959e6714af44caa212e102a5f06f0110e72/test/data/trivial-train.tsv", "trivial.dataset");

        public static string DownloadMlNetGeneratedRegressionDataset() =>
            DownloadIfNotExists("https://raw.githubusercontent.com/dotnet/machinelearning/e78971ea6fd736038b4c355b840e5cbabae8cb55/test/data/generated_regression_dataset.csv", "mlnet_generated_regression.dataset");

        public static string DownloadIrisDataset() =>
            DownloadIfNotExists("https://raw.githubusercontent.com/dotnet/machinelearning/54596ac/test/data/iris.txt", "iris.dataset");

        private static string DownloadIfNotExists(string baseGitPath, string dataFile)
        {
            // if file doesn't already exist, download it
            if (!File.Exists(dataFile))
            {
                using (var client = new WebClient())
                {
                    client.DownloadFile(new Uri($"{baseGitPath}"), dataFile);
                }
            }

            return dataFile;
        }
    }
}
