using System;
using System.IO;
using System.Net;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto.Test
{
    internal static class DatasetUtil
    {
        public const string UciAdultLabel = DefaultColumnNames.Label;
        public const string TrivialDatasetLabel = DefaultColumnNames.Label;
        public const string MlNetGeneratedRegressionLabel = "target";

        private static IDataView _uciAdultDataView;

        public static IDataView GetUciAdultDataView()
        {
            if(_uciAdultDataView == null)
            {
                var uciAdultDataFile = DownloadUciAdultDataset();
                _uciAdultDataView = (new MLContext()).Data.AutoRead(uciAdultDataFile, UciAdultLabel, true);
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

        private static string DownloadIfNotExists(string baseGitPath, string dataFile)
        {
            // if file doesn't already exist, download it
            if(!File.Exists(dataFile))
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
