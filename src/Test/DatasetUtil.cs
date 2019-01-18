using System;
using System.IO;
using System.Net;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto.Test
{
    internal static class DatasetUtil
    {
        public const string UciAdultLabel = DefaultColumnNames.Label;

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
        private static string DownloadUciAdultDataset() =>
            DownloadIfNotExists("https://raw.githubusercontent.com/dotnet/machinelearning/f0e639af5ffdc839aae8e65d19b5a9a1f0db634a/test/data/adult.tiny.with-schema.txt", "uciadult.dataset");

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
