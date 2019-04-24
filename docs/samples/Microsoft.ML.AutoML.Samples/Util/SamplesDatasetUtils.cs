using System;
using System.IO;
using System.Net;

namespace Microsoft.ML.AutoML.Samples
{
    class SamplesDatasetUtils
    {
        /// <summary>
        /// Downloads the opt digits train dataset from the ML.NET samples repo.
        /// </summary>
        public static string DownloadOptDigitsTrain()
            => Download("https://raw.githubusercontent.com/dotnet/machinelearning-samples/6434fa917ace08f6d683ee3ed10f3b309c70c475/datasets/optdigits-train.csv", "optdigits-train.csv");

        /// <summary>
        /// Downloads the opt digits test dataset from the ML.NET samples repo.
        /// </summary>
        public static string DownloadOptDigitsTest()
            => Download("https://raw.githubusercontent.com/dotnet/machinelearning-samples/6434fa917ace08f6d683ee3ed10f3b309c70c475/datasets/optdigits-test.csv", "optdigits-test.csv");

        private static string Download(string baseGitPath, string dataFile)
        {
            if (File.Exists(dataFile))
            {
                return dataFile;
            }

            using (WebClient client = new WebClient())
            {
                client.DownloadFile(new Uri($"{baseGitPath}"), dataFile);
            }

            return dataFile;
        }
    }
}
