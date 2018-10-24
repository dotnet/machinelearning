// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Net;

namespace Microsoft.ML.SamplesUtils
{
    public static class DatasetUtils
    {
        /// <summary>
        /// Downloads the housing dataset from the ML.Net repo.
        /// </summary>
        public static string DownloadHousingRegressionDataset()
        => Download("https://raw.githubusercontent.com/dotnet/machinelearning/024bd4452e1d3660214c757237a19d6123f951ca/test/data/housing.txt", "housing.txt");

        /// <summary>
        /// Downloads the adult dataset from the ML.NEt repo
        /// </summary>
        public static string DownloadSentimentDataset()
        => Download("https://github.com/dotnet/machinelearning/blob/76cb2cdf5cc8b6c88ca44b8969153836e589df04/test/data/wikipedia-detox-250-line-data.tsv", "sentiment.tsv");

        private static string Download(string baseGitPath, string dataFile)
        {
            using (WebClient client = new WebClient())
            {
                client.DownloadFile(new Uri($"{baseGitPath}"), dataFile);
            }

            return dataFile;
        }

        public class SampleInput
        {
            public float Feature0 { get; set; }
            public float Feature1 { get; set; }
            public float Feature2 { get; set; }
            public float Feature3 { get; set; }
            public float Target { get; set; }
        }

        public static IEnumerable<SampleInput> GetInputData()
        {
            var data = new List<SampleInput>();
            data.Add(new SampleInput { Feature0 = -2.75f, Feature1 = 0.77f, Feature2 = -0.61f, Feature3 = 0.14f, Target = 140.66f });
            data.Add(new SampleInput { Feature0 = -0.61f, Feature1 = -0.37f, Feature2 = -0.12f, Feature3 = 0.55f, Target = 148.12f });
            data.Add(new SampleInput { Feature0 = -0.85f, Feature1 = -0.91f, Feature2 = 1.81f, Feature3 = 0.02f, Target = 402.20f });

            return data;
        }
    }
}
