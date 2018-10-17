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
        public static string DownloadHousingRegressionDataset()
        {
            string baseGitPath = "https://raw.githubusercontent.com/dotnet/machinelearning/024bd4452e1d3660214c757237a19d6123f951ca/test/data/";

            // Downloading a regression dataset from github.com/dotnet/machinelearning
            string dataFile = "housing.txt";

            using (WebClient client = new WebClient())
            {
                client.DownloadFile(new Uri($"{baseGitPath}housing.txt"), dataFile);
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

        public class SampleSentimentData
        {
            public bool Sentiment { get; set; }
            public string SentimentText { get; set; }
        }

        public static IEnumerable<SampleSentimentData> GetSentimentData()
        {
            var data = new List<SampleSentimentData>();
            data.Add(new SampleSentimentData { Sentiment = true, SentimentText = "Best game I've ever played." });
            data.Add(new SampleSentimentData { Sentiment = false, SentimentText = "==RUDE== Dude" });
            data.Add(new SampleSentimentData { Sentiment = true, SentimentText = "Until the next game, this is the best Xbox game!" });

            return data;
        }
    }
}
