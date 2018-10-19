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
            data.Add(new SampleSentimentData { Sentiment = false, SentimentText = "==RUDE== Dude, 2" });
            data.Add(new SampleSentimentData { Sentiment = true, SentimentText = "Until the next game, this is the best Xbox game!" });

            return data;
        }

        public class SampleTopicsData
        {
            public string Review { get; set; }
            public string ReviewReverse { get; set; }
            public bool Label { get; set; }
        }

        public static IEnumerable<SampleTopicsData> GetTopicsData()
        {
            var data = new List<SampleTopicsData>();
            data.Add(new SampleTopicsData { Review = "animals birds cats dogs fish horse", ReviewReverse = "radiation galaxy universe duck", Label = true });
            data.Add(new SampleTopicsData { Review = "horse birds house fish duck cats", ReviewReverse = "space galaxy universe radiation", Label = false });
            data.Add(new SampleTopicsData { Review = "car truck driver bus pickup", ReviewReverse = "bus pickup", Label = true});
            data.Add(new SampleTopicsData { Review = "car truck driver bus pickup horse", ReviewReverse = "car truck", Label = false });

            return data;
        }

        public class SampleInfertData
        {
            public int RowNum { get; set; }
            public string Education { get; set; }
            public float Age { get; set; }
            public float Parity { get; set; }
            public float Induced { get; set; }
            public float Case { get; set; }

            public float Spontaneous { get; set; }
            public float Stratum { get; set; }
            public float PooledStratum { get; set; }
        }

        public static IEnumerable<SampleInfertData> GetInfertData()
        {
            var data = new List<SampleInfertData>();
            data.Add(new SampleInfertData {
                RowNum = 0, Education = "0-5yrs", Age = 26, Parity = 6, Induced = 1, Case = 1, Spontaneous = 2, Stratum = 1, PooledStratum = 3 });
            data.Add(new SampleInfertData {
                RowNum = 1, Education = "0-5yrs", Age = 42, Parity = 1, Induced = 1, Case = 1, Spontaneous = 0, Stratum = 2, PooledStratum = 1 });
            data.Add(new SampleInfertData {
                RowNum = 2, Education = "0-5yrs", Age = 39, Parity = 6, Induced = 2, Case = 1, Spontaneous = 0, Stratum = 3, PooledStratum = 4 });
            data.Add(new SampleInfertData {
                RowNum = 3, Education = "0-5yrs", Age = 34, Parity = 4, Induced = 2, Case = 1, Spontaneous = 0, Stratum = 4, PooledStratum = 2 });
            data.Add(new SampleInfertData {
                RowNum = 4, Education = "6-11yrs", Age = 35, Parity = 3, Induced = 1,  Case = 1, Spontaneous = 1, Stratum = 5, PooledStratum = 32 });
            return data;
        }
    }
}
