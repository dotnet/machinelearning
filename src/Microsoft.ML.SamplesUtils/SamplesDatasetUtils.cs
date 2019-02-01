// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Net;
using Microsoft.ML.Data;

namespace Microsoft.ML.SamplesUtils
{
    public static class DatasetUtils
    {
        /// <summary>
        /// Downloads the housing dataset from the ML.NET repo.
        /// </summary>
        public static string DownloadHousingRegressionDataset()
        => Download("https://raw.githubusercontent.com/dotnet/machinelearning/024bd4452e1d3660214c757237a19d6123f951ca/test/data/housing.txt", "housing.txt");

        /// <summary>
        /// Downloads the wikipedia detox dataset from the ML.NET repo.
        /// </summary>
        public static string DownloadSentimentDataset()
         => Download("https://raw.githubusercontent.com/dotnet/machinelearning/76cb2cdf5cc8b6c88ca44b8969153836e589df04/test/data/wikipedia-detox-250-line-data.tsv", "sentiment.tsv");

        /// <summary>
        /// Downloads the adult dataset from the ML.NET repo.
        /// </summary>
        public static string DownloadAdultDataset()
            => Download("https://raw.githubusercontent.com/dotnet/machinelearning/244a8c2ac832657af282aa312d568211698790aa/test/data/adult.train", "adult.txt");

        /// <summary>
        /// Downloads the breast cancer dataset from the ML.NET repo.
        /// </summary>
        public static string DownloadBreastCancerDataset()
            => Download("https://raw.githubusercontent.com/dotnet/machinelearning/76cb2cdf5cc8b6c88ca44b8969153836e589df04/test/data/breast-cancer.txt", "breast-cancer.txt");

        private static string Download(string baseGitPath, string dataFile)
        {
            using (WebClient client = new WebClient())
            {
                client.DownloadFile(new Uri($"{baseGitPath}"), dataFile);
            }

            return dataFile;
        }

        /// <summary>
        /// A simple set of features that help generate the Target column, according to a function.
        /// Used for the transformers/estimators working on numeric data.
        /// </summary>
        public class SampleInput
        {
            public float Feature0 { get; set; }
            public float Feature1 { get; set; }
            public float Feature2 { get; set; }
            public float Feature3 { get; set; }
            public float Target { get; set; }
        }

        /// <summary>
        /// Returns a sample of a numeric dataset.
        /// </summary>
        public static IEnumerable<SampleInput> GetInputData()
        {
            var data = new List<SampleInput>();
            data.Add(new SampleInput { Feature0 = -2.75f, Feature1 = 0.77f, Feature2 = -0.61f, Feature3 = 0.14f, Target = 140.66f });
            data.Add(new SampleInput { Feature0 = -0.61f, Feature1 = -0.37f, Feature2 = -0.12f, Feature3 = 0.55f, Target = 148.12f });
            data.Add(new SampleInput { Feature0 = -0.85f, Feature1 = -0.91f, Feature2 = 1.81f, Feature3 = 0.02f, Target = 402.20f });

            return data;
        }

        /// <summary>
        /// A dataset that contains a tweet and the sentiment assigned to that tweet: 0 - negative and 1 - positive sentiment.
        /// </summary>
        public class SampleSentimentData
        {
            public bool Sentiment { get; set; }
            public string SentimentText { get; set; }
        }

        /// <summary>
        /// Returns a sample of the sentiment dataset.
        /// </summary>
        public static IEnumerable<SampleSentimentData> GetSentimentData()
        {
            var data = new List<SampleSentimentData>();
            data.Add(new SampleSentimentData { Sentiment = true, SentimentText = "Best game I've ever played." });
            data.Add(new SampleSentimentData { Sentiment = false, SentimentText = "==RUDE== Dude, 2" });
            data.Add(new SampleSentimentData { Sentiment = true, SentimentText = "Until the next game, this is the best Xbox game!" });

            return data;
        }

        /// <summary>
        /// A dataset that contains one column with two set of keys assigned to a body of text: Review and ReviewReverse.
        /// The dataset will be used to classify how accurately the keys are assigned to the text.
        /// </summary>
        public class SampleTopicsData
        {
            public string Review { get; set; }
            public string ReviewReverse { get; set; }
            public bool Label { get; set; }
        }

        /// <summary>
        /// Returns a sample of the topics dataset.
        /// </summary>
        public static IEnumerable<SampleTopicsData> GetTopicsData()
        {
            var data = new List<SampleTopicsData>();
            data.Add(new SampleTopicsData { Review = "animals birds cats dogs fish horse", ReviewReverse = "radiation galaxy universe duck", Label = true });
            data.Add(new SampleTopicsData { Review = "horse birds house fish duck cats", ReviewReverse = "space galaxy universe radiation", Label = false });
            data.Add(new SampleTopicsData { Review = "car truck driver bus pickup", ReviewReverse = "bus pickup", Label = true });
            data.Add(new SampleTopicsData { Review = "car truck driver bus pickup horse", ReviewReverse = "car truck", Label = false });

            return data;
        }

        public class SampleTemperatureData
        {
            public DateTime Date {get; set; }
            public float Temperature { get; set; }
        }

        public static IEnumerable<SampleTemperatureData> GetSampleTemperatureData()
        {
            var data = new List<SampleTemperatureData>();
            data.Add(new SampleTemperatureData { Date = new DateTime(2012,1,1), Temperature = 39.0F });
            data.Add(new SampleTemperatureData { Date = new DateTime(2012,1,2), Temperature = 82.0F });
            data.Add(new SampleTemperatureData { Date = new DateTime(2012,1,3), Temperature = 75.0F });
            data.Add(new SampleTemperatureData { Date = new DateTime(2012,1,4), Temperature = 67.0F });
            data.Add(new SampleTemperatureData { Date = new DateTime(2012,1,5), Temperature = 75.0F });
            return data;
        }

        /// <summary>
        /// Represents the column of the infertility dataset.
        /// </summary>
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

        /// <summary>
        /// Returns a few rows of the infertility dataset.
        /// </summary>
        public static IEnumerable<SampleInfertData> GetInfertData()
        {
            var data = new List<SampleInfertData>();
            data.Add(new SampleInfertData
            {
                RowNum = 0,
                Education = "0-5yrs",
                Age = 26,
                Parity = 6,
                Induced = 1,
                Case = 1,
                Spontaneous = 2,
                Stratum = 1,
                PooledStratum = 3
            });
            data.Add(new SampleInfertData
            {
                RowNum = 1,
                Education = "0-5yrs",
                Age = 42,
                Parity = 1,
                Induced = 1,
                Case = 1,
                Spontaneous = 0,
                Stratum = 2,
                PooledStratum = 1
            });
            data.Add(new SampleInfertData
            {
                RowNum = 2,
                Education = "12+yrs",
                Age = 39,
                Parity = 6,
                Induced = 2,
                Case = 1,
                Spontaneous = 0,
                Stratum = 3,
                PooledStratum = 4
            });
            data.Add(new SampleInfertData
            {
                RowNum = 3,
                Education = "0-5yrs",
                Age = 34,
                Parity = 4,
                Induced = 2,
                Case = 1,
                Spontaneous = 0,
                Stratum = 4,
                PooledStratum = 2
            });
            data.Add(new SampleInfertData
            {
                RowNum = 4,
                Education = "6-11yrs",
                Age = 35,
                Parity = 3,
                Induced = 1,
                Case = 1,
                Spontaneous = 1,
                Stratum = 5,
                PooledStratum = 32
            });
            return data;
        }

        public class SampleVectorOfNumbersData
        {
            [VectorType(10)]

            public float[] Features { get; set; }
        }

        /// <summary>
        /// Returns a few rows of the infertility dataset.
        /// </summary>
        public static IEnumerable<SampleVectorOfNumbersData> GetVectorOfNumbersData()
        {
            var data = new List<SampleVectorOfNumbersData>();
            data.Add(new SampleVectorOfNumbersData { Features = new float[10] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 } });
            data.Add(new SampleVectorOfNumbersData { Features = new float[10] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 0 } });
            data.Add(new SampleVectorOfNumbersData
            {
                Features = new float[10] { 2, 3, 4, 5, 6, 7, 8, 9, 0, 1 }
            });
            data.Add(new SampleVectorOfNumbersData
            {
                Features = new float[10] { 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, }
            });
            data.Add(new SampleVectorOfNumbersData
            {
                Features = new float[10] { 5, 6, 7, 8, 9, 0, 1, 2, 3, 4 }
            });
            data.Add(new SampleVectorOfNumbersData
            {
                Features = new float[10] { 6, 7, 8, 9, 0, 1, 2, 3, 4, 5 }
            });
            return data;
        }

        private const int _simpleBinaryClassSampleFeatureLength = 10;

        public class BinaryLabelFloatFeatureVectorSample
        {
            public bool Label;

            [VectorType(_simpleBinaryClassSampleFeatureLength)]
            public float[] Features;
        }

        public static  IEnumerable<BinaryLabelFloatFeatureVectorSample> GenerateBinaryLabelFloatFeatureVectorSamples(int exampleCount)
        {
            var rnd = new Random(0);
            var data = new List<BinaryLabelFloatFeatureVectorSample>();
            for (int i = 0; i < exampleCount; ++i)
            {
                // Initialize an example with a random label and an empty feature vector.
                var sample = new BinaryLabelFloatFeatureVectorSample() { Label = rnd.Next() % 2 == 0, Features = new float[_simpleBinaryClassSampleFeatureLength] };
                // Fill feature vector according the assigned label.
                for (int j = 0; j < _simpleBinaryClassSampleFeatureLength; ++j)
                {
                    var value = (float)rnd.NextDouble();
                    // Positive class gets larger feature value.
                    if (sample.Label)
                        value += 0.2f;
                    sample.Features[j] = value;
                }

                data.Add(sample);
            }
            return data;
        }

        public class FloatLabelFloatFeatureVectorSample
        {
            public float Label;

            [VectorType(_simpleBinaryClassSampleFeatureLength)]
            public float[] Features;
        }

        public static  IEnumerable<FloatLabelFloatFeatureVectorSample> GenerateFloatLabelFloatFeatureVectorSamples(int exampleCount)
        {
            var rnd = new Random(0);
            var data = new List<FloatLabelFloatFeatureVectorSample>();
            for (int i = 0; i < exampleCount; ++i)
            {
                // Initialize an example with a random label and an empty feature vector.
                var sample = new FloatLabelFloatFeatureVectorSample() { Label = rnd.Next() % 2, Features = new float[_simpleBinaryClassSampleFeatureLength] };
                // Fill feature vector according the assigned label.
                for (int j = 0; j < _simpleBinaryClassSampleFeatureLength; ++j)
                {
                    var value = (float)rnd.NextDouble();
                    // Positive class gets larger feature value.
                    if (sample.Label == 0)
                        value += 0.2f;
                    sample.Features[j] = value;
                }

                data.Add(sample);
            }
            return data;
        }

        public class FfmExample
        {
            public bool Label;

            [VectorType(_simpleBinaryClassSampleFeatureLength)]
            public float[] Field0;

            [VectorType(_simpleBinaryClassSampleFeatureLength)]
            public float[] Field1;

            [VectorType(_simpleBinaryClassSampleFeatureLength)]
            public float[] Field2;
        }

        public static  IEnumerable<FfmExample> GenerateFfmSamples(int exampleCount)
        {
            var rnd = new Random(0);
            var data = new List<FfmExample>();
            for (int i = 0; i < exampleCount; ++i)
            {
                // Initialize an example with a random label and an empty feature vector.
                var sample = new FfmExample() { Label = rnd.Next() % 2 == 0,
                    Field0 = new float[_simpleBinaryClassSampleFeatureLength],
                    Field1 = new float[_simpleBinaryClassSampleFeatureLength],
                    Field2 = new float[_simpleBinaryClassSampleFeatureLength] };
                // Fill feature vector according the assigned label.
                for (int j = 0; j < 10; ++j)
                {
                    var value0 = (float)rnd.NextDouble();
                    // Positive class gets larger feature value.
                    if (sample.Label)
                        value0 += 0.2f;
                    sample.Field0[j] = value0;

                    var value1 = (float)rnd.NextDouble();
                    // Positive class gets smaller feature value.
                    if (sample.Label)
                        value1 -= 0.2f;
                    sample.Field1[j] = value1;

                    var value2 = (float)rnd.NextDouble();
                    // Positive class gets larger feature value.
                    if (sample.Label)
                        value2 += 0.8f;
                    sample.Field2[j] = value2;
                }

                data.Add(sample);
            }
            return data;
        }

        /// <summary>
        /// feature vector's length in <see cref="MulticlassClassificationExample"/>.
        /// </summary>
        private const int _featureVectorLength = 10;

        public class MulticlassClassificationExample
        {
            [VectorType(_featureVectorLength)]
            public float[] Features;
            [ColumnName("Label")]
            public string Label;
            public uint LabelIndex;
            public uint PredictedLabelIndex;
            [VectorType(4)]
            // The probabilities of being "AA", "BB", "CC", and "DD".
            public float[] Scores;

            public MulticlassClassificationExample()
            {
                Features = new float[_featureVectorLength];
            }
        }

        /// <summary>
        /// Helper function used to generate random <see cref="GenerateRandomMulticlassClassificationExamples"/>s.
        /// </summary>
        /// <param name="count">Number of generated examples.</param>
        /// <returns>A list of random examples.</returns>
        public static List<MulticlassClassificationExample> GenerateRandomMulticlassClassificationExamples(int count)
        {
            var examples = new List<MulticlassClassificationExample>();
            var rnd = new Random(0);
            for (int i = 0; i < count; ++i)
            {
                var example = new MulticlassClassificationExample();
                var res = i % 4;
                // Generate random float feature values.
                for (int j = 0; j < _featureVectorLength; ++j)
                {
                    var value = (float)rnd.NextDouble() + res * 0.2f;
                    example.Features[j] = value;
                }

                // Generate label based on feature sum.
                if (res == 0)
                    example.Label = "AA";
                else if (res == 1)
                    example.Label = "BB";
                else if (res == 2)
                    example.Label = "CC";
                else
                    example.Label = "DD";

                // The following three attributes are just placeholder for storing prediction results.
                example.LabelIndex = default;
                example.PredictedLabelIndex = default;
                example.Scores = new float[4];

                examples.Add(example);
            }
            return examples;
        }
    }
}
