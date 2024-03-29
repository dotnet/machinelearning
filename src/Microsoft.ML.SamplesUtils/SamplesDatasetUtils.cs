﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Runtime.InteropServices.ComTypes;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace Microsoft.ML.SamplesUtils
{
    public static class DatasetUtils
    {
        public static string GetFilePathFromDataDirectory(string fileName)
        {
#if NETFRAMEWORK
            string directory = AppDomain.CurrentDomain.BaseDirectory;
#else
            string directory = AppContext.BaseDirectory;
#endif

            while (!Directory.Exists(Path.Combine(directory, ".git")) && directory != null)
            {
                directory = Directory.GetParent(directory).FullName;
            }

            if (directory == null)
            {
                throw new DirectoryNotFoundException("Could not find the ML.NET repository");
            }
            return Path.Combine(directory, "test", "data", fileName);
        }

        /// <summary>
        /// Returns the path to the housing dataset from the ML.NET repo.
        /// </summary>
        public static string GetHousingRegressionDataset() => GetFilePathFromDataDirectory("housing.txt");

        public static IDataView LoadHousingRegressionDataset(MLContext mlContext)
        {
            // Obtains the path to the file
            string dataFile = GetHousingRegressionDataset();

            // Define the columns to load
            var loader = mlContext.Data.CreateTextLoader(
                columns: new[]
                    {
                        new TextLoader.Column("MedianHomeValue", DataKind.Single, 0),
                        new TextLoader.Column("CrimesPerCapita", DataKind.Single, 1),
                        new TextLoader.Column("PercentResidental", DataKind.Single, 2),
                        new TextLoader.Column("PercentNonRetail", DataKind.Single, 3),
                        new TextLoader.Column("CharlesRiver", DataKind.Single, 4),
                        new TextLoader.Column("NitricOxides", DataKind.Single, 5),
                        new TextLoader.Column("RoomsPerDwelling", DataKind.Single, 6),
                        new TextLoader.Column("PercentPre40s", DataKind.Single, 7),
                        new TextLoader.Column("EmploymentDistance", DataKind.Single, 8),
                        new TextLoader.Column("HighwayDistance", DataKind.Single, 9),
                        new TextLoader.Column("TaxRate", DataKind.Single, 10),
                        new TextLoader.Column("TeacherRatio", DataKind.Single, 11),
                    },
                hasHeader: true
            );

            // Load the data into an IDataView
            var data = loader.Load(dataFile);

            return data;
        }

        /// <summary>
        /// Returns the path to the adult dataset from the ML.NET repo.
        /// </summary>
        public static string GetAdultDataset() => GetFilePathFromDataDirectory("adult.txt");

        /// <summary>
        /// Returns the path to the Adult UCI dataset and featurizes it to be suitable for classification tasks.
        /// </summary>
        /// <param name="mlContext"><see cref="MLContext"/> used for data loading and processing.</param>
        /// <returns>Featurized dataset.</returns>
        /// <remarks>
        /// For more details about this dataset, please see https://archive.ics.uci.edu/ml/datasets/adult.
        /// </remarks>
        public static IDataView LoadFeaturizedAdultDataset(MLContext mlContext)
        {
            // Create data featurizing pipeline
            var pipeline = mlContext.Transforms.CopyColumns("Label", "IsOver50K")
                // Convert categorical features to one-hot vectors
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("workclass"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("education"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("marital-status"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("occupation"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("relationship"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("ethnicity"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("native-country"))
                // Combine all features into one feature vector
                .Append(mlContext.Transforms.Concatenate("Features", "workclass", "education", "marital-status",
                    "occupation", "relationship", "ethnicity", "native-country", "age", "education-num",
                    "capital-gain", "capital-loss", "hours-per-week"))
                // Min-max normalize all the features
                .Append(mlContext.Transforms.NormalizeMinMax("Features"));

            var data = LoadRawAdultDataset(mlContext);
            var featurizedData = pipeline.Fit(data).Transform(data);
            return featurizedData;
        }

        /// <summary>
        /// Returns the path to the Adult UCI dataset and featurizes it to be suitable for classification tasks.
        /// </summary>
        /// <param name="mlContext"><see cref="MLContext"/> used for data loading and processing.</param>
        /// <returns>Raw dataset.</returns>
        /// <remarks>
        /// For more details about this dataset, please see https://archive.ics.uci.edu/ml/datasets/adult.
        /// </remarks>
        public static IDataView LoadRawAdultDataset(MLContext mlContext)
        {
            // Obtains the path to the file
            string dataFile = GetAdultDataset();

            // Define the columns to load
            var loader = mlContext.Data.CreateTextLoader(
                columns: new[]
                    {
                        new TextLoader.Column("age", DataKind.Single, 0),
                        new TextLoader.Column("workclass", DataKind.String, 1),
                        new TextLoader.Column("fnlwgt", DataKind.Single, 2),
                        new TextLoader.Column("education", DataKind.String, 3),
                        new TextLoader.Column("education-num", DataKind.Single, 4),
                        new TextLoader.Column("marital-status", DataKind.String, 5),
                        new TextLoader.Column("occupation", DataKind.String, 6),
                        new TextLoader.Column("relationship", DataKind.String, 7),
                        new TextLoader.Column("ethnicity", DataKind.String, 8),
                        new TextLoader.Column("sex", DataKind.String, 9),
                        new TextLoader.Column("capital-gain", DataKind.Single, 10),
                        new TextLoader.Column("capital-loss", DataKind.Single, 11),
                        new TextLoader.Column("hours-per-week", DataKind.Single, 12),
                        new TextLoader.Column("native-country", DataKind.String, 13),
                        new TextLoader.Column("IsOver50K", DataKind.Boolean, 14),
                    },
                separatorChar: ',',
                hasHeader: true
            );

            return loader.Load(dataFile);
        }

        /// <summary>
        /// Returns the path to the breast cancer dataset from the ML.NET repo.
        /// </summary>
        public static string GetBreastCancerDataset() => GetFilePathFromDataDirectory("breast-cancer.txt");

        /// <summary>
        /// Returns the path to 4 sample images, and a tsv file with their names from the dotnet/machinelearning repo.
        /// </summary>
        public static string GetSampleImages() => GetFilePathFromDataDirectory("images/images.tsv");

        /// <summary>
        /// Downloads sentiment_model from the dotnet/machinelearning-testdata repo.
        /// </summary>
        /// <remarks>
        /// The model is downloaded from
        /// https://github.com/dotnet/machinelearning-testdata/blob/296625f4e49d50fcd6a48a0d92bea7584e198c0f/Microsoft.ML.TensorFlow.TestModels/sentiment_model
        /// The model is in 'SavedModel' format. For further explanation on how was the `sentiment_model` created
        /// c.f. https://github.com/dotnet/machinelearning-testdata/blob/296625f4e49d50fcd6a48a0d92bea7584e198c0f/Microsoft.ML.TensorFlow.TestModels/sentiment_model/README.md
        /// </remarks>
        public static string DownloadTensorFlowSentimentModel()
        {
            string remotePath = "https://github.com/dotnet/machinelearning-testdata/raw/296625f4e49d50fcd6a48a0d92bea7584e198c0f/Microsoft.ML.TensorFlow.TestModels/sentiment_model/";

            string path = "sentiment_model";
            if (!Directory.Exists(path))
                Directory.CreateDirectory(path);

            string varPath = Path.Combine(path, "variables");
            if (!Directory.Exists(varPath))
                Directory.CreateDirectory(varPath);

            Download(Path.Combine(remotePath, "saved_model.pb"), Path.Combine(path, "saved_model.pb")).Wait();
            Download(Path.Combine(remotePath, "imdb_word_index.csv"), Path.Combine(path, "imdb_word_index.csv")).Wait();
            Download(Path.Combine(remotePath, "variables", "variables.data-00000-of-00001"), Path.Combine(varPath, "variables.data-00000-of-00001")).Wait();
            Download(Path.Combine(remotePath, "variables", "variables.index"), Path.Combine(varPath, "variables.index")).Wait();

            return path;
        }

        private static async Task<string> Download(string baseGitPath, string dataFile)
        {
            if (File.Exists(dataFile))
                return dataFile;

            using (HttpClient client = new HttpClient())
            {
                var response = await client.GetStreamAsync(new Uri($"{baseGitPath}")).ConfigureAwait(false);
                using (var fs = new FileStream(dataFile, FileMode.CreateNew))
                {
                    await response.CopyToAsync(fs).ConfigureAwait(false);
                }
            }

            return dataFile;
        }

        private const int _simpleBinaryClassSampleFeatureLength = 10;

        /// <summary>
        /// Example with one binary label, 10 feature values and a weight (float).
        /// </summary>
        public class BinaryLabelFloatFeatureVectorFloatWeightSample
        {
            public bool Label;

            [VectorType(_simpleBinaryClassSampleFeatureLength)]
            public float[] Features;

            public float Weight;
        }

        /// <summary>
        /// Class used to capture prediction of <see cref="BinaryLabelFloatFeatureVectorFloatWeightSample"/> when
        /// calling <see cref="DataOperationsCatalog.CreateEnumerable{TRow}(IDataView, bool, bool, SchemaDefinition)"/> via on <see cref="MLContext"/>.
        /// </summary>
        public class CalibratedBinaryClassifierOutput
        {
            public bool Label;
            public float Score;
            public float Probability;
        }

        /// <summary>
        /// Class used to capture prediction of <see cref="BinaryLabelFloatFeatureVectorFloatWeightSample"/> when
        /// calling <see cref="DataOperationsCatalog.CreateEnumerable{TRow}(IDataView, bool, bool, SchemaDefinition)"/> via on <see cref="MLContext"/>.
        /// </summary>
        public class NonCalibratedBinaryClassifierOutput
        {
            public bool Label;
            public float Score;
        }

        public static IEnumerable<BinaryLabelFloatFeatureVectorFloatWeightSample> GenerateBinaryLabelFloatFeatureVectorFloatWeightSamples(int exampleCount)
        {
            var rnd = new Random(0);
            var data = new List<BinaryLabelFloatFeatureVectorFloatWeightSample>();
            for (int i = 0; i < exampleCount; ++i)
            {
                // Initialize an example with a random label and an empty feature vector.
                var sample = new BinaryLabelFloatFeatureVectorFloatWeightSample()
                {
                    Label = rnd.Next() % 2 == 0,
                    Features = new float[_simpleBinaryClassSampleFeatureLength],
                    Weight = (float)rnd.NextDouble()
                };

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

        public class FloatLabelFloatFeatureVectorUlongGroupIdSample
        {
            public float Label;

            [VectorType(_simpleBinaryClassSampleFeatureLength)]
            public float[] Features;

            [KeyType(ulong.MaxValue - 1)]
            public ulong GroupId;
        }

        public class FloatLabelFloatFeatureVectorSample
        {
            public float Label;

            [VectorType(_simpleBinaryClassSampleFeatureLength)]
            public float[] Features;
        }

        public static IEnumerable<FloatLabelFloatFeatureVectorUlongGroupIdSample> GenerateFloatLabelFloatFeatureVectorUlongGroupIdSamples(int exampleCount, double naRate = 0, ulong minGroupId = 1, ulong maxGroupId = 5)
        {
            var data = new List<FloatLabelFloatFeatureVectorUlongGroupIdSample>();
            var rnd = new Random(0);
            var intermediate = GenerateFloatLabelFloatFeatureVectorSamples(exampleCount, naRate).ToList();

            for (int i = 0; i < exampleCount; ++i)
            {
                var sample = new FloatLabelFloatFeatureVectorUlongGroupIdSample() { Label = intermediate[i].Label, Features = intermediate[i].Features, GroupId = (ulong)rnd.Next((int)minGroupId, (int)maxGroupId) };
                data.Add(sample);
            }

            return data;
        }

        public static IEnumerable<FloatLabelFloatFeatureVectorSample> GenerateFloatLabelFloatFeatureVectorSamples(int exampleCount, double naRate = 0)
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
                    float value = float.NaN;
                    if (naRate <= 0 || rnd.NextDouble() > naRate)
                    {
                        value = (float)rnd.NextDouble();
                        // Positive class gets larger feature value.
                        if (sample.Label == 0)
                            value += 0.2f;
                    }
                    sample.Features[j] = value;
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
        /// Helper function used to generate random <see cref="MulticlassClassificationExample"/> objects.
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
