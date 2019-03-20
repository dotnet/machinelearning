// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
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
        {
            var fileName = "housing.txt";
            if (!File.Exists(fileName))
                Download("https://raw.githubusercontent.com/dotnet/machinelearning/024bd4452e1d3660214c757237a19d6123f951ca/test/data/housing.txt", fileName);
            return fileName;
        }

        public static IDataView LoadHousingRegressionDataset(MLContext mlContext)
        {
            // Download the file
            string dataFile = DownloadHousingRegressionDataset();

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
        /// A class to hold the raw housing regression rows.
        /// </summary>
        public sealed class HousingRegression
        {
            public float MedianHomeValue { get; set; }
            public float CrimesPerCapita { get; set; }
            public float PercentResidental { get; set; }
            public float PercentNonRetail { get; set; }
            public float CharlesRiver { get; set; }
            public float NitricOxides { get; set; }
            public float RoomsPerDwelling { get; set; }
            public float PercentPre40s { get; set; }
            public float EmploymentDistance { get; set; }
            public float HighwayDistance { get; set; }
            public float TaxRate { get; set; }
            public float TeacherRatio { get; set; }
        }

        /// <summary>
        /// Downloads the wikipedia detox dataset from the ML.NET repo.
        /// </summary>
        public static string[] DownloadSentimentDataset()
        {
            var trainFile = Download("https://raw.githubusercontent.com/dotnet/machinelearning/76cb2cdf5cc8b6c88ca44b8969153836e589df04/test/data/wikipedia-detox-250-line-data.tsv", "sentiment.tsv");
            var testFile = Download("https://raw.githubusercontent.com/dotnet/machinelearning/76cb2cdf5cc8b6c88ca44b8969153836e589df04/test/data/wikipedia-detox-250-line-test.tsv", "sentimenttest.tsv");
            return new[] { trainFile, testFile };
        }

            /// <summary>
            /// Downloads the adult dataset from the ML.NET repo.
            /// </summary>
            public static string DownloadAdultDataset()
            => Download("https://raw.githubusercontent.com/dotnet/machinelearning/244a8c2ac832657af282aa312d568211698790aa/test/data/adult.train", "adult.txt");

        /// <summary>
        /// Downloads the  wikipedia detox dataset and featurizes it to be suitable for sentiment classification tasks.
        /// </summary>
        /// <param name="mlContext"><see cref="MLContext"/> used for data loading and processing.</param>
        /// <returns>Featurized train and test dataset.</returns>
        public static IDataView[] LoadFeaturizedSentimentDataset(MLContext mlContext)
        {
            // Download the files
            var dataFiles = DownloadSentimentDataset();

            // Define the columns to load
            var loader = mlContext.Data.CreateTextLoader(
                columns: new[]
                    {
                        new TextLoader.Column("Sentiment", DataKind.Boolean, 0),
                        new TextLoader.Column("SentimentText", DataKind.String, 1)
                    },
                hasHeader: true
            );

            // Create data featurizing pipeline
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "SentimentText");

            var data = loader.Load(dataFiles[0]);
            var model = pipeline.Fit(data);
            var featurizedDataTrain = model.Transform(data);
            var featurizedDataTest = model.Transform(loader.Load(dataFiles[1]));
            return new[] { featurizedDataTrain, featurizedDataTest };
        }

        /// <summary>
        /// Downloads the Adult UCI dataset and featurizes it to be suitable for classification tasks.
        /// </summary>
        /// <param name="mlContext"><see cref="MLContext"/> used for data loading and processing.</param>
        /// <returns>Featurized dataset.</returns>
        /// <remarks>
        /// For more details about this dataset, please see https://archive.ics.uci.edu/ml/datasets/adult.
        /// </remarks>
        public static IDataView LoadFeaturizedAdultDataset(MLContext mlContext)
        {
            // Download the file
            string dataFile = DownloadAdultDataset();

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
                        new TextLoader.Column("native-country", DataKind.Single, 13),
                        new TextLoader.Column("IsOver50K", DataKind.Boolean, 14),
                    },
                separatorChar: ',',
                hasHeader: true
            );

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
                .Append(mlContext.Transforms.Normalize("Features"));

            var data = loader.Load(dataFile);
            var featurizedData = pipeline.Fit(data).Transform(data);
            return featurizedData;
        }

        public static string DownloadMslrWeb10k()
        {
            var fileName = "MSLRWeb10KTrain10kRows.tsv";
            if (!File.Exists(fileName))
                Download("https://tlcresources.blob.core.windows.net/datasets/MSLR-WEB10K/MSLR-WEB10K%2BFold1.TRAIN.SMALL_10k-rows.tsv", fileName);
            return fileName;
        }

        public static IDataView LoadFeaturizedMslrWeb10kDataset(MLContext mlContext)
        {
            // Download the training and validation files.
            string dataFile = DownloadMslrWeb10k();

            // Create the loader to load the data.
            var loader = mlContext.Data.CreateTextLoader(
                columns: new[]
                {
                    new TextLoader.Column("Label", DataKind.Single, 0),
                    new TextLoader.Column("GroupId", DataKind.String, 1),
                    new TextLoader.Column("Features", DataKind.Single, new[] { new TextLoader.Range(2, 138) })
                }
            );

            // Load the raw dataset.
            var data = loader.Load(dataFile);

            // Create the featurization pipeline. First, hash the GroupId column.
            var pipeline = mlContext.Transforms.Conversion.Hash("GroupId")
                // Replace missing values in Features column with the default replacement value for its type.
                .Append(mlContext.Transforms.ReplaceMissingValues("Features"));

            // Fit the pipeline and transform the dataset.
            var transformedData = pipeline.Fit(data).Transform(data);

            return transformedData;
        }

        /// <summary>
        /// Downloads the breast cancer dataset from the ML.NET repo.
        /// </summary>
        public static string DownloadBreastCancerDataset()
            => Download("https://raw.githubusercontent.com/dotnet/machinelearning/76cb2cdf5cc8b6c88ca44b8969153836e589df04/test/data/breast-cancer.txt", "breast-cancer.txt");

        /// <summary>
        /// Downloads 4 images, and a tsv file with their names from the dotnet/machinelearning repo.
        /// </summary>
        public static string DownloadImages()
        {
            string path = "images";

            var dirInfo = Directory.CreateDirectory(path);

            string pathEscaped = $"{path}{Path.DirectorySeparatorChar}";

            Download("https://raw.githubusercontent.com/dotnet/machinelearning/284e02cadf5342aa0c36f31d62fc6fa15bc06885/test/data/images/banana.jpg", $"{pathEscaped}banana.jpg");
            Download("https://raw.githubusercontent.com/dotnet/machinelearning/284e02cadf5342aa0c36f31d62fc6fa15bc06885/test/data/images/hotdog.jpg", $"{pathEscaped}hotdog.jpg");
            Download("https://raw.githubusercontent.com/dotnet/machinelearning/284e02cadf5342aa0c36f31d62fc6fa15bc06885/test/data/images/images.tsv", $"{pathEscaped}images.tsv");
            Download("https://raw.githubusercontent.com/dotnet/machinelearning/284e02cadf5342aa0c36f31d62fc6fa15bc06885/test/data/images/tomato.bmp", $"{pathEscaped}tomato.bmp");
            Download("https://raw.githubusercontent.com/dotnet/machinelearning/284e02cadf5342aa0c36f31d62fc6fa15bc06885/test/data/images/tomato.jpg", $"{pathEscaped}tomato.jpg");

            return $"{path}{Path.DirectorySeparatorChar}images.tsv";
        }

        /// <summary>
        /// Downloads sentiment_model from the dotnet/machinelearning-testdata repo.
        /// </summary>
        /// <remarks>
        /// The model is downloaded from
        /// https://github.com/dotnet/machinelearning-testdata/blob/master/Microsoft.ML.TensorFlow.TestModels/sentiment_model
        /// The model is in 'SavedModel' format. For further explanation on how was the `sentiment_model` created
        /// c.f. https://github.com/dotnet/machinelearning-testdata/blob/master/Microsoft.ML.TensorFlow.TestModels/sentiment_model/README.md
        /// </remarks>
        public static string DownloadTensorFlowSentimentModel()
        {
            string remotePath = "https://github.com/dotnet/machinelearning-testdata/raw/master/Microsoft.ML.TensorFlow.TestModels/sentiment_model/";

            string path = "sentiment_model";
            if (!Directory.Exists(path))
                Directory.CreateDirectory(path);

            string varPath = Path.Combine(path, "variables");
            if (!Directory.Exists(varPath))
                Directory.CreateDirectory(varPath);

            Download(Path.Combine(remotePath, "saved_model.pb"), Path.Combine(path, "saved_model.pb"));
            Download(Path.Combine(remotePath, "imdb_word_index.csv"), Path.Combine(path, "imdb_word_index.csv"));
            Download(Path.Combine(remotePath, "variables", "variables.data-00000-of-00001"), Path.Combine(varPath, "variables.data-00000-of-00001"));
            Download(Path.Combine(remotePath, "variables", "variables.index"), Path.Combine(varPath, "variables.index"));

            return path;
        }

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
            public DateTime Date { get; set; }
            public float Temperature { get; set; }
        }

        public class SampleTemperatureDataWithLatitude
        {
            public float Latitude { get; set; }
            public DateTime Date { get; set; }
            public float Temperature { get; set; }
        }

        /// <summary>
        /// Get a fake temperature dataset.
        /// </summary>
        /// <param name="exampleCount">The number of examples to return.</param>
        /// <returns>An enumerable of <see cref="SampleTemperatureData"/>.</returns>
        public static IEnumerable<SampleTemperatureData> GetSampleTemperatureData(int exampleCount)
        {
            var rng = new Random(1234321);
            var date = new DateTime(2012, 1, 1);
            float temperature = 39.0f;

            for (int i = 0; i < exampleCount; i++)
            {
                date = date.AddDays(1);
                temperature += rng.Next(-5, 5);
                yield return new SampleTemperatureData { Date = date, Temperature = temperature };
            }
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

        /// <summary>
        /// Example with one binary label and 10 feature values.
        /// </summary>
        public class BinaryLabelFloatFeatureVectorSample
        {
            public bool Label;

            [VectorType(_simpleBinaryClassSampleFeatureLength)]
            public float[] Features;
        }

        /// <summary>
        /// Class used to capture prediction of <see cref="BinaryLabelFloatFeatureVectorSample"/> when
        /// calling <see cref="DataOperationsCatalog.CreateEnumerable{TRow}(IDataView, bool, bool, SchemaDefinition)"/> via on <see cref="MLContext"/>.
        /// </summary>
        public class CalibratedBinaryClassifierOutput
        {
            public bool Label;
            public float Score;
            public float Probability;
        }

        /// <summary>
        /// Class used to capture prediction of <see cref="BinaryLabelFloatFeatureVectorSample"/> when
        /// calling <see cref="DataOperationsCatalog.CreateEnumerable{TRow}(IDataView, bool, bool, SchemaDefinition)"/> via on <see cref="MLContext"/>.
        /// </summary>
        public class NonCalibratedBinaryClassifierOutput
        {
            public bool Label;
            public float Score;
        }

        public static IEnumerable<BinaryLabelFloatFeatureVectorSample> GenerateBinaryLabelFloatFeatureVectorSamples(int exampleCount)
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

        public static IEnumerable<FfmExample> GenerateFfmSamples(int exampleCount)
        {
            var rnd = new Random(0);
            var data = new List<FfmExample>();
            for (int i = 0; i < exampleCount; ++i)
            {
                // Initialize an example with a random label and an empty feature vector.
                var sample = new FfmExample()
                {
                    Label = rnd.Next() % 2 == 0,
                    Field0 = new float[_simpleBinaryClassSampleFeatureLength],
                    Field1 = new float[_simpleBinaryClassSampleFeatureLength],
                    Field2 = new float[_simpleBinaryClassSampleFeatureLength]
                };
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

        // The following variables defines the shape of a matrix. Its shape is _synthesizedMatrixRowCount-by-_synthesizedMatrixColumnCount.
        // Because in ML.NET key type's minimal value is zero, the first row index is always zero in C# data structure (e.g., MatrixColumnIndex=0
        // and MatrixRowIndex=0 in MatrixElement below specifies the value at the upper-left corner in the training matrix). If user's row index
        // starts with 1, their row index 1 would be mapped to the 2nd row in matrix factorization module and their first row may contain no values.
        // This behavior is also true to column index.
        private const uint _synthesizedMatrixFirstColumnIndex = 1;
        private const uint _synthesizedMatrixFirstRowIndex = 1;
        private const uint _synthesizedMatrixColumnCount = 60;
        private const uint _synthesizedMatrixRowCount = 100;

        // A data structure used to encode a single value in matrix
        public class MatrixElement
        {
            // Matrix column index is at most _synthesizedMatrixColumnCount + _synthesizedMatrixFirstColumnIndex.
            [KeyType(_synthesizedMatrixColumnCount + _synthesizedMatrixFirstColumnIndex)]
            public uint MatrixColumnIndex;
            // Matrix row index is at most _synthesizedMatrixRowCount + _synthesizedMatrixFirstRowIndex.
            [KeyType(_synthesizedMatrixRowCount + _synthesizedMatrixFirstRowIndex)]
            public uint MatrixRowIndex;
            // The value at the column MatrixColumnIndex and row MatrixRowIndex.
            public float Value;
        }

        // A data structure used to encode prediction result. Comparing with MatrixElement, The field Value in MatrixElement is
        // renamed to Score because Score is the default name of matrix factorization's output.
        public class MatrixElementForScore
        {
            [KeyType(_synthesizedMatrixColumnCount + _synthesizedMatrixFirstColumnIndex)]
            public uint MatrixColumnIndex;
            [KeyType(_synthesizedMatrixRowCount + _synthesizedMatrixFirstRowIndex)]
            public uint MatrixRowIndex;
            public float Score;
        }

        // Create an in-memory matrix as a list of tuples (column index, row index, value).
        public static List<MatrixElement> GetRecommendationData()
        {
            var dataMatrix = new List<MatrixElement>();
            for (uint i = _synthesizedMatrixFirstColumnIndex; i < _synthesizedMatrixFirstColumnIndex + _synthesizedMatrixColumnCount; ++i)
                for (uint j = _synthesizedMatrixFirstRowIndex; j < _synthesizedMatrixFirstRowIndex + _synthesizedMatrixRowCount; ++j)
                    dataMatrix.Add(new MatrixElement() { MatrixColumnIndex = i, MatrixRowIndex = j, Value = (i + j) % 5 });
            return dataMatrix;
        }
    }
}
