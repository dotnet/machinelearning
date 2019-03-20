// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Functional.Tests.Datasets;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Functional.Tests
{
    public class DataTransformation : BaseTestClass
    {
        public DataTransformation(ITestOutputHelper output) : base(output)
        {
        }

        /// <summary>
        /// Extensibility: Add a new column that is a function of other columns.
        /// </summary>
        [Fact]
        void ExtensibilityAddAColumnAsAFunctionOfMultipleColumns()
        {
            // Concurrency must be 1 to assure that the mapping is done sequentially.
            var mlContext = new MLContext(seed: 1);

            // Load the Iris dataset
            var data = mlContext.Data.LoadFromTextFile<Iris>(
                GetDataPath(TestDatasets.iris.trainFilename),
                hasHeader: TestDatasets.iris.fileHasHeader,
                separatorChar: TestDatasets.iris.fileSeparator);

            // Subsample it down to the first 10 rows.
            int numSamples = 10;
            data = mlContext.Data.TakeRows(data, numSamples);

            // Create a stand-alone function to produce a random number.
            float angiospermCosine(float petalWidth, float petalLength, float sepalWidth, float sepalLength)
            {
                var petalMagnitude = Math.Sqrt(petalWidth * petalWidth + petalLength * petalLength);
                var sepalMagnitude = Math.Sqrt(sepalWidth * sepalWidth + sepalLength * sepalLength);
                return (float)((petalWidth * sepalWidth + petalLength * sepalLength) / (petalMagnitude * sepalMagnitude));
            }

            // Create a function that generates a column.
            Action<Iris, IrisWithOneExtraColumn> generateGroupId = (input, output) =>
            {
                output.Label = input.Label;
                output.Float1 = angiospermCosine(input.PetalLength, input.PetalWidth, input.SepalLength, input.SepalWidth);
                output.PetalLength = input.PetalLength;
                output.PetalWidth = input.PetalWidth;
                output.SepalLength = input.SepalLength;
                output.SepalWidth = input.SepalWidth;
            };

            // Create a pipeline to execute the custom function.
            var pipeline = mlContext.Transforms.CustomMapping(generateGroupId, null);

            // Transform the data.
            var transformedData = pipeline.Fit(data).Transform(data);

            // Verify that the column has the correct data.
            var transformedRows = mlContext.Data.CreateEnumerable<IrisWithOneExtraColumn>(transformedData, reuseRowObject: true);
            foreach (var row in transformedRows)
            {
                var cosineDistance = angiospermCosine(row.PetalLength, row.PetalWidth, row.SepalLength, row.SepalWidth);
                Assert.Equal(cosineDistance, row.Float1);
            }
        }

        /// <summary>
        /// Extensibility: Add multiple new columns.
        /// </summary>
        [Fact]
        void ExtensibilityAddingTwoColumns()
        {
            // Concurrency must be 1 to assure that the mapping is done sequentially.
            var mlContext = new MLContext(seed: 1);

            // Load the Iris dataset
            var data = mlContext.Data.LoadFromTextFile<Iris>(
                GetDataPath(TestDatasets.iris.trainFilename),
                hasHeader: TestDatasets.iris.fileHasHeader,
                separatorChar: TestDatasets.iris.fileSeparator);

            // Subsample it down to the first 10 rows.
            int numSamples = 10;
            data = mlContext.Data.TakeRows(data, numSamples);

            // Create a function that generates a column.
            Action<Iris, IrisWithTwoExtraColumns> generateGroupId = (input, output) =>
            {
                output.Label = input.Label;
                output.Float1 = GetRandomNumber(1 + input.Label + input.PetalLength + input.PetalWidth + input.SepalLength + input.SepalWidth);
                output.Float2 = GetRandomNumber(2 + input.Label + input.PetalLength + input.PetalWidth + input.SepalLength + input.SepalWidth);
                output.PetalLength = input.PetalLength;
                output.PetalWidth = input.PetalWidth;
                output.SepalLength = input.SepalLength;
                output.SepalWidth = input.SepalWidth;
            };

            // Create a pipeline to execute the custom function.
            var pipeline = mlContext.Transforms.CustomMapping(generateGroupId, null);

            // Transform the data.
            var transformedData = pipeline.Fit(data).Transform(data);

            // Verify that the column has the correct data.
            var transformedRows = mlContext.Data.CreateEnumerable<IrisWithTwoExtraColumns>(transformedData, reuseRowObject: true);
            foreach (var row in transformedRows)
            {
                var randomNumber1 = GetRandomNumber(1 + row.Label + row.PetalLength + row.PetalWidth + row.SepalLength + row.SepalWidth);
                var randomNumber2 = GetRandomNumber(2 + row.Label + row.PetalLength + row.PetalWidth + row.SepalLength + row.SepalWidth);
                Assert.Equal(randomNumber1, row.Float1);
                Assert.Equal(randomNumber2, row.Float2);
            }
        }

        /// <summary>
        /// Extensibility: Featurize text using custom word-grams, char-grams, and normalization.
        /// </summary>
        [Fact]
        void ExtensibilityModifyTextFeaturization()
        {
            // Concurrency must be 1 to assure that the mapping is done sequentially.
            var mlContext = new MLContext(seed: 1);

            var data = mlContext.Data.LoadFromTextFile<TweetSentiment>(GetDataPath(TestDatasets.Sentiment.trainFilename),
                hasHeader: TestDatasets.Sentiment.fileHasHeader,
                separatorChar: TestDatasets.Sentiment.fileSeparator);

            // Create a training pipeline.
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features",
                    new TextFeaturizingEstimator.Options
                    {
                        CharFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 3, UseAllLengths = false },
                        WordFeatureExtractor = new WordBagEstimator.Options(),
                        Norm = TextFeaturizingEstimator.NormFunction.L1
                    }, "SentimentText")
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.BinaryClassification.Trainers.SdcaCalibrated(
                    new SdcaCalibratedBinaryTrainer.Options { NumberOfThreads = 1 }));

            // Train the model.
            var model = pipeline.Fit(data);

            // Evaluate the model.
            var scoredData = model.Transform(data);
            var metrics = mlContext.BinaryClassification.Evaluate(scoredData);

            // Check that the metrics returned are valid.
            Common.AssertMetrics(metrics);
        }

        /// <summary>
        /// Extensibility: Apply a normalizer to columns in the dataset.
        /// </summary>
        [Fact]
        void ExtensibilityNormalizeColumns()
        {
            // Concurrency must be 1 to assure that the mapping is done sequentially.
            var mlContext = new MLContext(seed: 1);

            // Load the Iris dataset.
            var data = mlContext.Data.LoadFromTextFile<Iris>(
                GetDataPath(TestDatasets.iris.trainFilename),
                hasHeader: TestDatasets.iris.fileHasHeader,
                separatorChar: TestDatasets.iris.fileSeparator);

            // Compose the transformation.
            var pipeline = mlContext.Transforms.Concatenate("Features", Iris.Features)
                .Append(mlContext.Transforms.Normalize("Features", mode: NormalizingEstimator.NormalizationMode.MinMax));
            
            // Transform the data.
            var transformedData = pipeline.Fit(data).Transform(data);

            // Validate that the data was normalized to between -1 and 1.
            var dataEnumerator = mlContext.Data.CreateEnumerable<FeatureColumn>(transformedData, true);
            foreach (var row in dataEnumerator)
                // Verify per-slot normalization.
                for (int i = 0; i < row.Features.Length; i++)
                    Assert.InRange(row.Features[i], -1, 1);
        }

        private float GetRandomNumber(float number)
        {
            var seed = (int)(10 * number);
            var rng = new Random(seed);
            return (float)rng.NextDouble();
        }
    }
}