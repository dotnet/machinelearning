// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Functional.Tests.Datasets;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Functional.Tests
{
    public class Debugging : BaseTestClass
    {
        public Debugging(ITestOutputHelper output) : base(output)
        {
        }

        /// <summary>
        /// Debugging: The individual pipeline steps can be inspected to see what is happening to 
        /// data as it flows through.
        /// </summary>
        /// <remarks>
        /// It should, possibly through the debugger, be not such a pain to actually
        /// see what is happening to your data when you apply this or that transform. For example, if I
        /// were to have the text "Help I'm a bug!" I should be able to see the steps where it is
        /// normalized to "help i'm a bug" then tokenized into ["help", "i'm", "a", "bug"] then
        /// mapped into term numbers [203, 25, 3, 511] then projected into the sparse
        /// float vector {3:1, 25:1, 203:1, 511:1}, etc. etc.
        /// </remarks>
        [Fact]
        void InspectIntermediatePipelineSteps()
        {
            var mlContext = new MLContext(seed: 1);

            var data = mlContext.Data.LoadFromEnumerable<TweetSentiment>(
                new TweetSentiment[]
                {
                    new TweetSentiment { Sentiment = true, SentimentText = "I love ML.NET." },
                    new TweetSentiment { Sentiment = true, SentimentText = "I love TLC." },
                    new TweetSentiment { Sentiment = false, SentimentText = "I dislike fika." }
                });

            // create a training pipeline.
            var pipeline = mlContext.Transforms.Text.FeaturizeText(
                    "Features",
                    new TextFeaturizingEstimator.Options
                    {
                        KeepPunctuations = false,
                        OutputTokensColumnName = "FeaturizeTextTokens",
                        CharFeatureExtractor = null, // new WordBagEstimator.Options { NgramLength = 0, SkipLength = -1 },
                        WordFeatureExtractor = new WordBagEstimator.Options { NgramLength = 1},
                        Norm = TextFeaturizingEstimator.NormFunction.None
                    },
                    "SentimentText");

            // Fit the pipeline to the data.
            var model = pipeline.Fit(data);

            // Transform the data.
            var transformedData = model.Transform(data);

            var preview = transformedData.Preview();

            // Verify that columns can be inspected.
            // Validate the tokens column.
            var tokensColumn = transformedData.GetColumn<string[]>(transformedData.Schema["FeaturizeTextTokens"]);
            var expectedTokens = new string[3][]
            {
                new string[] {"i", "love", "mlnet"},
                new string[] {"i", "love", "tlc"},
                new string[] {"i", "dislike", "fika"},
            };
            int i = 0;
            foreach (var rowTokens in tokensColumn)
                Assert.Equal(expectedTokens[i++], rowTokens);

            // Validate the Features column.
            var featuresColumn = transformedData.GetColumn<float[]>(transformedData.Schema["Features"]);
            var expectedFeatures = new float[3][]
            {
                new float[6] { 1, 1, 1, 0, 0 ,0 },
                new float[6] { 1, 1, 0, 1, 0, 0 },
                new float[6] { 1, 0, 0, 0, 1, 1 }
            };
            i = 0;
            foreach (var rowFeatures in featuresColumn)
                Assert.Equal(expectedFeatures[i++], rowFeatures);
        }

        /// <summary>
        /// Debugging: The schema of the pipeline can be inspected.
        /// </summary>
        [Fact]
        public void InspectPipelineSchema()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset.
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(GetDataPath(TestDatasets.housing.trainFilename), hasHeader: true);

            // Define a pipeline
            var pipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.Regression.Trainers.Sdca(
                    new SdcaRegressionTrainer.Options { NumberOfThreads = 1, MaximumNumberOfIterations = 20 }));

            // Fit the pipeline to the data.
            var model = pipeline.Fit(data);

            // Inspect the model schema, and verify that a Score column is produced.
            var outputSchema = model.GetOutputSchema(data.Schema);
            var columnNames = new string[outputSchema.Count];
            int i = 0;
            foreach (var column in outputSchema)
                columnNames[i++] = column.Name;
            Assert.Contains("Score", columnNames);
        }

        /// <summary>
        /// Debugging: The schema read in can be verified by inspecting the data.
        /// </summary>
        [Fact]
        public void InspectSchemaUponLoadingData()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset.
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(GetDataPath(TestDatasets.housing.trainFilename), hasHeader: true);

            // Verify the column names.
            int i = 0;
            foreach (var column in data.Schema)
            {
                if (i == 0)
                    Assert.Equal("Label", column.Name);
                else
                    Assert.Equal(HousingRegression.Features[i-1], column.Name);
                i++;
            }

            // Verify that I can cast it to the right schema by inspecting the first row.
            foreach (var row in mlContext.Data.CreateEnumerable<HousingRegression>(mlContext.Data.TakeRows(data, 1), true))
            {
                // Validate there was data in the row by checking that some values were not zero since zero is the default.
                var rowSum = row.MedianHomeValue;
                foreach (var property in HousingRegression.Features)
                    rowSum += (float) row.GetType().GetProperty(property).GetValue(row, null);

                Assert.NotEqual(0, rowSum);
            }
        }

        /// <summary>
        /// Debugging: The progress of training can be accessed.
        /// </summary>
        [Fact]
        public void ViewTrainingOutput()
        {
            var mlContext = new MLContext(seed: 1);

            // Attach a listener.
            var logWatcher = new LogWatcher();
            mlContext.Log += logWatcher.ObserveEvent;

            // Get the dataset.
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(GetDataPath(TestDatasets.housing.trainFilename), hasHeader: true);

            // Define a pipeline
            var pipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.Regression.Trainers.Sdca(
                    new SdcaRegressionTrainer.Options { NumberOfThreads = 1, MaximumNumberOfIterations = 20 }));

            // Fit the pipeline to the data.
            var model = pipeline.Fit(data);

            // Validate that we can read lines from the file.
            var expectedLines = new string[3] {
                @"[Source=SdcaTrainerBase; Training, Kind=Info] Auto-tuning parameters: L2 = 0.001.",
                @"[Source=SdcaTrainerBase; Training, Kind=Info] Auto-tuning parameters: L1Threshold (L1/L2) = 0.",
                @"[Source=SdcaTrainerBase; Training, Kind=Info] Using best model from iteration 7."};
            foreach (var line in expectedLines)
            {
                Assert.Contains(line, logWatcher.Lines);
                Assert.Equal(1, logWatcher.Lines[line]);
            }
        }

        internal class LogWatcher {

            public readonly IDictionary<string, int> Lines;

            public LogWatcher()
            {
                Lines = new Dictionary<string, int>();
            }
            
            public void ObserveEvent(object sender, LoggingEventArgs e)
            {
                if (Lines.ContainsKey(e.Message))
                    Lines[e.Message]++;
                else
                    Lines[e.Message] = 1;
            }
        }
    }
}
