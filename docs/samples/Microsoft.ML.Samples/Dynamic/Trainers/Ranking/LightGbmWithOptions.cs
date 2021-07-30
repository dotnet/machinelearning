using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

namespace Samples.Dynamic.Trainers.Ranking
{
    public static class LightGbmWithOptions
    {
        // This example requires installation of additional NuGet package for 
        // Microsoft.ML.FastTree at
        // https://www.nuget.org/packages/Microsoft.ML.FastTree/
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available operations
            // and as the source of randomness. Setting the seed to a fixed number
            // in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Create a list of training data points.
            var dataPoints = GenerateRandomDataPoints(1000);

            // Convert the list of data points to an IDataView object, which is
            // consumable by ML.NET API.
            var trainingData = mlContext.Data.LoadFromEnumerable(dataPoints);

            // Define trainer options.
            var options = new LightGbmRankingTrainer.Options
            {
                NumberOfLeaves = 4,
                MinimumExampleCountPerGroup = 10,
                LearningRate = 0.1,
                NumberOfIterations = 2,
                Booster = new GradientBooster.Options
                {
                    FeatureFraction = 0.9
                },
                RowGroupColumnName = "GroupId"
            };

            // Define the trainer.
            var pipeline = mlContext.Ranking.Trainers.LightGbm(options);

            // Train the model.
            var model = pipeline.Fit(trainingData);

            // Create testing data. Use different random seed to make it different
            // from training data.
            var testData = mlContext.Data.LoadFromEnumerable(
                GenerateRandomDataPoints(500, seed: 123));

            // Run the model on test data set.
            var transformedTestData = model.Transform(testData);

            // Take the top 5 rows.
            var topTransformedTestData = mlContext.Data.TakeRows(
                transformedTestData, 5);

            // Convert IDataView object to a list.
            var predictions = mlContext.Data.CreateEnumerable<Prediction>(
                topTransformedTestData, reuseRowObject: false).ToList();

            // Print 5 predictions.
            foreach (var p in predictions)
                Console.WriteLine($"Label: {p.Label}, Score: {p.Score}");

            // Expected output:
            //   Label: 5, Score: 0.05836755
            //   Label: 1, Score: -0.06531862
            //   Label: 3, Score: -0.004557075
            //   Label: 3, Score: -0.009396422
            //   Label: 1, Score: -0.05871891

            // Evaluate the overall metrics.
            var metrics = mlContext.Ranking.Evaluate(transformedTestData);
            PrintMetrics(metrics);

            // Expected output:
            //   DCG: @1:28.83, @2:46.36, @3:56.18
            //   NDCG: @1:0.69, @2:0.72, @3:0.74
        }

        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count,
            int seed = 0, int groupSize = 10)
        {
            var random = new Random(seed);
            float randomFloat() => (float)random.NextDouble();
            for (int i = 0; i < count; i++)
            {
                var label = random.Next(0, 5);
                yield return new DataPoint
                {
                    Label = (uint)label,
                    GroupId = (uint)(i / groupSize),
                    // Create random features that are correlated with the label.
                    // For data points with larger labels, the feature values are
                    // slightly increased by adding a constant.
                    Features = Enumerable.Repeat(label, 50).Select(
                        x => randomFloat() + x * 0.1f).ToArray()
                };
            }
        }

        // Example with label, groupId, and 50 feature values. A data set is a
        // collection of such examples.
        private class DataPoint
        {
            [KeyType(5)]
            public uint Label { get; set; }
            [KeyType(100)]
            public uint GroupId { get; set; }
            [VectorType(50)]
            public float[] Features { get; set; }
        }

        // Class used to capture predictions.
        private class Prediction
        {
            // Original label.
            public uint Label { get; set; }
            // Score produced from the trainer.
            public float Score { get; set; }
        }

        // Pretty-print RankerMetrics objects.
        public static void PrintMetrics(RankingMetrics metrics)
        {
            Console.WriteLine("DCG: " + string.Join(", ",
                metrics.DiscountedCumulativeGains.Select(
                    (d, i) => (i + 1) + ":" + d + ":F2").ToArray()));
            Console.WriteLine("NDCG: " + string.Join(", ",
                metrics.NormalizedDiscountedCumulativeGains.Select(
                    (d, i) => (i + 1) + ":" + d + ":F2").ToArray()));
        }
    }
}
