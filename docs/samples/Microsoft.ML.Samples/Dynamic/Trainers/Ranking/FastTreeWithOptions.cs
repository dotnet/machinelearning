using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;

namespace Samples.Dynamic.Trainers.Ranking
{
    public static class FastTreeWithOptions
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
            var options = new FastTreeRankingTrainer.Options
            {
                // Use NdcgAt3 for early stopping.
                EarlyStoppingMetric = EarlyStoppingRankingMetric.NdcgAt3,
                // Create a simpler model by penalizing usage of new features.
                FeatureFirstUsePenalty = 0.1,
                // Reduce the number of trees to 50.
                NumberOfTrees = 50,
                // Specify the row group column name.
                RowGroupColumnName = "GroupId"
            };

            // Define the trainer.
            var pipeline = mlContext.Ranking.Trainers.FastTree(options);

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
            //   Label: 5, Score: 8.807633
            //   Label: 1, Score: -10.71331
            //   Label: 3, Score: -8.134147
            //   Label: 3, Score: -6.545538
            //   Label: 1, Score: -10.27982

            // Evaluate the overall metrics.
            var metrics = mlContext.Ranking.Evaluate(transformedTestData);
            PrintMetrics(metrics);

            // Expected output:
            //   DCG: @1:40.57, @2:61.21, @3:74.11
            //   NDCG: @1:0.96, @2:0.95, @3:0.97
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
