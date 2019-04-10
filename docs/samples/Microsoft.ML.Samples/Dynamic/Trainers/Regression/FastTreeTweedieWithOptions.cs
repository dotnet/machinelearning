﻿using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;

namespace Samples.Dynamic.Trainers.Regression
{
    public static class FastTreeTweedieWithOptions
    {
        // This example requires installation of additional NuGet package
        // <a href="https://www.nuget.org/packages/Microsoft.ML.FastTree/">Microsoft.ML.FastTree</a>.
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            // Setting the seed to a fixed number in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Create a list of training examples.
            var examples = GenerateRandomDataPoints(1000);

            // Convert the examples list to an IDataView object, which is consumable by ML.NET API.
            var trainingData = mlContext.Data.LoadFromEnumerable(examples);

            // Define trainer options.
            var options = new FastTreeTweedieTrainer.Options
            {
                // Use L2Norm for early stopping.
                EarlyStoppingMetric = EarlyStoppingMetric.L2Norm,
                // Create a simpler model by penalizing usage of new features.
                FeatureFirstUsePenalty = 0.1,
                // Reduce the number of trees to 50.
                NumberOfTrees = 50
            };

            // Define the trainer.
            var pipeline = mlContext.Regression.Trainers.FastTreeTweedie(options);

            // Train the model.
            var model = pipeline.Fit(trainingData);

            // Create testing examples. Use different random seed to make it different from training data.
            var testData = mlContext.Data.LoadFromEnumerable(GenerateRandomDataPoints(500, seed:123));

            // Run the model on test data set.
            var transformedTestData = model.Transform(testData);

            // Convert IDataView object to a list.
            var predictions = mlContext.Data.CreateEnumerable<Prediction>(transformedTestData, reuseRowObject: false).ToList();

            // Look at 5 predictions
            foreach (var p in predictions.Take(5))
                Console.WriteLine($"Label: {p.Label:F3}, Prediction: {p.Score:F3}");

            // Expected output:
            //   Label: 0.985, Prediction: 0.954
            //   Label: 0.155, Prediction: 0.103
            //   Label: 0.515, Prediction: 0.450
            //   Label: 0.566, Prediction: 0.515
            //   Label: 0.096, Prediction: 0.078

            // Evaluate the overall metrics
            var metrics = mlContext.Regression.Evaluate(transformedTestData);
            Microsoft.ML.SamplesUtils.ConsoleUtils.PrintMetrics(metrics);

            // Expected output:
            //   Mean Absolute Error: 0.05
            //   Mean Squared Error: 0.00
            //   Root Mean Squared Error: 0.07
            //   RSquared: 0.95
        }

        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count, int seed=0)
        {
            var random = new Random(seed);
            float randomFloat() => (float)random.NextDouble();
            for (int i = 0; i < count; i++)
            {
                var label = randomFloat();
                yield return new DataPoint
                {
                    Label = label,
                    // Create random features that are correlated with label.
                    Features = Enumerable.Repeat(label, 50).Select(x => x + randomFloat()).ToArray()
                };
            }
        }

        // Example with label and 50 feature values. A data set is a collection of such examples.
        private class DataPoint
        {
            public float Label { get; set; }
            [VectorType(50)]
            public float[] Features { get; set; }
        }

        // Class used to capture predictions.
        private class Prediction
        {
            // Original label.
            public float Label { get; set; }
            // Predicted score from the trainer.
            public float Score { get; set; }
        }
    }
}