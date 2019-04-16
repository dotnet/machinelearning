﻿using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

namespace Samples.Dynamic.Trainers.Regression
{
    public static class LightGbmWithOptions
    {
        // This example requires installation of additional NuGet package
        // <a href="https://www.nuget.org/packages/Microsoft.ML.LightGbm/">Microsoft.ML.LightGBM</a>. 
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            // Setting the seed to a fixed number in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Create a list of training data points.
            var dataPoints = GenerateRandomDataPoints(1000);

            // Convert the list of data points to an IDataView object, which is consumable by ML.NET API.
            var trainingData = mlContext.Data.LoadFromEnumerable(dataPoints);

            // Define trainer options.
            var options = new LightGbmRegressionTrainer.Options
            {
                LabelColumnName = nameof(DataPoint.Label),
                FeatureColumnName = nameof(DataPoint.Features),
                // How many leaves a single tree should have.
                NumberOfLeaves = 4,
                // Each leaf contains at least this number of training data points.
                MinimumExampleCountPerLeaf = 6,
                // The step size per update. Using a large value might reduce the
                // training time but also increase the algorithm's numerical stability.
                LearningRate = 0.001,
                Booster = new Microsoft.ML.Trainers.LightGbm.GossBooster.Options()
                {
                    TopRate = 0.3,
                    OtherRate = 0.2
               }
            };

            // Define the trainer.
            var pipeline = mlContext.Regression.Trainers.LightGbm(options);

            // Train the model.
            var model = pipeline.Fit(trainingData);

            // Create testing data. Use different random seed to make it different from training data.
            var testData = mlContext.Data.LoadFromEnumerable(GenerateRandomDataPoints(5, seed: 123));

            // Run the model on test data set.
            var transformedTestData = model.Transform(testData);

            // Convert IDataView object to a list.
            var predictions = mlContext.Data.CreateEnumerable<Prediction>(transformedTestData, reuseRowObject: false).ToList();

            // Look at 5 predictions for the Label, side by side with the actual Label for comparison.
            foreach (var p in predictions)
                Console.WriteLine($"Label: {p.Label:F3}, Prediction: {p.Score:F3}");

            // Expected output:
            //   Label: 0.985, Prediction: 0.866
            //   Label: 0.155, Prediction: 0.171
            //   Label: 0.515, Prediction: 0.470
            //   Label: 0.566, Prediction: 0.476
            //   Label: 0.096, Prediction: 0.140

            // Evaluate the overall metrics
            var metrics = mlContext.Regression.Evaluate(transformedTestData);
            PrintMetrics(metrics);

            // Expected output:
            //   Mean Absolute Error: 0.04
            //   Mean Squared Error: 0.00
            //   Root Mean Squared Error: 0.06
            //   RSquared: 0.97 (closer to 1 is better. The worest case is 0)
        }

        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count, int seed=0)
        {
            var random = new Random(seed);
            for (int i = 0; i < count; i++)
            {
                float label = (float)random.NextDouble();
                yield return new DataPoint
                {
                    Label = label,
                    // Create random features that are correlated with the label.
                    Features = Enumerable.Repeat(label, 50).Select(x => x + (float)random.NextDouble()).ToArray()
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

        // Print some evaluation metrics to regression problems.
        private static void PrintMetrics(RegressionMetrics metrics)
        {
            Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError:F2}");
            Console.WriteLine($"Mean Squared Error: {metrics.MeanSquaredError:F2}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError:F2}");
            Console.WriteLine($"RSquared: {metrics.RSquared:F2}");
        }
    }
}

