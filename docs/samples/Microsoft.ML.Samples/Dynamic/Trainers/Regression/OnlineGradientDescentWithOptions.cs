using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace Samples.Dynamic.Trainers.Regression
{
    public static class OnlineGradientDescentWithOptions
    {
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
            var options = new OnlineGradientDescentTrainer.Options
            {
                LabelColumnName = nameof(DataPoint.Label),
                FeatureColumnName = nameof(DataPoint.Features),
                // Change the loss function.
                LossFunction = new TweedieLoss(),
                // Give an extra gain to more recent updates.
                RecencyGain = 0.1f,
                // Turn off lazy updates.
                LazyUpdate = false,
                // Specify scale for initial weights.
                InitialWeightsDiameter = 0.2f
            };

            // Define the trainer.
            var pipeline =
                mlContext.Regression.Trainers.OnlineGradientDescent(options);

            // Train the model.
            var model = pipeline.Fit(trainingData);

            // Create testing data. Use different random seed to make it different
            // from training data.
            var testData = mlContext.Data.LoadFromEnumerable(
                GenerateRandomDataPoints(5, seed: 123));

            // Run the model on test data set.
            var transformedTestData = model.Transform(testData);

            // Convert IDataView object to a list.
            var predictions = mlContext.Data.CreateEnumerable<Prediction>(
                transformedTestData, reuseRowObject: false).ToList();

            // Look at 5 predictions for the Label, side by side with the actual
            // Label for comparison.
            foreach (var p in predictions)
                Console.WriteLine($"Label: {p.Label:F3}, Prediction: {p.Score:F3}");

            // This trainer is not numerically stable.
            // Please see issue #2425.

            // Evaluate the overall metrics
            var metrics = mlContext.Regression.Evaluate(transformedTestData);
            PrintMetrics(metrics);

            // This trainer is not numerically stable. Please see
            // issue #2425.
        }

        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count,
            int seed = 0)
        {
            var random = new Random(seed);
            for (int i = 0; i < count; i++)
            {
                float label = (float)random.NextDouble();
                yield return new DataPoint
                {
                    Label = label,
                    // Create random features that are correlated with the label.
                    Features = Enumerable.Repeat(label, 50).Select(
                        x => x + (float)random.NextDouble()).ToArray()
                };
            }
        }

        // Example with label and 50 feature values. A data set is a collection of
        // such examples.
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
            Console.WriteLine("Mean Absolute Error: " + metrics.MeanAbsoluteError);
            Console.WriteLine("Mean Squared Error: " + metrics.MeanSquaredError);
            Console.WriteLine(
                "Root Mean Squared Error: " + metrics.RootMeanSquaredError);

            Console.WriteLine("RSquared: " + metrics.RSquared);
        }
    }
}

