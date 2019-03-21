﻿using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic.Trainers.BinaryClassification
{
    public static class FastForest
    {
        // This example requires installation of additional NuGet package
        // <a href="https://www.nuget.org/packages/Microsoft.ML.FastTree/">Microsoft.ML.FastTree</a>.
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

            // Define the trainer.
            var pipeline = mlContext.BinaryClassification.Trainers.FastForest();

            // Train the model.
            var model = pipeline.Fit(trainingData);

            // Create testing data. Use different random seed to make it different from training data.
            var testData = mlContext.Data.LoadFromEnumerable(GenerateRandomDataPoints(500, seed:123));

            // Run the model on test data set.
            var transformedTestData = model.Transform(testData);

            // Convert IDataView object to a list.
            var predictions = mlContext.Data.CreateEnumerable<Prediction>(transformedTestData, reuseRowObject: false).ToList();

            // Look at 5 predictions
            foreach (var p in predictions.Take(5))
                Console.WriteLine($"Label: {p.Label}, Prediction: {p.PredictedLabel}");

            // Expected output:
            //   Label: True, Prediction: True
            //   Label: False, Prediction: False
            //   Label: True, Prediction: True
            //   Label: True, Prediction: True
            //   Label: False, Prediction: False
            
            // Evaluate the overall metrics
            var metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(transformedTestData);
            SamplesUtils.ConsoleUtils.PrintMetrics(metrics);
            
            // Expected output:
            //   Accuracy: 0.74
            //   AUC: 0.83
            //   F1 Score: 0.74
            //   Negative Precision: 0.78
            //   Negative Recall: 0.71
            //   Positive Precision: 0.71
            //   Positive Recall: 0.78
        }

        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count, int seed=0)
        {
            var random = new Random(seed);
            float randomFloat() => (float)random.NextDouble();
            for (int i = 0; i < count; i++)
            {
                var label = randomFloat() > 0.5f;
                yield return new DataPoint
                {
                    Label = label,
                    // Create random features that are correlated with the label.
                    // For data points with false label, the feature values are slightly increased by adding a constant.
                    Features = Enumerable.Repeat(label, 50).Select(x => x ? randomFloat() : randomFloat() + 0.03f).ToArray()
                };
            }
        }

        // Example with label and 50 feature values. A data set is a collection of such examples.
        private class DataPoint
        {
            public bool Label { get; set; }
            [VectorType(50)]
            public float[] Features { get; set; }
        }

        // Class used to capture predictions.
        private class Prediction
        {
            // Original label.
            public bool Label { get; set; }
            // Predicted label from the trainer.
            public bool PredictedLabel { get; set; }
        }
    }
}

