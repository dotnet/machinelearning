﻿using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace Samples.Dynamic.Trainers.Clustering
{
    public static class KMeansWithOptions
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            // Setting the seed to a fixed number in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Convert the list of data points to an IDataView object, which is consumable by ML.NET API.
            var dataPoints = GenerateRandomDataPoints(1000, 0);

            // Convert the list of data points to an IDataView object, which is consumable by ML.NET API.
            var trainingData = mlContext.Data.LoadFromEnumerable(dataPoints);

            // Define trainer options.
            var options = new KMeansTrainer.Options
            {
                NumberOfClusters = 2,
                OptimizationTolerance = 1e-6f,
                NumberOfThreads = 1
            };

            // Define the trainer.
            var pipeline = mlContext.Clustering.Trainers.KMeans(options);

            // Train the model.
            var model = pipeline.Fit(trainingData);

            // Create testing data. Use different random seed to make it different from training data.
            var testData = mlContext.Data.LoadFromEnumerable(GenerateRandomDataPoints(500, seed: 123));

            // Run the model on test data set.
            var transformedTestData = model.Transform(testData);

            // Convert IDataView object to a list.
            var predictions = mlContext.Data.CreateEnumerable<Prediction>(transformedTestData, reuseRowObject: false).ToList();

            // Print 5 predictions. Note that the label is only used as a comparison wiht the predicted label.
            // It is not used during training.
            foreach (var p in predictions.Take(2))
                Console.WriteLine($"Label: {p.Label}, Prediction: {p.PredictedLabel}");
            foreach (var p in predictions.TakeLast(3))
                Console.WriteLine($"Label: {p.Label}, Prediction: {p.PredictedLabel}");

            // Expected output:
            //   Label: 1, Prediction: 1
            //   Label: 1, Prediction: 1
            //   Label: 2, Prediction: 2
            //   Label: 2, Prediction: 2
            //   Label: 2, Prediction: 2

            // Evaluate the overall metrics
            var metrics = mlContext.Clustering.Evaluate(transformedTestData, "Label", "Score", "Features");
            PrintMetrics(metrics);
            
            // Expected output:
            //   Normalized Mutual Information: 0.92
            //   Average Distance: 4.18
            //   Davies Bouldin Index: 2.87

            // Get cluster centroids and the number of clusters k from KMeansModelParameters.
            VBuffer<float>[] centroids = default;

            var modelParams = model.Model;
            modelParams.GetClusterCentroids(ref centroids, out int k);
            Console.WriteLine($"The first 3 coordinates of the first centroid are: ({string.Join(", ", centroids[0].GetValues().ToArray().Take(3))})");
            Console.WriteLine($"The first 3 coordinates of the second centroid are: ({string.Join(", ", centroids[1].GetValues().ToArray().Take(3))})");

            // Expected output:
            //   The first 3 coordinates of the first centroid are: (0.5840713, 0.5678288, 0.6221277)
            //   The first 3 coordinates of the second centroid are: (0.3705794, 0.4289133, 0.4001645)
        }

        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count, int seed = 0)
        {
            var random = new Random(seed);
            float randomFloat() => (float)random.NextDouble();
            for (int i = 0; i < count; i++)
            {
                int label = i < count / 2 ? 0 : 1;
                yield return new DataPoint
                {
                    Label = (uint)label,
                    // Create random features with two clusters.
                    // The first half has feature values centered around 0.6 the second half has values centered around 0.4.
                    Features = Enumerable.Repeat(label, 50).Select(index => label == 0 ? randomFloat() + 0.1f : randomFloat() - 0.1f).ToArray()
                };
            }
        }

        // Example with label and 50 feature values. A data set is a collection of such examples.
        private class DataPoint
        {
            // The label is not used during training, just for comparison with the predicted label.
            [KeyType(2)]
            public uint Label { get; set; }

            [VectorType(50)]
            public float[] Features { get; set; }
        }

        // Class used to capture predictions.
        private class Prediction
        {
            // Original label (not used during training, just for comparison).
            public uint Label { get; set; }
            // Predicted label from the trainer.
            public uint PredictedLabel { get; set; }
        }

        // Pretty-print of ClusteringMetrics object.
        private static void PrintMetrics(ClusteringMetrics metrics)
        {
            Console.WriteLine($"Normalized Mutual Information: {metrics.NormalizedMutualInformation:F2}");
            Console.WriteLine($"Average Distance: {metrics.AverageDistance:F2}");
            Console.WriteLine($"Davies Bouldin Index: {metrics.DaviesBouldinIndex:F2}");
        }
    }
}
