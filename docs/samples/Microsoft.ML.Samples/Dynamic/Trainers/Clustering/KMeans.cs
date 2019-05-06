using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic.Trainers.Clustering
{
    public static class KMeans
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available operations
            // and as the source of randomness. Setting the seed to a fixed number
            // in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Create a list of training data points.
            var dataPoints = GenerateRandomDataPoints(1000, 123);

            // Convert the list of data points to an IDataView object, which is
            // consumable by ML.NET API.
            IDataView trainingData = mlContext.Data.LoadFromEnumerable(dataPoints);

            // Define the trainer.
            var pipeline = mlContext.Clustering.Trainers.KMeans(
                numberOfClusters: 2);

            // Train the model.
            var model = pipeline.Fit(trainingData);

            // Create testing data. Use a different random seed to make it different
            // from the training data.
            var testData = mlContext.Data.LoadFromEnumerable(
                GenerateRandomDataPoints(500, seed: 123));

            // Run the model on test data set.
            var transformedTestData = model.Transform(testData);

            // Convert IDataView object to a list.
            var predictions = mlContext.Data.CreateEnumerable<Prediction>(
                transformedTestData, reuseRowObject: false).ToList();

            // Print 5 predictions. Note that the label is only used as a comparison
            // with the predicted label. It is not used during training.
            foreach (var p in predictions.Take(2))
                Console.WriteLine(
                    $"Label: {p.Label}, Prediction: {p.PredictedLabel}");

            foreach (var p in predictions.TakeLast(3))
                Console.WriteLine(
                    $"Label: {p.Label}, Prediction: {p.PredictedLabel}");

            // Expected output:
            //   Label: 1, Prediction: 1
            //   Label: 1, Prediction: 1
            //   Label: 2, Prediction: 2
            //   Label: 2, Prediction: 2
            //   Label: 2, Prediction: 2

            // Evaluate the overall metrics
            var metrics = mlContext.Clustering.Evaluate(
                transformedTestData, "Label", "Score", "Features");

            PrintMetrics(metrics);

            // Expected output:
            //   Normalized Mutual Information: 0.95
            //   Average Distance: 4.17
            //   Davies Bouldin Index: 2.87

            // Get the cluster centroids and the number of clusters k from
            // KMeansModelParameters.
            VBuffer<float>[] centroids = default;

            var modelParams = model.Model;
            modelParams.GetClusterCentroids(ref centroids, out int k);
            Console.WriteLine(
                $"The first 3 coordinates of the first centroid are: " +
                string.Join(", ", centroids[0].GetValues().ToArray().Take(3)));

            Console.WriteLine(
                $"The first 3 coordinates of the second centroid are: " +
                string.Join(", ", centroids[1].GetValues().ToArray().Take(3)));

            // Expected output similar to:
            //   The first 3 coordinates of the first centroid are: (0.6035213, 0.6017533, 0.5964218)
            //   The first 3 coordinates of the second centroid are: (0.4031044, 0.4175443, 0.4082336)
        }

        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count,
            int seed = 0)
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
                    // The first half has feature values centered around 0.6, while
                    // the second half has values centered around 0.4.
                    Features = Enumerable.Repeat(label, 50)
                        .Select(index => label == 0 ? randomFloat() + 0.1f :
                            randomFloat() - 0.1f).ToArray()
                };
            }
        }

        // Example with label and 50 feature values. A data set is a collection of
        // such examples.
        private class DataPoint
        {
            // The label is not used during training, just for comparison with the
            // predicted label.
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
            Console.WriteLine($"Normalized Mutual Information: " +
                $"{metrics.NormalizedMutualInformation:F2}");

            Console.WriteLine($"Average Distance: " +
                $"{metrics.AverageDistance:F2}");

            Console.WriteLine($"Davies Bouldin Index: " +
                $"{metrics.DaviesBouldinIndex:F2}");
        }
    }
}
