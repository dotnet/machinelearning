using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public static class LogLossPerClass
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

            // Define the trainer.
            var pipeline =
                // Convert the string labels into key types.
                mlContext.Transforms.Conversion
                .MapValueToKey(nameof(DataPoint.Label))
                // Apply a multiclass trainer.
                .Append(mlContext.MulticlassClassification.Trainers
                .LightGbm());

            // Train the model.
            var model = pipeline.Fit(trainingData);

            // Create testing data. Use different random seed to make it different
            // from training data.
            var testData = mlContext.Data
                .LoadFromEnumerable(GenerateRandomDataPoints(500, seed: 123));

            // Run the model on test data set.
            var transformedTestData = model.Transform(testData);

            // Evaluate the overall metrics
            var metrics = mlContext.MulticlassClassification
                .Evaluate(transformedTestData);

            // Find the original label values.
            VBuffer<uint> keys = default;
            transformedTestData.Schema["PredictedLabel"].GetKeyValues(ref keys);
            var originalLabels = keys.DenseValues().ToArray();
            for (var i = 0; i < originalLabels.Length; i++)
                Console.WriteLine($"LogLoss for label " +
                    $"{originalLabels[i]}: {metrics.PerClassLogLoss[i]:F4}");

            // Expected output:
            //   LogLoss for label 7: 0.2578
            //   LogLoss for label 8: 0.2504
            //   LogLoss for label 2: 0.1121
            //   LogLoss for label 9: 0.2229
            //   LogLoss for label 6: 0.1737
            //   LogLoss for label 1: 0.2645
            //   LogLoss for label 3: 0.2235
            //   LogLoss for label 5: 0.1128
            //   LogLoss for label 4: 0.1442
        }

        // Generates data points with random features and labels 1 to 9.
        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count,
            int seed = 0)

        {
            var random = new Random(seed);
            float randomFloat() => (float)(random.NextDouble() - 0.5);
            for (int i = 0; i < count; i++)
            {
                // Generate Labels that are integers 1, 2 or 3
                var label = random.Next(1, 10);
                yield return new DataPoint
                {
                    Label = (uint)label,
                    // Create random features that are correlated with the label.
                    // The feature values are slightly increased by adding a
                    // constant multiple of label.
                    Features = Enumerable.Repeat(label, 20)
                        .Select(x => randomFloat() + label * 0.2f).ToArray()

                };
            }
        }

        // Example with label and 20 feature values. A data set is a collection of
        // such examples.
        private class DataPoint
        {
            public uint Label { get; set; }
            [VectorType(20)]
            public float[] Features { get; set; }
        }

        // Class used to capture predictions.
        private class Prediction
        {
            // Original label.
            public uint Label { get; set; }
            // Predicted label from the trainer.
            public uint PredictedLabel { get; set; }
        }
    }
}
