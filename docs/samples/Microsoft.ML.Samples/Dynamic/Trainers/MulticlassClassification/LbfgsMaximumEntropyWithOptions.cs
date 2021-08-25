using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace Samples.Dynamic.Trainers.MulticlassClassification
{
    public static class LbfgsMaximumEntropyWithOptions
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
            var options = new LbfgsMaximumEntropyMulticlassTrainer.Options
            {
                HistorySize = 50,
                L1Regularization = 0.1f,
                NumberOfThreads = 1
            };

            // Define the trainer.
            var pipeline =
                // Convert the string labels into key types.
                mlContext.Transforms.Conversion.MapValueToKey("Label")
                // Apply LbfgsMaximumEntropy multiclass trainer.
                .Append(mlContext.MulticlassClassification.Trainers
                .LbfgsMaximumEntropy(options));

            // Train the model.
            var model = pipeline.Fit(trainingData);

            // Create testing data. Use different random seed to make it different
            // from training data.
            var testData = mlContext.Data
                .LoadFromEnumerable(GenerateRandomDataPoints(500, seed: 123));

            // Run the model on test data set.
            var transformedTestData = model.Transform(testData);

            // Convert IDataView object to a list.
            var predictions = mlContext.Data
                .CreateEnumerable<Prediction>(transformedTestData,
                reuseRowObject: false).ToList();

            // Look at 5 predictions
            foreach (var p in predictions.Take(5))
                Console.WriteLine($"Label: {p.Label}, " +
                    $"Prediction: {p.PredictedLabel}");

            // Expected output:
            //   Label: 1, Prediction: 1
            //   Label: 2, Prediction: 2
            //   Label: 3, Prediction: 2
            //   Label: 2, Prediction: 2
            //   Label: 3, Prediction: 3

            // Evaluate the overall metrics
            var metrics = mlContext.MulticlassClassification
                .Evaluate(transformedTestData);

            PrintMetrics(metrics);

            // Expected output:
            //   Micro Accuracy: 0.91
            //   Macro Accuracy: 0.91
            //   Log Loss: 0.22
            //   Log Loss Reduction: 0.80

            //   Confusion table
            //             ||========================
            //   PREDICTED ||     0 |     1 |     2 | Recall
            //   TRUTH     ||========================
            //           0 ||   147 |     0 |    13 | 0.9188
            //           1 ||     0 |   165 |    12 | 0.9322
            //           2 ||    11 |     7 |   145 | 0.8896
            //             ||========================
            //   Precision ||0.9304 |0.9593 |0.8529 |
        }

        // Generates random uniform doubles in [-0.5, 0.5)
        // range with labels 1, 2 or 3.
        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count,
            int seed = 0)

        {
            var random = new Random(seed);
            float randomFloat() => (float)(random.NextDouble() - 0.5);
            for (int i = 0; i < count; i++)
            {
                // Generate Labels that are integers 1, 2 or 3
                var label = random.Next(1, 4);
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

        // Pretty-print MulticlassClassificationMetrics objects.
        public static void PrintMetrics(MulticlassClassificationMetrics metrics)
        {
            Console.WriteLine($"Micro Accuracy: {metrics.MicroAccuracy:F2}");
            Console.WriteLine($"Macro Accuracy: {metrics.MacroAccuracy:F2}");
            Console.WriteLine($"Log Loss: {metrics.LogLoss:F2}");
            Console.WriteLine(
                $"Log Loss Reduction: {metrics.LogLossReduction:F2}\n");

            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
        }
    }
}

