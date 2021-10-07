using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace Samples.Dynamic.Trainers.BinaryClassification
{
    public static class SdcaLogisticRegressionWithOptions
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

            // ML.NET doesn't cache data set by default. Therefore, if one reads a
            // data set from a file and accesses it many times, it can be slow due
            // to expensive featurization and disk operations. When the considered
            // data can fit into memory, a solution is to cache the data in memory.
            // Caching is especially helpful when working with iterative algorithms 
            // which needs many data passes.
            trainingData = mlContext.Data.Cache(trainingData);

            // Define trainer options.
            var options = new SdcaLogisticRegressionBinaryTrainer.Options()
            {
                // Make the convergence tolerance tighter.
                ConvergenceTolerance = 0.05f,
                // Increase the maximum number of passes over training data.
                MaximumNumberOfIterations = 30,
                // Give the instances of the positive class slightly more weight.
                PositiveInstanceWeight = 1.2f,
            };

            // Define the trainer.
            var pipeline = mlContext.BinaryClassification.Trainers
                .SdcaLogisticRegression(options);

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

            // Print 5 predictions.
            foreach (var p in predictions.Take(5))
                Console.WriteLine($"Label: {p.Label}, "
                    + $"Prediction: {p.PredictedLabel}");

            // Expected output:
            //   Label: True, Prediction: True
            //   Label: False, Prediction: False
            //   Label: True, Prediction: True
            //   Label: True, Prediction: True
            //   Label: False, Prediction: True

            // Evaluate the overall metrics.
            var metrics = mlContext.BinaryClassification
                .Evaluate(transformedTestData);

            PrintMetrics(metrics);

            // Expected output:
            //   Accuracy: 0.60
            //   AUC: 0.67
            //   F1 Score: 0.65
            //   Negative Precision: 0.69
            //   Negative Recall: 0.45
            //   Positive Precision: 0.56
            //   Positive Recall: 0.77
            //   TEST POSITIVE RATIO:    0.4760 (238.0/(238.0+262.0))
            //
            //   Confusion table
            //             ||======================
            //   PREDICTED || positive | negative | Recall
            //   TRUTH     ||======================
            //    positive ||      165 |       73 | 0.6933
            //    negative ||      112 |      150 | 0.5725
            //             ||======================
            //   Precision ||   0.5957 |   0.6726 |
        }

        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count,
            int seed = 0)

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
                    // For data points with false label, the feature values are
                    // slightly increased by adding a constant.
                    Features = Enumerable.Repeat(label, 50)
                        .Select(x => x ? randomFloat() : randomFloat() +
                        0.03f).ToArray()

                };
            }
        }

        // Example with label and 50 feature values. A data set is a collection of
        // such examples.
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

        // Pretty-print BinaryClassificationMetrics objects.
        private static void PrintMetrics(BinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"Accuracy: {metrics.Accuracy:F2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F2}");
            Console.WriteLine($"Negative Precision: " +
                $"{metrics.NegativePrecision:F2}");

            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall:F2}");
            Console.WriteLine($"Positive Precision: " +
                $"{metrics.PositivePrecision:F2}");

            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:F2}\n");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
        }
    }
}

