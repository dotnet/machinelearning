using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic.Trainers.MulticlassClassification
{
    public static class NaiveBayes
    {
        // Naive Bayes classifier is based on Bayes' theorem. 
        // It assumes independence among the presence of features in a class even
        // though they may be dependent on each other. It is  a multi-class trainer
        // that accepts binary feature values of type float, i.e., feature values
        // are either true or false. Specifically a feature value greater than zero
        // is treated as true, zero or less is treated as false.
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
                // Apply NaiveBayes multiclass trainer.
                .Append(mlContext.MulticlassClassification.Trainers
                .NaiveBayes());

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
            //   Label: 3, Prediction: 3
            //   Label: 2, Prediction: 2
            //   Label: 3, Prediction: 3

            // Evaluate the overall metrics
            var metrics = mlContext.MulticlassClassification
                .Evaluate(transformedTestData);

            PrintMetrics(metrics);

            // Expected output:
            //   Micro Accuracy: 0.88
            //   Macro Accuracy: 0.88
            //   Log Loss: 34.54
            //   Log Loss Reduction: -30.47

            //   Confusion table
            //             ||========================
            //   PREDICTED ||     0 |     1 |     2 | Recall
            //   TRUTH     ||========================
            //           0 ||   160 |     0 |     0 | 1.0000
            //           1 ||     0 |   145 |    32 | 0.8192
            //           2 ||     9 |    21 |   133 | 0.8160
            //             ||========================
            //   Precision ||0.9467 |0.8735 |0.8061 |
        }


        // Generates random uniform doubles in [-0.5, 0.5) range with labels
        // 1, 2 or 3. For NaiveBayes values greater than zero are treated as true,
        // zero or less are treated as false.
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

