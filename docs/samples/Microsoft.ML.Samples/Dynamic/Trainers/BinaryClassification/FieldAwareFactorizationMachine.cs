using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic.Trainers.BinaryClassification
{
    public static class FieldAwareFactorizationMachine
    {
        // This example first train a field-aware factorization to binary
        // classification, measure the trained model's quality, and finally
        // use the trained model to make prediction.
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available operations
            // and as the source of randomness. Setting the seed to a fixed number
            // in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Create a list of training data points.
            IEnumerable<DataPoint> data = GenerateRandomDataPoints(500);

            // Convert the list of data points to an IDataView object, which is
            // consumable by ML.NET API.
            var trainingData = mlContext.Data.LoadFromEnumerable(data);

            // Define the trainer.
            // This trainer trains field-aware factorization (FFM)
            // for binary classification.
            // See https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf for the theory
            // behind and 
            // https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf for the
            // training algorithm implemented in ML.NET.
            var pipeline = mlContext.BinaryClassification.Trainers
                .FieldAwareFactorizationMachine(
                // Specify three feature columns!
                new[] {nameof(DataPoint.Field0), nameof(DataPoint.Field1),
                nameof(DataPoint.Field2) },
                // Specify binary label's column name.
                nameof(DataPoint.Label));

            // Train the model.
            var model = pipeline.Fit(trainingData);

            // Run the model on training data set.
            var transformedTrainingData = model.Transform(trainingData);

            // Measure the quality of the trained model.
            var metrics = mlContext.BinaryClassification
                .Evaluate(transformedTrainingData);

            // Show the quality metrics.
            PrintMetrics(metrics);

            // Expected output:
            //   Accuracy: 0.99
            //   AUC: 1.00
            //   F1 Score: 0.99
            //   Negative Precision: 1.00
            //   Negative Recall: 0.98
            //   Positive Precision: 0.98
            //   Positive Recall: 1.00
            //   Log Loss: 0.17
            //   Log Loss Reduction: 0.83
            //   Entropy: 1.00
            //
            //   TEST POSITIVE RATIO:    0.4760 (238.0/(238.0+262.0))
            //   Confusion table
            //             ||======================
            //   PREDICTED || positive | negative | Recall
            //   TRUTH     ||======================
            //    positive ||      193 |       45 | 0.8109
            //    negative ||       52 |      210 | 0.8015
            //             ||======================
            //   Precision ||   0.7878 |   0.8235 |

            // Create prediction function from the trained model.
            var engine = mlContext.Model
                .CreatePredictionEngine<DataPoint, Result>(model);

            // Make some predictions.
            foreach (var dataPoint in data.Take(5))
            {
                var result = engine.Predict(dataPoint);
                Console.WriteLine($"Actual label: {dataPoint.Label}, "
                    + $"predicted label: {result.PredictedLabel}, "
                    + $"score of being positive class: {result.Score}, "
                    + $"and probability of beling positive class: "
                    + $"{result.Probability}.");

            }

            // Expected output:
            //   Actual label: True, predicted label: True, score of being positive class: 1.115094, and probability of being positive class: 0.7530775.
            //   Actual label: False, predicted label: False, score of being positive class: -3.478797, and probability of being positive class: 0.02992158.
            //   Actual label: True, predicted label: True, score of being positive class: 3.191896, and probability of being positive class: 0.9605282.
            //   Actual label: False, predicted label: False, score of being positive class: -3.400863, and probability of being positive class: 0.03226851.
            //   Actual label: True, predicted label: True, score of being positive class: 4.06056, and probability of being positive class: 0.9830528.
        }

        // Number of features per field.
        const int featureLength = 5;

        // This class defines objects fed to the trained model.
        private class DataPoint
        {
            // Label.
            public bool Label { get; set; }

            // Features from the first field. Note that different fields can have
            // different numbers of features.
            [VectorType(featureLength)]
            public float[] Field0 { get; set; }

            // Features from the second field. 
            [VectorType(featureLength)]
            public float[] Field1 { get; set; }

            // Features from the thrid field. 
            [VectorType(featureLength)]
            public float[] Field2 { get; set; }
        }

        // This class defines objects produced by trained model. The trained model
        // maps a DataPoint to a Result.
        public class Result
        {
            // Label.
            public bool Label { get; set; }
            // Predicted label.
            public bool PredictedLabel { get; set; }
            // Predicted score.
            public float Score { get; set; }
            // Probability of belonging to positive class.
            public float Probability { get; set; }
        }

        // Function used to create toy data sets.
        private static IEnumerable<DataPoint> GenerateRandomDataPoints(
            int exampleCount, int seed = 0)

        {
            var rnd = new Random(seed);
            var data = new List<DataPoint>();
            for (int i = 0; i < exampleCount; ++i)
            {
                // Initialize an example with a random label and an empty feature
                // vector.
                var sample = new DataPoint()
                {
                    Label = rnd.Next() % 2 == 0,
                    Field0 = new float[featureLength],
                    Field1 = new float[featureLength],
                    Field2 = new float[featureLength]
                };

                // Fill feature vectors according the assigned label.
                // Notice that features from different fields have different biases
                // and therefore different distributions. In practices such as game
                // recommendation, one may use one field to store features from user
                // profile and another field to store features from game profile.
                for (int j = 0; j < featureLength; ++j)
                {
                    var value0 = (float)rnd.NextDouble();
                    // Positive class gets larger feature value.
                    if (sample.Label)
                        value0 += 0.2f;
                    sample.Field0[j] = value0;

                    var value1 = (float)rnd.NextDouble();
                    // Positive class gets smaller feature value.
                    if (sample.Label)
                        value1 -= 0.2f;
                    sample.Field1[j] = value1;

                    var value2 = (float)rnd.NextDouble();
                    // Positive class gets larger feature value.
                    if (sample.Label)
                        value2 += 0.8f;
                    sample.Field2[j] = value2;
                }

                data.Add(sample);
            }
            return data;
        }

        // Function used to show evaluation metrics such as accuracy of predictions.
        private static void PrintMetrics(
            CalibratedBinaryClassificationMetrics metrics)

        {
            Console.WriteLine($"Accuracy: {metrics.Accuracy:F2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F2}");
            Console.WriteLine($"Negative Precision: " +
                $"{metrics.NegativePrecision:F2}");

            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall:F2}");
            Console.WriteLine($"Positive Precision: " +
                $"{metrics.PositivePrecision:F2}");

            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:F2}");
            Console.WriteLine($"Log Loss: {metrics.LogLoss:F2}");
            Console.WriteLine($"Log Loss Reduction: {metrics.LogLossReduction:F2}");
            Console.WriteLine($"Entropy: {metrics.Entropy:F2}");
        }
    }
}

