using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML.Samples
{
    public static class AutoMLExperiment
    {
        public static async Task RunAsync()
        {
            var seed = 0;

            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available operations
            // and as the source of randomness. Setting the seed to a fixed number
            // in this example to make outputs deterministic.
            var context = new MLContext(seed);

            // Create a list of training data points and convert it to IDataView.
            var data = GenerateRandomBinaryClassificationDataPoints(100, seed);
            var dataView = context.Data.LoadFromEnumerable(data);

            var trainTestSplit = context.Data.TrainTestSplit(dataView);

            // Define the sweepable pipeline using predefined binary trainers and search space.
            var pipeline = context.Auto().BinaryClassification(labelColumnName: "Label", featureColumnName: "Features");

            // Create an AutoML experiment
            var experiment = context.Auto().CreateExperiment();

            // Redirect AutoML log to console
            context.Log += (object o, LoggingEventArgs e) =>
            {
                if (e.Source == nameof(AutoMLExperiment) && e.Kind > Runtime.ChannelMessageKind.Trace)
                {
                    Console.WriteLine(e.RawMessage);
                }
            };

            // Config experiment to optimize "Accuracy" metric on given dataset.
            // This experiment will run hyper-parameter optimization on given pipeline
            experiment.SetPipeline(pipeline)
                      .SetDataset(trainTestSplit.TrainSet, fold: 5) // use 5-fold cross validation to evaluate each trial
                      .SetBinaryClassificationMetric(BinaryClassificationMetric.Accuracy, "Label")
                      .SetMaxModelToExplore(100); // explore 100 trials

            // start automl experiment
            var result = await experiment.RunAsync();

            // Expected output samples during training:
            //      Update Running Trial - Id: 0
            //      Update Completed Trial - Id: 0 - Metric: 0.5536912515402218 - Pipeline: FastTreeBinary - Duration: 595 - Peak CPU: 0.00 % -Peak Memory in MB: 35.81
            //      Update Best Trial - Id: 0 - Metric: 0.5536912515402218 - Pipeline: FastTreeBinary

            // evaluate test dataset on best model.
            var bestModel = result.Model;
            var eval = bestModel.Transform(trainTestSplit.TestSet);
            var metrics = context.BinaryClassification.Evaluate(eval);

            PrintMetrics(metrics);

            // Expected output:
            //  Accuracy: 0.67
            //  AUC: 0.75
            //  F1 Score: 0.33
            //  Negative Precision: 0.88
            //  Negative Recall: 0.70
            //  Positive Precision: 0.25
            //  Positive Recall: 0.50

            //  TEST POSITIVE RATIO: 0.1667(2.0 / (2.0 + 10.0))
            //  Confusion table
            //            ||======================
            //  PREDICTED || positive | negative | Recall
            //  TRUTH     ||======================
            //   positive || 1 | 1 | 0.5000
            //   negative || 3 | 7 | 0.7000
            //            ||======================
            //  Precision || 0.2500 | 0.8750 |
        }

        private static IEnumerable<BinaryClassificationDataPoint> GenerateRandomBinaryClassificationDataPoints(int count,
            int seed = 0)

        {
            var random = new Random(seed);
            float randomFloat() => (float)random.NextDouble();
            for (int i = 0; i < count; i++)
            {
                var label = randomFloat() > 0.5f;
                yield return new BinaryClassificationDataPoint
                {
                    Label = label,
                    // Create random features that are correlated with the label.
                    // For data points with false label, the feature values are
                    // slightly increased by adding a constant.
                    Features = Enumerable.Repeat(label, 50)
                        .Select(x => x ? randomFloat() : randomFloat() +
                        0.1f).ToArray()

                };
            }
        }

        // Example with label and 50 feature values. A data set is a collection of
        // such examples.
        private class BinaryClassificationDataPoint
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
