using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;

using Microsoft.ML.Trainers.XGBoost;
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.Samples.XGBoost
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available operations
            // and as the source of randomness. Setting the seed to a fixed number
            // in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Create a list of training data points.
            var dataPoints = GenerateRandomDataPoints(10/*00*/);

            foreach (var dataPoint in dataPoints)
            {
                var feats = dataPoint.Features;
                if (feats != null)
                {
                    string strVec = string.Join(",", feats.Select(x => x.ToString()).ToArray());
                    Console.WriteLine($"features: [{strVec}], label: {dataPoint.Label}");
                }
            }

            // Convert the list of data points to an IDataView object, which is
            // consumable by ML.NET API.
            var trainingData = mlContext.Data.LoadFromEnumerable(dataPoints);

#if true
            // Define the trainer.
            var pipeline = mlContext.Regression.Trainers.
                XGBoost(
                labelColumnName: nameof(DataPoint.Label),
                featureColumnName: nameof(DataPoint.Features),
        numberOfLeaves: 8
        );
#else
            var pipeline = mlContext.Regression.Trainers.XGBoost(
             new XGBoostRegressionTrainer.Options
             {
                 //                    LabelColumnName = labelName,
                 NumberOfLeaves = 4,
                 MinimumExampleCountPerLeaf = 6,
                 LearningRate = 0.001,
                 Booster = new DartBooster.Options()
                 {
                     TreeDropFraction = 0.124
                 }
             });
#endif

            // Train the model.
            var model = pipeline.Fit(trainingData);

#if false
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

            // Expected output:
            //   Label: 0.985, Prediction: 0.864
            //   Label: 0.155, Prediction: 0.164
            //   Label: 0.515, Prediction: 0.470
            //   Label: 0.566, Prediction: 0.501
            //   Label: 0.096, Prediction: 0.138

            // Evaluate the overall metrics
            var metrics = mlContext.Regression.Evaluate(transformedTestData);
            PrintMetrics(metrics);

            // Expected output:
            //   Mean Absolute Error: 0.10
            //   Mean Squared Error: 0.01
            //   Root Mean Squared Error: 0.11
            //   RSquared: 0.89 (closer to 1 is better. The worst case is 0)
#else
            //            var v =
            //	    XGBoost.XGBoostVersion v;
            var vm = XGBoostUtils.XgbMajorVersion();
            Console.WriteLine($"The output of checking the version is [{vm.Major}.{vm.Minor}]");
#endif

            Console.WriteLine($"The build information on the XGBoost library is {XGBoostUtils.BuildInfo()}");
            Console.WriteLine("*** Done!!!");
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
                    Features = Enumerable.Repeat(label, 10).Select(
                    //Features = Enumerable.Repeat(label, 50).Select(
                        x => x + (float)random.NextDouble()).ToArray()
                };
            }
        }

        // Example with label and 50 feature values. A data set is a collection of
        // such examples.
        private class DataPoint
        {
            public float Label { get; set; }
            //[VectorType(50)]
            [VectorType(10)]
            public float[]? Features { get; set; }
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
