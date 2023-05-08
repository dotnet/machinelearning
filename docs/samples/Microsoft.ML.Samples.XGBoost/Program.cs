using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;

using Microsoft.ML.Trainers.XGBoost;
using Newtonsoft.Json.Linq;
using Microsoft.Data.Analysis;
using Microsoft.ML.SamplesUtils;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace Microsoft.ML.Samples.XGBoost
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            var vm = XGBoostUtils.XgbMajorVersion();
            Console.WriteLine($"The output of checking the version is [{vm.Major}.{vm.Minor}]");


            var housingDSet = DatasetUtils.GetFilePathFromDataDirectory("boston_housing.csv");
            Console.WriteLine($"I'm reading a dataset from {housingDSet}");
            Console.WriteLine($"The build information on the XGBoost library is {XGBoostUtils.BuildInfo()}");

            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available operations
            // and as the source of randomness. Setting the seed to a fixed number
            // in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

#if false
            // Create a list of training data points.
            var dataPoints = GenerateRandomDataPoints(250);

#if false
            foreach (var dataPoint in dataPoints)
            {
                var feats = dataPoint.Features;
                if (feats != null)
                {
                    string strVec = string.Join(",", feats.Select(x => x.ToString()).ToArray());
                    Console.WriteLine($"features: [{strVec}], label: {dataPoint.Label}");
                }
            }
#endif


            // Convert the list of data points to an IDataView object, which is
            // consumable by ML.NET API.
            var trainingData = mlContext.Data.LoadFromEnumerable(dataPoints);
#else
            var df = DataFrame.LoadCsv(housingDSet);
            var trainingData = (df as IDataView);
            foreach (var c in trainingData.Schema)
            {
                Console.WriteLine($"Column {c.Name} is of type {c.Type}");
            }

            string[] featureColNames = { "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT" };
            Console.WriteLine($"Feature columns is of length: {featureColNames.Length}.");

            Console.WriteLine($"The schema of this is {trainingData.Schema}");

            var prepPipeline = mlContext.Transforms.Concatenate("Features", featureColNames);
            var trainingDataClean = prepPipeline.Fit(trainingData).Transform(trainingData);
            Console.WriteLine($"After preprocessing, the schema is: {trainingDataClean.Schema}.");
#endif

#if true
            // Define the trainer.
            var pipeline = mlContext.Regression.Trainers.
                XGBoost(
                //labelColumnName: nameof(DataPoint.Label),
                labelColumnName: "MEDV",
        //featureColumnName: nameof(DataPoint.Features),
        featureColumnName: "Features",
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

#if false
            try
            {
#endif
            // Train the model.
            var model = pipeline.Fit(/* trainingData */ trainingDataClean);
#if false
        }
            catch (Exception ex)
            {
                Console.WriteLine($"In top level: Exception: {ex.Message}");
            }
#endif

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
#endif

#if false
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
#endif
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
                    Features = Enumerable.Repeat(label, 12).Select(
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
            [VectorType(12)]
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
