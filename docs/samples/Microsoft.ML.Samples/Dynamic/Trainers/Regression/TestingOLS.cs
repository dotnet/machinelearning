using System;
using System.IO;
using System.Reflection;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;


namespace Samples.Dynamic.Trainers.Regression
{
    public static class TestingOLS
    {
        public static void Example()
        {
            Console.WriteLine($"Running from TestingOLS... from {Assembly.GetExecutingAssembly().Location}...");

            var selectedBackend = Environment.GetEnvironmentVariable("MLNET_BACKEND");
            Console.WriteLine($"found the backend to be [{selectedBackend}]");

            var selectedTask = Environment.GetEnvironmentVariable("MLNET_BENCHMARK_TASK");
            var selectedDataset = Environment.GetEnvironmentVariable("MLNET_BENCHMARK_DATASET");

            if (selectedTask == null || selectedDataset == null)
            {
                Console.WriteLine($"*Both* variables MLNET_BENCHMARK_TASK and MLNET_BENCHMARK_DATASET need to be specified");
                return;
            }

            var tg = System.Diagnostics.Stopwatch.StartNew();
            var t0 = System.Diagnostics.Stopwatch.StartNew();

            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available operations
            // and as the source of randomness. Setting the seed to a fixed number
            // in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

#if false
            // Create a list of training data points.
            var dataPoints = GenerateRandomDataPoints(1000);
            // Convert the list of data points to an IDataView object, which is
            // consumable by ML.NET API.
            var trainingData = mlContext.Data.LoadFromEnumerable(dataPoints);
#endif
            var data = LoadData(mlContext, selectedDataset, selectedTask);
            var featuresArray = GetFeaturesArray(data[0]);
            Console.WriteLine($"Found [{featuresArray.Length}] features.");
            IDataView trainingData, testingData;

            Console.WriteLine($"Arranging data for task: {selectedTask} (configuring preprocessing).");
            if (selectedTask == "mcl")
            {
                var preprocessingModel = mlContext.Transforms.Concatenate("Features", featuresArray).Append(mlContext.Transforms.Conversion.MapValueToKey("target"));
                trainingData = preprocessingModel.Fit(data[0]).Transform(data[0]);
                testingData = preprocessingModel.Fit(data[0]).Transform(data[1]);
            }
            else
            {
                var preprocessingModel = mlContext.Transforms.Concatenate("Features", featuresArray);
                trainingData = preprocessingModel.Fit(data[0]).Transform(data[0]);
                testingData = preprocessingModel.Fit(data[0]).Transform(data[1]);
            }
            t0.Stop();

            if (selectedTask == "reg") // regression
            {
                Console.WriteLine("Carrying out regression");

                var t1 = System.Diagnostics.Stopwatch.StartNew();
                var trainer = mlContext.Regression.Trainers.Ols(labelColumnName: "target", featureColumnName: "Features");
                var model = trainer.Fit(trainingData);
                t1.Stop();
#if false
                var t2 = System.Diagnostics.Stopwatch.StartNew();
                IDataView predictions = model.Transform(testingData);
                t2.Stop();

                var t3 = System.Diagnostics.Stopwatch.StartNew();
                List<double> metricsList = new List<double>();
                var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "target", scoreColumnName: "Score");
                t3.Stop();

                tg.Stop();

                Console.WriteLine("Dataset,Task,All time[ms],Reading time[ms],Fitting time[ms],Prediction time[ms],Evaluation time[ms],MAE,RMSE,R2");
                Console.Write($"{args[0]},{args[1]},{tg.Elapsed.TotalMilliseconds},{t0.Elapsed.TotalMilliseconds},{t1.Elapsed.TotalMilliseconds},{t2.Elapsed.TotalMilliseconds},{t3.Elapsed.TotalMilliseconds},");
                Console.Write($"{metrics.MeanAbsoluteError},{metrics.RootMeanSquaredError},{metrics.RSquared}\n");
#endif

            }
            else if (selectedTask == "bin")
            {
                Console.WriteLine("Carrying out binary classification");
                var t1 = System.Diagnostics.Stopwatch.StartNew();
                var options = new LbfgsLogisticRegressionBinaryTrainer.Options()
                {
                    LabelColumnName = "target",
                    FeatureColumnName = "Features",
                    L1Regularization = 0.05f,
                    L2Regularization = 0.05f,
                    HistorySize = 20,
                    OptimizationTolerance = 1e-6f,
                    MaximumNumberOfIterations = 100
                };

                var trainer = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(options);
                var model = trainer.Fit(trainingData);
                t1.Stop();

                var t2 = System.Diagnostics.Stopwatch.StartNew();
                IDataView predictions = model.Transform(testingData);
                t2.Stop();

                var t3 = System.Diagnostics.Stopwatch.StartNew();
                var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "target");
                t3.Stop();

                tg.Stop();

                Console.WriteLine("Dataset,Task,All time[ms],Reading time[ms],Fitting time[ms],Prediction time[ms],Evaluation time[ms],LogLoss,Accuracy,ROC-AUC");
                Console.Write($"{selectedDataset},{selectedTask},{tg.Elapsed.TotalMilliseconds},{t0.Elapsed.TotalMilliseconds},{t1.Elapsed.TotalMilliseconds},{t2.Elapsed.TotalMilliseconds},{t3.Elapsed.TotalMilliseconds},");
                Console.Write($"{metrics.LogLoss},{metrics.Accuracy},{metrics.AreaUnderRocCurve}\n");
            }
            else if (selectedTask == "mcl") // multi-class classification
            {
                Console.WriteLine("Carrying out multi-class classification");
                var t1 = System.Diagnostics.Stopwatch.StartNew();
                var options = new LbfgsMaximumEntropyMulticlassTrainer.Options()
                {
                    LabelColumnName = "target",
                    FeatureColumnName = "Features",
                    L1Regularization = 0.05f,
                    L2Regularization = 0.05f,
                    HistorySize = 20,
                    OptimizationTolerance = 1e-6f,
                    MaximumNumberOfIterations = 100
                };
                var trainer = mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(options);
                var model = trainer.Fit(trainingData);
                t1.Stop();

                var t2 = System.Diagnostics.Stopwatch.StartNew();
                IDataView predictions = model.Transform(testingData);
                t2.Stop();

                var t3 = System.Diagnostics.Stopwatch.StartNew();
                var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "target");
                t3.Stop();

                tg.Stop();

                Console.WriteLine("Dataset,Task,All time[ms],Reading time[ms],Fitting time[ms],Prediction time[ms],Evaluation time[ms],LogLoss,MicroAccuracy,MacroAccuracy");
                Console.Write($"{selectedDataset},{selectedTask},{tg.Elapsed.TotalMilliseconds},{t0.Elapsed.TotalMilliseconds},{t1.Elapsed.TotalMilliseconds},{t2.Elapsed.TotalMilliseconds},{t3.Elapsed.TotalMilliseconds},");
                Console.Write($"{metrics.LogLoss},{metrics.MicroAccuracy},{metrics.MacroAccuracy}\n");
            }

            Console.WriteLine("Done with TestingOLS...");
        }

        public static IDataView[] LoadData(MLContext mlContext, string dataset, string task, string label = "target", char separator = ',')
        {
            //string modelLocation = Microsoft.ML.SamplesUtils.DatasetUtils.GetHousingRegressionDataset();
            string datasetLocation = Microsoft.ML.SamplesUtils.DatasetUtils.GetFilePathFromDataDirectory($"{dataset}_train.csv");
            if (!File.Exists(datasetLocation))
            {
                Console.WriteLine($"Cannot find dataset from expected [{datasetLocation}] location");
                return null; // TODO: throw an exception instead
            }

            System.IO.StreamReader file = new System.IO.StreamReader(datasetLocation);
            string header = file.ReadLine();
            file.Close();
            string[] headerArray = header.Split(separator);
            List<TextLoader.Column> columns = new List<TextLoader.Column>();
            foreach (string column in headerArray)
            {
                if (column == label && task != "reg")
                {
                    if (task == "bin")
                    {
                        columns.Add(new TextLoader.Column(column, DataKind.Boolean, Array.IndexOf(headerArray, column)));
                    }
                    else
                    {
                        columns.Add(new TextLoader.Column(column, DataKind.UInt32, Array.IndexOf(headerArray, column)));
                    }
                }
                else
                {
                    columns.Add(new TextLoader.Column(column, DataKind.Single, Array.IndexOf(headerArray, column)));
                }
            }

            var loader = mlContext.Data.CreateTextLoader(
                separatorChar: separator,
                hasHeader: true,
                columns: columns.ToArray()
            );

            string datasetLocationTrain = Microsoft.ML.SamplesUtils.DatasetUtils.GetFilePathFromDataDirectory($"{dataset}_train.csv");
            string datasetLocationTest = Microsoft.ML.SamplesUtils.DatasetUtils.GetFilePathFromDataDirectory($"{dataset}_test.csv");

            List<IDataView> dataList = new List<IDataView>();
#if false
            dataList.Add(loader.Load($"{dataset}_train.csv"));
            dataList.Add(loader.Load($"{dataset}_test.csv"));
#else
            dataList.Add(loader.Load(datasetLocationTrain));
            dataList.Add(loader.Load(datasetLocationTest));

#endif
            return dataList.ToArray();
        }

        public static string[] GetFeaturesArray(IDataView data, string labelName = "target")
        {
            List<string> featuresList = new List<string>();
            var nColumns = data.Schema.Count;
            var columnsEnumerator = data.Schema.GetEnumerator();
            for (int i = 0; i < nColumns; i++)
            {
                columnsEnumerator.MoveNext();
                if (columnsEnumerator.Current.Name != labelName)
                    featuresList.Add(columnsEnumerator.Current.Name);
            }

            return featuresList.ToArray();
        }

#if false
        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count, int seed = 0)
        {
            var random = new Random(seed);
            for (int i = 0; i < count; i++)
            {
                float label = (float)random.NextDouble();
                yield return new DataPoint
                {
                    Label = label,
                    // Create random features that are correlated with the label.
                    Features = Enumerable.Repeat(label, 50).Select(
                        x => x + (float)random.NextDouble()).ToArray()
                };
            }
        }

        // Example with label and 50 feature values. A data set is a collection of
        // such examples.
        private class DataPoint
        {
            public float Label { get; set; }
            [VectorType(50)]
            public float[] Features { get; set; }
        }
#if false
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
#endif
#endif

    }
}

#if false
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

class Program
{
    static void Main(string[] args)
    {
        var tg = System.Diagnostics.Stopwatch.StartNew();
        var t0 = System.Diagnostics.Stopwatch.StartNew();
        MLContext mlContext = new MLContext();
        var data = LoadData(mlContext, args[0]);
        var featuresArray = GetFeaturesArray(data[0]);
        var preprocessingModel = mlContext.Transforms.Concatenate("Features", featuresArray);
        var trainingData = preprocessingModel.Fit(data[0]).Transform(data[0]);
        var testingData = preprocessingModel.Fit(data[0]).Transform(data[1]);
        t0.Stop();

        var t1 = System.Diagnostics.Stopwatch.StartNew();
        var trainer = mlContext.Regression.Trainers.Ols(labelColumnName: "target", featureColumnName: "Features");
        var model = trainer.Fit(trainingData);
        t1.Stop();

        var t2 = System.Diagnostics.Stopwatch.StartNew();
        IDataView predictions = model.Transform(testingData);
        t2.Stop();

        var t3 = System.Diagnostics.Stopwatch.StartNew();
        List<double> metricsList = new List<double>();
        var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "target", scoreColumnName: "Score");
        t3.Stop();
        tg.Stop();

        Console.WriteLine("Dataset,All time[ms],Reading time[ms],Fitting time[ms],Prediction time[ms],Evaluation time[ms],MAE,RMSE,R2");
        Console.Write($"{args[0]},{tg.Elapsed.TotalMilliseconds},{t0.Elapsed.TotalMilliseconds},{t1.Elapsed.TotalMilliseconds},{t2.Elapsed.TotalMilliseconds},{t3.Elapsed.TotalMilliseconds},");
        Console.Write($"{metrics.MeanAbsoluteError},{metrics.RootMeanSquaredError},{metrics.RSquared}\n");
    }
}
#endif

#if false
class Program
{
    static void Main(string[] args)
    {
        var tg = System.Diagnostics.Stopwatch.StartNew();
        var t0 = System.Diagnostics.Stopwatch.StartNew();
        MLContext mlContext = new MLContext();
        var data = LoadData(mlContext, args[0], args[1]);
        var featuresArray = GetFeaturesArray(data[0]);
        IDataView trainingData, testingData;

        t0.Stop();


    }
}
#endif
