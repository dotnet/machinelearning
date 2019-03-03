// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Auto;
using Microsoft.ML.Data;
using Samples.Helpers;

namespace Samples
{
    static class CustomizeTraining
    {
        private static string BaseDatasetsLocation = Path.Combine("..", "..", "..", "..", "src", "Samples", "Data");
        private static string TrainDataPath = Path.Combine(BaseDatasetsLocation, "taxi-fare-train.csv");
        private static string TestDataPath = Path.Combine(BaseDatasetsLocation, "taxi-fare-test.csv");
        private static string ModelPath = Path.Combine(BaseDatasetsLocation, "TaxiFareModel.zip");
        private static string LabelColumn = "fare_amount";

        public static void Run()
        {
            MLContext mlContext = new MLContext();

            // STEP 1: Infer columns
            ColumnInferenceResults columnInference = mlContext.Auto().InferColumns(TrainDataPath, LabelColumn, ',');
            ConsoleHelper.Print(columnInference);

            // STEP 2: Load data
            TextLoader textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            IDataView trainDataView = textLoader.Load(TrainDataPath);
            IDataView testDataView = textLoader.Load(TestDataPath);

            var experimentSettings = new RegressionExperimentSettings();
            experimentSettings.MaxExperimentTimeInSeconds = 20;
            experimentSettings.ProgressHandler = new ProgressHandler();

            // STEP 3: Using a different optimizing metric instead of RSquared and use only LightGbm
            Console.WriteLine($"Starting an experiment with MeanSquaredError optimizing metric and using LightGbm trainer only");
            experimentSettings.OptimizingMetric = RegressionMetric.MeanSquaredError;
            experimentSettings.Trainers.Clear();
            experimentSettings.Trainers.Add(RegressionTrainer.LightGbm);

            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }
    }
}
