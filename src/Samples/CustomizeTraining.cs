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
            TextLoader textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderArgs);
            IDataView trainDataView = textLoader.Read(TrainDataPath);
            IDataView testDataView = textLoader.Read(TestDataPath);

            // STEP 3: Using a different optimizing metric instead of default R2 and whitelisting only LightGbm
            Console.WriteLine($"Starting an experiment with L2 optimizing metric and whitelisting LightGbm trainer");
            RegressionExperiment autoExperiment = mlContext.Auto().CreateRegressionExperiment(new RegressionExperimentSettings()
            {
                MaxExperimentTimeInSeconds = 20,
                OptimizingMetric = RegressionMetric.L2,
                WhitelistedTrainers = new[] { RegressionTrainer.LightGbm },
                ProgressHandler = new ProgressHandler()
            });
            autoExperiment.Execute(trainDataView, LabelColumn);

            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }
    }
}
