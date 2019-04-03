// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Auto;
using Microsoft.ML.Data;

namespace Samples
{
    static class AdvancedExperimentSettings
    {
        private static string BaseDatasetsLocation = "Data";
        private static string TrainDataPath = Path.Combine(BaseDatasetsLocation, "taxi-fare-train.csv");
        private static string TestDataPath = Path.Combine(BaseDatasetsLocation, "taxi-fare-test.csv");
        private static string ModelPath = Path.Combine(BaseDatasetsLocation, "TaxiFareModel.zip");
        private static string LabelColumn = "FareAmount";

        public static void Run()
        {
            MLContext mlContext = new MLContext();
            
            // STEP 1: Create text loader options
            var textLoaderOptions = new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("VendorId", DataKind.String, 0),
                    new TextLoader.Column("RateCode", DataKind.Single, 1),
                    new TextLoader.Column("PassengerCount", DataKind.Single, 2),
                    new TextLoader.Column("TripTimeInSeconds", DataKind.Single, 3),
                    new TextLoader.Column("TripDistance", DataKind.Single, 4),
                    new TextLoader.Column("PaymentType", DataKind.String, 5),
                    new TextLoader.Column("FareAmount", DataKind.Single, 6),
                },
                HasHeader = true,
                Separators = new[] { ',' }
            };

            // STEP 2: Load data
            TextLoader textLoader = mlContext.Data.CreateTextLoader(textLoaderOptions);
            IDataView trainDataView = textLoader.Load(TrainDataPath);
            IDataView testDataView = textLoader.Load(TestDataPath);

            var experimentSettings = new RegressionExperimentSettings();
            experimentSettings.MaxExperimentTimeInSeconds = 20;
            experimentSettings.ProgressHandler = new ProgressHandler();

            // STEP 3: Using a different optimizing metric instead of RSquared and use only LightGbm
            experimentSettings.OptimizingMetric = RegressionMetric.MeanSquaredError;
            experimentSettings.Trainers.Clear();
            experimentSettings.Trainers.Add(RegressionTrainer.LightGbm);

            // STEP 4: Start AutoML experiment
            Console.WriteLine($"Starting an experiment with MeanSquaredError optimizing metric and using LightGbm trainer only\r\n");
            RegressionExperiment autoExperiment = mlContext.Auto().CreateRegressionExperiment(experimentSettings);
            autoExperiment.Execute(trainDataView, LabelColumn);

            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }
    }
}
