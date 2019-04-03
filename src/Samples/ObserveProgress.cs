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
    static class ObserveProgress
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

            // STEP 3: Auto inference with a callback configured
            RegressionExperiment autoExperiment = mlContext.Auto().CreateRegressionExperiment(new RegressionExperimentSettings()
            {
                MaxExperimentTimeInSeconds = 60
            });
            autoExperiment.Execute(trainDataView, LabelColumn, progressHandler: new ProgressHandler());

            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }
    }

    class ProgressHandler : IProgress<RunDetails<RegressionMetrics>>
    {
        int iterationIndex;
        private bool _initialized = false;

        public ProgressHandler()
        {
        }

        public void Report(RunDetails<RegressionMetrics> iterationResult)
        {
            if (!_initialized)
            {
                ConsolePrinter.PrintRegressionMetricsHeader();
                _initialized = true;
            }
            iterationIndex++;
            ConsolePrinter.PrintRegressionMetrics(iterationIndex, iterationResult.TrainerName, iterationResult.ValidationMetrics);
        }
    }

    class ConsolePrinter
    {
        public static void PrintRegressionMetrics(int iteration, string trainerName, RegressionMetrics metrics)
        {
            Console.WriteLine($"{iteration,-3}{trainerName,-35}{metrics.RSquared,-10:0.###}{metrics.LossFunction,-8:0.##}{metrics.MeanAbsoluteError,-15:#.##}{metrics.MeanSquaredError,-15:#.##}{metrics.RootMeanSquaredError,-10:#.##}");
        }

        public static void PrintRegressionMetricsHeader()
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for regression models     ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"{" ",-3}{"Trainer",-35}{"R2-Score",-10}{"LossFn",-8}{"Absolute-loss",-15}{"Squared-loss",-15}{"RMS-loss",-10}");
            Console.WriteLine();
        }
    }
}
