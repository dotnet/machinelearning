// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Auto;
using Microsoft.ML.Data;

namespace Samples
{
    static class Cancellation
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

            // STEP 3: Auto inference with a cancellation token in a new task
            Stopwatch stopwatch = Stopwatch.StartNew();
            CancellationTokenSource cts = new CancellationTokenSource();
            var experiment = mlContext.Auto()
                .CreateRegressionExperiment(new RegressionExperimentSettings()
                {
                    MaxExperimentTimeInSeconds = 3600,
                    CancellationToken = cts.Token
                });
            IEnumerable<RunResult<RegressionMetrics>> runResults = new List<RunResult<RegressionMetrics>>();
            Console.WriteLine($"Running AutoML experiment...");
            Task experimentTask = Task.Run(() =>
                {
                    runResults = experiment.Execute(trainDataView, LabelColumn);
                });

            // STEP 4: Stop the experiment run after any key is pressed
            Console.WriteLine($"Press any key to stop the experiment run...");
            Console.ReadKey();
            cts.Cancel();
            experimentTask.Wait();

            Console.WriteLine($"{runResults.Count()} models were returned after {stopwatch.Elapsed.TotalSeconds:0.00} seconds");

            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }
    }
}
