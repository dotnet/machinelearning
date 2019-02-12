using System;
using System.Diagnostics;
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
        private static string BaseDatasetsLocation = @"../../../../src/Samples/Data";
        private static string TrainDataPath = $"{BaseDatasetsLocation}/taxi-fare-train.csv";
        private static string TestDataPath = $"{BaseDatasetsLocation}/taxi-fare-test.csv";
        private static string ModelPath = $"{BaseDatasetsLocation}/TaxiFareModel.zip";
        private static string LabelColumnName = "fare_amount";

        public static void Run()
        {
            //Create ML Context with seed for repeteable/deterministic results
            MLContext mlContext = new MLContext(seed: 0);

            // STEP 1: Infer columns
            var columnInference = mlContext.Data.InferColumns(TrainDataPath, LabelColumnName, ',');

            // STEP 2: Load data
            TextLoader textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderArgs);
            IDataView trainDataView = textLoader.Read(TrainDataPath);
            IDataView testDataView = textLoader.Read(TestDataPath);

            int cancelAfterInSeconds = 20;
            CancellationTokenSource cts = new CancellationTokenSource();
            cts.CancelAfter(cancelAfterInSeconds * 1000);

            Stopwatch watch = Stopwatch.StartNew();

            // STEP 3: Autofit with a cancellation token
            Console.WriteLine($"Invoking Regression.AutoFit");
            var autoFitResults = mlContext.Regression.AutoFit(trainDataView,
                                                        LabelColumnName,
                                                        timeoutInSeconds: 1,
                                                        cancellationToken: cts.Token);

            Console.WriteLine($"{autoFitResults.Count} models were returned after {cancelAfterInSeconds} seconds");

            Console.WriteLine("Press any key to continue..");
            Console.ReadLine();
        }
    }
}
