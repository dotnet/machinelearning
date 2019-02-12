using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Auto;
using Microsoft.ML.Data;

namespace Samples
{
    static class ProgressHandler
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

            // STEP 3: Autofit with a callback configured
            var autoFitResults = mlContext.Regression.AutoFit(trainDataView, 
                                                        LabelColumnName, 
                                                        timeoutInSeconds: 1, 
                                                        progressCallback: new Progress());

            Console.WriteLine("Press any key to continue..");
            Console.ReadLine();
        }

        class Progress : IProgress<AutoFitRunResult<RegressionMetrics>>
        {
            int iterationIndex;
            public Progress()
            {
                ConsolePrinter.PrintRegressionMetricsHeader();
            }

            public void Report(AutoFitRunResult<RegressionMetrics> iterationResult)
            {
                iterationIndex++;
                ConsolePrinter.PrintRegressionMetrics(iterationIndex, iterationResult.TrainerName, iterationResult.Metrics);
            }
        }

        class ConsolePrinter
        {
            public static void PrintRegressionMetrics(int iteration, string trainerName, RegressionMetrics metrics)
            {
                Console.WriteLine($"{iteration,-3}{trainerName,-35}{metrics.RSquared,-10:0.###}{metrics.LossFn,-8:0.##}{metrics.L1,-15:#.##}{metrics.L2,-15:#.##}{metrics.Rms,-10:#.##}");
            }

            public static void PrintActualVersusPredictedValue(int index, float fareAmount, float score)
            {
                Console.WriteLine($"{index,-5}{fareAmount,-20}{score,-20}");
            }

            public static void PrintRegressionMetricsHeader()
            {
                Console.WriteLine($"*************************************************");
                Console.WriteLine($"*       Metrics for regression models     ");
                Console.WriteLine($"*------------------------------------------------");
                Console.WriteLine($"{" ",-3}{"Trainer",-35}{"R2-Score",-10}{"LossFn",-8}{"Absolute-loss",-15}{"Squared-loss",-15}{"RMS-loss",-10}");
                Console.WriteLine();
            }

            public static void PrintActualVersusPredictedHeader()
            {
                Console.WriteLine();
                Console.WriteLine($"*************************************************");
                Console.WriteLine($"*       Actual fare Vs predicted fare using the model picked by automl");
                Console.WriteLine($"*------------------------------------------------");
                Console.WriteLine($"{"Row",-5}{"Actual",-20}{"Predicted",-20}");
            }

            public static void PrintBestPipelineHeader()
            {
                Console.WriteLine();
                Console.WriteLine($"*************************************************");
                Console.WriteLine($"*       Best pipeline      ");
                Console.WriteLine($"*------------------------------------------------");
            }
        }
    }
}
