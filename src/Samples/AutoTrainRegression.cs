using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Auto;
using System.IO;
using System.Linq;

namespace Samples
{
    static class AutoTrainRegression
    {
        private static string BaseDatasetsLocation = @"../../../../src/Samples/Data";
        private static string TrainDataPath = $"{BaseDatasetsLocation}/taxi-fare-train.csv";
        private static string TestDataPath = $"{BaseDatasetsLocation}/taxi-fare-test.csv";
        private static string ModelPath = $"{BaseDatasetsLocation}/TaxiFareModel.zip";

        public static void Run()
        {
            //Create ML Context with seed for repeteable/deterministic results
            MLContext mlContext = new MLContext(seed: 0);

            // STEP 1: Common data loading configuration
            TextLoader textLoader = mlContext.Data.CreateTextReader(new[]
                                                                    {
                                                                        new TextLoader.Column("VendorId", DataKind.Text, 0),
                                                                        new TextLoader.Column("RateCode", DataKind.Text, 1),
                                                                        new TextLoader.Column("PassengerCount", DataKind.R4, 2),
                                                                        new TextLoader.Column("TripTime", DataKind.R4, 3),
                                                                        new TextLoader.Column("TripDistance", DataKind.R4, 4),
                                                                        new TextLoader.Column("PaymentType", DataKind.Text, 5),
                                                                        new TextLoader.Column("FareAmount", DataKind.R4, 6)
                                                                    },
                                                                     hasHeader: true,
                                                                     separatorChar: ','
                                                                    );

            IDataView trainDataView = textLoader.Read(TrainDataPath);
            IDataView testDataView = textLoader.Read(TestDataPath);

            // STEP 2: Auto featurize, auto train and auto hyperparameter tuning
            var autoFitResults = mlContext.Regression.AutoFit(trainDataView, "FareAmount", timeoutInMinutes:1);

            // STEP 3: Print metrics for each iteration 
            int iterationIndex = 0;
            PrintRegressionMetricsHeader();

            IDataView testDataViewWithBestScore = null;
            IterationResult<RegressionMetrics> bestIteration = null;
            double bestScore = 0;

            foreach (var iterationResult in autoFitResults)
            {
                if (iterationResult.Exception != null)
                {
                    Console.WriteLine(iterationResult.Exception);
                    continue;
                }

                IDataView testDataViewWithScore = iterationResult.Model.Transform(testDataView);
                var testMetrics = mlContext.Regression.Evaluate(testDataViewWithScore, label: DefaultColumnNames.Label, DefaultColumnNames.Score);
                if (bestScore < iterationResult.Metrics.RSquared)
                {
                    bestScore = iterationResult.Metrics.RSquared;
                    bestIteration = iterationResult;
                    testDataViewWithBestScore = testDataViewWithScore;
                }

                ++iterationIndex;
                PrintRegressionMetrics(iterationIndex, "validation metrics", iterationResult.Metrics);
                PrintRegressionMetrics(iterationIndex, "test metrics      ", testMetrics);
                Console.WriteLine();
            }

            // STEP 4: Compare and print actual value vs predicted value for top 5 rows from validation data
            PrintActualVersusPredictedHeader();
            IEnumerable<float> fareAmounts = testDataViewWithBestScore.GetColumn<float>(mlContext, "FareAmount");
            IEnumerable<float> scores = testDataViewWithBestScore.GetColumn<float>(mlContext, "Score");
            int rowCount = 1;
            do
            {
                PrintActualVersusPredictedValue(rowCount, fareAmounts.ElementAt(rowCount), scores.ElementAt(rowCount));
                
            } while (rowCount++ <= 5);

            // STEP 5: Save the best model for later deployment and inferencing
            using (var fs = File.Create(ModelPath))
                bestIteration.Model.SaveTo(mlContext, fs);

            Console.WriteLine("Press any key to continue..");
            Console.ReadLine();
        }

        static void PrintRegressionMetrics(int iteration, string typeOfMetrics, RegressionMetrics metrics)
        {
            Console.WriteLine($"{iteration}       {typeOfMetrics}       {metrics.LossFn:0.##}        {metrics.RSquared:0.##}        {metrics.L1:#.##}        {metrics.L2:#.##}        {metrics.Rms:#.##}");
        }

        static void PrintActualVersusPredictedValue(int index, float fareAmount, float score)
        {
            Console.WriteLine($"{index}         {fareAmount}            {score}");
        }

        static void PrintRegressionMetricsHeader()
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for regression model      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"iteration       type        LossFn      R2-Score   Absolute-loss  Squared-loss   RMS-loss");
        }

        static void PrintActualVersusPredictedHeader()
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Actual value Vs predicted value      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"Row    ActualFareAmount  PredictedFareAmount");
        }
    }
}
