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
    static class EarlyStopping
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
            var autoFitResults = mlContext.Regression.AutoFit(trainDataView, "FareAmount", timeoutInMinutes: 3);

            IterationResult<RegressionMetrics> bestIteration = null;
            double bestScore = 0;
            int totalIterations = 0;
            int iterationsWithoutScoreImprovement = 0;

            foreach (var iterationResult in autoFitResults)
            {
                totalIterations++;
                IDataView testDataViewWithScore = iterationResult.Model.Transform(testDataView);
                var testMetrics = mlContext.Regression.Evaluate(testDataViewWithScore, label: DefaultColumnNames.Label, DefaultColumnNames.Score);
                Console.WriteLine($"iteration{ totalIterations} score:{iterationResult.Metrics.RSquared}");
                if (bestScore < iterationResult?.Metrics.RSquared)
                {
                    bestScore = iterationResult.Metrics.RSquared;
                    bestIteration = iterationResult;
                    iterationsWithoutScoreImprovement = 0;
                }
                else
                {
                    iterationsWithoutScoreImprovement++;
                }

                // Stop iterations when one of the criteria is met 
                // 1) Best score is above 0.95 
                // 2) Score hasn't improved in last 10 iterations
                // 3) Total iterations has exceeded 30
                if (bestScore > 0.95 || 
                    totalIterations > 30 || 
                    iterationsWithoutScoreImprovement > 10)
                {
                    Console.WriteLine("Stopping early");
                    break;
                }
            }

            Console.WriteLine($"total iterations:{totalIterations}      bestscore:{bestScore}       iterations without improvement:{iterationsWithoutScoreImprovement}");

            Console.WriteLine("Press any key to continue..");
            Console.ReadLine();
        }
    }
}
