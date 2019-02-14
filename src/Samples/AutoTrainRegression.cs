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
    static class AutoTrainRegression
    {
        private static string BaseDatasetsLocation = @"../../../../src/Samples/Data";
        private static string TrainDataPath = $"{BaseDatasetsLocation}/taxi-fare-train.csv";
        private static string TestDataPath = $"{BaseDatasetsLocation}/taxi-fare-test.csv";
        private static string ModelPath = $"{BaseDatasetsLocation}/TaxiFareModel.zip";
        private static string LabelColumnName = "fare_amount";

        public static void Run()
        {
            MLContext mlContext = new MLContext();

            // STEP 1: Common data loading configuration
            var columnInference = mlContext.Data.InferColumns(TrainDataPath, LabelColumnName, ',');

            // STEP 2: Load data
            TextLoader textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderArgs);
            IDataView trainDataView = textLoader.Read(TrainDataPath);
            IDataView testDataView = textLoader.Read(TestDataPath);

            // STEP 3: Auto featurize, auto train and auto hyperparameter tuning
            Console.WriteLine($"Invoking Regression.AutoFit");
            var autoFitResults = mlContext.Regression.AutoFit(trainDataView, LabelColumnName, timeoutInSeconds: 60);
            
            // STEP 4: Compare and print actual value vs predicted value for top 5 rows from validation data
            var best = autoFitResults.Best();
            Console.WriteLine($"RSquared of best model from validation data: {best.Metrics.RSquared}");

            // STEP 5: Evaluate test data
            IDataView testDataViewWithBestScore = best.Model.Transform(testDataView);
            var testMetrics = mlContext.Regression.Evaluate(testDataViewWithBestScore, label: DefaultColumnNames.Label, DefaultColumnNames.Score);
            Console.WriteLine($"RSquared of best model from test data: {best.Metrics.RSquared}");

            // STEP 6: Save the best model for later deployment and inferencing
            using (var fs = File.Create(ModelPath))
                best.Model.SaveTo(mlContext, fs);

            Console.WriteLine("Press any key to continue..");
            Console.ReadLine();
        }
    }
}
