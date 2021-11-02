using System;
using System.IO;
using System.Linq;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML.Samples
{
    public static class RegressionExperiment
    {
        private static string TrainDataPath = "<Path to your train dataset goes here>";
        private static string TestDataPath = "<Path to your test dataset goes here>";
        private static string ModelPath = @"<Desired model output directory goes here>\TaxiFareModel.zip";
        private static string LabelColumnName = "FareAmount";
        private static uint ExperimentTime = 60;

        public static void Run()
        {
            MLContext mlContext = new MLContext();

            // STEP 1: Load data
            IDataView trainDataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(TrainDataPath, hasHeader: true, separatorChar: ',');
            IDataView testDataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(TestDataPath, hasHeader: true, separatorChar: ',');

            // STEP 2: Run AutoML experiment
            Console.WriteLine($"Running AutoML regression experiment for {ExperimentTime} seconds...");
            ExperimentResult<RegressionMetrics> experimentResult = mlContext.Auto()
                .CreateRegressionExperiment(ExperimentTime)
                .Execute(trainDataView, LabelColumnName);

            // STEP 3: Print metric from best model
            RunDetail<RegressionMetrics> bestRun = experimentResult.BestRun;
            Console.WriteLine($"Total models produced: {experimentResult.RunDetails.Count()}");
            Console.WriteLine($"Best model's trainer: {bestRun.TrainerName}");
            Console.WriteLine($"Metrics of best model from validation data --");
            PrintMetrics(bestRun.ValidationMetrics);

            // STEP 5: Evaluate test data
            IDataView testDataViewWithBestScore = bestRun.Model.Transform(testDataView);
            RegressionMetrics testMetrics = mlContext.Regression.Evaluate(testDataViewWithBestScore, labelColumnName: LabelColumnName);
            Console.WriteLine($"Metrics of best model on test data --");
            PrintMetrics(testMetrics);

            // STEP 6: Save the best model for later deployment and inferencing
            using (FileStream fs = File.Create(ModelPath))
                mlContext.Model.Save(bestRun.Model, trainDataView.Schema, fs);

            // STEP 7: Create prediction engine from the best trained model
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(bestRun.Model);

            // STEP 8: Initialize a new test taxi trip, and get the predicted fare
            var testTaxiTrip = new TaxiTrip
            {
                VendorId = "VTS",
                RateCode = 1,
                PassengerCount = 1,
                TripTimeInSeconds = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD"
            };
            var prediction = predictionEngine.Predict(testTaxiTrip);
            Console.WriteLine($"Predicted fare for test taxi trip: {prediction.FareAmount}");

            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }

        private static void PrintMetrics(RegressionMetrics metrics)
        {
            Console.WriteLine($"MeanAbsoluteError: {metrics.MeanAbsoluteError}");
            Console.WriteLine($"MeanSquaredError: {metrics.MeanSquaredError}");
            Console.WriteLine($"RootMeanSquaredError: {metrics.RootMeanSquaredError}");
            Console.WriteLine($"RSquared: {metrics.RSquared}");
        }
    }
}
