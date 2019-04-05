// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Auto;
using Microsoft.ML.Data;
using Samples.DataStructures;

namespace Samples
{
    static class AutoTrainRegression
    {
        private static string BaseDatasetsLocation = "Data";
        private static string TrainDataPath = Path.Combine(BaseDatasetsLocation, "taxi-fare-train.csv");
        private static string TestDataPath = Path.Combine(BaseDatasetsLocation, "taxi-fare-test.csv");
        private static string ModelPath = Path.Combine(BaseDatasetsLocation, "TaxiFareModel.zip");
        private static string LabelColumn = "FareAmount";
        private static uint ExperimentTime = 60;

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

            // STEP 3: Auto featurize, auto train and auto hyperparameter tune
            Console.WriteLine($"Running AutoML regression experiment for {ExperimentTime} seconds...");
            IEnumerable<RunDetail<RegressionMetrics>> runDetails = mlContext.Auto()
                                                                   .CreateRegressionExperiment(ExperimentTime)
                                                                   .Execute(trainDataView, LabelColumn);

            // STEP 4: Print metric from best model
            RunDetail<RegressionMetrics> best = runDetails.Best();
            Console.WriteLine($"Total models produced: {runDetails.Count()}");
            Console.WriteLine($"Best model's trainer: {best.TrainerName}");
            Console.WriteLine($"RSquared of best model from validation data: {best.ValidationMetrics.RSquared}");

            // STEP 5: Evaluate test data
            IDataView testDataViewWithBestScore = best.Model.Transform(testDataView);
            RegressionMetrics testMetrics = mlContext.Regression.Evaluate(testDataViewWithBestScore, labelColumnName: LabelColumn);
            Console.WriteLine($"RSquared of best model on test data: {testMetrics.RSquared}");

            // STEP 6: Save the best model for later deployment and inferencing
            using (FileStream fs = File.Create(ModelPath))
                mlContext.Model.Save(best.Model, textLoader, fs);

            // STEP 7: Create prediction engine from the best trained model
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(best.Model);

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
    }
}
