// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Auto;
using Microsoft.ML.Data;

namespace Samples
{
    static class RefitBestModel
    {
        private static string BaseDatasetsLocation = "Data";
        private static string TrainDataPath = Path.Combine(BaseDatasetsLocation, "taxi-fare-train.csv");
        private static string SmallTrainDataPath = Path.Combine(BaseDatasetsLocation, "taxi-fare-small-train.csv");
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

            // STEP 3: Subsample training data, for faster AutoML experimentation time
            IDataView smallTrainDataView = textLoader.Load(SmallTrainDataPath);

            // STEP 4: Auto-featurization, model selection, and hyperparameter tuning
            Console.WriteLine($"Running AutoML regression classification experiment for {ExperimentTime} seconds...");
            IEnumerable<RunResult<RegressionMetrics>> runResults = mlContext.Auto()
                                                                   .CreateRegressionExperiment(ExperimentTime)
                                                                   .Execute(smallTrainDataView, LabelColumn);

            // STEP 5: Refit best model on entire training data
            RunResult<RegressionMetrics> best = runResults.Best();
            var refitBestModel = best.Estimator.Fit(trainDataView);

            // STEP 6: Evaluate test data
            IDataView testDataViewWithBestScore = refitBestModel.Transform(testDataView);
            RegressionMetrics testMetrics = mlContext.Regression.Evaluate(testDataViewWithBestScore, label: LabelColumn);
            Console.WriteLine($"RSquared of the re-fit model on test data: {testMetrics.RSquared}");

            // STEP 7: Save the re-fit best model for later deployment and inferencing
            using (FileStream fs = File.Create(ModelPath))
                refitBestModel.SaveTo(mlContext, fs);

            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }
    }
}
