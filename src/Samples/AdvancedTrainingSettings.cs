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

namespace Samples
{
    static class AdvancedTrainingSettings
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

            // STEP 3: Build a pre-featurizer for use in our AutoML experiment
            IEstimator<ITransformer> preFeaturizer = mlContext.Transforms.Conversion.MapValue("IsCash", 
                new[] { new KeyValuePair<string, bool>("CSH", true) }, "PaymentType");

            // STEP 4: Initialize custom column information for use in AutoML experiment
            ColumnInformation columnInformation = new ColumnInformation() { LabelColumnName = LabelColumn };
            columnInformation.CategoricalColumnNames.Add("VendorId");
            columnInformation.IgnoredColumnNames.Add("PaymentType");

            // STEP 5: Run AutoML experiment
            Console.WriteLine($"Running AutoML regression experiment for {ExperimentTime} seconds...");
            IEnumerable<RunDetails<RegressionMetrics>> runDetails = mlContext.Auto()
                                                                   .CreateRegressionExperiment(ExperimentTime)
                                                                   .Execute(trainDataView, columnInformation, preFeaturizer);

            // STEP 6: Print metric from best model
            RunDetails<RegressionMetrics> best = runDetails.Best();
            Console.WriteLine($"Total models produced: {runDetails.Count()}");
            Console.WriteLine($"Best model's trainer: {best.TrainerName}");
            Console.WriteLine($"RSquared of best model from validation data: {best.ValidationMetrics.RSquared}");

            // STEP 7: Save the best model for later deployment and inferencing
            using (FileStream fs = File.Create(ModelPath))
                mlContext.Model.Save(best.Model, textLoader, fs);

            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }
    }
}
