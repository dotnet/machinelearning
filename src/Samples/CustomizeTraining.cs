// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Auto;
using Microsoft.ML.Data;

namespace Samples
{
    static class CustomizeTraining
    {
        private static string BaseDatasetsLocation = @"../../../../src/Samples/Data";
        private static string TrainDataPath = $"{BaseDatasetsLocation}/taxi-fare-train.csv";
        private static string TestDataPath = $"{BaseDatasetsLocation}/taxi-fare-test.csv";
        private static string ModelPath = $"{BaseDatasetsLocation}/TaxiFareModel.zip";
        private static string LabelColumnName = "fare_amount";

        public static void Run()
        {
            MLContext mlContext = new MLContext();

            // STEP 1: Infer columns
            var columnInference = mlContext.AutoInference().InferColumns(TrainDataPath, LabelColumnName, ',');

            // STEP 2: Load data
            TextLoader textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderArgs);
            IDataView trainDataView = textLoader.Read(TrainDataPath);
            IDataView testDataView = textLoader.Read(TestDataPath);

            // STEP 3: Autofit with a callback configured
            var autoFitExperiment = mlContext.AutoInference().CreateRegressionExperiment(new RegressionExperimentSettings()
            {
                MaxInferenceTimeInSeconds = 20,
                OptimizingMetric = RegressionMetric.L2,
                WhitelistedTrainers = new[] { RegressionTrainer.LightGbm },
                ProgressCallback = new Progress()
            });
            autoFitExperiment.Execute(trainDataView, LabelColumnName);

            Console.WriteLine("Press any key to continue..");
            Console.ReadLine();
        }
    }
}
