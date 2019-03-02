// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Auto;
using Microsoft.ML.Data;
using Samples.Helpers;

namespace Samples
{
    public class AutoTrainMulticlassClassification
    {
        private static string BaseDatasetsLocation = Path.Combine("..", "..", "..", "..", "src", "Samples", "Data");
        private static string TrainDataPath = Path.Combine(BaseDatasetsLocation, "optdigits-train.csv");
        private static string TestDataPath = Path.Combine(BaseDatasetsLocation, "optdigits-test.csv");
        private static string ModelPath = Path.Combine(BaseDatasetsLocation, "OptDigits.zip");
        private static uint ExperimentTime = 60;

        public static void Run()
        {
            MLContext mlContext = new MLContext();

            // STEP 1: Infer columns
            ColumnInferenceResults columnInference = mlContext.Auto().InferColumns(TrainDataPath);
            ConsoleHelper.Print(columnInference);

            // STEP 2: Load data
            TextLoader textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            IDataView trainDataView = textLoader.Load(TrainDataPath);
            IDataView testDataView = textLoader.Load(TestDataPath);

            // STEP 3: Auto featurize, auto train and auto hyperparameter tune
            Console.WriteLine($"Running AutoML multiclass classification experiment for {ExperimentTime} seconds...");
            IEnumerable<RunResult<MultiClassClassifierMetrics>> runResults = mlContext.Auto()
                                                                             .CreateMulticlassClassificationExperiment(60)
                                                                             .Execute(trainDataView);

            // STEP 4: Print metric from the best model
            RunResult<MultiClassClassifierMetrics> best = runResults.Best();
            Console.WriteLine($"Total models produced: {runResults.Count()}");
            Console.WriteLine($"Best model's trainer: {best.TrainerName}");
            Console.WriteLine($"AccuracyMacro of best model from validation data: {best.ValidationMetrics.AccuracyMacro}");

            // STEP 5: Evaluate test data
            IDataView testDataViewWithBestScore = best.Model.Transform(testDataView);
            MultiClassClassifierMetrics testMetrics = mlContext.MulticlassClassification.Evaluate(testDataViewWithBestScore);
            Console.WriteLine($"AccuracyMacro of best model on test data: {testMetrics.AccuracyMacro}");

            // STEP 6: Save the best model for later deployment and inferencing
            using (FileStream fs = File.Create(ModelPath))
                best.Model.SaveTo(mlContext, fs);

            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }
    }
}
