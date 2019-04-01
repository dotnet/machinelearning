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
using Samples.DataStructures;

namespace Samples
{
    public class AutoTrainMulticlassClassification
    {
        private static string BaseDatasetsLocation = "Data";
        private static string TrainDataPath = Path.Combine(BaseDatasetsLocation, "optdigits-train.csv");
        private static string TestDataPath = Path.Combine(BaseDatasetsLocation, "optdigits-test.csv");
        private static string ModelPath = Path.Combine(BaseDatasetsLocation, "OptDigits.zip");
        private static uint ExperimentTime = 60;

        public static void Run()
        {
            MLContext mlContext = new MLContext();

            // STEP 1: Create text loader options
            var textLoaderOptions = new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("PixelValues", DataKind.Single, 0, 63),
                    new TextLoader.Column("Label", DataKind.Single, 64),
                },
                HasHeader = true,
                Separators = new[] { ',' }
            };

            // STEP 2: Load data
            TextLoader textLoader = mlContext.Data.CreateTextLoader(textLoaderOptions);
            IDataView trainDataView = textLoader.Load(TrainDataPath);
            IDataView testDataView = textLoader.Load(TestDataPath);

            // STEP 3: Auto featurize, auto train and auto hyperparameter tune
            Console.WriteLine($"Running AutoML multiclass classification experiment for {ExperimentTime} seconds...");
            IEnumerable<RunResult<MultiClassClassifierMetrics>> runResults = mlContext.Auto()
                                                                             .CreateMulticlassClassificationExperiment(ExperimentTime)
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

            // STEP 7: Create prediction engine from the best trained model
            var predictionEngine = best.Model.CreatePredictionEngine<PixelData, PixelPrediction>(mlContext);

            // STEP 8: Initialize new pixel data, and get the predicted number
            var testPixelData = new PixelData
            {
                PixelValues = new float[] { 0, 0, 1, 8, 15, 10, 0, 0, 0, 3, 13, 15, 14, 14, 0, 0, 0, 5, 10, 0, 10, 12, 0, 0, 0, 0, 3, 5, 15, 10, 2, 0, 0, 0, 16, 16, 16, 16, 12, 0, 0, 1, 8, 12, 14, 8, 3, 0, 0, 0, 0, 10, 13, 0, 0, 0, 0, 0, 0, 11, 9, 0, 0, 0 }
            };
            var prediction = predictionEngine.Predict(testPixelData);
            Console.WriteLine($"Predicted number for test pixels: {prediction.Prediction}");

            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }
    }
}
