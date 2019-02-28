// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Auto;
using Microsoft.ML.Data;

namespace Samples
{
    public class AutoTrainBinaryClassification
    {
        private static string BaseDatasetsLocation = @"../../../../src/Samples/Data";
        private static string TrainDataPath = $"{BaseDatasetsLocation}/wikipedia-detox-250-line-data.tsv";
        private static string TestDataPath = $"{BaseDatasetsLocation}/wikipedia-detox-250-line-test.tsv";
        private static string ModelPath = $"{BaseDatasetsLocation}/SentimentModel.zip";
        private static string LabelColumn = "Sentiment";
        private static uint ExperimentTime = 60;

        public static void Run()
        {
            MLContext mlContext = new MLContext();

            // STEP 1: Infer columns
            var columnInference = mlContext.Auto().InferColumns(TrainDataPath, LabelColumn);

            // STEP 2: Load data
            var textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderArgs);
            var trainDataView = textLoader.Read(TrainDataPath);
            var testDataView = textLoader.Read(TestDataPath);

            // STEP 3: Auto featurize, auto train and auto hyperparameter tune
            Console.WriteLine($"Running AutoML binary classification experiment for {ExperimentTime} seconds...");
            var runResults = mlContext.Auto()
                .CreateBinaryClassificationExperiment(ExperimentTime)
                .Execute(trainDataView, LabelColumn);

            // STEP 4: Print metric from the best model
            var best = runResults.Best();
            Console.WriteLine($"Total models produced: {runResults.Count()}");
            Console.WriteLine($"Best model's trainer: {best.TrainerName}");
            Console.WriteLine($"Accuracy of best model from validation data: {best.ValidationMetrics.Accuracy}");

            // STEP 5: Evaluate test data
            var testDataViewWithBestScore = best.Model.Transform(testDataView);
            var testMetrics = mlContext.BinaryClassification.EvaluateNonCalibrated(testDataViewWithBestScore, label: LabelColumn);
            Console.WriteLine($"Accuracy of best model on test data: {testMetrics.Accuracy}");

            // STEP 6: Save the best model for later deployment and inferencing
            using (var fs = File.Create(ModelPath))
                best.Model.SaveTo(mlContext, fs);

            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }
    }
}
