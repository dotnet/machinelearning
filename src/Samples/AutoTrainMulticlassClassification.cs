// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Auto;
using Microsoft.ML.Data;

namespace Samples
{
    public class AutoTrainMulticlassClassification
    {
        private static string BaseDatasetsLocation = @"../../../../src/Samples/Data";
        private static string TrainDataPath = $"{BaseDatasetsLocation}/iris-train.txt";
        private static string TestDataPath = $"{BaseDatasetsLocation}/iris-test.txt";
        private static string ModelPath = $"{BaseDatasetsLocation}/IrisClassificationModel.zip";

        public static void Run()
        {
            MLContext mlContext = new MLContext();

            // STEP 1: Infer columns
            var columnInference = mlContext.AutoInference().InferColumns(TrainDataPath);

            // STEP 2: Load data
            TextLoader textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderArgs);
            IDataView trainDataView = textLoader.Read(TrainDataPath);
            IDataView testDataView = textLoader.Read(TestDataPath);

            // STEP 3: Auto featurize, auto train and auto hyperparameter tune
            Console.WriteLine($"Invoking new AutoML multiclass classification experiment...");
            var runResults = mlContext.AutoInference()
                .CreateMulticlassClassificationExperiment(60)
                .Execute(trainDataView);

            // STEP 4: Print metric from the best model
            var best = runResults.Best();
            Console.WriteLine($"AccuracyMacro of best model from validation data: {best.Metrics.AccuracyMacro}");

            // STEP 5: Evaluate test data
            IDataView testDataViewWithBestScore = best.Model.Transform(testDataView);
            var testMetrics = mlContext.MulticlassClassification.Evaluate(testDataViewWithBestScore);
            Console.WriteLine($"AccuracyMacro of best model on test data: {testMetrics.AccuracyMacro}");

            // STEP 6: Save the best model for later deployment and inferencing
            using (var fs = File.Create(ModelPath))
                best.Model.SaveTo(mlContext, fs);

            Console.WriteLine("Press any key to continue...");
            Console.ReadLine();
        }
    }
}
