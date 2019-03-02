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
    public class AutoTrainBinaryClassification
    {
        private static string BaseDatasetsLocation = Path.Combine("..", "..", "..", "..", "src", "Samples", "Data");
        private static string TrainDataPath = Path.Combine(BaseDatasetsLocation, "wikipedia-detox-250-line-data.tsv");
        private static string TestDataPath = Path.Combine(BaseDatasetsLocation, "wikipedia-detox-250-line-test.tsv");
        private static string ModelPath = Path.Combine(BaseDatasetsLocation, "SentimentModel.zip");
        private static string LabelColumn = "Sentiment";
        private static uint ExperimentTime = 60;

        public static void Run()
        {
            MLContext mlContext = new MLContext();

            // STEP 1: Infer columns
            ColumnInferenceResults columnInference = mlContext.Auto().InferColumns(TrainDataPath, LabelColumn);
            ConsoleHelper.Print(columnInference);

            // STEP 2: Load data
            TextLoader textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            IDataView trainDataView = textLoader.Load(TrainDataPath);
            IDataView testDataView = textLoader.Load(TestDataPath);

            // STEP 3: Auto featurize, auto train and auto hyperparameter tune
            Console.WriteLine($"Running AutoML binary classification experiment for {ExperimentTime} seconds...");
            IEnumerable<RunResult<BinaryClassificationMetrics>> runResults = mlContext.Auto()
                                                                             .CreateBinaryClassificationExperiment(ExperimentTime)
                                                                             .Execute(trainDataView, LabelColumn);

            // STEP 4: Print metric from the best model
            RunResult<BinaryClassificationMetrics> best = runResults.Best();
            Console.WriteLine($"Total models produced: {runResults.Count()}");
            Console.WriteLine($"Best model's trainer: {best.TrainerName}");
            Console.WriteLine($"Accuracy of best model from validation data: {best.ValidationMetrics.Accuracy}");

            // STEP 5: Evaluate test data
            IDataView testDataViewWithBestScore = best.Model.Transform(testDataView);
            BinaryClassificationMetrics testMetrics = mlContext.BinaryClassification.EvaluateNonCalibrated(testDataViewWithBestScore, label: LabelColumn);
            Console.WriteLine($"Accuracy of best model on test data: {testMetrics.Accuracy}");

            // STEP 6: Save the best model for later deployment and inferencing
            using (FileStream fs = File.Create(ModelPath))
                best.Model.SaveTo(mlContext, fs);

            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }
    }
}
