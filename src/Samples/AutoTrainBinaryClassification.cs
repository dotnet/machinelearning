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
    public class AutoTrainBinaryClassification
    {
        private static string BaseDatasetsLocation = "Data";
        private static string TrainDataPath = Path.Combine(BaseDatasetsLocation, "wikipedia-detox-250-line-data.tsv");
        private static string TestDataPath = Path.Combine(BaseDatasetsLocation, "wikipedia-detox-250-line-test.tsv");
        private static string ModelPath = Path.Combine(BaseDatasetsLocation, "SentimentModel.zip");
        private static uint ExperimentTime = 60;

        public static void Run()
        {
            MLContext mlContext = new MLContext();

            // STEP 1: Create text loader options
            var textLoaderOptions = new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("Label", DataKind.Boolean, 0),
                    new TextLoader.Column("Text", DataKind.String, 1),
                },
                HasHeader = true,
                Separators = new[] { '\t' }
            };

            // STEP 2: Load data
            TextLoader textLoader = mlContext.Data.CreateTextLoader(textLoaderOptions);
            IDataView trainDataView = textLoader.Load(TrainDataPath);
            IDataView testDataView = textLoader.Load(TestDataPath);

            // STEP 3: Auto featurize, auto train and auto hyperparameter tune
            Console.WriteLine($"Running AutoML binary classification experiment for {ExperimentTime} seconds...");
            IEnumerable<RunDetail<BinaryClassificationMetrics>> runDetails = mlContext.Auto()
                                                                             .CreateBinaryClassificationExperiment(ExperimentTime)
                                                                             .Execute(trainDataView);

            // STEP 4: Print metric from the best model
            RunDetail<BinaryClassificationMetrics> best = runDetails.Best();
            Console.WriteLine($"Total models produced: {runDetails.Count()}");
            Console.WriteLine($"Best model's trainer: {best.TrainerName}");
            Console.WriteLine($"Accuracy of best model from validation data: {best.ValidationMetrics.Accuracy}");

            // STEP 5: Evaluate test data
            IDataView testDataViewWithBestScore = best.Model.Transform(testDataView);
            BinaryClassificationMetrics testMetrics = mlContext.BinaryClassification.EvaluateNonCalibrated(testDataViewWithBestScore);
            Console.WriteLine($"Accuracy of best model on test data: {testMetrics.Accuracy}");

            // STEP 6: Save the best model for later deployment and inferencing
            using (FileStream fs = File.Create(ModelPath))
                mlContext.Model.Save(best.Model, textLoader, fs);

            // STEP 7: Create prediction engine from the best trained model
            var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(best.Model);

            // STEP 8: Initialize a new sentiment issue, and get the predicted sentiment
            var testSentimentIssue = new SentimentIssue
            {
                Text = "I hope this helps."
            };
            var prediction = predictionEngine.Predict(testSentimentIssue);
            Console.WriteLine($"Predicted sentiment for test issue: {prediction.Prediction}");

            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }
    }
}
