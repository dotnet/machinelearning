using System;
using System.IO;
using System.Linq;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML.Samples
{
    public static class BinaryClassificationExperiment
    {
        private static string TrainDataPath = "<Path to your train dataset goes here>";
        private static string TestDataPath = "<Path to your test dataset goes here>";
        private static string ModelPath = @"<Desired model output directory goes here>\SentimentModel.zip";
        private static uint ExperimentTime = 60;

        public static void Run()
        {
            MLContext mlContext = new MLContext();

            // STEP 1: Load data
            IDataView trainDataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(TrainDataPath, hasHeader: true);
            IDataView testDataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(TestDataPath, hasHeader: true);

            // STEP 2: Run AutoML experiment
            Console.WriteLine($"Running AutoML binary classification experiment for {ExperimentTime} seconds...");
            ExperimentResult<BinaryClassificationMetrics> experimentResult = mlContext.Auto()
                .CreateBinaryClassificationExperiment(ExperimentTime)
                .Execute(trainDataView);

            // STEP 3: Print metric from the best model
            RunDetail<BinaryClassificationMetrics> bestRun = experimentResult.BestRun;
            Console.WriteLine($"Total models produced: {experimentResult.RunDetails.Count()}");
            Console.WriteLine($"Best model's trainer: {bestRun.TrainerName}");
            Console.WriteLine($"Metrics of best model from validation data --");
            PrintMetrics(bestRun.ValidationMetrics);

            // STEP 4: Evaluate test data
            IDataView testDataViewWithBestScore = bestRun.Model.Transform(testDataView);
            BinaryClassificationMetrics testMetrics = mlContext.BinaryClassification.EvaluateNonCalibrated(testDataViewWithBestScore);
            Console.WriteLine($"Metrics of best model on test data --");
            PrintMetrics(testMetrics);

            // STEP 5: Save the best model for later deployment and inferencing
            using (FileStream fs = File.Create(ModelPath))
                mlContext.Model.Save(bestRun.Model, trainDataView.Schema, fs);

            // STEP 6: Create prediction engine from the best trained model
            var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(bestRun.Model);

            // STEP 7: Initialize a new sentiment issue, and get the predicted sentiment
            var testSentimentIssue = new SentimentIssue
            {
                Text = "I hope this helps."
            };
            var prediction = predictionEngine.Predict(testSentimentIssue);
            Console.WriteLine($"Predicted sentiment for test issue: {prediction.Prediction}");

            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }

        private static void PrintMetrics(BinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"Accuracy: {metrics.Accuracy}");
            Console.WriteLine($"AreaUnderPrecisionRecallCurve: {metrics.AreaUnderPrecisionRecallCurve}");
            Console.WriteLine($"AreaUnderRocCurve: {metrics.AreaUnderRocCurve}");
            Console.WriteLine($"F1Score: {metrics.F1Score}");
            Console.WriteLine($"NegativePrecision: {metrics.NegativePrecision}");
            Console.WriteLine($"NegativeRecall: {metrics.NegativeRecall}");
            Console.WriteLine($"PositivePrecision: {metrics.PositivePrecision}");
            Console.WriteLine($"PositiveRecall: {metrics.PositiveRecall}");
        }
    }
}
