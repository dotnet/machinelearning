using System;
using System.IO;
using System.Linq;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML.Samples
{
    public static class MulticlassClassificationExperiment
    {
        private static string TrainDataPath = "<Path to your train dataset goes here>";
        private static string TestDataPath = "<Path to your test dataset goes here>";
        private static string ModelPath = @"<Desired model output directory goes here>\OptDigitsModel.zip";
        private static string LabelColumnName = "Number";
        private static uint ExperimentTime = 60;

        public static void Run()
        {
            MLContext mlContext = new MLContext();

            // STEP 1: Load data
            IDataView trainDataView = mlContext.Data.LoadFromTextFile<PixelData>(TrainDataPath, separatorChar: ',');
            IDataView testDataView = mlContext.Data.LoadFromTextFile<PixelData>(TestDataPath, separatorChar: ',');

            // STEP 2: Run AutoML experiment
            Console.WriteLine($"Running AutoML multiclass classification experiment for {ExperimentTime} seconds...");
            ExperimentResult<MulticlassClassificationMetrics> experimentResult = mlContext.Auto()
                .CreateMulticlassClassificationExperiment(ExperimentTime)
                .Execute(trainDataView, LabelColumnName);

            // STEP 3: Print metric from the best model
            RunDetail<MulticlassClassificationMetrics> bestRun = experimentResult.BestRun;
            Console.WriteLine($"Total models produced: {experimentResult.RunDetails.Count()}");
            Console.WriteLine($"Best model's trainer: {bestRun.TrainerName}");
            Console.WriteLine($"Metrics of best model from validation data --");
            PrintMetrics(bestRun.ValidationMetrics);

            // STEP 4: Evaluate test data
            IDataView testDataViewWithBestScore = bestRun.Model.Transform(testDataView);
            MulticlassClassificationMetrics testMetrics = mlContext.MulticlassClassification.Evaluate(testDataViewWithBestScore, labelColumnName: LabelColumnName);
            Console.WriteLine($"Metrics of best model on test data --");
            PrintMetrics(testMetrics);

            // STEP 5: Save the best model for later deployment and inferencing
            using (FileStream fs = File.Create(ModelPath))
                mlContext.Model.Save(bestRun.Model, trainDataView.Schema, fs);

            // STEP 6: Create prediction engine from the best trained model
            var predictionEngine = mlContext.Model.CreatePredictionEngine<PixelData, PixelPrediction>(bestRun.Model);

            // STEP 7: Initialize new pixel data, and get the predicted number
            var testPixelData = new PixelData
            {
                PixelValues = new float[] { 0, 0, 1, 8, 15, 10, 0, 0, 0, 3, 13, 15, 14, 14, 0, 0, 0, 5, 10, 0, 10, 12, 0, 0, 0, 0, 3, 5, 15, 10, 2, 0, 0, 0, 16, 16, 16, 16, 12, 0, 0, 1, 8, 12, 14, 8, 3, 0, 0, 0, 0, 10, 13, 0, 0, 0, 0, 0, 0, 11, 9, 0, 0, 0 }
            };
            var prediction = predictionEngine.Predict(testPixelData);
            Console.WriteLine($"Predicted number for test pixels: {prediction.Prediction}");

            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }

        private static void PrintMetrics(MulticlassClassificationMetrics metrics)
        {
            Console.WriteLine($"LogLoss: {metrics.LogLoss}");
            Console.WriteLine($"LogLossReduction: {metrics.LogLossReduction}");
            Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy}");
            Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy}");
        }
    }
}
