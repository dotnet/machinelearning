using System;
using System.Collections.Generic;
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
        private static string LabelColumnName = "Label";

        public static void Run()
        {
            //Create ML Context with seed for repeteable/deterministic results
            MLContext mlContext = new MLContext(seed: 0);

            // STEP 1: Infer columns
            var columnInference = mlContext.Data.InferColumns(TrainDataPath, LabelColumnName, '\t');

            // STEP 2: Load data
            TextLoader textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderArgs);
            IDataView trainDataView = textLoader.Read(TrainDataPath);
            IDataView testDataView = textLoader.Read(TestDataPath);

            // STEP 3: Auto featurize, auto train and auto hyperparameter tuning
            Console.WriteLine($"Invoking BinaryClassification.AutoFit");
            var autoFitResults = mlContext.BinaryClassification.AutoFit(trainDataView, timeoutInSeconds: 60);

            // STEP 4: Print metric from the best model
            var best = autoFitResults.Best();
            Console.WriteLine($"Accuracy of best model from validation data {best.Metrics.Accuracy}");

            // STEP 5: Evaluate test data
            IDataView testDataViewWithBestScore = best.Model.Transform(testDataView);
            var testMetrics = mlContext.Regression.Evaluate(testDataViewWithBestScore, label: DefaultColumnNames.Label, DefaultColumnNames.Score);
            Console.WriteLine($"Accuracy of best model from test data {best.Metrics.Accuracy}");

            // STEP 6: Save the best model for later deployment and inferencing
            using (var fs = File.Create(ModelPath))
                best.Model.SaveTo(mlContext, fs);

            Console.WriteLine("Press any key to continue..");
            Console.ReadLine();
        }
    }
}
