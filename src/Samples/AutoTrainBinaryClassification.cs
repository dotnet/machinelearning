using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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

        public static void Run()
        {
            //Create ML Context with seed for repeteable/deterministic results
            MLContext mlContext = new MLContext(seed: 0);

            // STEP 1: Common data loading configuration
            TextLoader textLoader = mlContext.Data.CreateTextReader(
                                                       columns: new[]
                                                                   {
                                                                    new TextLoader.Column("Label", DataKind.Bool, 0),
                                                                    new TextLoader.Column("Text", DataKind.Text, 1)
                                                                   },
                                                       hasHeader: true,
                                                       separatorChar: '\t'
                                                       );

            IDataView trainDataView = textLoader.Read(TrainDataPath);
            IDataView testDataView = textLoader.Read(TestDataPath);

            // STEP 2: Auto featurize, auto train and auto hyperparameter tuning
            var autoFitResults = mlContext.BinaryClassification.AutoFit(trainDataView, timeoutInMinutes: 1);

            // STEP 3: Print metrics for each iteration 
            int iterationIndex = 0;
            PrintBinaryClassificationMetricsHeader();

            IDataView testDataViewWithBestScore = null;
            IterationResult<BinaryClassificationMetrics> bestIteration = null;
            double bestScore = 0;

            foreach (var iterationResult in autoFitResults)
            {
                if (iterationResult.Exception != null)
                {
                    Console.WriteLine(iterationResult.Exception);
                    continue;
                }

                IDataView testDataViewWithScore = iterationResult.Model.Transform(testDataView);
                var testMetrics = mlContext.BinaryClassification.EvaluateNonCalibrated(testDataViewWithScore, label: "Label", score: "Score");
                if (bestScore < iterationResult.Metrics.Accuracy)
                {
                    bestScore = iterationResult.Metrics.Accuracy;
                    bestIteration = iterationResult;
                    testDataViewWithBestScore = testDataViewWithScore;
                }

                ++iterationIndex;
                PrintBinaryClassificationMetrics(iterationIndex, "validation metrics", iterationResult.Metrics);
                PrintBinaryClassificationMetrics(iterationIndex, "test metrics      ", testMetrics);
                Console.WriteLine();
            }

            // STEP 4: Compare and print actual value vs predicted value for top 5 rows from validation data
            PrintActualVersusPredictedHeader();
            IEnumerable<bool> labels = testDataViewWithBestScore.GetColumn<bool>(mlContext, DefaultColumnNames.Label);
            IEnumerable<float> scores = testDataViewWithBestScore.GetColumn<float>(mlContext, DefaultColumnNames.Score);
            int rowCount = 1;
            do
            {
                PrintActualVersusPredictedValue(rowCount, labels.ElementAt(rowCount), scores.ElementAt(rowCount));

            } while (rowCount++ <= 5);

            // STEP 5: Save the best model for later deployment and inferencing
            using (var fs = File.Create(ModelPath))
                bestIteration.Model.SaveTo(mlContext, fs);

            Console.WriteLine("Press any key to continue..");
            Console.ReadLine();
        }
        
        static void PrintBinaryClassificationMetrics(int iteration, string typeOfMetrics, BinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"{iteration}       {typeOfMetrics}       {metrics.Accuracy:P2}        {metrics.Auc:P2}        {metrics.F1Score:P2}        {metrics.PositivePrecision:#.##}        {metrics.PositiveRecall:#.##}");
        }

        static void PrintActualVersusPredictedValue(int index, bool label, float score)
        {
            Console.WriteLine($"{index}        {label}        {(score == 0 ? false : true)}");
        }

        static void PrintBinaryClassificationMetricsHeader()
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for binary classification model      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"iteration   type        Accuracy      Auc    F1Score   PositivePrecision    PositiveRecall");
        }

        static void PrintActualVersusPredictedHeader()
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Actual value Vs predicted value      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"Row    Actual Label  Predicted Label");
        }
    }
}
