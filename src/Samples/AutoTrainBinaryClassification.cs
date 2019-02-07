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

        public static void Run()
        {
            //Create ML Context with seed for repeteable/deterministic results
            MLContext mlContext = new MLContext(seed: 0);

            // STEP 1: Common data loading configuration
            TextLoader textLoader = mlContext.Data.CreateTextLoader(
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
                PrintBinaryClassificationMetrics(iterationIndex, iterationResult.TrainerName, "validation", iterationResult.Metrics);
                PrintBinaryClassificationMetrics(iterationIndex, iterationResult.TrainerName, "test", testMetrics);
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
        
        static void PrintBinaryClassificationMetrics(int iteration, string trainerName, string typeOfMetrics, BinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"{iteration,-3}{trainerName,-35}{typeOfMetrics,-15}{metrics.Accuracy,-15:P2}{metrics.Auc,-15:P2}{metrics.F1Score,-8:P2}{metrics.PositivePrecision,-15:#.##}{metrics.PositiveRecall,-12:#.##}");
        }

        static void PrintActualVersusPredictedValue(int index, bool label, float score)
        {
            Console.WriteLine($"{index,-5}{label,-15}{(score == 0 ? false : true),-15}");
        }

        static void PrintBinaryClassificationMetricsHeader()
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for binary classification model      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"{" ",-3}{"Trainer",-35}{"Type",-15}{"Accuracy",-15}{"Auc",-15}{"F1Score",-8}{"P-Precision",-15}{"P-Recall",-12:#.##}");
        }

        static void PrintActualVersusPredictedHeader()
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Actual value Vs predicted value      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"{"Row",-5}{"Actual",-15}{"Predicted",-15}");
        }
    }
}
