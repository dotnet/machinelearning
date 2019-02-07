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
    public class AutoTrainMulticlassClassification
    {
        private static string BaseDatasetsLocation = @"../../../../src/Samples/Data";
        private static string TrainDataPath = $"{BaseDatasetsLocation}/iris-train.txt";
        private static string TestDataPath = $"{BaseDatasetsLocation}/iris-test.txt";
        private static string ModelPath = $"{BaseDatasetsLocation}/IrisClassificationModel.zip";

        public static void Run()
        {
            //Create ML Context with seed for repeteable/deterministic results
            MLContext mlContext = new MLContext(seed: 0);

            // STEP 1: Common data loading configuration
            var textLoader = mlContext.Data.CreateTextLoader(
                                                                new TextLoader.Arguments()
                                                                {
                                                                    Separators = new[] { '\t' },
                                                                    HasHeader = true,
                                                                    Column = new[]
                                                                    {
                                                                        new TextLoader.Column("Label", DataKind.R4, 0),
                                                                        new TextLoader.Column("SepalLength", DataKind.R4, 1),
                                                                        new TextLoader.Column("SepalWidth", DataKind.R4, 2),
                                                                        new TextLoader.Column("PetalLength", DataKind.R4, 3),
                                                                        new TextLoader.Column("PetalWidth", DataKind.R4, 4),
                                                                    }
                                                                });

            IDataView trainDataView = textLoader.Read(TrainDataPath);
            IDataView testDataView = textLoader.Read(TestDataPath);

            // STEP 2: Auto featurize, auto train and auto hyperparameter tuning
            var autoFitResults = mlContext.MulticlassClassification.AutoFit(trainDataView, timeoutInMinutes: 1);

            // STEP 3: Print metrics for each iteration 
            int iterationIndex = 0;
            PrintMulticlassClassificationMetricsHeader();

            IDataView testDataViewWithBestScore = null;
            IterationResult<MultiClassClassifierMetrics> bestIteration = null;
            double bestScore = 0;

            foreach (var iterationResult in autoFitResults)
            {
                if (iterationResult.Exception != null)
                {
                    Console.WriteLine(iterationResult.Exception);
                    continue;
                }

                IDataView testDataViewWithScore = iterationResult.Model.Transform(testDataView);
                var testMetrics = mlContext.MulticlassClassification.Evaluate(testDataViewWithScore, label: "Label", score: "Score");
                if (bestScore < iterationResult.Metrics.AccuracyMacro)
                {
                    bestScore = iterationResult.Metrics.AccuracyMacro;
                    bestIteration = iterationResult;
                    testDataViewWithBestScore = testDataViewWithScore;
                }

                ++iterationIndex;
                PrintMulticlassClassificationMetrics(iterationIndex, iterationResult.TrainerName, "validation", iterationResult.Metrics);
                PrintMulticlassClassificationMetrics(iterationIndex, iterationResult.TrainerName, "test", testMetrics);
                Console.WriteLine();
            }

            // STEP 4: Compare and print actual value vs predicted value for top 5 rows from validation data
            PrintActualVersusPredictedHeader();
            IEnumerable<uint> labels = testDataViewWithBestScore.GetColumn<uint>(mlContext, DefaultColumnNames.Label);
            IEnumerable<uint> scores = testDataViewWithBestScore.GetColumn<uint>(mlContext, DefaultColumnNames.PredictedLabel);
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

        static void PrintMulticlassClassificationMetrics(int iteration, string trainerName, string typeOfMetrics, MultiClassClassifierMetrics metrics)
        {
            Console.WriteLine($"{iteration,-3}{trainerName,-35}{typeOfMetrics,-15}{metrics.AccuracyMacro,-15:0.####}{metrics.AccuracyMicro,-15:0.####}{metrics.LogLossReduction,-15:0.##}");
        }

        static void PrintActualVersusPredictedValue(int index, uint label, uint predictedLabel)
        {
            Console.WriteLine($"{index,-5}{label,-15}{predictedLabel,15}");
        }

        static void PrintMulticlassClassificationMetricsHeader()
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for multiclass classification model      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"{"  ",-3}{"Trainer",-35}{"Type",-15}{"AccuracyMacro",-15}{"AccuracyMicro",-15}{"LogLossReduction",-15}");
        }

        static void PrintActualVersusPredictedHeader()
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Actual value Vs predicted value      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"{"Row",-5}{"Actual",-15}{"Predicted",15}");
        }
    }
}
