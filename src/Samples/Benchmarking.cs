using System;
using Microsoft.ML;
using Microsoft.ML.Auto;

namespace Samples
{
    public static class Benchmarking
    {
        const string DatasetName = "VirusPrediction";
        const string Label = "WnvPresent";
        const string DatasetPathPrefix = @"D:\SplitDatasets\";

        static readonly string TrainDataPath = $"{DatasetPathPrefix}{DatasetName}_train.csv";
        static readonly string ValidationDataPath = $"{DatasetPathPrefix}{DatasetName}_valid.csv";
        static readonly string TestDataPath = $"{DatasetPathPrefix}{DatasetName}_test.csv";

        public static void Run()
        {
            var context = new MLContext();
            var columnInference = context.Data.InferColumns(TrainDataPath, Label, true);
            var textLoader = context.Data.CreateTextReader(columnInference);
            var trainData = textLoader.Read(TrainDataPath);
            var validationData = textLoader.Read(ValidationDataPath);
            var testData = textLoader.Read(TestDataPath);
            var best = context.BinaryClassification.AutoFit(trainData, Label, validationData, settings:
                new AutoFitSettings()
                {
                    StoppingCriteria = new ExperimentStoppingCriteria()
                    {
                        MaxIterations = 200,
                        TimeOutInMinutes = 1000000000
                    }
                });
            var scoredTestData = best.BestPipeline.Model.Transform(testData);
            var testDataMetrics = context.BinaryClassification.EvaluateNonCalibrated(scoredTestData);

            Console.WriteLine(testDataMetrics.Accuracy);
            Console.ReadLine();
        }
    }
}
