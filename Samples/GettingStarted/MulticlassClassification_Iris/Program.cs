using System;
using System.IO;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML;
using System.Threading.Tasks;

namespace MulticlassClassification_Iris
{
    public static partial class Program
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string TrainDataPath => Path.Combine(AppPath, @"..\..\..\..\datasets\", "iris_train.txt");
        private static string TestDataPath => Path.Combine(AppPath, @"..\..\..\..\datasets\", "iris_test.txt");
        private static string ModelPath => Path.Combine(AppPath,  "Models", "IrisModel.zip");

        private static async Task Main(string[] args)
        {
            var model = await TrainAsync();

            Evaluate(model);

            Console.WriteLine();
            var prediction = model.Predict(TestIrisData.Iris1);
            Console.WriteLine($"Actual: type 1.     Predicted probability: type 1: {prediction.Score[0]:0.####}");
            Console.WriteLine($"                                           type 2: {prediction.Score[1]:0.####}");
            Console.WriteLine($"                                           type 3: {prediction.Score[2]:0.####}");
            Console.WriteLine();
            
            prediction = model.Predict(TestIrisData.Iris2);
            Console.WriteLine($"Actual: type 3.     Predicted probability: type 2: {prediction.Score[0]:0.####}");
            Console.WriteLine($"                                           type 2: {prediction.Score[1]:0.####}");
            Console.WriteLine($"                                           type 3: {prediction.Score[2]:0.####}");
            Console.WriteLine();

            prediction = model.Predict(TestIrisData.Iris3);
            Console.WriteLine($"Actual: type 2.     Predicted probability: type 1: {prediction.Score[0]:0.####}");
            Console.WriteLine($"                                           type 2: {prediction.Score[1]:0.####}");
            Console.WriteLine($"                                           type 3: {prediction.Score[2]:0.####}");
            
            Console.ReadLine();
        }

        internal static async Task<PredictionModel<IrisData, IrisPrediction>> TrainAsync()
        {
            var pipeline = new LearningPipeline
            {
                new TextLoader<IrisData>(TrainDataPath, useHeader: false),
                new ColumnConcatenator("Features",
                    "SepalLength",
                    "SepalWidth",
                    "PetalLength",
                    "PetalWidth"),
                new StochasticDualCoordinateAscentClassifier()
            };

            Console.WriteLine("=============== Training model ===============");

            var model = pipeline.Train<IrisData, IrisPrediction>();

            await model.WriteAsync(ModelPath);

            Console.WriteLine("=============== End training ===============");
            Console.WriteLine("The model is saved to {0}", ModelPath);

            return model;
        }

        private static void Evaluate(PredictionModel<IrisData, IrisPrediction> model)
        {
            var testData = new TextLoader<IrisData>(TestDataPath, useHeader: false);

            var evaluator = new ClassificationEvaluator {OutputTopKAcc = 3};
            
            Console.WriteLine("=============== Evaluating model ===============");

            var metrics = evaluator.Evaluate(model, testData);
            Console.WriteLine("Metrics:");
            Console.WriteLine($"    AccuracyMacro = {metrics.AccuracyMacro:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    AccuracyMicro = {metrics.AccuracyMicro:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 1 = {metrics.PerClassLogLoss[0]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 2 = {metrics.PerClassLogLoss[1]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 3 = {metrics.PerClassLogLoss[2]:0.####}, the closer to 0, the better");
            Console.WriteLine();
            Console.WriteLine($"    ConfusionMatrix:");

            // Print confusion matrix
            for (var i = 0; i < metrics.ConfusionMatrix.Order; i++)
            {
                for (var j = 0; j < metrics.ConfusionMatrix.ClassNames.Count; j++)
                {
                    Console.Write("\t" + metrics.ConfusionMatrix[i, j] + "\t");
                }
                Console.WriteLine();
            }

            Console.WriteLine("=============== End evaluating ===============");
            Console.WriteLine();
        }
    }
}