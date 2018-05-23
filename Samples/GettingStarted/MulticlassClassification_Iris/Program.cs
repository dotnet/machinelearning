using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MulticlassClassification_Iris
{
    public static partial class Program
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string TrainDataPath => Path.Combine(AppPath, "..", "..", "..", "..",  "datasets", "iris_train.txt");
        private static string TestDataPath => Path.Combine(AppPath,  "..", "..", "..", "..", "datasets", "iris_test.txt");
        private static string ModelPath => Path.Combine(AppPath,  "Models", "IrisModel.zip");

        private static async Task Main(string[] args)
        {
            // ML task includes 3 steps: training a ML model, evaluating how good it is,
            // and if the quality is acceptable, using this model for predictions.
            var model = await TrainAsync();

            Evaluate(model);

            Console.WriteLine();
            var prediction = model.Predict(TestIrisData.Iris1);
            Console.WriteLine($"Actual: setosa.     Predicted probability: setosa:      {prediction.Score[0]:0.####}");
            Console.WriteLine($"                                           versicolor:  {prediction.Score[1]:0.####}");
            Console.WriteLine($"                                           virginica:   {prediction.Score[2]:0.####}");
            Console.WriteLine();
            
            prediction = model.Predict(TestIrisData.Iris2);
            Console.WriteLine($"Actual: virginica.  Predicted probability: setosa:      {prediction.Score[0]:0.####}");
            Console.WriteLine($"                                           versicolor:  {prediction.Score[1]:0.####}");
            Console.WriteLine($"                                           virginica:   {prediction.Score[2]:0.####}");
            Console.WriteLine();

            prediction = model.Predict(TestIrisData.Iris3);
            Console.WriteLine($"Actual: versicolor. Predicted probability: setosa:      {prediction.Score[0]:0.####}");
            Console.WriteLine($"                                           versicolor:  {prediction.Score[1]:0.####}");
            Console.WriteLine($"                                           virginica:   {prediction.Score[2]:0.####}");
            
            Console.ReadLine();
        }

        internal static async Task<PredictionModel<IrisData, IrisPrediction>> TrainAsync()
        {
            // LearningPipeline holds all steps of the learning process: data, transforms, learners.
            var pipeline = new LearningPipeline
            {
                // The TextLoader loads a dataset. The schema of the dataset is specified by passing a class containing
                // all the column names and their types. This will be used to create the model, and train it. 
                new TextLoader<IrisData>(TrainDataPath, useHeader: false),
                
                // Transforms
                // When ML model starts training, it looks for two columns: Label and Features.
                // Label:   values that should be predicted. If you have a field named Label in your data type,
                //              like in this example, no extra actions required.
                //          If you don’t have it, copy the column you want to predict with ColumnCopier transform:
                //              new ColumnCopier(("FareAmount", "Label"))
                // Features: all data used for prediction. At the end of all transforms you need to concatenate
                //              all columns except the one you want to predict into Features column with
                //              ColumnConcatenator transform:
                new ColumnConcatenator("Features",
                    "SepalLength",
                    "SepalWidth",
                    "PetalLength",
                    "PetalWidth"),
                // StochasticDualCoordinateAscentClassifier is an algorithm that will be used to train the model.
                new StochasticDualCoordinateAscentClassifier()
            };

            Console.WriteLine("=============== Training model ===============");
            // The pipeline is trained on the dataset that has been loaded and transformed.
            var model = pipeline.Train<IrisData, IrisPrediction>();

            // Saving the model as a .zip file.
            await model.WriteAsync(ModelPath);

            Console.WriteLine("=============== End training ===============");
            Console.WriteLine("The model is saved to {0}", ModelPath);

            return model;
        }

        private static void Evaluate(PredictionModel<IrisData, IrisPrediction> model)
        {
            // To evaluate how good the model predicts values, the model is ran against new set
            // of data (test data) that was not involved in training.
            var testData = new TextLoader<IrisData>(TestDataPath, useHeader: false);

            // ClassificationEvaluator performs evaluation for Multiclass Classification type of ML problems.
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