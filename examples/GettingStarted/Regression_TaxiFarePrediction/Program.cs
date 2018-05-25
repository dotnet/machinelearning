using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace Regression_TaxiFarePrediction
{
    internal static class Program
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string TrainDataPath => Path.Combine(AppPath, "datasets", "taxi-fare-train.csv");
        private static string TestDataPath => Path.Combine(AppPath,  "datasets", "taxi-fare-test.csv");
        private static string ModelPath => Path.Combine(AppPath, "TaxiFareModel.zip");

        private static async Task Main(string[] args)
        {
            // ML task includes 3 steps: training a ML model, evaluating how good it is,
            // and if the quality is acceptable, using this model for predictions.
            var model = await TrainAsync();

            Evaluate(model);

            var prediction = model.Predict(TestTaxiTrips.Trip1);
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 29.5");

            Console.ReadLine();
        }

        private static async Task<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> TrainAsync()
        {
            // LearningPipeline holds all steps of the learning process: data, transforms, learners.
            var pipeline = new LearningPipeline
            {
                // The TextLoader loads a dataset. The schema of the dataset is specified by passing a class containing
                // all the column names and their types. This will be used to create the model, and train it.
                new TextLoader<TaxiTrip>(TrainDataPath, useHeader: true, separator: ","),
                
                // Transforms
                // When ML model starts training, it looks for two columns: Label and Features.
                // Label:   values that should be predicted. If you have a field named Label in your data type,
                //              no extra actions required.
                //          If you don’t have it, like in this example, copy the column you want to predict with
                //              ColumnCopier transform:
                new ColumnCopier(("FareAmount", "Label")),
                
                // CategoricalOneHotVectorizer transforms categorical (string) values into 0/1 vectors
                new CategoricalOneHotVectorizer("VendorId",
                    "RateCode",
                    "PaymentType"),
                // Features: all data used for prediction. At the end of all transforms you need to concatenate
                //              all columns except the one you want to predict into Features column with
                //              ColumnConcatenator transform:
                new ColumnConcatenator("Features",
                    "VendorId",
                    "RateCode",
                    "PassengerCount",
                    "TripDistance",
                    "PaymentType"),
                //FastTreeRegressor is an algorithm that will be used to train the model.
                new FastTreeRegressor()
            };

            Console.WriteLine("=============== Training model ===============");
            // The pipeline is trained on the dataset that has been loaded and transformed.
            var model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();
            
            // Saving the model as a .zip file.
            await model.WriteAsync(ModelPath);
            
            Console.WriteLine("=============== End training ===============");
            Console.WriteLine("The model is saved to {0}", ModelPath);

            return model;
        }

        private static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
        {
            // To evaluate how good the model predicts values, the model is ran against new set
            // of data (test data) that was not involved in training.
            var testData = new TextLoader<TaxiTrip>(TestDataPath, useHeader: true, separator: ",");

            // RegressionEvaluator performs evaluation for Regression type of ML problems.
            var evaluator = new RegressionEvaluator();
            
            Console.WriteLine("=============== Evaluating model ===============");

            var metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine($"Rms = {metrics.Rms}, ideally should be around 2.8, can be improved with larger dataset");
            Console.WriteLine($"RSquared = {metrics.RSquared}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine("=============== End evaluating ===============");
            Console.WriteLine();
        }
    }
}