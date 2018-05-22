using System;
using System.IO;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML;
using System.Threading.Tasks;

namespace Regression_TaxiFarePrediction
{
    internal static class Program
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string TrainDataPath => Path.Combine(AppPath, @"..\..\..\..\datasets\", "taxi-fare-train.csv");
        private static string TestDataPath => Path.Combine(AppPath, @"..\..\..\..\datasets\", "taxi-fare-test.csv");
        private static string ModelPath => Path.Combine(AppPath,  "Models", "TaxiFareModel.zip");

        private static async Task Main(string[] args)
        {
            var model = await TrainAsync();

            Evaluate(model);

            var prediction = model.Predict(TestTaxiTrips.Trip1);
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 29.5");

            Console.ReadLine();
        }

        private static async Task<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> TrainAsync()
        {
            var pipeline = new LearningPipeline
            {
                new TextLoader<TaxiTrip>(TrainDataPath, useHeader: true, separator: ","),
                new ColumnCopier(("FareAmount", "Label")),
                new CategoricalOneHotVectorizer("VendorId",
                    "RateCode",
                    "PaymentType"),
                new ColumnConcatenator("Features",
                    "VendorId",
                    "RateCode",
                    "PassengerCount",
                    "TripDistance",
                    "PaymentType"),
                new FastTreeRegressor()
            };

            Console.WriteLine("=============== Training model ===============");

            var model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();
            
            await model.WriteAsync(ModelPath);
            
            Console.WriteLine("=============== End training ===============");
            Console.WriteLine("The model is saved to {0}", ModelPath);

            return model;
        }

        private static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
        {
            var testData = new TextLoader<TaxiTrip>(TestDataPath, useHeader: true, separator: ",");

            var evaluator = new RegressionEvaluator();
            
            Console.WriteLine("=============== Evaluating model ===============");

            var metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine($"Rms = {metrics.Rms}, expected to be around 2.8");
            Console.WriteLine($"RSquared = {metrics.RSquared}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine("=============== End evaluating ===============");
            Console.WriteLine();
        }
    }
}