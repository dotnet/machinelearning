using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Data.Analysis;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using static Microsoft.ML.Transforms.OneHotEncodingEstimator.OutputKind;

namespace Microsoft.ML.AutoML.Samples
{
    /// <summary>
    /// Regression automl experiment using taxi fare dataset.
    /// </summary>
    public static class TaxiFare
    {
        public static void Run()
        {
            //Load File
            var trainDataPath = @"C:\Users\xiaoyuz\Desktop\taxi-fare-train.csv";
            var df = DataFrame.LoadCsv(trainDataPath);
            var mlContext = new MLContext();

            // Append the trainer to the data processing pipeline
            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(new[] { new InputOutputColumnPair(@"vendor_id", @"vendor_id"), new InputOutputColumnPair(@"payment_type", @"payment_type") })
                             .Append(mlContext.Transforms.ReplaceMissingValues(new[] { new InputOutputColumnPair(@"rate_code", @"rate_code"), new InputOutputColumnPair(@"passenger_count", @"passenger_count"), new InputOutputColumnPair(@"trip_time_in_secs", @"trip_time_in_secs"), new InputOutputColumnPair(@"trip_distance", @"trip_distance") }))
                             .Append(mlContext.Transforms.Concatenate(@"Features", new[] { @"vendor_id", @"payment_type", @"rate_code", @"passenger_count", @"trip_time_in_secs", @"trip_distance" }))
                             .Append(mlContext.Auto().Regression(labelColumnName: "fare_amount"));

            // Configure AutoML
            var trainTestSplit = mlContext.Data.TrainTestSplit(df, 0.1);

            var experiment = mlContext.Auto().CreateExperiment()
                               .SetPipeline(pipeline)
                               .SetTrainingTimeInSeconds(50)
                               .SetDataset(trainTestSplit.TrainSet, trainTestSplit.TestSet)
                               .SetEvaluateMetric(RegressionMetric.RSquared, "fare_amount", "Score");

            mlContext.Log += (o, e) =>
            {
                if (e.Source.StartsWith("AutoMLExperiment"))
                {
                    Console.WriteLine(e.RawMessage);
                }
            };

            // Start Experiment
            var res = experiment.Run().Result;

        }
    }
}
