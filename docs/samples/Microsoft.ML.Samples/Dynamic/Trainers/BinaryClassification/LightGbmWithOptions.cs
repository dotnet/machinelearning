﻿using Microsoft.ML.Trainers.LightGbm;

namespace Microsoft.ML.Samples.Dynamic.Trainers.BinaryClassification
{
    class LightGbmWithOptions
    {
        // This example requires installation of additional nuget package <a href="https://www.nuget.org/packages/Microsoft.ML.LightGbm/">Microsoft.ML.LightGbm</a>.
        public static void Example()
        {
            // Creating the ML.Net IHostEnvironment object, needed for the pipeline
            var mlContext = new MLContext();

            // Download and featurize the dataset.
            var dataview = SamplesUtils.DatasetUtils.LoadFeaturizedAdultDataset(mlContext);

            // Leave out 10% of data for testing.
            var split = mlContext.Data.TrainTestSplit(dataview, testFraction: 0.1);

            // Create the pipeline with LightGbm Estimator using advanced options.
            var pipeline = mlContext.BinaryClassification.Trainers.LightGbm(
                                new LightGbmBinaryTrainer.Options
                                {
                                    Booster = new GossBooster.Options
                                    {
                                        TopRate = 0.3,
                                        OtherRate = 0.2
                                    }
                                });

            // Fit this Pipeline to the Training Data.
            var model = pipeline.Fit(split.TrainSet);

            // Evaluate how the model is doing on the test data.
            var dataWithPredictions = model.Transform(split.TestSet);

            var metrics = mlContext.BinaryClassification.Evaluate(dataWithPredictions);
            SamplesUtils.ConsoleUtils.PrintMetrics(metrics);

            // Expected output:
            //   Accuracy: 0.88
            //   AUC: 0.93
            //   F1 Score: 0.71
            //   Negative Precision: 0.90
            //   Negative Recall: 0.94
            //   Positive Precision: 0.76
            //   Positive Recall: 0.67
        }
    }
}
