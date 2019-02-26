﻿using Microsoft.ML.LightGBM;
using static Microsoft.ML.LightGBM.Options;

namespace Microsoft.ML.Samples.Dynamic.Trainers.Ranking
{
    public class LightGbmWithOptions
    {
        // This example requires installation of additional nuget package <a href="https://www.nuget.org/packages/Microsoft.ML.LightGBM/">Microsoft.ML.LightGBM</a>.
        public static void Example()
        {
            // Creating the ML.Net IHostEnvironment object, needed for the pipeline.
            var mlContext = new MLContext();

            // Download and featurize the train and validation datasets.
            var dataview = SamplesUtils.DatasetUtils.LoadFeaturizedMslrWeb10kDataset(mlContext);

            // Leave out 10% of the dataset for testing. Since this is a ranking problem, we must ensure that the split
            // respects the GroupId column, i.e. rows with the same GroupId are either all in the train split or all in
            // the test split. The samplingKeyColumn parameter in Ranking.TrainTestSplit is used for this purpose.
            var split = mlContext.Ranking.TrainTestSplit(dataview, testFraction: 0.1, samplingKeyColumn: "GroupId");

            // Create the Estimator pipeline. For simplicity, we will train a small tree with 4 leaves and 2 boosting iterations.
            var pipeline = mlContext.Ranking.Trainers.LightGbm(
                new Options
                {
                    NumLeaves = 4,
                    MinDataPerLeaf = 10,
                    LearningRate = 0.1,
                    NumBoostRound = 2,
                    Booster = new TreeBooster.Options
                    {
                        FeatureFraction = 0.9
                    }
                });

            // Fit this pipeline to the training Data.
            var model = pipeline.Fit(split.TrainSet);

            // Evaluate how the model is doing on the test data.
            var dataWithPredictions = model.Transform(split.TestSet);

            var metrics = mlContext.Ranking.Evaluate(dataWithPredictions);
            SamplesUtils.ConsoleUtils.PrintMetrics(metrics);

            // Expected output:
            //   DCG: @1:1.71, @2:3.88, @3:7.93
            //   NDCG: @1:7.98, @2:12.14, @3:16.62
        }
    }
}
