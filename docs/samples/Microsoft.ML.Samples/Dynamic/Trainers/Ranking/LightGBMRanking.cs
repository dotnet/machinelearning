using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Samples.Dynamic
{
    public class LightGbmRanking
    {
        // This example requires installation of additional nuget package <a href="https://www.nuget.org/packages/Microsoft.ML.LightGBM/">Microsoft.ML.LightGBM</a>.
        public static void Example()
        {
            // Creating the ML.Net IHostEnvironment object, needed for the pipeline.
            var mlContext = new MLContext();

            // Download and featurize the train and validation datasets.
            (var trainData, var validationData) = SamplesUtils.DatasetUtils.LoadFeaturizedMslrWeb10kTrainAndValidate(mlContext);

            // Create the Estimator pipeline. For simplicity, we will train a small tree with 4 leaves and 2 boosting iterations.
            var pipeline = mlContext.Ranking.Trainers.LightGbm(
                labelColumn: "Label",
                featureColumn: "Features",
                groupIdColumn: "GroupId",
                numLeaves: 4,
                minDataPerLeaf: 10,
                learningRate: 0.1,
                numBoostRound: 2);

            // Fit this Pipeline to the Training Data.
            var model = pipeline.Fit(trainData);

            // Evaluate how the model is doing on the test data.
            var dataWithPredictions = model.Transform(validationData);

            var metrics = mlContext.Ranking.Evaluate(dataWithPredictions, "Label", "GroupId");
            SamplesUtils.ConsoleUtils.PrintMetrics(metrics);

            // Output:
            // DCG @N: 1.38, 3.11, 4.94
            // NDCG @N: 7.13, 10.12, 12.62
        }
    }
}
