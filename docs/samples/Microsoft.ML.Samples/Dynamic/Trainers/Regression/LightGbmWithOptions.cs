using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

namespace Microsoft.ML.Samples.Dynamic.Trainers.Regression
{
    class LightGbmWithOptions
    {
        // This example requires installation of additional nuget package <a href="https://www.nuget.org/packages/Microsoft.ML.LightGbm/">Microsoft.ML.LightGbm</a>.
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Download and load the housing dataset into an IDataView.
            var dataView = SamplesUtils.DatasetUtils.LoadHousingRegressionDataset(mlContext);

            //////////////////// Data Preview ////////////////////
            /// Only 6 columns are displayed here.
            // MedianHomeValue    CrimesPerCapita    PercentResidental    PercentNonRetail    CharlesRiver    NitricOxides    RoomsPerDwelling    PercentPre40s     ...
            // 24.00              0.00632            18.00                2.310               0               0.5380          6.5750              65.20             ...
            // 21.60              0.02731            00.00                7.070               0               0.4690          6.4210              78.90             ...
            // 34.70              0.02729            00.00                7.070               0               0.4690          7.1850              61.10             ...

            var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.1);

            // Create a pipeline with LightGbm estimator with advanced options.
            // Here we only need LightGbm trainer as data is already processed
            // in a form consumable by the trainer.
            var labelName = "MedianHomeValue";
            var featureNames = dataView.Schema
                .Select(column => column.Name) // Get the column names
                .Where(name => name != labelName) // Drop the Label
                .ToArray();
            var pipeline = mlContext.Transforms.Concatenate("Features", featureNames)
                           .Append(mlContext.Regression.Trainers.LightGbm(new LightGbmRegressionTrainer.Options
                           {
                               LabelColumnName = labelName,
                               NumberOfLeaves = 4,
                               MinimumExampleCountPerLeaf = 6,
                               LearningRate = 0.001,
                               Booster = new GossBooster.Options()
                               {
                                   TopRate = 0.3,
                                   OtherRate = 0.2
                               }
                           }));

            // Fit this pipeline to the training data.
            var model = pipeline.Fit(split.TrainSet);

            // Get the feature importance based on the information gain used during training.
            VBuffer<float> weights = default;
            model.LastTransformer.Model.GetFeatureWeights(ref weights);
            var weightsValues = weights.DenseValues().ToArray();
            Console.WriteLine($"weight 0 - {weightsValues[0]}"); // CrimesPerCapita  (weight 0) = 0.1898361
            Console.WriteLine($"weight 5 - {weightsValues[5]}"); // RoomsPerDwelling (weight 5) = 1

            // Evaluate how the model is doing on the test data.
            var dataWithPredictions = model.Transform(split.TestSet);
            var metrics = mlContext.Regression.Evaluate(dataWithPredictions, labelColumnName: labelName);
            SamplesUtils.ConsoleUtils.PrintMetrics(metrics);

            // Expected output
            //   L1: 4.97
            //   L2: 51.37
            //   LossFunction: 51.37
            //   RMS: 7.17
            //   RSquared: 0.08
        }
    }
}
