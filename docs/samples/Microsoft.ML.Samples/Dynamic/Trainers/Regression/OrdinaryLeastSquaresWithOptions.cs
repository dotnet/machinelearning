using System;
using Microsoft.ML.Data;
using Microsoft.ML.SamplesUtils;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.Samples.Dynamic.Trainers.Regression
{
    public static class OrdinaryLeastSquaresWithOptions
    {
        // This example requires installation of additional nuget package <a href="https://www.nuget.org/packages/Microsoft.ML.Mkl.Components/">Microsoft.ML.Mkl.Components</a>.
        // In this examples we will use the housing price dataset. The goal is to predict median home value.
        // For more details about this dataset, please see https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
        public static void Example()
        {
            // Downloading a regression dataset from github.com/dotnet/machinelearning
            string dataFile = SamplesUtils.DatasetUtils.DownloadHousingRegressionDataset();

            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext(seed: 3);

            // Creating a data loader, based on the format of the data
            // The data is tab separated with all numeric columns.
            // The first column being the label and rest are numeric features
            // Here only seven numeric columns are used as features
            var dataView = mlContext.Data.LoadFromTextFile(dataFile, new TextLoader.Options
            {
                Separators = new[] { '\t' },
                HasHeader = true,
                Columns = new[]
               {
                    new TextLoader.Column("Label", DataKind.Single, 0),
                    new TextLoader.Column("Features", DataKind.Single, 1, 6)
                }
            });

            //////////////////// Data Preview ////////////////////
            // MedianHomeValue    CrimesPerCapita    PercentResidental    PercentNonRetail    CharlesRiver    NitricOxides    RoomsPerDwelling    PercentPre40s
            // 24.00              0.00632            18.00                2.310               0               0.5380          6.5750              65.20
            // 21.60              0.02731            00.00                7.070               0               0.4690          6.4210              78.90
            // 34.70              0.02729            00.00                7.070               0               0.4690          7.1850              61.10

            var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            // Create the estimator, here we only need OrdinaryLeastSquares trainer 
            // as data is already processed in a form consumable by the trainer
            var pipeline = mlContext.Regression.Trainers.Ols(new OlsTrainer.Options()
            {
                L2Regularization = 0.1f,
                CalculateStatistics = false
            });
            var model = pipeline.Fit(split.TrainSet);

            // Check the weights that the model learned
            var weightsValues = model.Model.Weights;
            Console.WriteLine($"weight 0 - {weightsValues[0]}"); // CrimesPerCapita  (weight 0) = -0.1783206
            Console.WriteLine($"weight 3 - {weightsValues[3]}"); // CharlesRiver (weight 1) = 3.118422
            var dataWithPredictions = model.Transform(split.TestSet);
            var metrics = mlContext.Regression.Evaluate(dataWithPredictions);

            ConsoleUtils.PrintMetrics(metrics);
            
            // Expected output:
            //   L1: 4.14
            //   L2: 32.35
            //   LossFunction: 32.35
            //   RMS: 5.69
            //   RSquared: 0.56
        }
    }
}
