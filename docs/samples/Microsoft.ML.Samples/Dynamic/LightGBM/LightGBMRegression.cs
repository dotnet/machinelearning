using System;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic.LightGBM
{
    class LightGbmRegression
    {
        public static void LightGbmRegressionExample()
        {
            // Downloading a regression dataset from github.com/dotnet/machinelearning
            // this will create a housing.txt file in the filsystem this code will run
            // you can open the file to see the data. 
            string dataFile = SamplesUtils.DatasetUtils.DownloadHousingRegressionDataset();

            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Creating a data reader, based on the format of the data
            // The data is tab separated with all numeric columns.
            // The first column being the label and rest are numeric features
            // Here only seven numeric columns are used as features
            var dataView = mlContext.Data.ReadFromTextFile(dataFile, new TextLoader.Arguments
            {
                Separators = new[] { '\t' },
                HasHeader = true,
                Columns = new[]
               {
                    new TextLoader.Column("Label", DataKind.R4, 0),
                    new TextLoader.Column("Features", DataKind.R4, 1, 6)
                }
            });

            //////////////////// Data Preview ////////////////////
            // MedianHomeValue    CrimesPerCapita    PercentResidental    PercentNonRetail    CharlesRiver    NitricOxides    RoomsPerDwelling    PercentPre40s
            // 24.00              0.00632            18.00                2.310               0               0.5380          6.5750              65.20
            // 21.60              0.02731            00.00                7.070               0               0.4690          6.4210              78.90
            // 34.70              0.02729            00.00                7.070               0               0.4690          7.1850              61.10

            var (trainData, testData) = mlContext.Regression.TrainTestSplit(dataView, testFraction: 0.1);

            // Create the estimator, here we only need LightGbm trainer 
            // as data is already processed in a form consumable by the trainer
            var pipeline = mlContext.Regression.Trainers.LightGbm(
                                            numLeaves: 4,
                                            minDataPerLeaf: 6,
                                            learningRate: 0.001);

            // Fit this pipeline to the training data
            var model = pipeline.Fit(trainData);

            // Check the weights that the model learned
            VBuffer<float> weights = default;
            model.Model.GetFeatureWeights(ref weights);

            var weightsValues = weights.GetValues();
            Console.WriteLine($"weight 0 - {weightsValues[0]}"); // CrimesPerCapita  (weight 0) = 0.1898361
            Console.WriteLine($"weight 1 - {weightsValues[5]}"); // RoomsPerDwelling (weight 1) = 1

            // Evaluate how the model is doing on the test data
            var dataWithPredictions = model.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(dataWithPredictions);

            Console.WriteLine($"L1 - {metrics.L1}");    // 4.9669731
            Console.WriteLine($"L2 - {metrics.L2}");    // 51.37296
            Console.WriteLine($"LossFunction - {metrics.LossFn}");  // 51.37296
            Console.WriteLine($"RMS - {metrics.Rms}");              // 7.167493
            Console.WriteLine($"RSquared - {metrics.RSquared}");    // 0.079478
        }
    }
}
