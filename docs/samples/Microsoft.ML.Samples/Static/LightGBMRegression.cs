using System;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Trainers.LightGbm.StaticPipe;
using Microsoft.ML.StaticPipe;

namespace Microsoft.ML.Samples.Static
{
    public class LightGbmRegressionExample
    {
        public static void Example()
        {
            // Downloading a regression dataset from github.com/dotnet/machinelearning
            // this will create a housing.txt file in the filsystem.
            // You can open the file to see the data. 
            string dataFile = SamplesUtils.DatasetUtils.DownloadHousingRegressionDataset();

            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Creating a data loader, based on the format of the data
            var loader = TextLoaderStatic.CreateLoader(mlContext, c => (
                        label: c.LoadFloat(0),
                        features: c.LoadFloat(1, 6)
                    ),
                separator: '\t', hasHeader: true);

            // Load the data, and leave 10% out, so we can use them for testing
            var data = loader.Load(new MultiFileSource(dataFile));
            var (trainData, testData) = mlContext.Data.TrainTestSplit(data, testFraction: 0.1);

            // The predictor that gets produced out of training
            LightGbmRegressionModelParameters pred = null;

            // Create the estimator
            var learningPipeline = loader.MakeNewEstimator()
                .Append(r => (r.label, score: mlContext.Regression.Trainers.LightGbm(
                                            r.label,
                                            r.features,
                                            numberOfLeaves: 4,
                                            minimumExampleCountPerLeaf: 6,
                                            learningRate: 0.001,
                                        onFit: p => pred = p)
                                )
                        );

            // Fit this pipeline to the training data
            var model = learningPipeline.Fit(trainData);

            // Check the weights that the model learned
            VBuffer<float> weights = default;
            pred.GetFeatureWeights(ref weights);

            var weightsValues = weights.GetValues();
            Console.WriteLine($"weight 0 - {weightsValues[0]}");
            Console.WriteLine($"weight 1 - {weightsValues[1]}");

            // Evaluate how the model is doing on the test data
            var dataWithPredictions = model.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(dataWithPredictions, r => r.label, r => r.score);

            Console.WriteLine($"L1 - {metrics.MeanAbsoluteError}");    // 4.9669731
            Console.WriteLine($"L2 - {metrics.MeanSquaredError}");    // 51.37296
            Console.WriteLine($"LossFunction - {metrics.LossFunction}");  // 51.37296
            Console.WriteLine($"RMS - {metrics.RootMeanSquaredError}");              // 7.167493
            Console.WriteLine($"RSquared - {metrics.RSquared}");    // 0.079478
        }
    }
}
