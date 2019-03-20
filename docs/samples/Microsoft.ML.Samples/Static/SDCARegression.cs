using System;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.Samples.Static
{
    public class SdcaRegressionExample
    {
        public static void SdcaRegression()
        {
            // Downloading a regression dataset from github.com/dotnet/machinelearning
            // this will create a housing.txt file in the filsystem this code will run
            // you can open the file to see the data. 
            string dataFile = SamplesUtils.DatasetUtils.DownloadHousingRegressionDataset();

            // Creating the ML.Net IHostEnvironment object, needed for the pipeline
            var mlContext = new MLContext();

            // Creating a data loader, based on the format of the data
            var loader = TextLoaderStatic.CreateLoader(mlContext, c => (
                        label: c.LoadFloat(0),
                        features: c.LoadFloat(1, 6)
                    ),
                separator: '\t', hasHeader: true);

            // Load the data, and leave 10% out, so we can use them for testing
            var data = loader.Load(dataFile);
            var (trainData, testData) = mlContext.Data.TrainTestSplit(data, testFraction: 0.1);

            // The predictor that gets produced out of training
            LinearRegressionModelParameters pred = null;

            // Create the estimator
            var learningPipeline = loader.MakeNewEstimator()
                .Append(r => (r.label, score: mlContext.Regression.Trainers.Sdca(
                                            r.label,
                                            r.features,
                                            l1Threshold: 0f,
                                            numberOfIterations: 100,
                                        onFit: p => pred = p)
                                )
                        );

            // Fit this pipeline to the training data
            var model = learningPipeline.Fit(trainData);

            // Check the weights that the model learned
            var weights = pred.Weights;

            Console.WriteLine($"weight 0 - {weights[0]}");
            Console.WriteLine($"weight 1 - {weights[1]}");

            // Evaluate how the model is doing on the test data
            var dataWithPredictions = model.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(dataWithPredictions, r => r.label, r => r.score);

            Console.WriteLine($"L1 - {metrics.MeanAbsoluteError}");  // 3.7226085
            Console.WriteLine($"L2 - {metrics.MeanSquaredError}");  // 24.250636
            Console.WriteLine($"LossFunction - {metrics.LossFunction}");  // 24.25063
            Console.WriteLine($"RMS - {metrics.RootMeanSquaredError}");  // 4.924493
            Console.WriteLine($"RSquared - {metrics.RSquared}");  // 0.565467
        }
    }
}
