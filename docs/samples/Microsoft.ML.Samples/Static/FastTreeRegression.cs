using System;
using System.Linq;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Trainers.FastTree;

namespace Microsoft.ML.Samples.Static
{
    public class FastTreeRegressionExample
    {
        // This example requires installation of additional nuget package <a href="https://www.nuget.org/packages/Microsoft.ML.FastTree/">Microsoft.ML.FastTree</a>.
        public static void FastTreeRegression()
        {
            // Downloading a regression dataset from github.com/dotnet/machinelearning
            // this will create a housing.txt file in the filsystem this code will run
            // you can open the file to see the data. 
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
            var data = loader.Load(dataFile);

            // The predictor that gets produced out of training
            FastTreeRegressionModelParameters pred = null;

            // Create the estimator
            var learningPipeline = loader.MakeNewEstimator()
                .Append(r => (r.label, score: mlContext.Regression.Trainers.FastTree(
                                            r.label,
                                            r.features,
                                            numberOfTrees: 100, // try: (int) 20-2000
                                            numberOfLeaves: 20, // try: (int) 2-128
                                            minimumExampleCountPerLeaf: 10, // try: (int) 1-100
                                            learningRate: 0.2, // try: (float) 0.025-0.4    
                                        onFit: p => pred = p)
                                )
                        );

            var cvResults = mlContext.Regression.CrossValidate(data, learningPipeline, r => r.label, numFolds: 5);
            var averagedMetrics = (
                L1: cvResults.Select(r => r.metrics.MeanAbsoluteError).Average(),
                L2: cvResults.Select(r => r.metrics.MeanSquaredError).Average(),
                LossFn: cvResults.Select(r => r.metrics.LossFunction).Average(),
                Rms: cvResults.Select(r => r.metrics.RootMeanSquaredError).Average(),
                RSquared: cvResults.Select(r => r.metrics.RSquared).Average()
            );
            Console.WriteLine($"L1 - {averagedMetrics.L1}");    // 3.091095
            Console.WriteLine($"L2 - {averagedMetrics.L2}");    // 20.351073
            Console.WriteLine($"LossFunction - {averagedMetrics.LossFn}");  // 20.351074
            Console.WriteLine($"RMS - {averagedMetrics.Rms}");              // 4.478358
            Console.WriteLine($"RSquared - {averagedMetrics.RSquared}");    // 0.754977
        }

    }
}
