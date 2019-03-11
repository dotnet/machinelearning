using System;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic.Trainers.Regression
{
    public static class StochasticDualCoordinateAscent
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            // Setting the seed to a fixed number in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Create in-memory examples as C# native class and convert to IDataView
            var data = SamplesUtils.DatasetUtils.GenerateFloatLabelFloatFeatureVectorSamples(1000);
            var dataView = mlContext.Data.LoadFromEnumerable(data);

            // Split the data into training and test sets. Only training set is used in fitting
            // the created pipeline. Metrics are computed on the test.
            var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.1);

            // Train the model.
            var pipeline = mlContext.Regression.Trainers.Sdca();
            var model = pipeline.Fit(split.TrainSet);

            // Do prediction on the test set.
            var dataWithPredictions = model.Transform(split.TestSet);

            // Evaluate the trained model using the test set.
            var metrics = mlContext.Regression.Evaluate(dataWithPredictions);
            SamplesUtils.ConsoleUtils.PrintMetrics(metrics);

            // Expected output:
            //   L1: 0.27
            //   L2: 0.11
            //   LossFunction: 0.11
            //   RMS: 0.33
            //   RSquared: 0.56
        }
    }
}
