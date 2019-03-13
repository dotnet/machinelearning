using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.Samples.Dynamic.Trainers.Regression
{
    public static class StochasticDualCoordinateAscentWithOptions
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

            // Create trainer options.
            var options = new SdcaRegressionTrainer.Options
            {
                // Make the convergence tolerance tighter.
                ConvergenceTolerance = 0.02f,
                // Increase the maximum number of passes over training data.
                MaximumNumberOfIterations = 30,
                // Increase learning rate for bias
                BiasLearningRate = 0.1f
            };

            // Train the model.
            var pipeline = mlContext.Regression.Trainers.Sdca(options);
            var model = pipeline.Fit(split.TrainSet);

            // Do prediction on the test set.
            var dataWithPredictions = model.Transform(split.TestSet);

            // Evaluate the trained model using the test set.
            var metrics = mlContext.Regression.Evaluate(dataWithPredictions);
            SamplesUtils.ConsoleUtils.PrintMetrics(metrics);

            // Expected output:
            //   L1: 0.26
            //   L2: 0.11
            //   LossFunction: 0.11
            //   RMS: 0.33
            //   RSquared: 0.56
        }
    }
}
