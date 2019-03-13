using Microsoft.ML.Trainers;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class StochasticGradientDescentNonCalibratedWithOptions
    {
        // In this examples we will use the adult income dataset. The goal is to predict
        // if a person's income is above $50K or not, based on demographic information about that person.
        // For more details about this dataset, please see https://archive.ics.uci.edu/ml/datasets/adult.
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            // Setting the seed to a fixed number in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Download and featurize the dataset.
            var data = SamplesUtils.DatasetUtils.LoadFeaturizedAdultDataset(mlContext);

            // Leave out 10% of data for testing.
            var trainTestData = mlContext.Data.TrainTestSplit(data, testFraction: 0.1);

            // Create data training pipeline.
            var pipeline = mlContext.BinaryClassification
                .Trainers.SgdNonCalibrated(
                        new SgdNonCalibratedTrainer.Options
                        {
                            InitialLearningRate = 0.01,
                            NumberOfIterations = 10,
                            L2Regularization = 1e-7f
                        }
                       );

            // Fit this pipeline to the training data.
            var model = pipeline.Fit(trainTestData.TrainSet);

            // Evaluate how the model is doing on the test data.
            var dataWithPredictions = model.Transform(trainTestData.TestSet);
            var metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(dataWithPredictions);
            SamplesUtils.ConsoleUtils.PrintMetrics(metrics);

            // Expected output:
            //  Accuracy: 0.85
            //  AUC: 0.89
            //  F1 Score: 0.63
            //  Negative Precision: 0.88
            //  Negative Recall: 0.93
            //  Positive Precision: 0.71
            //  Positive Recall: 0.57
        }
    }
}
