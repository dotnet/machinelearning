namespace Microsoft.ML.Samples.Dynamic.Trainers.BinaryClassification.Calibrators
{
    public static class Naive
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            // Setting the seed to a fixed number in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Download and featurize the dataset.
            var data = SamplesUtils.DatasetUtils.LoadFeaturizedAdultDataset(mlContext);

            // Leave out 10% of data for testing.
            var trainTestData = mlContext.BinaryClassification.TrainTestSplit(data, testFraction: 0.1);

            // Create data training pipeline.
            var pipeline = mlContext.BinaryClassification.Trainers.AveragedPerceptron(numIterations: 10).
                Append(mlContext.BinaryClassification.Calibrators.Naive());

            // Fit this pipeline to the training data.
            var model = pipeline.Fit(trainTestData.TrainSet);

            // Evaluate how the model is doing on the test data.
            var dataWithPredictions = model.Transform(trainTestData.TestSet);

            var metrics = mlContext.BinaryClassification.Evaluate(dataWithPredictions);
            SamplesUtils.ConsoleUtils.PrintMetrics(metrics);
        }
    }
}
