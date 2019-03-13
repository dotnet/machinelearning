namespace Microsoft.ML.Samples.Dynamic.Trainers.BinaryClassification
{
    public static class SymbolicStochasticGradientDescent
    {
        // This example requires installation of additional nuget package <a href="https://www.nuget.org/packages/Microsoft.ML.Mkl.Components/">Microsoft.ML.Mkl.Components</a>.
        // In this example we will use the adult income dataset. The goal is to predict
        // if a person's income is above $50K or not, based on demographic information about that person.
        // For more details about this dataset, please see https://archive.ics.uci.edu/ml/datasets/adult
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            // Setting the seed to a fixed number in this examples to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Download and featurize the dataset.
            var data = SamplesUtils.DatasetUtils.LoadFeaturizedAdultDataset(mlContext);

            // Leave out 10% of data for testing.
            var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.1);
            // Create data training pipeline.
            var pipeline = mlContext.BinaryClassification.Trainers.SymbolicSgd(labelColumnName: "IsOver50K", numberOfIterations: 25);
            var model = pipeline.Fit(split.TrainSet);

            // Evaluate how the model is doing on the test data.
            var dataWithPredictions = model.Transform(split.TestSet);
            var metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(dataWithPredictions);
            SamplesUtils.ConsoleUtils.PrintMetrics(metrics);
            
            // Expected output:
            //   Accuracy: 0.85
            //   AUC: 0.90
            //   F1 Score: 0.64
            //   Negative Precision: 0.88
            //   Negative Recall: 0.93
            //   Positive Precision: 0.72
            //   Positive Recall: 0.58
        }
    }
}
