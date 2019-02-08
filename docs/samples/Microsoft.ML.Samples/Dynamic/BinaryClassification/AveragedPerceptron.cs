using Microsoft.ML;

namespace Microsoft.ML.Samples.Dynamic.BinaryClassification
{
    public static class AveragedPerceptron
    {
        public static void Example()
        {
            // In this examples we will use the adult income dataset. The goal is to predict
            // if a person's income is above $50K or not, based on different pieces of information about that person.
            // For more details about this dataset, please see https://archive.ics.uci.edu/ml/datasets/adult

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            // Setting the seed to a fixed number in this examples to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Download the dataset and load it as IDataView
            var data = SamplesUtils.DatasetUtils.LoadAdultDataset(mlContext);

            // Leave out 10% of data for testing
            var (trainData, testData) = mlContext.BinaryClassification.TrainTestSplit(data, testFraction: 0.1);

            // Create data processing pipeline
            var pipeline =
                // Convert categorical features to one-hot vectors
                mlContext.Transforms.Categorical.OneHotEncoding("workclass")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("education"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("marital-status"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("occupation"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("relationship"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("ethnicity"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("native-country"))
                // Combine all features into one feature vector
                .Append(mlContext.Transforms.Concatenate("Features", "workclass", "education", "marital-status",
                    "occupation", "relationship", "ethnicity", "native-country", "age", "education-num", 
                    "capital-gain", "capital-loss", "hours-per-week"))
                // Min-max normalized all the features
                .Append(mlContext.Transforms.Normalize("Features"))
                // Add the trainer
                .Append(mlContext.BinaryClassification.Trainers.AveragedPerceptron("IsOver50K", "Features"));

            // Fit this pipeline to the training data
            var model = pipeline.Fit(trainData);

            // Evaluate how the model is doing on the test data
            var dataWithPredictions = model.Transform(testData);
            var metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(dataWithPredictions, "IsOver50K");
            SamplesUtils.ConsoleUtils.PrintBinaryClassificationMetrics(metrics);

            // Output:
            // Accuracy: 0.85
            // AUC: 0.90
            // F1 Score: 0.66
            // Negative Precision: 0.89
            // Negative Recall: 0.91
            // Positive Precision: 0.69
            // Positive Recall: 0.63
        }
    }
}