using System;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Categorical;

namespace Microsoft.ML.Samples.Dynamic.Trainers
{
    public class LightGbmBinaryClassification
    {
        public static void Example()
        {
            // Creating the ML.Net IHostEnvironment object, needed for the pipeline
            var mlContext = new MLContext();

            // Download the dataset and load it as IDataView
            var dataview = SamplesUtils.DatasetUtils.LoadAdultDataset(mlContext);

            // Leave out 10% of data for testing
            var (trainData, testData) = mlContext.BinaryClassification.TrainTestSplit(dataview, testFraction: 0.1);

            // Create the Estimator
            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(new OneHotEncodingEstimator.ColumnInfo[]
                {
                    new OneHotEncodingEstimator.ColumnInfo("marital-status"),
                    new OneHotEncodingEstimator.ColumnInfo("occupation"),
                    new OneHotEncodingEstimator.ColumnInfo("relationship"),
                    new OneHotEncodingEstimator.ColumnInfo("ethnicity"),
                    new OneHotEncodingEstimator.ColumnInfo("sex"),
                    new OneHotEncodingEstimator.ColumnInfo("native-country"),
                })
            .Append(mlContext.Transforms.FeatureSelection.SelectFeaturesBasedOnCount("native-country", count: 10))
            .Append(mlContext.Transforms.Concatenate("Features",
                                                        "age",
                                                        "education-num",
                                                        "marital-status",
                                                        "relationship",
                                                        "ethnicity",
                                                        "sex",
                                                        "hours-per-week",
                                                        "native-country"))
            .Append(mlContext.Transforms.Normalize("Features"))
            .Append(mlContext.BinaryClassification.Trainers.LightGbm("IsOver50K", "Features"));

            // Fit this Pipeline to the Training Data
            var model = pipeline.Fit(trainData);

            // Evaluate how the model is doing on the test data
            var dataWithPredictions = model.Transform(testData);

            var metrics = mlContext.BinaryClassification.Evaluate(dataWithPredictions, "IsOver50K");
            SamplesUtils.ConsoleUtils.PrintMetrics(metrics);

            // Output:
            // Accuracy: 0.84
            // AUC: 0.88
            // F1 Score: 0.62
            // Negative Precision: 0.88
            // Negative Recall: 0.91
            // Positive Precision: 0.68
            // Positive Recall: 0.59
        }
    }
}