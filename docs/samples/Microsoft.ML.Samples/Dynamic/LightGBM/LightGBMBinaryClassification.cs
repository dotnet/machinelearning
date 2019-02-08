using System;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Categorical;

namespace Microsoft.ML.Samples.Dynamic.LightGBM
{
    public class LightGbmBinaryClassification
    {
        public static void LightGbmBinaryClassificationExample()
        {
            // Downloading a classification dataset from github.com/dotnet/machinelearning.
            // It will be stored in the same path as the executable
            string dataFilePath = SamplesUtils.DatasetUtils.DownloadAdultDataset();

            // Data Preview
            // 1. Column: age (numeric)
            // 2. Column: workclass (text/categorical)
            // 3. Column: fnlwgt (numeric)
            // 4. Column: education (text/categorical)
            // 5. Column: education-num (numeric)
            // 6. Column: marital-status (text/categorical)
            // 7. Column: occupation (text/categorical)
            // 8. Column: relationship (text/categorical)
            // 9. Column: ethnicity (text/categorical)
            // 10. Column: sex (text/categorical)
            // 11. Column: capital-gain (numeric)
            // 12. Column: capital-loss (numeric)
            // 13. Column: hours-per-week (numeric)
            // 14. Column: native-country (text/categorical)
            // 15. Column: Column [Label]: IsOver50K (boolean)

            // Creating the ML.Net IHostEnvironment object, needed for the pipeline
            var mlContext = new MLContext();

            var reader = mlContext.Data.ReadFromTextFile(dataFilePath, new TextLoader.Arguments
            {
                Separators = new[] { ',' },
                HasHeader = true,
                Columns = new[]
                {
                    new TextLoader.Column("age", DataKind.R4, 0),
                    new TextLoader.Column("workclass", DataKind.Text, 1),
                    new TextLoader.Column("fnlwgt", DataKind.R4, 2),
                    new TextLoader.Column("education", DataKind.Text, 3),
                    new TextLoader.Column("education-num", DataKind.R4, 4),
                    new TextLoader.Column("marital-status", DataKind.Text, 5),
                    new TextLoader.Column("occupation", DataKind.Text, 6),
                    new TextLoader.Column("relationship", DataKind.Text, 7),
                    new TextLoader.Column("ethnicity", DataKind.Text, 8),
                    new TextLoader.Column("sex", DataKind.Text, 9),
                    new TextLoader.Column("capital-gain", DataKind.R4, 10),
                    new TextLoader.Column("capital-loss", DataKind.R4, 11),
                    new TextLoader.Column("hours-per-week", DataKind.R4, 12),
                    new TextLoader.Column("native-country", DataKind.Text, 13),
                    new TextLoader.Column("Label", DataKind.Bool, 14)
                }
            });

            // Read the data, and leave 10% out, so we can use them for testing
            var (trainData, testData) = mlContext.BinaryClassification.TrainTestSplit(reader, testFraction: 0.1);

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
            .Append(mlContext.BinaryClassification.Trainers.LightGbm());

            // Fit this Pipeline to the Training Data
            var model = pipeline.Fit(trainData);

            // Evaluate how the model is doing on the test data
            var dataWithPredictions = model.Transform(testData);

            var metrics = mlContext.BinaryClassification.Evaluate(dataWithPredictions);

            Console.WriteLine($"Accuracy: {metrics.Accuracy}"); // 0.84
            Console.WriteLine($"AUC: {metrics.Auc}"); // 0.88
            Console.WriteLine($"F1 Score: {metrics.F1Score}"); // 0.62

            Console.WriteLine($"Negative Precision: {metrics.NegativePrecision}"); // 0.88
            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall}"); // 0.91
            Console.WriteLine($"Positive Precision: {metrics.PositivePrecision}"); // 0.67
            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall}"); // 0.58       
        }
    }
}