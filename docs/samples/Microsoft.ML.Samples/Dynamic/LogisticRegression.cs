using System;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public class LogisticRegressionExample
    {
        public static void LogisticRegression()
        {
            var ml = new MLContext();

            // Downloading a classification dataset from github.com/dotnet/machinelearning.
            // It will be stored in the same path as the executable
            var dataFilePath = SamplesUtils.DatasetUtils.DownloadAdultDataset();

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

            var reader = ml.Data.CreateTextLoader(new TextLoader.Arguments
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

            IDataView data = reader.Read(dataFilePath);

            var (trainData, testData) = ml.BinaryClassification.TrainTestSplit(data, testFraction: 0.2);

            var pipeline = ml.Transforms.Concatenate("Text", "workclass", "education", "marital-status",
                    "relationship", "ethnicity", "sex", "native-country")
                .Append(ml.Transforms.Text.FeaturizeText("TextFeatures", "Text"))
                .Append(ml.Transforms.Concatenate("Features", "TextFeatures", "age", "fnlwgt",
                    "education-num", "capital-gain", "capital-loss", "hours-per-week"))
                .Append(ml.BinaryClassification.Trainers.LogisticRegression());

            var model = pipeline.Fit(trainData);

            var dataWithPredictions = model.Transform(testData);

            var metrics = ml.BinaryClassification.Evaluate(dataWithPredictions);

            Console.WriteLine($"Accuracy: {metrics.Accuracy}"); // 0.80
            Console.WriteLine($"AUC: {metrics.Auc}"); // 0.64
            Console.WriteLine($"F1 Score: {metrics.F1Score}"); // 0.39

            Console.WriteLine($"Negative Precision: {metrics.NegativePrecision}"); // 0.81
            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall}"); // 0.96
            Console.WriteLine($"Positive Precision: {metrics.PositivePrecision}"); // 0.68
            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall}"); // 0.27

            Console.ReadLine();
        }
    }
}
