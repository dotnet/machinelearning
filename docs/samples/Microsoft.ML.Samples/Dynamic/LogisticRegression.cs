using System;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class LogisticRegressionExample
    {
        public static void Example()
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

            var loader = ml.Data.CreateTextLoader(new TextLoader.Options
            {
                Separators = new[] { ',' },
                HasHeader = true,
                Columns = new[]
                {
                    new TextLoader.Column("age", DataKind.Single, 0),
                    new TextLoader.Column("workclass", DataKind.String, 1),
                    new TextLoader.Column("fnlwgt", DataKind.Single, 2),
                    new TextLoader.Column("education", DataKind.String, 3),
                    new TextLoader.Column("education-num", DataKind.Single, 4),
                    new TextLoader.Column("marital-status", DataKind.String, 5),
                    new TextLoader.Column("occupation", DataKind.String, 6),
                    new TextLoader.Column("relationship", DataKind.String, 7),
                    new TextLoader.Column("ethnicity", DataKind.String, 8),
                    new TextLoader.Column("sex", DataKind.String, 9),
                    new TextLoader.Column("capital-gain", DataKind.Single, 10),
                    new TextLoader.Column("capital-loss", DataKind.Single, 11),
                    new TextLoader.Column("hours-per-week", DataKind.Single, 12),
                    new TextLoader.Column("native-country", DataKind.String, 13),
                    new TextLoader.Column("Label", DataKind.Boolean, 14)
                }
            });

            IDataView data = loader.Load(dataFilePath);

            var split = ml.Data.TrainTestSplit(data, testFraction: 0.2);

            var pipeline = ml.Transforms.Concatenate("Text", "workclass", "education", "marital-status",
                    "relationship", "ethnicity", "sex", "native-country")
                .Append(ml.Transforms.Text.FeaturizeText("TextFeatures", "Text"))
                .Append(ml.Transforms.Concatenate("Features", "TextFeatures", "age", "fnlwgt",
                    "education-num", "capital-gain", "capital-loss", "hours-per-week"))
                .Append(ml.BinaryClassification.Trainers.LogisticRegression());

            var model = pipeline.Fit(split.TrainSet);

            var dataWithPredictions = model.Transform(split.TestSet);

            var metrics = ml.BinaryClassification.Evaluate(dataWithPredictions);

            Console.WriteLine($"Accuracy: {metrics.Accuracy}"); // 0.80
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve}"); // 0.64
            Console.WriteLine($"F1 Score: {metrics.F1Score}"); // 0.39

            Console.WriteLine($"Negative Precision: {metrics.NegativePrecision}"); // 0.81
            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall}"); // 0.96
            Console.WriteLine($"Positive Precision: {metrics.PositivePrecision}"); // 0.68
            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall}"); // 0.27

            Console.ReadLine();
        }
    }
}
