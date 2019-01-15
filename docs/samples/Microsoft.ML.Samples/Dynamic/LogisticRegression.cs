using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public class LogisticRegression_example
    {
        public static void LogisticRegression()
        {
            var ml = new MLContext();

            // Downloading a classification dataset from github.com/dotnet/machinelearning.
            // It will be stored in the same path as the executable
            var dataFilePath = SamplesUtils.DatasetUtils.DownloadAdultDataset();

            // Data Preview
            // 1. Column [Label]: IsOver50K (boolean)
            // 2. Column: workclass (text/categorical)
            // 3. Column: education (text/categorical)
            // 4. Column: marital-status (text/categorical)
            // 5. Column: occupation (text/categorical)
            // 6. Column: relationship (text/categorical)
            // 7. Column: ethnicity (text/categorical)
            // 8. Column: sex (text/categorical)
            // 9. Column: native-country-region (text/categorical)
            // 10. Column: age (numeric)
            // 11. Column: fnlwgt (numeric)
            // 12. Column: education-num (numeric)
            // 13. Column: capital-gain (numeric)
            // 14. Column: capital-loss (numeric)
            // 15. Column: hours-per-week (numeric)

            var reader = ml.Data.CreateTextReader(new TextLoader.Arguments
            {
                Separator = ",",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.Bool, 0),
                    new TextLoader.Column("workclass", DataKind.Text, 1),
                    new TextLoader.Column("education", DataKind.Text, 2),
                    new TextLoader.Column("marital-status", DataKind.Text, 3),
                    new TextLoader.Column("occupation", DataKind.Text, 4),
                    new TextLoader.Column("relationship", DataKind.Text, 5),
                    new TextLoader.Column("ethnicity", DataKind.Text, 6),
                    new TextLoader.Column("sex", DataKind.Text, 7),
                    new TextLoader.Column("native-country-region", DataKind.Text, 8),
                    new TextLoader.Column("age", DataKind.R4, 9),
                    new TextLoader.Column("fnlwgt", DataKind.R4, 10),
                    new TextLoader.Column("education-num", DataKind.R4, 11),
                    new TextLoader.Column("capital-gain", DataKind.R4, 12),
                    new TextLoader.Column("capital-loss", DataKind.R4, 13),
                    new TextLoader.Column("hours-per-week", DataKind.R4, 14)
                }
            });

            IDataView data = reader.Read(dataFilePath);

            var (trainData, testData) = ml.BinaryClassification.TrainTestSplit(data, testFraction: 0.2);

            var pipeline = ml.Transforms.Concatenate("Text", "workclass", "education", "marital-status",
                    "relationship", "ethnicity", "sex", "native-country-region")
                .Append(ml.Transforms.Text.FeaturizeText("Text", "TextFeatures"))
                .Append(ml.Transforms.Concatenate("Features", "TextFeatures", "age", "fnlwgt", 
                    "education-num", "capital-gain", "capital-loss", "hours-per-week"));

            var model = pipeline.Fit(trainData);

            var dataWithPredictions = model.Transform(testData);

            var crossValidation = ml.BinaryClassification.CrossValidate(dataWithPredictions, pipeline);

            var averageAuc = crossValidation.Average(i => i.metrics.Auc);

            Console.WriteLine($"Average cross validation AUC - {averageAuc}");
        }
    }
}
