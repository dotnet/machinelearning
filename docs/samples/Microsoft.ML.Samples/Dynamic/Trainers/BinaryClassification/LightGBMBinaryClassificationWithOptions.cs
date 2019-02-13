using Microsoft.ML.LightGBM;
using Microsoft.ML.Transforms.Categorical;
using static Microsoft.ML.LightGBM.Options;

namespace Microsoft.ML.Samples.Dynamic
{
    class LightGbmBinaryClassificationWithOptions
    {
        /// <summary>
        /// This example require installation of addition nuget package <a href="https://www.nuget.org/packages/Microsoft.ML.LightGBM/">Microsoft.ML.LightGBM</a>
        /// </summary>
        public static void Example()
        {
            // Creating the ML.Net IHostEnvironment object, needed for the pipeline
            var mlContext = new MLContext();

            // Download the dataset and load it as IDataView
            var dataview = SamplesUtils.DatasetUtils.LoadAdultDataset(mlContext);

            // Leave out 10% of data for testing
            var split = mlContext.BinaryClassification.TrainTestSplit(dataview, testFraction: 0.1);

            // Create the pipeline with LightGbm Estimator using advanced options
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
            .Append(mlContext.BinaryClassification.Trainers.LightGbm(
                new Options
                {
                    LabelColumn = "IsOver50K",
                    FeatureColumn = "Features",
                    Booster = new GossBooster.Arguments
                    {
                        TopRate = 0.3,
                        OtherRate = 0.2
                    }
                }));

            // Fit this Pipeline to the Training Data
            var model = pipeline.Fit(split.TrainSet);

            // Evaluate how the model is doing on the test data
            var dataWithPredictions = model.Transform(split.TestSet);

            var metrics = mlContext.BinaryClassification.Evaluate(dataWithPredictions, "IsOver50K");
            SamplesUtils.ConsoleUtils.PrintMetrics(metrics);

            // Output:
            // Accuracy: 0.84
            // AUC: 0.88
            // F1 Score: 0.62
            // Negative Precision: 0.88
            // Negative Recall: 0.92
            // Positive Precision: 0.67
            // Positive Recall: 0.58
        }
    }
}
