using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using System;

namespace Microsoft.ML.Samples.Dynamic
{
    public class FeatureContributionCalculationTransform_RegressionExample
    {
        public static void FeatureContributionCalculationTransform_Regression()
        {
            // Downloading the dataset from github.com/dotnet/machinelearning.
            // This will create a sentiment.tsv file in the filesystem.
            // You can open this file, if you want to see the data. 
            string dataFile = SamplesUtils.DatasetUtils.DownloadHousingRegressionDataset();

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Step 1: Read the data as an IDataView.
            // First, we define the reader: specify the data columns and where to find them in the text file.
            var reader = mlContext.Data.TextReader(new TextLoader.Arguments()
                {
                    Separator = "tab",
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoader.Column("MedianHomeValue", DataKind.R4, 0),
                        new TextLoader.Column("CrimesPerCapita", DataKind.R4, 1),
                        new TextLoader.Column("PercentResidental", DataKind.R4, 2),
                        new TextLoader.Column("PercentNonRetail", DataKind.R4, 3),
                        new TextLoader.Column("CharlesRiver", DataKind.R4, 4),
                        new TextLoader.Column("NitricOxides", DataKind.R4, 5),
                        new TextLoader.Column("RoomsPerDwelling", DataKind.R4, 6),
                        new TextLoader.Column("PercentPre40s", DataKind.R4, 7),
                        new TextLoader.Column("EmploymentDistance", DataKind.R4, 8),
                        new TextLoader.Column("HighwayDistance", DataKind.R4, 9),
                        new TextLoader.Column("TaxRate", DataKind.R4, 10),
                        new TextLoader.Column("TeacherRatio", DataKind.R4, 11),
                    }
                });
            
            // Read the data
            var data = reader.Read(dataFile);

            // Step 2: Pipeline
            // Concatenate the features to create a Feature vector.
            // Then append a linear model, setting the "MedianHomeValue" column as the label of the dataset,
            // the "Features" column produced by concatenation as the features column.
            var transformPipeline = mlContext.Transforms.Concatenate("Features", "CrimesPerCapita", "PercentResidental",
                "PercentNonRetail", "CharlesRiver", "NitricOxides", "RoomsPerDwelling", "PercentPre40s",
                "EmploymentDistance", "HighwayDistance", "TaxRate", "TeacherRatio");
            var learner = mlContext.Regression.Trainers.StochasticDualCoordinateAscent(
                        labelColumn: "MedianHomeValue", featureColumn: "Features");

            var transformedData = transformPipeline.Fit(data).Transform(data);

            var model = learner.Fit(transformedData);

            // Create a Feature Contribution Calculator
            // Calculate the feature contributions for all features
            // And don't normalize the contribution scores
            var args = new FeatureContributionCalculationTransform.Arguments()
            {
                Top = 11,
                Normalize = false
            };
            var featureContributionCalculator = FeatureContributionCalculationTransform.Create(mlContext, args, transformedData, model.Model, model.FeatureColumn);

            // Let's extract the weights from the linear model to use as a comparison
            var weights = new VBuffer<float>();
            model.Model.GetFeatureWeights(ref weights);

            // Let's now walk through the first ten reconds and see which feature drove the values the most
            // Get prediction scores and contributions
            var scoringEnumerator = featureContributionCalculator.AsEnumerable<HousingRegressionScoreAndContribution>(mlContext, true).GetEnumerator();
            int index = 0;
            Console.WriteLine("Label\tScore\tBiggestFeature\tValue\tWeight\tContribution\tPercent");
            while (scoringEnumerator.MoveNext() && index < 10)
            {
                var row = scoringEnumerator.Current;

                // Get the feature index with the biggest contribution
                var featureOfInterest = GetMostContributingFeature(row.FeatureContributions);

                // And the corresponding information about the feature
                var value = row.Features[featureOfInterest];
                var contribution = row.FeatureContributions[featureOfInterest];
                var percentContribution = 100 * contribution / row.Score;
                var name = data.Schema.GetColumnName(featureOfInterest + 1);
                var weight = weights.GetValues()[featureOfInterest];

                Console.WriteLine("{0:0.00}\t{1:0.00}\t{2}\t{3:0.00}\t{4:0.00}\t{5:0.00}\t{6:0.00}",
                    row.MedianHomeValue,
                    row.Score,
                    name,
                    value,
                    weight,
                    contribution,
                    percentContribution
                    );

                index++;
            }

            // For bulk scoring, the ApplyToData API can also be used
            var scoredData = featureContributionCalculator.ApplyToData(mlContext, transformedData);
            var preview = scoredData.Preview(100);
        }

        private static int GetMostContributingFeature(float[] featureContributions)
        {
            int index = 0;
            float currentValue = float.NegativeInfinity;
            for (int i = 0; i < featureContributions.Length; i++)
                if (featureContributions[i] > currentValue)
                {
                    currentValue = featureContributions[i];
                    index = i;
                }
            return index;
        }

        private sealed class HousingRegressionScoreAndContribution
        {
            public float MedianHomeValue { get; set; }

            [VectorType(11)]
            public float[] Features { get; set; }

            public float Score { get; set; }

            [VectorType(4)]
            public float[] FeatureContributions { get; set; }
        }
    }
}
