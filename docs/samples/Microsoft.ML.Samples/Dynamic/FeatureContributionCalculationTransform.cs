using System;
using Microsoft.ML.Data;
using Microsoft.ML.SamplesUtils;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class FeatureContributionCalculationTransform
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Read the Housing regression dataset
            var data = DatasetUtils.LoadHousingRegressionDataset(mlContext);

            // Create a pipeline.
            // Concatenate the features to create a Feature vector.
            // Then append a linear model, setting the "MedianHomeValue" column as the label of the dataset,
            // the "Features" column produced by concatenation as the features column.
            var transformPipeline = mlContext.Transforms.Concatenate("Features", "CrimesPerCapita", "PercentResidental",
                "PercentNonRetail", "CharlesRiver", "NitricOxides", "RoomsPerDwelling", "PercentPre40s",
                "EmploymentDistance", "HighwayDistance", "TaxRate", "TeacherRatio");
            var learner = mlContext.Regression.Trainers.Ols(
                        labelColumnName: "MedianHomeValue", featureColumnName: "Features");

            var transformedData = transformPipeline.Fit(data).Transform(data);

            // Now we train the model and score it on the transformed data.
            var model = learner.Fit(transformedData);
            var scoredData = model.Transform(transformedData);

            // Create a Feature Contribution Calculator
            // Calculate the feature contributions for all features given trained model parameters
            // And don't normalize the contribution scores
            var featureContributionCalculator = mlContext.Transforms.CalculateFeatureContribution(model, numberOfPositiveContributions: 11, normalize: false);
            var outputData = featureContributionCalculator.Fit(scoredData).Transform(scoredData);

            // FeatureContributionCalculatingEstimator can be use as an intermediary step in a pipeline. 
            // The features retained by FeatureContributionCalculatingEstimator will be in the FeatureContribution column.
            var pipeline = mlContext.Transforms.CalculateFeatureContribution(model, numberOfPositiveContributions: 11)
                .Append(mlContext.Regression.Trainers.Ols(featureColumnName: "FeatureContributions"));
            var outData = featureContributionCalculator.Fit(scoredData).Transform(scoredData);

            // Let's extract the weights from the linear model to use as a comparison
            var weights = model.Model.Weights;

            // Let's now walk through the first ten records and see which feature drove the values the most
            // Get prediction scores and contributions
            var scoringEnumerator = mlContext.Data.CreateEnumerable<HousingRegressionScoreAndContribution>(outputData, true).GetEnumerator();
            int index = 0;
            Console.WriteLine("Label\tScore\tBiggestFeature\tValue\tWeight\tContribution");
            while (scoringEnumerator.MoveNext() && index < 10)
            {
                var row = scoringEnumerator.Current;

                // Get the feature index with the biggest contribution
                var featureOfInterest = GetMostContributingFeature(row.FeatureContributions);

                // And the corresponding information about the feature
                var value = row.Features[featureOfInterest];
                var contribution = row.FeatureContributions[featureOfInterest];
                var name = data.Schema[featureOfInterest + 1].Name;
                var weight = weights[featureOfInterest];

                Console.WriteLine("{0:0.00}\t{1:0.00}\t{2}\t{3:0.00}\t{4:0.00}\t{5:0.00}",
                    row.MedianHomeValue,
                    row.Score,
                    name,
                    value,
                    weight,
                    contribution
                    );

                index++;
            }
            Console.ReadLine();

            // The output of the above code is:
            // Label Score   BiggestFeature Value   Weight Contribution
            // 24.00   27.74   RoomsPerDwelling        6.58    98.55   39.95
            // 21.60   23.85   RoomsPerDwelling        6.42    98.55   39.01
            // 34.70   29.29   RoomsPerDwelling        7.19    98.55   43.65
            // 33.40   27.17   RoomsPerDwelling        7.00    98.55   42.52
            // 36.20   27.68   RoomsPerDwelling        7.15    98.55   43.42
            // 28.70   23.13   RoomsPerDwelling        6.43    98.55   39.07
            // 22.90   22.71   RoomsPerDwelling        6.01    98.55   36.53
            // 27.10   21.72   RoomsPerDwelling        6.17    98.55   37.50
            // 16.50   18.04   RoomsPerDwelling        5.63    98.55   34.21
            // 18.90   20.14   RoomsPerDwelling        6.00    98.55   36.48
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
