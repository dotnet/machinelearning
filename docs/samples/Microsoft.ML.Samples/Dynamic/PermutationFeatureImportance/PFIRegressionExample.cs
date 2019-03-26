using System;
using System.Linq;

namespace Microsoft.ML.Samples.Dynamic.PermutationFeatureImportance
{
    public static class PfiRegression
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Step 1: Read the data
            var data = PfiHelper.GetHousingRegressionIDataView(mlContext, out string labelName, out string[] featureNames);

            // Step 2: Pipeline
            // Concatenate the features to create a Feature vector.
            // Normalize the data set so that for each feature, its maximum value is 1 while its minimum value is 0.
            // Then append a linear regression trainer.
            var pipeline = mlContext.Transforms.Concatenate("Features", featureNames)
                    .Append(mlContext.Transforms.Normalize("Features"))
                    .Append(mlContext.Regression.Trainers.Ols(
                        labelColumnName: labelName, featureColumnName: "Features"));
            var model = pipeline.Fit(data);

            // Extract the model from the pipeline
            var linearPredictor = model.LastTransformer;
            var weights = PfiHelper.GetLinearModelWeights(linearPredictor.Model);

            // Compute the permutation metrics using the properly normalized data.
            var transformedData = model.Transform(data);
            var permutationMetrics = mlContext.Regression.PermutationFeatureImportance(
                linearPredictor, transformedData, labelColumnName: labelName, permutationCount: 3);

            // Now let's look at which features are most important to the model overall
            // Get the feature indices sorted by their impact on R-Squared
            var sortedIndices = permutationMetrics.Select((metrics, index) => new { index, metrics.RSquared })
                .OrderByDescending(feature => Math.Abs(feature.RSquared.Mean))
                .Select(feature => feature.index);

            // Print out the permutation results, with the model weights, in order of their impact:
            // Expected console output for 100 permutations:
            //    Feature             Model Weight    Change in R-Squared    95% Confidence Interval of the Mean
            //    RoomsPerDwelling      53.35           -0.4298                 0.005705
            //    EmploymentDistance   -19.21           -0.2609                 0.004591
            //    NitricOxides         -19.32           -0.1569                 0.003701
            //    HighwayDistance        6.11           -0.1173                 0.0025
            //    TeacherRatio         -21.92           -0.1106                 0.002207
            //    TaxRate               -8.68           -0.1008                 0.002083
            //    CrimesPerCapita      -16.37           -0.05988                0.00178
            //    PercentPre40s         -4.52           -0.03836                0.001432
            //    PercentResidental      3.91           -0.02006                0.001079
            //    CharlesRiver           3.49           -0.01839                0.000841
            //    PercentNonRetail      -1.17           -0.002111               0.0003176
            //
            // Let's dig into these results a little bit. First, if you look at the weights of the model, they generally correlate
            // with the results of PFI, but there are some significant misorderings. For example, "Tax Rate" and "Highway Distance" 
            // have relatively small model weights, but the permutation analysis shows these feature to have a larger effect
            // on the accuracy of the model than higher-weighted features. To understand why the weights don't reflect the same 
            // feature importance as PFI, we need to go back to the basics of linear models: one of the assumptions of a linear 
            // model is that the features are uncorrelated. Now, the features in this dataset are clearly correlated: the tax rate
            // for a house and the student-to-teacher ratio at the nearest school, for example, are often coupled through school 
            // levies. The tax rate, distance to a highway, and the crime rate would also seem to be correlated through social 
            // dynamics. We could draw out similar relationships for all variables in this dataset. The reason why the linear 
            // model weights don't reflect the same feature importance as PFI is that the solution to the linear model redistributes 
            // weights between correlated variables in unpredictable ways, so that the weights themselves are no longer a good 
            // measure of feature importance.
            Console.WriteLine("Feature\tModel Weight\tChange in R-Squared\t95% Confidence Interval of the Mean");
            var rSquared = permutationMetrics.Select(x => x.RSquared).ToArray(); // Fetch r-squared as an array
            foreach (int i in sortedIndices)
            {
                Console.WriteLine($"{featureNames[i]}\t{weights[i]:0.00}\t{rSquared[i].Mean:G4}\t{1.96 * rSquared[i].StandardError:G4}");
            }
        }
    }
}
