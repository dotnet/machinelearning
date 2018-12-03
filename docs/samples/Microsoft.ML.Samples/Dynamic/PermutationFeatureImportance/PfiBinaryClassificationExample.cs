using Microsoft.ML.Runtime.Learners;
using System;
using System.Linq;

namespace Microsoft.ML.Samples.Dynamic.PermutationFeatureImportance
{
    public class PfiBinaryClassificationExample
    {
        public static void RunExample()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Step 1: Read the data
            var data = PfiHelper.GetHousingRegressionIDataView(mlContext, 
                out string labelName, out string[] featureNames, binaryPrediction: true);

            // Step 2: Pipeline
            // Concatenate the features to create a Feature vector.
            // Normalize the data set so that for each feature, its maximum value is 1 while its minimum value is 0.
            // Then append a logistic regression trainer.
            var pipeline = mlContext.Transforms.Concatenate("Features", featureNames)
                    .Append(mlContext.Transforms.Normalize("Features"))
                    .Append(mlContext.BinaryClassification.Trainers.LogisticRegression(
                        labelColumn: labelName, featureColumn: "Features"));
            var model = pipeline.Fit(data);

            // Extract the model from the pipeline
            var linearPredictor = model.LastTransformer;
            // Linear models for binary classification are wrapped by a calibrator as a generic predictor
            //  To access it directly, we must extract it out and cast it to the proper class
            var weights = PfiHelper.GetLinearModelWeights(linearPredictor.Model.SubPredictor as LinearBinaryModelParameters);

            // Compute the permutation metrics using the properly normalized data.
            var transformedData = model.Transform(data);
            var permutationMetrics = mlContext.BinaryClassification.PermutationFeatureImportance(
                linearPredictor, transformedData, label: labelName, features: "Features");

            // Now let's look at which features are most important to the model overall
            // Get the feature indices sorted by their impact on AUC
            var sortedIndices = permutationMetrics.Select((metrics, index) => new { index, metrics.Auc })
                .OrderByDescending(feature => Math.Abs(feature.Auc.Mean))
                .Select(feature => feature.index);

            // Print out the permutation results, with the model weights, in order of their impact:
            // Expected console output:
            //    Feature            Model Weight Change in AUC
            //    PercentPre40s         -1.96 -0.04582
            //    RoomsPerDwelling       3.71 -0.04516
            //    EmploymentDistance    -1.31 -0.02375
            //    TeacherRatio          -2.46 -0.01476
            //    CharlesRiver           0.66 -0.008683
            //    PercentNonRetail      -1.58 -0.007314
            //    PercentResidental      0.60  0.003979
            //    TaxRate               -0.95  0.002739
            //    NitricOxides          -0.32  0.001917
            //    CrimesPerCapita       -0.04 -3.222E-05
            //    HighwayDistance        0.00  0
            //
            // Let's look at these results.
            // First, if you look at the weights of the model, they generally correlate with the results of PFI,
            // but there are some significant misorderings. See the discussion in the Regression example for an
            // explanation of why this happens and how to interpret it.
            // Second, the logistic regression learner uses L1 regularization by default. Here, it causes the "HighWay Distance"
            // feature to be zeroed out from the model. PFI assigns zero importance to this variable, as expected.
            // Third, some features showed an *increase* in AUC. This means that the model actually improved 
            // when these features were shuffled. This is actually expected when the effects are small (here on the order of 10^-3).
            // This is due to the random nature of permutations. To reduce computational costs, PFI performs a single
            // permutation per feature, which means that the change in AUC is just from one sample of the data.
            // If each feature were permuted many times and the average computed, the resuting average change in AUC
            // would be small and negative for these features, or zero if the features truly were meaningless.
            // To see observe this behavior yourself, try adding a second call to PFI and compare the results, or
            // rerun the script with a different seed set in the MLContext(), like so:
            //  `var mlContext = new MLContext(seed: 12345);`
            Console.WriteLine("Feature\tModel Weight\tChange in AUC\t95% Confidence in the Mean Change in AUC");
            var auc = permutationMetrics.Select(x => x.Auc).ToArray(); // Fetch AUC as an array
            foreach (int i in sortedIndices)
            {
                Console.WriteLine($"{featureNames[i]}\t{weights[i]:0.00}\t{auc[i].Mean:G4}\t{1.96 * auc[i].StandardError:G4}");
            }
            // DON"T CHECK IN UNTIL TEXT IS COMPLETE
        }
    }
}
