using System;
using System.Linq;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.Samples.Dynamic.PermutationFeatureImportance
{
    public static class PfiBinaryClassification
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext(seed:999123);

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
                        labelColumnName: labelName, featureColumnName: "Features"));
            var model = pipeline.Fit(data);

            // Extract the model from the pipeline
            var linearPredictor = model.LastTransformer;
            // Linear models for binary classification are wrapped by a calibrator as a generic predictor
            //  To access it directly, we must extract it out and cast it to the proper class
            var weights = PfiHelper.GetLinearModelWeights(linearPredictor.Model.SubModel as LinearBinaryModelParameters);

            // Compute the permutation metrics using the properly normalized data.
            var transformedData = model.Transform(data);
            var permutationMetrics = mlContext.BinaryClassification.PermutationFeatureImportance(
                linearPredictor, transformedData, labelColumnName: labelName, permutationCount: 3);

            // Now let's look at which features are most important to the model overall.
            // Get the feature indices sorted by their impact on AreaUnderRocCurve.
            var sortedIndices = permutationMetrics.Select((metrics, index) => new { index, metrics.AreaUnderRocCurve })
                .OrderByDescending(feature => Math.Abs(feature.AreaUnderRocCurve.Mean))
                .Select(feature => feature.index);

            // Print out the permutation results, with the model weights, in order of their impact:
            // Expected console output (for 100 permutations):
            //    Feature            Model Weight    Change in AUC   95% Confidence in the Mean Change in AUC
            //    PercentPre40s      -1.96            -0.06316        0.002377
            //    RoomsPerDwelling    3.71            -0.04385        0.001245
            //    EmploymentDistance -1.31            -0.02139        0.0006867
            //    TeacherRatio       -2.46            -0.0203         0.0009566
            //    PercentNonRetail   -1.58            -0.01846        0.001586
            //    CharlesRiver        0.66            -0.008605       0.0005136
            //    PercentResidental   0.60             0.002483       0.0004818
            //    TaxRate            -0.95            -0.00221        0.0007394
            //    NitricOxides       -0.32             0.00101        0.0001428
            //    CrimesPerCapita    -0.04            -3.029E-05      1.678E-05
            //    HighwayDistance     0.00             0              0
            // Let's look at these results.
            // First, if you look at the weights of the model, they generally correlate with the results of PFI,
            // but there are some significant misorderings. See the discussion in the Regression example for an
            // explanation of why this happens and how to interpret it.
            // Second, the logistic regression learner uses L1 regularization by default. Here, it causes the "HighWay Distance"
            // feature to be zeroed out from the model. PFI assigns zero importance to this variable, as expected.
            // Third, some features show an *increase* in AUC. This means that the model actually improved 
            // when these features were shuffled. This is a sign to investigate these features further.
            Console.WriteLine("Feature\tModel Weight\tChange in AUC\t95% Confidence in the Mean Change in AUC");
            var auc = permutationMetrics.Select(x => x.AreaUnderRocCurve).ToArray(); // Fetch AUC as an array
            foreach (int i in sortedIndices)
            {
                Console.WriteLine($"{featureNames[i]}\t{weights[i]:0.00}\t{auc[i].Mean:G4}\t{1.96 * auc[i].StandardError:G4}");
            }
        }
    }
}
