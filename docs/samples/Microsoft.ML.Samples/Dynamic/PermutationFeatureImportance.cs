using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Trainers.HalLearners;
using System;
using System.Linq;

namespace Microsoft.ML.Samples.Dynamic
{
    public class PermutationFeatureImportance_Examples
    {
        public static void PFI_Regression()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Step 1: Read the data
            var data = GetHousingRegressionIDataView(mlContext, out string labelName, out string[] featureNames);

            // Step 2: Pipeline
            // Concatenate the features to create a Feature vector.
            // Normalize the data set so that for each feature, its maximum value is 1 while its minimum value is 0.
            // Then append a linear regression trainer.
            var pipeline = mlContext.Transforms.Concatenate("Features", featureNames)
                    .Append(mlContext.Transforms.Normalize("Features"))
                    .Append(mlContext.Regression.Trainers.OrdinaryLeastSquares(
                        labelColumn: labelName, featureColumn: "Features"));

            var model = pipeline.Fit(data);
            // Extract the model from the pipeline
            var linearPredictor = model.LastTransformer;
            var weights = GetLinearModelWeights(linearPredictor.Model);

            // Compute the permutation metrics using the properly normalized data.
            var transformedData = model.Transform(data);
            var permutationMetrics = mlContext.Regression.PermutationFeatureImportance(
                linearPredictor, transformedData, label: labelName, features: "Features", permutationCount: 3);

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
        public static void PFI_BinaryClassification()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Step 1: Read the data
            var data = GetHousingRegressionIDataView(mlContext, out string labelName, out string[] featureNames, binaryPrediction: true);

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
            var weights = GetLinearModelWeights(linearPredictor.Model.SubPredictor as LinearBinaryModelParameters);

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

        private static IDataView GetHousingRegressionIDataView(MLContext mlContext, out string labelName, out string[] featureNames, bool binaryPrediction = false)
        {
            // Download the dataset from github.com/dotnet/machinelearning.
            // This will create a housing.txt file in the filesystem.
            // You can open this file to see the data. 
            string dataFile = SamplesUtils.DatasetUtils.DownloadHousingRegressionDataset();

            // Read the data as an IDataView.
            // First, we define the reader: specify the data columns and where to find them in the text file.
            // The data file is composed of rows of data, with each row having 11 numerical columns
            // separated by whitespace.
            var reader = mlContext.Data.CreateTextReader(
                columns: new[]
                    {
                        // Read the first column (indexed by 0) in the data file as an R4 (float)
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
                    },
                hasHeader: true
            );

            // Read the data
            var data = reader.Read(dataFile);
            var labelColumn = "MedianHomeValue";

            if (binaryPrediction)
            {
                labelColumn = nameof(BinaryOutputRow.AboveAverage);
                data = mlContext.Transforms.CustomMappingTransformer(GreaterThanAverage, null).Transform(data);
                data = mlContext.Transforms.DropColumns("MedianHomeValue").Fit(data).Transform(data);
            }

            labelName = labelColumn;
            featureNames = data.Schema.AsEnumerable()
                .Select(column => column.Name) // Get the column names
                .Where(name => name != labelColumn) // Drop the Label
                .ToArray();

            return data;
        }

        // Define a class for all the input columns that we intend to consume.
        private class ContinuousInputRow
        {
            public float MedianHomeValue { get; set; }
        }

        // Define a class for all output columns that we intend to produce.
        private class BinaryOutputRow
        {
            public bool AboveAverage { get; set; }
        }

        // Define an Action to apply a custom mapping from one object to the other
        private readonly static Action<ContinuousInputRow, BinaryOutputRow> GreaterThanAverage = (input, output) 
            => output.AboveAverage = input.MedianHomeValue > 22.6;

        private static float[] GetLinearModelWeights(OlsLinearRegressionModelParameters linearModel)
        {
            return linearModel.Weights.ToArray();
        }

        private static float[] GetLinearModelWeights(LinearBinaryModelParameters linearModel)
        {
            return linearModel.Weights.ToArray();
        }
    }
}
