using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using System;
using System.Linq;

namespace Microsoft.ML.Samples.Dynamic
{
    public class PFI_RegressionExample
    {
        public static void PFI_Regression()
        {
            // Download the dataset from github.com/dotnet/machinelearning.
            // This will create a housing.txt file in the filesystem.
            // You can open this file to see the data. 
            string dataFile = SamplesUtils.DatasetUtils.DownloadHousingRegressionDataset();

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Step 1: Read the data as an IDataView.
            // First, we define the reader: specify the data columns and where to find them in the text file.
            // The data file is composed of rows of data, with each row having 11 numerical columns
            // separated by whitespace.
            var reader = mlContext.Data.TextReader(new TextLoader.Arguments()
                {
                    Separator = "tab",
                    HasHeader = true,
                    Column = new[]
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
                    }
                });
            
            // Read the data
            var data = reader.Read(dataFile);

            // Step 2: Pipeline
            // Concatenate the features to create a Feature vector.
            // Normalize the data set so that for each feature, its maximum value is 1 while its minimum value is 0.
            // Then append a linear regression trainer, setting the "MedianHomeValue" column as the label of the dataset,
            // the "Features" column produced by concatenation as the features of the dataset.
            var labelName = "MedianHomeValue";
            var pipeline = mlContext.Transforms.Concatenate("Features", "CrimesPerCapita", "PercentResidental",
                        "PercentNonRetail", "CharlesRiver", "NitricOxides", "RoomsPerDwelling", "PercentPre40s",
                        "EmploymentDistance", "HighwayDistance", "TaxRate", "TeacherRatio")
                    .Append(mlContext.Transforms.Normalize("Features"))
                    .Append(mlContext.Regression.Trainers.StochasticDualCoordinateAscent(
                        labelColumn: labelName, featureColumn: "Features"));
            var model = pipeline.Fit(data);

            // Extract the model from the pipeline
            var linearPredictor = model.LastTransformer;
            var weights = GetLinearModelWeights(linearPredictor.Model);

            // Compute the permutation metrics using the properly-featurized data.
            var transformedData = model.Transform(data);
            var permutationMetrics = mlContext.Regression.PermutationFeatureImportance(
                linearPredictor, transformedData, label: labelName, features: "Features");

            // Now let's look at which features are most important to the model overall
            // First, we have to prepare the data:
            // Get the feature names as an IEnumerable
            var featureNames = data.Schema.GetColumns()
                .Select(tuple => tuple.column.Name) // Get the column names
                .Where(name => name != labelName) // Drop the Label
                .ToArray();

            // Get the feature indices sorted by their impact on R-Squared
            var sortedIndices = permutationMetrics.Select((metrics, index) => new { index, metrics.RSquared })
                .OrderByDescending(feature => Math.Abs(feature.RSquared))
                .Select(feature => feature.index);

            // Print out the permutation results, with the model weights, in order of their impact:
            // Expected console output:
            //    Feature            Model Weight    Change in R - Squared
            //    RoomsPerDwelling      50.80 -0.3695
            //    EmploymentDistance   -17.79 -0.2238
            //    TeacherRatio         -19.83 -0.1228
            //    TaxRate              -8.60  -0.1042
            //    NitricOxides         -15.95 -0.1025
            //    HighwayDistance        5.37 -0.09345
            //    CrimesPerCapita      -15.05 -0.05797
            //    PercentPre40s         -4.64 -0.0385
            //    PercentResidental      3.98 -0.02184
            //    CharlesRiver           3.38 -0.01487
            //    PercentNonRetail      -1.94 -0.007231
            //
            // Let's dig into these results a little bit. First, if you look at the weights of the model, they generally correlate
            // with the results of PFI, but there are some significant misorderings. For example, "Tax Rate" is weighted lower than
            // "Nitric Oxides" and "Crimes Per Capita", but the permutation analysis shows this feature to have a larger effect
            // on the accuracy of the model even though it has a relatively small weight. To understand why the weights don't 
            // reflect the same feature importance as PFI, we need to go back to the basics of linear models: one of the 
            // assumptions of a linear model is that the features are uncorrelated. Now, the features in this dataset are clearly 
            // correlated: the tax rate for a house and the student-to-teacher ratio at the nearest school, for example, are often 
            // coupled through school levies. The tax rate, presence of pollution (e.g. nitric oxides), and the crime rate would also
            // seem to be correlated with each other through social dynamics. We could draw out similar relationships for all the 
            // variables in this dataset. The reason why the linear model weights don't reflect the same feature importance as PFI
            // is that the solution to the linear model redistributes weights between correlated variables in unpredictable ways, so
            // that the weights themselves are no longer a good measure of feature importance.
            Console.WriteLine("Feature\tModel Weight\tChange in R-Squared");
            var rSquared = permutationMetrics.Select(x => x.RSquared).ToArray(); // Fetch r-squared as an array
            foreach (int i in sortedIndices)
            {
                Console.WriteLine($"{featureNames[i]}\t{weights[i]:0.00}\t{rSquared[i]:G4}");
            }
        }

        private static float[] GetLinearModelWeights(LinearRegressionPredictor linearModel)
        {
            var weights = new VBuffer<float>();
            linearModel.GetFeatureWeights(ref weights);
            return weights.GetValues().ToArray();
        }
    }
}
