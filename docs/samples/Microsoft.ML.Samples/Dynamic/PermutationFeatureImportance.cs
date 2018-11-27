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
            // Then append a gam regressor, setting the "MedianHomeValue" column as the label of the dataset,
            // the "Features" column produced by concatenation as the features column.
            var labelName = "MedianHomeValue";
            var pipeline = mlContext.Transforms.Concatenate("Features", "CrimesPerCapita", "PercentResidental",
                    "PercentNonRetail", "CharlesRiver", "NitricOxides", "RoomsPerDwelling", "PercentPre40s",
                    "EmploymentDistance", "HighwayDistance", "TaxRate", "TeacherRatio")
                    .Append(mlContext.Regression.Trainers.StochasticDualCoordinateAscent(
                        labelColumn: labelName, featureColumn: "Features"));
            var fitPipeline = pipeline.Fit(data);

            // Extract the model from the pipeline
            var linearPredictor = fitPipeline.LastTransformer;
            var weights = GetLinearModelWeights(linearPredictor.Model);

            // Compute the permutation metrics using the properly-featurized data.
            var transformedData = fitPipeline.Transform(data);
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

            // Print out the permutation results, with the model weights, in order of their impact
            Console.WriteLine("Feature\tModel Weight\tChange in R-Squared");
            var rSquared = permutationMetrics.Select(x => x.RSquared).ToArray(); // Fetch r-squared as an array
            foreach (int i in sortedIndices)
            {
                Console.WriteLine("{0}\t{1:0.00}\t{2:G4}", featureNames[i], weights[i], rSquared[i]);
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
