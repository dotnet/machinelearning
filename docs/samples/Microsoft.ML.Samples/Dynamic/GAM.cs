using Microsoft.ML.Runtime.Data;
using System;

namespace Microsoft.ML.Samples.Dynamic
{
    public class GAM_BinaryClassificationExample
    {
        public static void GAM_BinaryClassification()
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
            // Then append a gam regressor, setting the "MedianHomeValue" column as the label of the dataset,
            // the "Features" column produced by concatenation as the features column,
            // and use a small number of bins to make it easy to visualize in the console window.
            // For real appplications, it is recommended to start with the default number of bins.
            var pipeline = mlContext.Transforms.Concatenate("Features", "CrimesPerCapita", "PercentResidental",
                "PercentNonRetail", "CharlesRiver", "NitricOxides", "RoomsPerDwelling", "PercentPre40s",
                "EmploymentDistance", "HighwayDistance", "TaxRate", "TeacherRatio")
                    .Append(mlContext.Regression.Trainers.GeneralizedAdditiveModels(
                        labelColumn: "MedianHomeValue", featureColumn: "Features", maxBins: 16));
            var fitPipeline = pipeline.Fit(data);

            // Extract the model from the pipeline
            var gamModel = fitPipeline.LastTransformer.Model;

            // Step 3: Investigate the properties of the model

            // The intercept for the GAM models represent the average prediction for the training data
            var intercept = gamModel.Intercept;
            Console.WriteLine($"Average predicted cost: {intercept}");

            // Each feature represents the deviation from the average prediction as a function of the feature value
            
            // Let's take a look at the TeacherRatio
            var teacherRatioBinUpperBounds = gamModel.GetFeatureBinUpperBounds(10);
            var teacherRatioFeatureWeights = gamModel.GetFeatureWeights(10);

            Console.Write("We can see that smaller class sizes are predictive of a higher house value, while ");
            Console.WriteLine("student-teacher ratios higher than about 18 lead to lower predictions in house value.");

            Console.WriteLine(string.Empty);
            Console.WriteLine("Student-Teacher Ratio");
            for (int i = 0; i < teacherRatioBinUpperBounds.Length; i++)
            {
                Console.WriteLine($"x < {teacherRatioBinUpperBounds[i]:0.00} => {teacherRatioFeatureWeights[i]:0.000}");
            }
            Console.WriteLine(string.Empty);

            Console.Write("Note that these measurements are noisy (see student-teacher ratios > ");
            Console.Write($"{ teacherRatioBinUpperBounds[teacherRatioBinUpperBounds.Length - 2]:0.00}). ");
            Console.Write("Common practice is to use resampling methods to estimate confidence at each bin. ");
            Console.Write("See for example, Tan, Caruana, Hooker, and Lou. ");
            Console.WriteLine("\"Distill-and-Compare: Auditing Black-Box Models Using Transparent Model Distillation.\" arXiv:1710.06169.");
        }
    }
}
