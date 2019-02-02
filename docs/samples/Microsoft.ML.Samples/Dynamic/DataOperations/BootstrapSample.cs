using System;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public class BootstrapSample
    {
        public static void Example()
        {
            // Downloading the dataset from github.com/dotnet/machinelearning.
            // This will create a housing.txt file in the filesystem.
            // You can open this file, if you want to see the data. 
            string dataFile = SamplesUtils.DatasetUtils.DownloadHousingRegressionDataset();

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Define the reader: specify the data columns and where to find them in the text file.
            var reader = mlContext.Data.CreateTextLoader(
                columns: new[]
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
                    },
                hasHeader: true
            );
            
            // Read the data
            var data = reader.Read(dataFile);

            // Create a training set by taking a bootstrap sample. The sample has about the same number of rows as the input dataset
            // (it's an approximate streaming bootstrap), but will only use about 63% unique rows per sample (i.e. 1-e^-1).
            var train = mlContext.Data.BootstrapSample(data, seed: 632);
            // And create a training set consisting of the complementary set of rows unused above by using the complement parameter
            // with the same seed as before.
            var test = mlContext.Data.BootstrapSample(data, complement: true, seed: 632);

            // Create a learning pipeline
            var labelName = "MedianHomeValue";
            var featuresName = "Features";
            var featureNames = data.Schema
                .Select(column => column.Name) // Get the column names
                .Where(name => name != labelName) // Drop the Label
                .ToArray();
            var pipeline = mlContext.Transforms.Concatenate(featuresName, featureNames)
                    .Append(mlContext.Transforms.Normalize(Transforms.Normalizers.NormalizingEstimator.NormalizerMode.MeanVariance))
                    .Append(mlContext.Regression.Trainers.StochasticDualCoordinateAscent(labelColumn: labelName, featureColumn: featuresName));

            // Train it
            var model = pipeline.Fit(train);

            // Score it
            var scoredTrain = model.Transform(train);
            var scoredTest = model.Transform(test);

            // Evaluate the results
            var trainEval = mlContext.Regression.Evaluate(scoredTrain, label: labelName);
            var testEval = mlContext.Regression.Evaluate(scoredTest, label: labelName);

            Console.WriteLine($"Train RMS: {trainEval.Rms:0.00} Test RMS: {testEval.Rms:0.00}");

            // Expected output:
            //  Train RMS: 17.57 Test RMS: 17.73

            // Exercise: Try to tune the learner hyperparameters to minimize the root mean squared loss.
        }
    }
}
