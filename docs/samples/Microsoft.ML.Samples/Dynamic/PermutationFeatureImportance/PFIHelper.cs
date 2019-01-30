using System;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Learners;
using Microsoft.ML.Trainers.HalLearners;

namespace Microsoft.ML.Samples.Dynamic.PermutationFeatureImportance
{
    public class PfiHelper
    {
        public static IDataView GetHousingRegressionIDataView(MLContext mlContext, out string labelName, out string[] featureNames, bool binaryPrediction = false)
        {
            // Download the dataset from github.com/dotnet/machinelearning.
            // This will create a housing.txt file in the filesystem.
            // You can open this file to see the data. 
            string dataFile = SamplesUtils.DatasetUtils.DownloadHousingRegressionDataset();

            // Read the data as an IDataView.
            // First, we define the reader: specify the data columns and where to find them in the text file.
            // The data file is composed of rows of data, with each row having 11 numerical columns
            // separated by whitespace.
            var reader = mlContext.Data.CreateTextLoader(
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

        public static float[] GetLinearModelWeights(OlsLinearRegressionModelParameters linearModel)
        {
            return linearModel.Weights.ToArray();
        }

        public static float[] GetLinearModelWeights(LinearBinaryModelParameters linearModel)
        {
            return linearModel.Weights.ToArray();
        }
    }
}
