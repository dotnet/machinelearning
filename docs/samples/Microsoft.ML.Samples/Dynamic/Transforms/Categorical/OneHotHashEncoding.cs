using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Samples.Dynamic.Transforms.Categorical
{
    public static class OneHotHashEncoding
    {
        public static void Example()
        {
            // Create a new ML context for ML.NET operations. It can be used for
            // exception tracking and logging as well as the source of randomness.
            var mlContext = new MLContext();

            // Create a small dataset as an IEnumerable.
            var samples = new[]
            {
                new DataPoint {Education = "0-5yrs"},
                new DataPoint {Education = "0-5yrs"},
                new DataPoint {Education = "6-11yrs"},
                new DataPoint {Education = "6-11yrs"},
                new DataPoint {Education = "11-15yrs"}
            };

            // Convert training data to an IDataView.
            IDataView data = mlContext.Data.LoadFromEnumerable(samples);

            // A pipeline for one hot hash encoding the 'Education' column.
            var pipeline = mlContext.Transforms.Categorical.OneHotHashEncoding(
                "EducationOneHotHashEncoded", "Education", numberOfBits: 3);

            // Fit and transform the data.
            IDataView hashEncodedData = pipeline.Fit(data).Transform(data);

            PrintDataColumn(hashEncodedData, "EducationOneHotHashEncoded");
            // We have 8 slots, because we used numberOfBits = 3.

            // 0 0 0 1 0 0 0 0
            // 0 0 0 1 0 0 0 0
            // 0 0 0 0 1 0 0 0
            // 0 0 0 0 1 0 0 0
            // 0 0 0 0 0 0 0 1

            // A pipeline for one hot hash encoding the 'Education' column
            // (using keying strategy).
            var keyPipeline = mlContext.Transforms.Categorical.OneHotHashEncoding(
                "EducationOneHotHashEncoded", "Education",
                OneHotEncodingEstimator.OutputKind.Key, 3);

            // Fit and transform the data.
            IDataView hashKeyEncodedData = keyPipeline.Fit(data).Transform(data);

            // Get the data of the newly created column for inspecting.
            var keyEncodedColumn =
                hashKeyEncodedData.GetColumn<uint>("EducationOneHotHashEncoded");

            Console.WriteLine(
                "One Hot Hash Encoding of single column 'Education', with key " +
                "type output.");

            // One Hot Hash Encoding of single column 'Education', with key type output.

            foreach (uint element in keyEncodedColumn)
                Console.WriteLine(element);

            // 4
            // 4
            // 5
            // 5
            // 8
        }

        private static void PrintDataColumn(IDataView transformedData,
            string columnName)
        {
            var countSelectColumn = transformedData.GetColumn<float[]>(
                transformedData.Schema[columnName]);

            foreach (var row in countSelectColumn)
            {
                for (var i = 0; i < row.Length; i++)
                    Console.Write($"{row[i]}\t");
                Console.WriteLine();
            }
        }

        private class DataPoint
        {
            public string Education { get; set; }
        }
    }
}
