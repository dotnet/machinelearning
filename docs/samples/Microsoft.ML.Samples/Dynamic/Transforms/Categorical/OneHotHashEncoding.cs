using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Samples.Dynamic
{
    public static class OneHotHashEncoding
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            var samples = new List<DataPoint>()
            {
                new DataPoint(){ Education = "0-5yrs" },
                new DataPoint(){ Education = "0-5yrs" },
                new DataPoint(){ Education = "6-11yrs" },
                new DataPoint(){ Education = "6-11yrs" },
                new DataPoint(){ Education = "11-15yrs" },
            };

            // Convert training data to IDataView.
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // A pipeline for one hot hash encoding the 'Education' column.
            var pipeline = mlContext.Transforms.Categorical.OneHotHashEncoding("EducationOneHotHashEncoded", "Education", numberOfBits: 3);

            // Fit and transform the data.
            var hashEncodedData = pipeline.Fit(data).Transform(data);

            PrintDataColumn(hashEncodedData, "EducationOneHotHashEncoded");
            // We have 8 slots, because we used numberOfBits = 3.

            // 0 0 0 1 0 0 0 0
            // 0 0 0 1 0 0 0 0
            // 0 0 0 0 1 0 0 0
            // 0 0 0 0 1 0 0 0
            // 0 0 0 0 0 0 0 1

            // A pipeline for one hot hash encoding the 'Education' column (using keying strategy).
            var keyPipeline = mlContext.Transforms.Categorical.OneHotHashEncoding("EducationOneHotHashEncoded", "Education", 
                outputKind: OneHotEncodingEstimator.OutputKind.Key,
                numberOfBits: 3);

            // Fit and transform the data.
            var hashKeyEncodedData = keyPipeline.Fit(data).Transform(data);

            // Getting the data of the newly created column, so we can preview it.
            var keyEncodedColumn = hashKeyEncodedData.GetColumn<uint>("EducationOneHotHashEncoded");

            Console.WriteLine("One Hot Hash Encoding of single column 'Education', with key type output.");
            foreach (var element in keyEncodedColumn)
                Console.WriteLine(element);

            // 4
            // 4
            // 5
            // 5
            // 8
        }

        private static void PrintDataColumn(IDataView transformedData, string columnName)
        {
            var countSelectColumn = transformedData.GetColumn<float[]>(transformedData.Schema[columnName]);

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