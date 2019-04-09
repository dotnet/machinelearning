using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.Transforms.OneHotEncodingEstimator;

namespace Samples.Dynamic
{
    public static class OneHotEncoding
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

            // A pipeline for one hot encoding the Education column.
            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding("EducationOneHotEncoded", "Education");

            // Fit and transform the data.
            var oneHotEncodedData = pipeline.Fit(data).Transform(data);

            PrintDataColumn(oneHotEncodedData, "EducationOneHotEncoded");
            // We have 3 slots, because there are three categories in the 'Education' column.
            // 1 0 0
            // 1 0 0
            // 0 1 0
            // 0 1 0
            // 0 0 1

            // A pipeline for one hot encoding the Education column (using keying).
            var keyPipeline = mlContext.Transforms.Categorical.OneHotEncoding("EducationOneHotEncoded", "Education", OutputKind.Key);

            // Fit and Transform data.
            oneHotEncodedData = keyPipeline.Fit(data).Transform(data);

            var keyEncodedColumn = oneHotEncodedData.GetColumn<uint>("EducationOneHotEncoded");

            Console.WriteLine("One Hot Encoding of single column 'Education', with key type output.");
            foreach (var element in keyEncodedColumn)
                Console.WriteLine(element);

            // 1
            // 1
            // 2
            // 2
            // 3
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