using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using static Microsoft.ML.Transforms.OneHotEncodingEstimator;

namespace Microsoft.ML.Samples.Dynamic
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
                new DataPoint(){ Label = 0, Education = "0-5yrs", ZipCode = "98005" },
                new DataPoint(){ Label = 1, Education = "0-5yrs", ZipCode = "98052" },
                new DataPoint(){ Label = 45, Education = "6-11yrs", ZipCode = "98005" },
                new DataPoint(){ Label = 50, Education = "6-11yrs", ZipCode = "98052" },
                new DataPoint(){ Label = 50, Education = "11-15yrs", ZipCode = "98005" },
            };

            // Convert training data to IDataView.
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // A pipeline for one hot encoding the Education column.
            var bagPipeline = mlContext.Transforms.Categorical.OneHotEncoding("EducationOneHotEncoded", "Education");
            
            // Fit and transform the data.
            var oneHotEncodedData = bagPipeline.Fit(data).Transform(data);

            // Getting the data of the newly created column, so we can preview it.
            var encodedDataColumn = oneHotEncodedData.GetColumn<float[]>("EducationOneHotEncoded");

            Console.WriteLine("One Hot Encoding of single column 'Education'");
            foreach (var row in encodedDataColumn)
            {
                for (var i = 0; i < row.Length; i++)
                    Console.Write($"{row[i]} ");
                Console.WriteLine();
            }

            // We have 3 slots, because there are three categories in the 'Education' column.
            
            // 1 0 0
            // 1 0 0
            // 0 1 0
            // 0 1 0
            // 0 0 1

            // A pipeline for one hot encoding the Education column (using keying).
            var keyPipeline = mlContext.Transforms.Categorical.OneHotEncoding("EducationOneHotEncoded", "Education", OutputKind.Key);
            
            // Fit and Transform data.
            var keyTransformer = keyPipeline.Fit(data).Transform(data);

            // Getting the data of the newly created column, so we can preview it.
            var keyEncodedColumn = keyTransformer.GetColumn<uint>("EducationOneHotEncoded");

            Console.WriteLine("One Hot Encoding of single column 'Education', with key type output.");
            foreach (var element in keyEncodedColumn)
                Console.WriteLine(element);

            // 1
            // 1
            // 2
            // 2
            // 3

            // Multi column example : A pipeline for one hot encoding two columns 'Education' and 'ZipCode' 
            var multiColumnKeyPipeline = mlContext.Transforms.Categorical.OneHotEncoding(
                new InputOutputColumnPair[] {
                    new InputOutputColumnPair("Education"),
                    new InputOutputColumnPair("ZipCode"),
                });

            // Fit and Transform data.
            var transformedData = multiColumnKeyPipeline.Fit(data).Transform(data);

            var convertedData = mlContext.Data.CreateEnumerable<TransformedData>(transformedData, true);

            Console.WriteLine("One Hot Encoding of two columns 'Education' and 'ZipCode'.");
            foreach (var item in convertedData)
                Console.WriteLine("{0}\t\t\t{1}", string.Join(" ", item.Education), string.Join(" ", item.ZipCode));

            // 1 0 0                   1 0
            // 1 0 0                   0 1
            // 0 1 0                   1 0
            // 0 1 0                   0 1
            // 0 0 1                   1 0
        }

        private class DataPoint
        {
            public float Label { get; set; }

            public string Education { get; set; }

            public string ZipCode { get; set; }
        }

        private class TransformedData
        {
            public float[] Education { get; set; }

            public float[] ZipCode { get; set; }
        }
    }
}