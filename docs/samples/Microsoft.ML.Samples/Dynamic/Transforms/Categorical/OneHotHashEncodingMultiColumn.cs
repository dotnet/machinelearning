using System;
using Microsoft.ML;

namespace Samples.Dynamic.Transforms.Categorical
{
    public static class OneHotHashEncodingMultiColumn
    {
        public static void Example()
        {
            // Create a new ML context for ML.NET operations. It can be used for
            // exception tracking and logging as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            var samples = new[]
            {
                new DataPoint {Education = "0-5yrs", ZipCode = "98005"},
                new DataPoint {Education = "0-5yrs", ZipCode = "98052"},
                new DataPoint {Education = "6-11yrs", ZipCode = "98005"},
                new DataPoint {Education = "6-11yrs", ZipCode = "98052"},
                new DataPoint {Education = "11-15yrs", ZipCode = "98005"}
            };

            // Convert training data to IDataView.
            IDataView data = mlContext.Data.LoadFromEnumerable(samples);

            // Multi column example: A pipeline for one hot has encoding two
            // columns 'Education' and 'ZipCode'.
            var multiColumnKeyPipeline =
                mlContext.Transforms.Categorical.OneHotHashEncoding(
                    new[]
                    {
                        new InputOutputColumnPair("Education"),
                        new InputOutputColumnPair("ZipCode")
                    },
                    numberOfBits: 3);

            // Fit and Transform the data.
            IDataView transformedData =
                multiColumnKeyPipeline.Fit(data).Transform(data);

            var convertedData =
                mlContext.Data.CreateEnumerable<TransformedData>(transformedData,
                    true);

            Console.WriteLine(
                "One Hot Hash Encoding of two columns 'Education' and 'ZipCode'.");

            // One Hot Hash Encoding of two columns 'Education' and 'ZipCode'.

            foreach (TransformedData item in convertedData)
                Console.WriteLine("{0}\t\t\t{1}", string.Join(" ", item.Education),
                    string.Join(" ", item.ZipCode));

            // We have 8 slots, because we used numberOfBits = 3.

            // 0 0 0 1 0 0 0 0                 0 0 0 0 0 0 0 1
            // 0 0 0 1 0 0 0 0                 1 0 0 0 0 0 0 0
            // 0 0 0 0 1 0 0 0                 0 0 0 0 0 0 0 1
            // 0 0 0 0 1 0 0 0                 1 0 0 0 0 0 0 0
            // 0 0 0 0 0 0 0 1                 0 0 0 0 0 0 0 1
        }

        private class DataPoint
        {
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
