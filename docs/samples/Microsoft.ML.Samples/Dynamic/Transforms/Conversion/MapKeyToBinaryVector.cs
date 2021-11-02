using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    class MapKeyToBinaryVector
    {
        /// This example demonstrates the use of MapKeyToVector by mapping keys to
        /// floats[] of 0 and 1, representing the number in binary format.
        /// Because the ML.NET KeyType maps the missing value to zero, counting
        /// starts at 1, so the uint values converted to KeyTypes will appear
        /// skewed by one.
        /// See https://github.com/dotnet/machinelearning/blob/main/docs/code/IDataViewTypeSystem.md#key-types
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            var rawData = new[] {
                new DataPoint() { Timeframe = 9 },
                new DataPoint() { Timeframe = 8 },
                new DataPoint() { Timeframe = 8 },
                new DataPoint() { Timeframe = 9 },
                new DataPoint() { Timeframe = 2 },
                new DataPoint() { Timeframe = 3 }
            };

            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // Constructs the ML.net pipeline
            var pipeline = mlContext.Transforms.Conversion.MapKeyToBinaryVector(
                "TimeframeVector", "Timeframe");

            // Fits the pipeline to the data.
            IDataView transformedData = pipeline.Fit(data).Transform(data);

            // Getting the resulting data as an IEnumerable.
            // This will contain the newly created columns.
            IEnumerable<TransformedData> features = mlContext.Data.CreateEnumerable<
                TransformedData>(transformedData, reuseRowObject: false);

            Console.WriteLine($" Timeframe           TimeframeVector");
            foreach (var featureRow in features)
                Console.WriteLine($"{featureRow.Timeframe}\t\t\t" +
                    $"{string.Join(',', featureRow.TimeframeVector)}");

            // Timeframe             TimeframeVector
            // 10                      0,1,0,0,1 //binary representation of 9, the original value
            // 9                       0,1,0,0,0 //binary representation of 8, the original value
            // 9                       0,1,0,0,0
            // 10                      0,1,0,0,1
            // 3                       0,0,0,1,0
            // 4                       0,0,0,1,1
        }

        private class DataPoint
        {
            [KeyType(10)]
            public uint Timeframe { get; set; }

        }

        private class TransformedData : DataPoint
        {
            public float[] TimeframeVector { get; set; }
        }
    }
}
