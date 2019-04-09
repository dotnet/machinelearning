using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    class MapKeyToVector
    {
        /// This example demonstrates the use of MapKeyToVector by mapping keys to floats[]. 
        /// Because the ML.NET KeyType maps the missing value to zero, counting starts at 1, so the uint values
        /// converted to KeyTypes will appear skewed by one. 
        /// See https://github.com/dotnet/machinelearning/blob/master/docs/code/IDataViewTypeSystem.md#key-types
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            var rawData = new[] {
                new DataPoint() { Timeframe = 9, Category = 5 },
                new DataPoint() { Timeframe = 8, Category = 4 },
                new DataPoint() { Timeframe = 8, Category = 4 },
                new DataPoint() { Timeframe = 9, Category = 3 },
                new DataPoint() { Timeframe = 2, Category = 3 },
                new DataPoint() { Timeframe = 3, Category = 5 }
            };

            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // Constructs the ML.net pipeline
            var pipeline = mlContext.Transforms.Conversion.MapKeyToVector("TimeframeVector", "Timeframe")
                           .Append(mlContext.Transforms.Conversion.MapKeyToVector("CategoryVector", "Category", outputCountVector: true));

            // Fits the pipeline to the data.
            IDataView transformedData = pipeline.Fit(data).Transform(data);

            // Getting the resulting data as an IEnumerable.
            // This will contain the newly created columns.
            IEnumerable<TransformedData> features = mlContext.Data.CreateEnumerable<TransformedData>(transformedData, reuseRowObject: false);

            Console.WriteLine($" Timeframe           TimeframeVector         Category    CategoryVector");
            foreach (var featureRow in features)
                Console.WriteLine($"{featureRow.Timeframe}\t\t\t{string.Join(',', featureRow.TimeframeVector)}\t\t\t{featureRow.Category}\t\t{string.Join(',', featureRow.CategoryVector)}");

            // TransformedData obtained post-transformation.
            //
            // Timeframe          TimeframeVector    Category    CategoryVector
            //  10              0,0,0,0,0,0,0,0,0,1       6          0,0,0,0,0
            //  9               0,0,0,0,0,0,0,0,1,0       5          0,0,0,0,1
            //  9               0,0,0,0,0,0,0,0,1,0       5          0,0,0,0,1
            //  10              0,0,0,0,0,0,0,0,0,1       4          0,0,0,1,0
            //  3               0,0,1,0,0,0,0,0,0,0       4          0,0,0,1,0
            //  4               0,0,0,1,0,0,0,0,0,0       6          0,0,0,0,0
         }

        private class DataPoint
        {
            [KeyType(10)]
            public uint Timeframe { get; set; }

            [KeyType(6)]
            public uint Category { get; set; }

        }

        private class TransformedData : DataPoint
        {
            public uint[] TimeframeVector { get; set; }
            public uint[] CategoryVector { get; set; }
        }
    }
}
