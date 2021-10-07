using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    class MapKeyToVector
    {
        /// This example demonstrates the use of MapKeyToVector by mapping keys to
        /// floats[]. Because the ML.NET KeyType maps the missing value to zero,
        /// counting starts at 1, so the uint values converted to KeyTypes will
        /// appear skewed by one. See https://github.com/dotnet/machinelearning/blob/main/docs/code/IDataViewTypeSystem.md#key-types
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            var rawData = new[] {
                new DataPoint() { Timeframe = 8, PartA=1, PartB=2},
                new DataPoint() { Timeframe = 7, PartA=2, PartB=1},
                new DataPoint() { Timeframe = 8, PartA=3, PartB=2},
                new DataPoint() { Timeframe = 3, PartA=3, PartB=3}
            };

            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // First transform just maps key type to indicator vector. i.e. it's
            // produces vector filled with zeros with size of key cardinality and
            // set 1 to corresponding key's value index in that array. After that we
            // concatenate two columns with single int values into vector of ints.
            // Third transform will create vector of keys, where key type is shared
            // across whole vector. Forth transform output data as count vector and
            // that vector would have size equal to shared key type cardinality and
            // put key counts to corresponding indexes in array. Fifth transform
            // output indicator vector for each key and concatenate them together.
            // Result vector would be size of key cardinality multiplied by size of
            // original vector.
            var pipeline = mlContext.Transforms.Conversion.MapKeyToVector(
                "TimeframeVector", "Timeframe")
                .Append(mlContext.Transforms.Concatenate("Parts", "PartA", "PartB"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Parts"))
                .Append(mlContext.Transforms.Conversion.MapKeyToVector(
                    "PartsCount", "Parts", outputCountVector: true))
                .Append(mlContext.Transforms.Conversion.MapKeyToVector(
                    "PartsNoCount", "Parts"));

            // Fits the pipeline to the data.
            IDataView transformedData = pipeline.Fit(data).Transform(data);

            // Getting the resulting data as an IEnumerable.
            // This will contain the newly created columns.
            IEnumerable<TransformedData> features = mlContext.Data.CreateEnumerable<
                TransformedData>(transformedData, reuseRowObject: false);

            Console.WriteLine("Timeframe  TimeframeVector    PartsCount  " +
                "PartsNoCount");

            foreach (var featureRow in features)
                Console.WriteLine(featureRow.Timeframe + "          " +
                    string.Join(',', featureRow.TimeframeVector.Select(x => x)) + "  "
                    + string.Join(',', featureRow.PartsCount.Select(x => x)) +
                    "       " + string.Join(',', featureRow.PartsNoCount.Select(
                    x => x)));

            // Expected output:
            //  Timeframe  TimeframeVector    PartsCount  PartsNoCount
            //  9          0,0,0,0,0,0,0,0,1  1,1,0       1,0,0,0,1,0
            //  8          0,0,0,0,0,0,0,1,0  1,1,0       0,1,0,1,0,0
            //  9          0,0,0,0,0,0,0,0,1  0,1,1       0,0,1,0,1,0
            //  4          0,0,0,1,0,0,0,0,0  0,0,2       0,0,1,0,0,1
        }

        private class DataPoint
        {
            [KeyType(9)]
            public uint Timeframe { get; set; }
            public int PartA { get; set; }
            public int PartB { get; set; }

        }

        private class TransformedData : DataPoint
        {
            public float[] TimeframeVector { get; set; }
            public float[] PartsCount { get; set; }
            public float[] PartsNoCount { get; set; }
        }
    }
}
