using System;
using System.Collections.Generic;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class MapValueToKeyInputOutputPair
    {
        /// This example demonstrates the use of the ValueMappingEstimator by mapping strings to other string values, or floats to strings. 
        /// This is useful to map types to a grouping. 
        /// It is possible to have multiple values map to the same category.
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            var rawData = new[] {
                new DataPoint() { StudyTime = "0-4yrs" , DevelopmentTime = "6-11yrs" },
                new DataPoint() { StudyTime = "6-11yrs" , DevelopmentTime = "6-11yrs" },
                new DataPoint() { StudyTime = "12-25yrs" , DevelopmentTime = "25+yrs" },
                new DataPoint() { StudyTime = "0-5yrs" , DevelopmentTime = "0-5yrs" }
            };

            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // Constructs the ML.net pipeline
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(new[] {
                new  InputOutputColumnPair("StudyTimeCategory", "StudyTime"),
                new  InputOutputColumnPair("DevelopmentTimeCategory", "DevelopmentTime")
                },
                keyOrdinality: Transforms.ValueToKeyMappingEstimator.KeyOrdinality.ByValue,
                addKeyValueAnnotationsAsText: true);

            // Fits the pipeline to the data.
            IDataView transformedData = pipeline.Fit(data).Transform(data);

            // Getting the resulting data as an IEnumerable.
            // This will contain the newly created columns.
            IEnumerable<TransformedData> features = mlContext.Data.CreateEnumerable<TransformedData>(transformedData, reuseRowObject: false);

            Console.WriteLine($" StudyTime   StudyTimeCategory   DevelopmentTime    DevelopmentTimeCategory");
            foreach (var featureRow in features)
                Console.WriteLine($"{featureRow.StudyTime}\t\t{featureRow.StudyTimeCategory}\t\t\t{featureRow.DevelopmentTime}\t\t{featureRow.DevelopmentTimeCategory}");

            // TransformedData obtained post-transformation.
            //
            //  StudyTime   StudyTimeCategory   DevelopmentTime    DevelopmentTimeCategory
            // 0-4yrs          1                6-11yrs                3
            // 6-11yrs         4                6-11yrs                3
            // 12-25yrs        3                25+yrs                 2
            // 0-5yrs          2                0-5yrs                 1


            // If we wanted to provide the mapping, rather than letting the transform create it, 
            // we could do so by creating an IDataView one column containing the values to map to. 
            // If the values in the dataset are not found in the lookup IDataView they will get mapped to the mising value, 0.
            // Create the lookup map data IEnumerable.  
            var lookupData = new[] {
                new LookupMap { Key = "0-4yrs" },
                new LookupMap { Key = "6-11yrs" },
                new LookupMap { Key = "25+yrs"  }

            };

            // Convert to IDataView
            var lookupIdvMap = mlContext.Data.LoadFromEnumerable(lookupData);

            // Constructs the ML.net pipeline
            var pipelineWithLookupMap = mlContext.Transforms.Conversion.MapValueToKey(new[] {
                new  InputOutputColumnPair("StudyTimeCategory", "StudyTime"),
                new  InputOutputColumnPair("DevelopmentTimeCategory", "DevelopmentTime")
                },
                keyData: lookupIdvMap);

            // Fits the pipeline to the data.
            transformedData = pipelineWithLookupMap.Fit(data).Transform(data);

            // Getting the resulting data as an IEnumerable.
            // This will contain the newly created columns.
            features = mlContext.Data.CreateEnumerable<TransformedData>(transformedData, reuseRowObject: false);

            Console.WriteLine($" StudyTime   StudyTimeCategory   DevelopmentTime    DevelopmentTimeCategory");
            foreach (var featureRow in features)
                Console.WriteLine($"{featureRow.StudyTime}\t\t{featureRow.StudyTimeCategory}\t\t\t{featureRow.DevelopmentTime}\t\t{featureRow.DevelopmentTimeCategory}");

            // StudyTime    StudyTimeCategory   DevelopmentTime     DevelopmentTimeCategory
            // 0 - 4yrs          1                  6 - 11yrs         2
            // 6 - 11yrs         2                  6 - 11yrs         2
            // 12 - 25yrs        0                  25 + yrs          3
            // 0 - 5yrs          0                  0 - 5yrs          0

        }

        private class DataPoint
        {
            public string StudyTime { get; set; }
            public string DevelopmentTime { get; set; }
        }
        private class TransformedData : DataPoint
        {
            public uint StudyTimeCategory { get; set; }
            public uint DevelopmentTimeCategory { get; set; }
        }
        // Type for the IDataView that will be serving as the map
        private class LookupMap
        {
            public string Key { get; set; }
        }
    }
}
