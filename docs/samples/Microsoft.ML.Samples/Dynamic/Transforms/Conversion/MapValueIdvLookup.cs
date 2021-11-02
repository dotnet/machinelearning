using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace Samples.Dynamic
{
    public static class MapValueIdvLookup
    {
        /// This example demonstrates the use of MapValue by mapping floats to
        /// strings, looking up the mapping in an IDataView. This is useful to map
        /// types to a grouping. 
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            var rawData = new[] {
                new DataPoint() { Price = 3.14f },
                new DataPoint() { Price = 2000f },
                new DataPoint() { Price = 1.19f },
                new DataPoint() { Price = 2.17f },
                new DataPoint() { Price = 33.784f },

            };

            // Convert to IDataView
            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // Create the lookup map data IEnumerable.   
            var lookupData = new[] {
                new LookupMap { Value = 3.14f, Category = "Low" },
                new LookupMap { Value = 1.19f , Category = "Low" },
                new LookupMap { Value = 2.17f , Category = "Low" },
                new LookupMap { Value = 33.784f, Category = "Medium" },
                new LookupMap { Value = 2000f, Category = "High"}

            };

            // Convert to IDataView
            var lookupIdvMap = mlContext.Data.LoadFromEnumerable(lookupData);

            // Constructs the ValueMappingEstimator making the ML.NET pipeline
            var pipeline = mlContext.Transforms.Conversion.MapValue("PriceCategory",
                lookupIdvMap, lookupIdvMap.Schema["Value"], lookupIdvMap.Schema[
                    "Category"], "Price");

            // Fits the ValueMappingEstimator and transforms the data converting the
            // Price to PriceCategory.
            IDataView transformedData = pipeline.Fit(data).Transform(data);

            // Getting the resulting data as an IEnumerable.
            IEnumerable<TransformedData> features = mlContext.Data.CreateEnumerable<
                TransformedData>(transformedData, reuseRowObject: false);

            Console.WriteLine($" Price   PriceCategory");
            foreach (var featureRow in features)
                Console.WriteLine($"{featureRow.Price}\t\t" +
                $"{featureRow.PriceCategory}");

            // TransformedData obtained post-transformation.
            //
            // Price        PriceCategory
            // 3.14            Low
            // 2000            High
            // 1.19            Low
            // 2.17            Low
            // 33.784          Medium
        }

        // Type for the IDataView that will be serving as the map
        private class LookupMap
        {
            public float Value { get; set; }
            public string Category { get; set; }
        }

        private class DataPoint
        {
            public float Price { get; set; }
        }

        private class TransformedData : DataPoint
        {
            public string PriceCategory { get; set; }
        }
    }
}
