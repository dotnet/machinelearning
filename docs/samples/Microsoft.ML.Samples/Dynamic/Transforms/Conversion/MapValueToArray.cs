using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace Samples.Dynamic
{
    public static class MapValueToArray
    {
        /// This example demonstrates the use of MapValue by mapping strings to
        /// array values, which allows for mapping data to numeric arrays. This
        /// functionality is useful when the generated column will serve as the
        /// Features column for a trainer. Most of the trainers take a numeric
        /// vector, as the Features column. In this example, we are mapping the
        /// Timeframe data to arbitrary integer arrays.
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            var rawData = new[] {
                new DataPoint() { Timeframe = "0-4yrs" },
                new DataPoint() { Timeframe = "6-11yrs" },
                new DataPoint() { Timeframe = "12-25yrs" },
                new DataPoint() { Timeframe = "0-5yrs" },
                new DataPoint() { Timeframe = "12-25yrs" },
                new DataPoint() { Timeframe = "25+yrs" },
            };

            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // Creating a list of key-value pairs to indicate the mapping between
            // the DataPoint values, and the arrays they should map to. 
            var timeframeMap = new Dictionary<string, int[]>();
            timeframeMap["0-4yrs"] = new int[] { 0, 5, 300 };
            timeframeMap["0-5yrs"] = new int[] { 0, 5, 300 };
            timeframeMap["6-11yrs"] = new int[] { 6, 11, 300 };
            timeframeMap["12-25yrs"] = new int[] { 12, 50, 300 };
            timeframeMap["25+yrs"] = new int[] { 12, 50, 300 };

            // Constructs the ValueMappingEstimator making the ML.NET pipeline.
            var pipeline = mlContext.Transforms.Conversion.MapValue("Features",
                timeframeMap, "Timeframe");

            // Fits the ValueMappingEstimator and transforms the data adding the
            // Features column.
            IDataView transformedData = pipeline.Fit(data).Transform(data);

            // Getting the resulting data as an IEnumerable.
            IEnumerable<TransformedData> featuresColumn = mlContext.Data
                .CreateEnumerable<TransformedData>(transformedData, reuseRowObject:
                false);

            Console.WriteLine($"Timeframe     Features");
            foreach (var featureRow in featuresColumn)
                Console.WriteLine($"{featureRow.Timeframe}\t\t " +
                $"{string.Join(",", featureRow.Features)}");

            // Timeframe      Features
            // 0-4yrs       0, 5, 300
            // 6-11yrs      6, 11, 300
            // 12-25yrs     12, 50, 300
            // 0-5yrs       0, 5, 300
            // 12-25yrs     12, 50,300
            // 25+yrs       12, 50, 300
        }

        public class DataPoint
        {
            public string Timeframe { get; set; }
        }

        public class TransformedData : DataPoint
        {
            public int[] Features { get; set; }
        }
    }
}
