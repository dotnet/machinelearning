using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace Samples.Dynamic
{
    public static class MapValue
    {
        /// This example demonstrates the use of the ValueMappingEstimator by 
        /// mapping strings to other string values, or floats to strings. This is
        /// useful to map types to a category. 
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            var rawData = new[] {
                new DataPoint() { Timeframe = "0-4yrs" , Score = 1 },
                new DataPoint() { Timeframe = "6-11yrs" , Score = 2 },
                new DataPoint() { Timeframe = "12-25yrs" , Score = 3 },
                new DataPoint() { Timeframe = "0-5yrs" , Score = 4 },
                new DataPoint() { Timeframe = "12-25yrs" , Score = 5 },
                new DataPoint() { Timeframe = "25+yrs" , Score = 5 },
            };

            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // Construct the mapping to other strings for the Timeframe column.  
            var timeframeMap = new Dictionary<string, string>();
            timeframeMap["0-4yrs"] = "Short";
            timeframeMap["0-5yrs"] = "Short";
            timeframeMap["6-11yrs"] = "Medium";
            timeframeMap["12-25yrs"] = "Long";
            timeframeMap["25+yrs"] = "Long";

            // Construct the mapping of strings to keys(uints) for the Timeframe
            // column. 
            var timeframeKeyMap = new Dictionary<string, uint>();
            timeframeKeyMap["0-4yrs"] = 1;
            timeframeKeyMap["0-5yrs"] = 1;
            timeframeKeyMap["6-11yrs"] = 2;
            timeframeKeyMap["12-25yrs"] = 3;
            timeframeKeyMap["25+yrs"] = 3;

            // Construct the mapping of ints to strings for the Score column. 
            var scoreMap = new Dictionary<int, string>();
            scoreMap[1] = "Low";
            scoreMap[2] = "Low";
            scoreMap[3] = "Average";
            scoreMap[4] = "High";
            scoreMap[5] = "High";

            // Constructs the ML.net pipeline
            var pipeline = mlContext.Transforms.Conversion.MapValue(
                "TimeframeCategory", timeframeMap, "Timeframe").Append(mlContext.
                Transforms.Conversion.MapValue("ScoreCategory", scoreMap, "Score"))
                // on the MapValue below, the treatValuesAsKeyType is set to true.
                // The type of the Label column will be a KeyDataViewType type, 
                // and it can be used as input for trainers performing multiclass
                // classification.
                .Append(mlContext.Transforms.Conversion.MapValue("Label",
                timeframeKeyMap, "Timeframe", treatValuesAsKeyType: true));

            // Fits the pipeline to the data.
            IDataView transformedData = pipeline.Fit(data).Transform(data);

            // Getting the resulting data as an IEnumerable.
            // This will contain the newly created columns.
            IEnumerable<TransformedData> features = mlContext.Data.CreateEnumerable<
                TransformedData>(transformedData, reuseRowObject: false);

            Console.WriteLine(" Timeframe   TimeframeCategory   Label    Score   " +
                "ScoreCategory");

            foreach (var featureRow in features)
                Console.WriteLine($"{featureRow.Timeframe}\t\t" +
                    $"{featureRow.TimeframeCategory}\t\t\t{featureRow.Label}\t\t" +
                    $"{featureRow.Score}\t{featureRow.ScoreCategory}");

            // TransformedData obtained post-transformation.
            //
            //  Timeframe   TimeframeCategory   Label    Score   ScoreCategory
            // 0-4yrs         Short              1       1       Low
            // 6-11yrs        Medium             2       2       Low
            // 12-25yrs       Long               3       3       Average
            // 0-5yrs         Short              1       4       High
            // 12-25yrs       Long               3       5       High
            // 25+yrs         Long               3       5       High
        }

        private class DataPoint
        {
            public string Timeframe { get; set; }
            public int Score { get; set; }
        }

        private class TransformedData : DataPoint
        {
            public string TimeframeCategory { get; set; }
            public string ScoreCategory { get; set; }
            public uint Label { get; set; }
        }
    }
}
