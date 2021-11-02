using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace Samples.Dynamic
{
    public static class MapValueToKeyMultiColumn
    {
        /// This example demonstrates the use of the ValueToKeyMappingEstimator, by
        /// mapping strings to KeyType values. For more on ML.NET KeyTypes see:
        /// https://github.com/dotnet/machinelearning/blob/main/docs/code/IDataViewTypeSystem.md#key-types
        /// It is possible to have multiple values map to the same category.
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            var rawData = new[] {
                new DataPoint() { StudyTime = "0-4yrs" , Course = "CS" },
                new DataPoint() { StudyTime = "6-11yrs" , Course = "CS" },
                new DataPoint() { StudyTime = "12-25yrs" , Course = "LA" },
                new DataPoint() { StudyTime = "0-5yrs" , Course = "DS" }
            };

            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // Constructs the ML.net pipeline
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(new[] {
                new  InputOutputColumnPair("StudyTimeCategory", "StudyTime"),
                new  InputOutputColumnPair("CourseCategory", "Course")
                },
                keyOrdinality: Microsoft.ML.Transforms.ValueToKeyMappingEstimator
                    .KeyOrdinality.ByValue, addKeyValueAnnotationsAsText: true);

            // Fits the pipeline to the data.
            IDataView transformedData = pipeline.Fit(data).Transform(data);

            // Getting the resulting data as an IEnumerable.
            // This will contain the newly created columns.
            IEnumerable<TransformedData> features = mlContext.Data.CreateEnumerable<
                TransformedData>(transformedData, reuseRowObject: false);

            Console.WriteLine($" StudyTime   StudyTimeCategory   Course    " +
                $"CourseCategory");

            foreach (var featureRow in features)
                Console.WriteLine($"{featureRow.StudyTime}\t\t" +
                    $"{featureRow.StudyTimeCategory}\t\t\t{featureRow.Course}\t\t" +
                    $"{featureRow.CourseCategory}");

            // TransformedData obtained post-transformation.
            //
            // StudyTime     StudyTimeCategory   Course    CourseCategory
            // 0-4yrs          1                   CS          1
            // 6-11yrs         4                   CS          1
            // 12-25yrs        3                   LA          3
            // 0-5yrs          2                   DS          2

            // If we wanted to provide the mapping, rather than letting the
            // transform create it, we could do so by creating an IDataView one
            // column containing the values to map to. If the values in the dataset
            // are not found in the lookup IDataView they will get mapped to the
            // missing value, 0. The keyData are shared among the columns, therefore
            // the keys are not contiguous for the column. Create the lookup map
            // data IEnumerable.
            var lookupData = new[] {
                new LookupMap { Key = "0-4yrs" },
                new LookupMap { Key = "6-11yrs" },
                new LookupMap { Key = "25+yrs"  },
                new LookupMap { Key = "CS" },
                new LookupMap { Key = "DS" },
                new LookupMap { Key = "LA"  }
            };

            // Convert to IDataView
            var lookupIdvMap = mlContext.Data.LoadFromEnumerable(lookupData);

            // Constructs the ML.net pipeline
            var pipelineWithLookupMap = mlContext.Transforms.Conversion
                .MapValueToKey(new[] {
                    new  InputOutputColumnPair("StudyTimeCategory", "StudyTime"),
                    new  InputOutputColumnPair("CourseCategory", "Course")
                    },
                    keyData: lookupIdvMap);

            // Fits the pipeline to the data.
            transformedData = pipelineWithLookupMap.Fit(data).Transform(data);

            // Getting the resulting data as an IEnumerable.
            // This will contain the newly created columns.
            features = mlContext.Data.CreateEnumerable<TransformedData>(
                transformedData, reuseRowObject: false);

            Console.WriteLine($" StudyTime   StudyTimeCategory  " +
                $"Course CourseCategory");

            foreach (var featureRow in features)
                Console.WriteLine($"{featureRow.StudyTime}\t\t" +
                    $"{featureRow.StudyTimeCategory}\t\t\t{featureRow.Course}\t\t" +
                    $"{featureRow.CourseCategory}");

            // StudyTime    StudyTimeCategory  Course     CourseCategory
            // 0 - 4yrs          1              CS              4
            // 6 - 11yrs         2              CS              4
            // 12 - 25yrs        0              LA              6
            // 0 - 5yrs          0              DS              5

        }

        private class DataPoint
        {
            public string StudyTime { get; set; }
            public string Course { get; set; }
        }

        private class TransformedData : DataPoint
        {
            public uint StudyTimeCategory { get; set; }
            public uint CourseCategory { get; set; }
        }

        // Type for the IDataView that will be serving as the map
        private class LookupMap
        {
            public string Key { get; set; }
        }
    }
}
