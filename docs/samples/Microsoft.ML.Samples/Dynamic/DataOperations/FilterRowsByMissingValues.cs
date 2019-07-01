using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.SamplesUtils;

namespace Samples.Dynamic
{
    public class FilterRowsByMissingValues
    {
        /// <summary>
        /// Sample class showing how to use FilterRowsByMissingValues.
        /// </summary>
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Create a small dataset as an IEnumerable.
            var samples = new List<DataPoint>()
            {
                new DataPoint(){ Feature1 = 21, Feature2 = new [] { 1, 2, float.NaN}
                    },

                new DataPoint(){ Feature1 = 40, Feature2 = new [] { 1f, 2f, 3f}  },
                new DataPoint(){ Feature1 = float.NaN, Feature2 = new [] { 1, 2,
                    float.NaN}  }

            };

            // Convert training data to IDataView.
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // Filter out any row with an NaN values in either column
            var filteredData = mlContext.Data
                .FilterRowsByMissingValues(data, new[] { "Feature1", "Feature2" });

            // Take a look at the resulting dataset and note that rows with NaNs are
            // filtered out. Only the second data point is left
            var enumerable = mlContext.Data
                .CreateEnumerable<DataPoint>(filteredData, reuseRowObject: true);

            Console.WriteLine($"Feature1   Feature2");

            foreach (var row in enumerable)
            {
                Console.WriteLine($"{row.Feature1}" +
                    $"\t({string.Join(", ", row.Feature2)})");
            }

            // Feature1     Feature2
            // 
            //   40         (1, 2, 3)

        }

        private class DataPoint
        {
            public float Feature1 { get; set; }

            public float[] Feature2 { get; set; }
        }
    }
}
