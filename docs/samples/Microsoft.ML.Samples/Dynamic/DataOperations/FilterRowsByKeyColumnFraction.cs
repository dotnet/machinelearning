using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace Samples.Dynamic
{
    /// <summary>
    /// Sample class showing how to use FilterRowsByKeyColumnFraction.
    /// </summary>
    public static class FilterRowsByKeyColumnFraction
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Create a small dataset as an IEnumerable.
            var samples = new List<DataPoint>()
            {
                new DataPoint(){ Age = 21 },
                new DataPoint(){ Age = 40 },
                new DataPoint(){ Age = 38 },
                new DataPoint(){ Age = 22 },
                new DataPoint(){ Age = 40 },
                new DataPoint(){ Age = 40 },
                new DataPoint(){ Age = 22 },
                new DataPoint(){ Age = 21 }
            };

            // Convert training data to IDataView.
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // Convert the uint values to KeyDataViewData
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Age");
            var transformedData = pipeline.Fit(data).Transform(data);

            // Before we apply a filter, examine all the records in the dataset.
            var enumerable = mlContext.Data
                .CreateEnumerable<DataPoint>(transformedData, reuseRowObject: true);

            Console.WriteLine($"Age");
            foreach (var row in enumerable)
            {
                Console.WriteLine($"{row.Age}");
            }

            // Age
            // 
            //  1
            //  2
            //  3
            //  4
            //  2
            //  2
            //  4
            //  1

            // Now filter down to half the keys, choosing the lower half of values. 
            // For the keys we have the sorted values: 1 1 2 2 2 3 4 4.
            // Projected in the [0, 1[ interval as per: (key - 0.5)/(Count of Keys)
            // the values of the keys for our data would be:
            // 0.125 0.125 0.375 0.375 0.375 0.625 0.875 0.875
            // so the keys resulting from filtering in the [0, 0.5 [ interval are
            // the ones with normalized values 0.125 and 0.375, respectively keys 
            // with values 1 and 2.
            var filteredHalfData = mlContext.Data
                .FilterRowsByKeyColumnFraction(transformedData, columnName: "Age",
                lowerBound: 0, upperBound: 0.5);

            var filteredHalfEnumerable = mlContext.Data
                .CreateEnumerable<DataPoint>(filteredHalfData,
                reuseRowObject: true);

            Console.WriteLine($"Age");
            foreach (var row in filteredHalfEnumerable)
            {
                Console.WriteLine($"{row.Age}");
            }

            // Age
            // 
            //  1
            //  2
            //  2
            //  2
            //  1

            // As mentioned above, the normalized keys are:
            // 0.125 0.125 0.375 0.375 0.375 0.625 0.875 0.875
            // so the keys resulting from filtering in the [0.3, 0.6 [ interval are
            // the ones with normalized value 0.375, respectively key with
            // value = 2.
            var filteredMiddleData = mlContext.Data
                .FilterRowsByKeyColumnFraction(transformedData, columnName: "Age",
                lowerBound: 0.3, upperBound: 0.6);

            // Look at the data and observe that values above 2 have been filtered
            // out
            var filteredMiddleEnumerable = mlContext.Data
                .CreateEnumerable<DataPoint>(filteredMiddleData,
                reuseRowObject: true);

            Console.WriteLine($"Age");
            foreach (var row in filteredMiddleEnumerable)
            {
                Console.WriteLine($"{row.Age}");
            }

            // Age
            //
            //  2
            //  2
            //  2
        }

        private class DataPoint
        {
            public uint Age { get; set; }
        }
    }
}
