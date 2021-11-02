using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace Samples.Dynamic
{
    public static class FilterRowsByColumn
    {
        // // Sample class showing how to filter out some rows in
        // IDataView.
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available
            // operations and as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            var enumerableOfData = GetSampleTemperatureData(10);
            var data = mlContext.Data.LoadFromEnumerable(enumerableOfData);

            // Before we apply a filter, examine all the records in the dataset.
            Console.WriteLine($"Date\tTemperature");
            foreach (var row in enumerableOfData)
            {
                Console.WriteLine(
            $"{row.Date.ToString("d")}\t{row.Temperature}");

            }
            Console.WriteLine();
            // Expected output:
            //  Date    Temperature
            //  1/2/2012        36
            //  1/3/2012        36
            //  1/4/2012        34
            //  1/5/2012        35
            //  1/6/2012        35
            //  1/7/2012        39
            //  1/8/2012        40
            //  1/9/2012        35
            //  1/10/2012       30
            //  1/11/2012       29

            // Filter the data by the values of the temperature. The lower bound is
            // inclusive, the upper exclusive.
            var filteredData = mlContext.Data
                .FilterRowsByColumn(data, columnName: "Temperature",
                lowerBound: 34, upperBound: 37);

            // Look at the filtered data and observe that values outside [34,37)
            // have been dropped.
            var enumerable = mlContext.Data
                .CreateEnumerable<SampleTemperatureData>(filteredData,
                reuseRowObject: true);

            Console.WriteLine($"Date\tTemperature");
            foreach (var row in enumerable)
            {
                Console.WriteLine(
            $"{row.Date.ToString("d")}\t{row.Temperature}");

            }

            // Expected output:
            //  Date    Temperature
            //  1/2/2012        36
            //  1/3/2012        36
            //  1/4/2012        34
            //  1/5/2012        35
            //  1/6/2012        35
            //  1/9/2012        35
        }

        private class SampleTemperatureData
        {
            public DateTime Date { get; set; }
            public float Temperature { get; set; }
        }

        /// <summary>
        /// Get a fake temperature dataset.
        /// </summary>
        /// <param name="exampleCount">The number of examples to return.</param>
        /// <returns>An enumerable of <see cref="SampleTemperatureData"/>.</returns>
        private static IEnumerable<SampleTemperatureData> GetSampleTemperatureData(
            int exampleCount)

        {
            var rng = new Random(1234321);
            var date = new DateTime(2012, 1, 1);
            float temperature = 39.0f;

            for (int i = 0; i < exampleCount; i++)
            {
                date = date.AddDays(1);
                temperature += rng.Next(-5, 5);
                yield return new SampleTemperatureData
                {
                    Date = date,
                    Temperature =
                    temperature
                };

            }
        }
    }
}

