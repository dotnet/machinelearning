using System;
using System.Collections.Generic;

namespace Microsoft.ML.Samples.Dynamic
{
    /// <summary>
    /// Sample class showing how to use FilterRowsByColumn.
    /// </summary>
    public static class FilterRowsByColumn
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            IEnumerable<SamplesUtils.DatasetUtils.SampleTemperatureData> enumerableOfData = SamplesUtils.DatasetUtils.GetSampleTemperatureData(10);
            var data = mlContext.Data.LoadFromEnumerable(enumerableOfData);

            // Before we apply a filter, examine all the records in the dataset.
            Console.WriteLine($"Date\tTemperature");
            foreach (var row in enumerableOfData)
            {
                Console.WriteLine($"{row.Date.ToString("d")}\t{row.Temperature}");
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

            // Filter the data by the values of the temperature. The lower bound is inclusive, the upper exclusive.
            var filteredData = mlContext.Data.FilterRowsByColumn(data, columnName: "Temperature", lowerBound: 34, upperBound: 37);

            // Look at the filtered data and observe that values outside [34,37) have been dropped.
            var enumerable = mlContext.Data.CreateEnumerable<SamplesUtils.DatasetUtils.SampleTemperatureData>(filteredData, reuseRowObject: true);
            Console.WriteLine($"Date\tTemperature");
            foreach (var row in enumerable)
            {
                Console.WriteLine($"{row.Date.ToString("d")}\t{row.Temperature}");
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
    }
}
