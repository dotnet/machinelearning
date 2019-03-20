using System;
using Microsoft.ML.SamplesUtils;

namespace Microsoft.ML.Samples.Dynamic
{
    /// <summary>
    /// Sample class showing how to use ShuffleRows.
    /// </summary>
    public static class ShuffleRows
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            var enumerableOfData = DatasetUtils.GetSampleTemperatureData(5);
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

            // Shuffle the dataset.
            var shuffledData = mlContext.Data.ShuffleRows(data, seed: 123);

            // Look at the shuffled data and observe that the rows are in a randomized order.
            var enumerable = mlContext.Data.CreateEnumerable<DatasetUtils.SampleTemperatureData>(shuffledData, reuseRowObject: true);
            Console.WriteLine($"Date\tTemperature");
            foreach (var row in enumerable)
            {
                Console.WriteLine($"{row.Date.ToString("d")}\t{row.Temperature}");
            }
            // Expected output:
            //  Date    Temperature
            //  1/4/2012        34
            //  1/2/2012        36
            //  1/5/2012        35
            //  1/3/2012        36
            //  1/6/2012        35
        }
    }
}
