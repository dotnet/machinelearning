using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace Samples.Dynamic
{
    public static class DataViewEnumerable
    {
        // A simple case of creating IDataView from
        //IEnumerable.
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging,
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            IEnumerable<SampleTemperatureData> enumerableOfData =
                GetSampleTemperatureData(5);

            // Load dataset into an IDataView. 
            IDataView data = mlContext.Data.LoadFromEnumerable(enumerableOfData);

            // We can now examine the records in the IDataView. We first create an
            // enumerable of rows in the IDataView.
            var rowEnumerable = mlContext.Data
                .CreateEnumerable<SampleTemperatureData>(data,
                reuseRowObject: true);

            // SampleTemperatureDataWithLatitude has the definition of a Latitude
            // column of type float. We can use the parameter ignoreMissingColumns
            // to true to ignore any missing columns in the IDataView. The produced
            // enumerable will have the Latitude field set to the default for the
            // data type, in this case 0. 
            var rowEnumerableIgnoreMissing = mlContext.Data
                .CreateEnumerable<SampleTemperatureDataWithLatitude>(data,
                reuseRowObject: true, ignoreMissingColumns: true);

            Console.WriteLine($"Date\tTemperature");
            foreach (var row in rowEnumerable)
                Console.WriteLine(
                    $"{row.Date.ToString("d")}\t{row.Temperature}");

            // Expected output:
            //  Date    Temperature
            //  1/2/2012        36
            //  1/3/2012        36
            //  1/4/2012        34
            //  1/5/2012        35
            //  1/6/2012        35

            Console.WriteLine($"Date\tTemperature\tLatitude");
            foreach (var row in rowEnumerableIgnoreMissing)
                Console.WriteLine($"{row.Date.ToString("d")}\t{row.Temperature}"
                    + $"\t{row.Latitude}");

            // Expected output:
            //  Date    Temperature     Latitude
            //  1/2/2012        36      0
            //  1/3/2012        36      0
            //  1/4/2012        34      0
            //  1/5/2012        35      0
            //  1/6/2012        35      0
        }

        private class SampleTemperatureData
        {
            public DateTime Date { get; set; }
            public float Temperature { get; set; }
        }

        private class SampleTemperatureDataWithLatitude
        {
            public float Latitude { get; set; }
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

