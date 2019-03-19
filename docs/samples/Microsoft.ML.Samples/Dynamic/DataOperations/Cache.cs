using System;
using Microsoft.ML.SamplesUtils;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class Cache
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            var data = DatasetUtils.LoadHousingRegressionDataset(mlContext);

            // Time how long it takes to page through the records if we don't cache.
            (int lines, double columnAverage, double elapsedSeconds) = TimeToScanIDataView(mlContext, data);
            Console.WriteLine($"Lines={lines}, averageOfColumn0={columnAverage:0.00} and took {elapsedSeconds} seconds.");
            // Expected output (time is approximate):
            // Lines=506, averageOfColumn0=564.17 and took 0.314 seconds.

            // Now create a cached view of the data.
            var cachedData = mlContext.Data.Cache(data);

            // Time how long it takes to page through the records the first time they're accessed after a cache is applied.
            // This iteration will be longer than subsequent calls, as the dataset is being accessed and stored for later.
            // Note that this operation may be relatively quick, as the system may have cached the file.
            (lines, columnAverage, elapsedSeconds) = TimeToScanIDataView(mlContext, cachedData);
            Console.WriteLine($"Lines={lines}, averageOfColumn0={columnAverage:0.00} and took {elapsedSeconds} seconds.");
            // Expected output (time is approximate):
            // Lines=506, averageOfColumn0=564.17 and took 0.056 seconds.

            // Time how long it takes to page through the records now that the data is cached. After the first iteration that caches the IDataView,
            // future iterations, like this one, are faster because they are pulling from data cached in memory.
            (lines, columnAverage, elapsedSeconds) = TimeToScanIDataView(mlContext, cachedData);
            Console.WriteLine($"Lines={lines}, averageOfColumn0={columnAverage:0.00} and took {elapsedSeconds} seconds.");
            // Expected output (time is approximate):
            // Lines=506, averageOfColumn0=564.17 and took 0.006 seconds.
        }

        private static (int lines, double columnAverage, double elapsedSeconds) TimeToScanIDataView(MLContext mlContext, IDataView data)
        {
            int lines = 0;
            double columnAverage = 0.0;
            var enumerable = mlContext.Data.CreateEnumerable<DatasetUtils.HousingRegression>(data, reuseRowObject: true);
            var watch = System.Diagnostics.Stopwatch.StartNew();
            foreach (var row in enumerable)
            {
                lines++;
                columnAverage += row.MedianHomeValue + row.CrimesPerCapita + row.PercentResidental + row.PercentNonRetail + row.CharlesRiver 
                    + row.NitricOxides + row.RoomsPerDwelling + row.PercentPre40s + row.EmploymentDistance 
                    + row.HighwayDistance + row.TaxRate + row.TeacherRatio;
            }
            watch.Stop();
            columnAverage /= lines;
            var elapsed = watch.Elapsed;

            return (lines, columnAverage, elapsed.Seconds);
        }       
    }
}
