using System;
using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class Cache
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            IEnumerable<SamplesUtils.DatasetUtils.FloatLabelFloatFeatureVectorSample> enumerableOfData = SamplesUtils.DatasetUtils.GenerateFloatLabelFloatFeatureVectorSamples(100000);
            var data = mlContext.Data.ReadFromEnumerable(enumerableOfData);

            // Time how long it takes to page through the records if we don't cache.
            int lines = 0;
            double averageOfColumn0 = 0.0;
            var watch = System.Diagnostics.Stopwatch.StartNew();
            var enumerable = mlContext.CreateEnumerable<SamplesUtils.DatasetUtils.FloatLabelFloatFeatureVectorSample>(data, reuseRowObject: true);
            foreach (var row in enumerable)
            {
                lines++;
                averageOfColumn0 += row.Features[0];
            }
            watch.Stop();
            averageOfColumn0 /= lines;
            var elapsedSeconds = watch.ElapsedMilliseconds / 1000.0;
            Console.WriteLine($"Lines={lines}, averageOfColumn0={averageOfColumn0:0.00} and took {elapsedSeconds} seconds.");
            // Expected output (time is approximate):
            //  Lines=100000, averageOfColumn0=0.60 and took 0.221 seconds.

            // Now cache the data.
            var cachedData = mlContext.Data.Cache(data);

            // Time how long it takes to page through the records the first time they're accessed after a cache.
            // This iteration will be longer, as the dataset is being accessed and stored for later.
            lines = 0;
            averageOfColumn0 = 0.0;
            watch = System.Diagnostics.Stopwatch.StartNew();
            enumerable = mlContext.CreateEnumerable<SamplesUtils.DatasetUtils.FloatLabelFloatFeatureVectorSample>(cachedData, reuseRowObject: true);
            foreach (var row in enumerable)
            {
                lines++;
                averageOfColumn0 += row.Features[0];
            }
            watch.Stop();
            averageOfColumn0 /= lines;
            elapsedSeconds = watch.ElapsedMilliseconds / 1000.0;
            Console.WriteLine($"Lines={lines}, averageOfColumn0={averageOfColumn0:0.00} and took {elapsedSeconds} seconds.");
            // Expected output (time is approximate):
            //  Lines=100000, averageOfColumn0=0.60 and took 0.357 seconds.

            // Time how long it takes to page through the records now that the data is cached. After the first iteration that caches the IDataView,
            // future iterations, like this one, are faster because they are pulling from data cached in memory.
            lines = 0;
            averageOfColumn0 = 0.0;
            watch = System.Diagnostics.Stopwatch.StartNew();
            enumerable = mlContext.CreateEnumerable<SamplesUtils.DatasetUtils.FloatLabelFloatFeatureVectorSample>(cachedData, reuseRowObject: true);
            foreach (var row in enumerable)
            {
                lines++;
                averageOfColumn0 += row.Features[0];
            }
            watch.Stop();
            averageOfColumn0 /= lines;
            elapsedSeconds = watch.ElapsedMilliseconds / 1000.0;
            Console.WriteLine($"Lines={lines}, averageOfColumn0={averageOfColumn0:0.00} and took {elapsedSeconds} seconds.");
            // Expected output (time is approximate):
            //  Lines=100000, averageOfColumn0=0.60 and took 0.137 seconds.
        }
    }
}
