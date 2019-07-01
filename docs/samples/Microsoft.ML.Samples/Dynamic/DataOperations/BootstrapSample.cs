using System;
using Microsoft.ML;

namespace Samples.Dynamic
{
    public static class BootstrapSample
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available operations 
            // and as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            var rawData = new[] {
                new DataPoint() { Label = true, Feature = 1.017325f},
                new DataPoint() { Label = false, Feature = 0.6326591f},
                new DataPoint() { Label = false, Feature = 0.0326252f},
                new DataPoint() { Label = false, Feature = 0.8426974f},
                new DataPoint() { Label = true, Feature = 0.9947656f},
                new DataPoint() { Label = true, Feature = 1.017325f},
            };

            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // Now take a bootstrap sample of this dataset to create a new dataset. 
            // The bootstrap is a resampling technique that creates a training set
            // of the same size by picking with replacement from the original
            // dataset. With the bootstrap, we expect that the resampled dataset
            // will have about 63% of the rows of the original dataset
            // (i.e. 1-e^-1), with some rows represented more than once.
            // BootstrapSample is a streaming implementation of the boostrap that
            // enables sampling from a dataset too large to hold in memory. To
            // enable streaming, BootstrapSample approximates the bootstrap by 
            // sampling each row according to a Poisson(1) distribution. Note that
            // this streaming approximation treats each row independently, thus the
            // resampled dataset is not guaranteed to be the same length as the 
            // input dataset. Let's take a look at the behavior of the
            // BootstrapSample by examining a few draws:
            for (int i = 0; i < 3; i++)
            {
                var resample = mlContext.Data.BootstrapSample(data, seed: i);

                var enumerable = mlContext.Data
                    .CreateEnumerable<DataPoint>(resample, reuseRowObject: false);

                Console.WriteLine($"Label\tFeature");
                foreach (var row in enumerable)
                {
                    Console.WriteLine($"{row.Label}\t{row.Feature}");
                }
                Console.WriteLine();
            }
            // Expected output:
            //  Label Feature
            //  True    1.017325
            //  False   0.6326591
            //  False   0.6326591
            //  False   0.6326591
            //  False   0.0326252
            //  False   0.0326252
            //  True    0.8426974
            //  True    0.8426974

            //  Label Feature
            //  True    1.017325
            //  True    1.017325
            //  False   0.6326591
            //  False   0.6326591
            //  False   0.0326252
            //  False   0.0326252
            //  False   0.0326252
            //  True    0.9947656

            //  Label Feature
            //  False   0.6326591
            //  False   0.0326252
            //  True    0.8426974
            //  True    0.8426974
            //  True    0.8426974
        }

        private class DataPoint
        {
            public bool Label { get; set; }

            public float Feature { get; set; }
        }
    }
}
