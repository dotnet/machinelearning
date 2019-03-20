using System;
using System.Collections.Generic;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class Bootstrap
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable and them read it as ML.NET's data type.
            IEnumerable<SamplesUtils.DatasetUtils.BinaryLabelFloatFeatureVectorSample> enumerableOfData = SamplesUtils.DatasetUtils.GenerateBinaryLabelFloatFeatureVectorSamples(5);
            var data = mlContext.Data.LoadFromEnumerable(enumerableOfData);

            // Look at the original dataset
            Console.WriteLine($"Label\tFeatures[0]");
            foreach (var row in enumerableOfData)
            {
                Console.WriteLine($"{row.Label}\t{row.Features[0]}");
            }
            Console.WriteLine();
            // Expected output:
            //  Label Features[0]
            //  True    1.017325
            //  False   0.6326591
            //  False   0.0326252
            //  True    0.8426974
            //  True    0.9947656

            // Now take a bootstrap sample of this dataset to create a new dataset. The bootstrap is a resampling technique that
            // creates a training set of the same size by picking with replacement from the original dataset. With the bootstrap, 
            // we expect that the resampled dataset will have about 63% of the rows of the original dataset (i.e. 1-e^-1), with some
            // rows represented more than once.
            // BootstrapSample is a streaming implementation of the boostrap that enables sampling from a dataset too large to hold in memory.
            // To enable streaming, BootstrapSample approximates the bootstrap by sampling each row according to a Poisson(1) distribution.
            // Note that this streaming approximation treats each row independently, thus the resampled dataset is not guaranteed to be the 
            // same length as the input dataset.
            // Let's take a look at the behavior of the BootstrapSample by examining a few draws:
            for (int i = 0; i < 3; i++)
            {
                var resample = mlContext.Data.BootstrapSample(data, seed: i);

                var enumerable = mlContext.Data.CreateEnumerable<SamplesUtils.DatasetUtils.BinaryLabelFloatFeatureVectorSample>(resample, reuseRowObject: false);
                Console.WriteLine($"Label\tFeatures[0]");
                foreach (var row in enumerable)
                {
                    Console.WriteLine($"{row.Label}\t{row.Features[0]}");
                }
                Console.WriteLine();
            }
            // Expected output:
            //  Label Features[0]
            //  True    1.017325
            //  False   0.6326591
            //  False   0.6326591
            //  False   0.6326591
            //  False   0.0326252
            //  False   0.0326252
            //  True    0.8426974
            //  True    0.8426974

            //  Label Features[0]
            //  True    1.017325
            //  True    1.017325
            //  False   0.6326591
            //  False   0.6326591
            //  False   0.0326252
            //  False   0.0326252
            //  False   0.0326252
            //  True    0.9947656

            //  Label Features[0]
            //  False   0.6326591
            //  False   0.0326252
            //  True    0.8426974
            //  True    0.8426974
            //  True    0.8426974
        }
    }
}
