using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.SamplesUtils;

namespace Microsoft.ML.Samples.Dynamic
{
    using MulticlassClassificationExample = DatasetUtils.MulticlassClassificationExample;

    /// <summary>
    /// Sample class showing how to use FilterRowsByKeyColumnFraction.
    /// </summary>
    public static class FilterRowsByKeyColumnFraction
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            IEnumerable<MulticlassClassificationExample> enumerableOfData = DatasetUtils.GenerateRandomMulticlassClassificationExamples(10);
            var data = mlContext.Data.LoadFromEnumerable(enumerableOfData);

            // Convert the string labels to keys
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label");
            var transformedData = pipeline.Fit(data).Transform(data);

            // Before we apply a filter, examine all the records in the dataset.
            var enumerable = mlContext.Data.CreateEnumerable<MulticlassWithKeyLabel>(transformedData, reuseRowObject: true);
            Console.WriteLine($"Label\tFeatures");
            foreach (var row in enumerable)
            {
                Console.WriteLine($"{row.Label}\t({string.Join(", ", row.Features)})");
            }
            Console.WriteLine();
            // Expected output:
            //  1 (0.7262433, 0.8173254, 0.7680227, 0.5581612, 0.2060332, 0.5588848, 0.9060271, 0.4421779, 0.9775497, 0.2737045)
            //  2 (0.4919063, 0.6673147, 0.8326591, 0.6695119, 1.182151, 0.230367, 1.06237, 1.195347, 0.8771811, 0.5145918)
            //  3 (1.216908, 1.248052, 1.391902, 0.4326252, 1.099942, 0.9262842, 1.334019, 1.08762, 0.9468155, 0.4811099)
            //  4 (0.7871246, 1.053327, 0.8971719, 1.588544, 1.242697, 1.362964, 0.6303943, 0.9810045, 0.9431419, 1.557455)
            //  1 (0.5051292, 0.7159725, 0.1189577, 0.2734515, 0.9070979, 0.7947656, 0.3371603, 0.4572088, 0.146825, 0.2213147)
            //  2 (0.6100733, 0.9187268, 0.8198303, 0.6879681, 0.3949134, 1.078192, 1.025423, 0.9353975, 1.058219, 0.879749)
            //  3 (1.024866, 0.6184068, 1.295362, 1.29644, 0.4865799, 1.238579, 0.5701429, 1.044115, 1.226814, 0.6191877)
            //  4 (1.599973, 1.081366, 1.252205, 1.319726, 1.409463, 0.7009354, 1.329094, 1.318451, 0.7255273, 1.505176)
            //  1 (0.1891238, 0.4768099, 0.5407953, 0.3255007, 0.6710367, 0.4683977, 0.8334969, 0.8092038, 0.7936304, 0.764506)
            //  2 (1.13754, 0.4949968, 0.7227853, 0.8633928, 0.532589, 0.4867224, 1.02061, 0.4225179, 0.3868716, 0.2685189)

            // Now filter down to half the keys, choosing the lower half of values
            var filteredData = mlContext.Data.FilterRowsByKeyColumnFraction(transformedData, columnName: "Label", lowerBound: 0, upperBound: 0.5);

            // Look at the data and observe that values above 2 have been filtered out
            var filteredEnumerable = mlContext.Data.CreateEnumerable<MulticlassWithKeyLabel>(filteredData, reuseRowObject: true);
            Console.WriteLine($"Label\tFeatures");
            foreach (var row in filteredEnumerable)
            {
                Console.WriteLine($"{row.Label}\t({string.Join(", ", row.Features)})");
            }
            // Expected output:
            //  1 (0.7262433, 0.8173254, 0.7680227, 0.5581612, 0.2060332, 0.5588848, 0.9060271, 0.4421779, 0.9775497, 0.2737045)
            //  2 (0.4919063, 0.6673147, 0.8326591, 0.6695119, 1.182151, 0.230367, 1.06237, 1.195347, 0.8771811, 0.5145918)
            //  1 (0.5051292, 0.7159725, 0.1189577, 0.2734515, 0.9070979, 0.7947656, 0.3371603, 0.4572088, 0.146825, 0.2213147)
            //  2 (0.6100733, 0.9187268, 0.8198303, 0.6879681, 0.3949134, 1.078192, 1.025423, 0.9353975, 1.058219, 0.879749)
            //  1 (0.1891238, 0.4768099, 0.5407953, 0.3255007, 0.6710367, 0.4683977, 0.8334969, 0.8092038, 0.7936304, 0.764506)
            //  2 (1.13754, 0.4949968, 0.7227853, 0.8633928, 0.532589, 0.4867224, 1.02061, 0.4225179, 0.3868716, 0.2685189)
        }

        private class MulticlassWithKeyLabel
        {
            public uint Label { get; set; }
            [VectorType(10)]
            public float[] Features { get; set; }
        }
    }
}
