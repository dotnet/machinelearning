using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Samples.Dynamic
{
    class ReplaceMissingValues
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable and convert it to an IDataView.
            var samples = new List<DataPoint>()
            {
                new DataPoint(){ Features = new float[3] {float.PositiveInfinity, 1,
                    0 } },

                new DataPoint(){ Features = new float[3] {0, float.NaN, 1} },
                new DataPoint(){ Features = new float[3] {-1, 2, -3} },
                new DataPoint(){ Features = new float[3] {-1, float.NaN, -3} },
            };
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // Here we use the default replacement mode, which replaces the value
            // with the default value for its type.
            var defaultPipeline = mlContext.Transforms.ReplaceMissingValues(
                "MissingReplaced", "Features", MissingValueReplacingEstimator
                .ReplacementMode.DefaultValue);

            // Now we can transform the data and look at the output to confirm the
            // behavior of the estimator. This operation doesn't actually evaluate
            // data until we read the data below.
            var defaultTransformer = defaultPipeline.Fit(data);
            var defaultTransformedData = defaultTransformer.Transform(data);

            // We can extract the newly created column as an IEnumerable of
            // SampleDataTransformed, the class we define below.
            var defaultRowEnumerable = mlContext.Data.CreateEnumerable<
                SampleDataTransformed>(defaultTransformedData, reuseRowObject:
                false);

            // And finally, we can write out the rows of the dataset, looking at the
            // columns of interest.
            foreach (var row in defaultRowEnumerable)
                Console.WriteLine("Features: [" + string.Join(", ", row.Features) +
                    "]\t MissingReplaced: [" + string.Join(", ", row
                    .MissingReplaced) + "]");

            // Expected output:
            // Features: [∞, 1, 0]      MissingReplaced: [∞, 1, 0]
            // Features: [0, NaN, 1]    MissingReplaced: [0, 0, 1]
            // Features: [-1, 2, -3]    MissingReplaced: [-1, 2, -3]
            // Features: [-1, NaN, -3]  MissingReplaced: [-1, 0, -3]

            // Here we use the mean replacement mode, which replaces the value with
            // the mean of the non values that were not missing.
            var meanPipeline = mlContext.Transforms.ReplaceMissingValues(
                "MissingReplaced", "Features", MissingValueReplacingEstimator
                .ReplacementMode.Mean);

            // Now we can transform the data and look at the output to confirm the
            // behavior of the estimator. This operation doesn't actually evaluate
            // data until we read the data below.
            var meanTransformer = meanPipeline.Fit(data);
            var meanTransformedData = meanTransformer.Transform(data);

            // We can extract the newly created column as an IEnumerable of
            // SampleDataTransformed, the class we define below.
            var meanRowEnumerable = mlContext.Data.CreateEnumerable<
                SampleDataTransformed>(meanTransformedData, reuseRowObject: false);

            // And finally, we can write out the rows of the dataset, looking at the
            // columns of interest.
            foreach (var row in meanRowEnumerable)
                Console.WriteLine("Features: [" + string.Join(", ", row.Features) +
                    "]\t MissingReplaced: [" + string.Join(", ", row
                    .MissingReplaced) + "]");

            // Expected output:
            // Features: [∞, 1, 0]      MissingReplaced: [∞, 1, 0]
            // Features: [0, NaN, 1]    MissingReplaced: [0, 1.5, 1]
            // Features: [-1, 2, -3]    MissingReplaced: [-1, 2, -3]
            // Features: [-1, NaN, -3]  MissingReplaced: [-1, 1.5, -3]
        }

        private class DataPoint
        {
            [VectorType(3)]
            public float[] Features { get; set; }
        }

        private sealed class SampleDataTransformed : DataPoint
        {
            [VectorType(3)]
            public float[] MissingReplaced { get; set; }
        }
    }
}
