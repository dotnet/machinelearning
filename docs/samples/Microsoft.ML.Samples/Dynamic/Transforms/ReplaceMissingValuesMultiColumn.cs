using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Samples.Dynamic
{
    class ReplaceMissingValuesMultiColumn
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable and convert it to an IDataView.
            var samples = new List<DataPoint>()
            {
                new DataPoint(){ Features1 = new float[3] {1, 1, 0}, Features2 =
                    new float[2] {1, 1} },

                new DataPoint(){ Features1 = new float[3] {0, float.NaN, 1},
                    Features2 = new float[2] {0, 1} },

                new DataPoint(){ Features1 = new float[3] {-1, float.NaN, -3},
                    Features2 = new float[2] {-1, float.NaN} },

                new DataPoint(){ Features1 = new float[3] {-1, 6, -3}, Features2 =
                    new float[2] {0, float.PositiveInfinity} },
            };
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // Here we use the default replacement mode, which replaces the value
            // with the default value for its type.
            var defaultPipeline = mlContext.Transforms.ReplaceMissingValues(new[] {
                new InputOutputColumnPair("MissingReplaced1", "Features1"),
                new InputOutputColumnPair("MissingReplaced2", "Features2")
            },
            MissingValueReplacingEstimator.ReplacementMode.DefaultValue);

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
                Console.WriteLine("Features1: [" + string.Join(", ", row
                    .Features1) + "]\t MissingReplaced1: [" + string.Join(", ", row
                    .MissingReplaced1) + "]\t Features2: [" + string.Join(", ", row
                    .Features2) + "]\t MissingReplaced2: [" + string.Join(", ", row
                    .MissingReplaced2) + "]");

            // Expected output:
            // Features1: [1, 1, 0]     MissingReplaced1: [1, 1, 0]     Features2: [1, 1]       MissingReplaced2: [1, 1]
            // Features1: [0, NaN, 1]   MissingReplaced1: [0, 0, 1]     Features2: [0, 1]       MissingReplaced2: [0, 1]
            // Features1: [-1, NaN, -3]         MissingReplaced1: [-1, 0, -3]   Features2: [-1, NaN]    MissingReplaced2: [-1, 0]
            // Features1: [-1, 6, -3]   MissingReplaced1: [-1, 6, -3]   Features2: [0, ∞]       MissingReplaced2: [0, ∞]

            // Here we use the mean replacement mode, which replaces the value with
            // the mean of the non values that were not missing.
            var meanPipeline = mlContext.Transforms.ReplaceMissingValues(new[] {
                new InputOutputColumnPair("MissingReplaced1", "Features1"),
                new InputOutputColumnPair("MissingReplaced2", "Features2")
            },
            MissingValueReplacingEstimator.ReplacementMode.Mean);

            // Now we can transform the data and look at the output to confirm the
            // behavior of the estimator.
            // This operation doesn't actually evaluate data until we read the data
            // below.
            var meanTransformer = meanPipeline.Fit(data);
            var meanTransformedData = meanTransformer.Transform(data);

            // We can extract the newly created column as an IEnumerable of
            // SampleDataTransformed, the class we define below.
            var meanRowEnumerable = mlContext.Data.CreateEnumerable<
                SampleDataTransformed>(meanTransformedData, reuseRowObject: false);

            // And finally, we can write out the rows of the dataset, looking at the
            // columns of interest.
            foreach (var row in meanRowEnumerable)
                Console.WriteLine("Features1: [" + string.Join(", ", row
                    .Features1) + "]\t MissingReplaced1: [" + string.Join(", ", row
                    .MissingReplaced1) + "]\t Features2: [" + string.Join(", ", row
                    .Features2) + "]\t MissingReplaced2: [" + string.Join(", ", row
                    .MissingReplaced2) + "]");

            // Expected output:
            // Features1: [1, 1, 0]     MissingReplaced1: [1, 1, 0]     Features2: [1, 1]       MissingReplaced2: [1, 1]
            // Features1: [0, NaN, 1]   MissingReplaced1: [0, 3.5, 1]   Features2: [0, 1]       MissingReplaced2: [0, 1]
            // Features1: [-1, NaN, -3]         MissingReplaced1: [-1, 3.5, -3]         Features2: [-1, NaN]    MissingReplaced2: [-1, 1]
            // Features1: [-1, 6, -3]   MissingReplaced1: [-1, 6, -3]   Features2: [0, ∞]       MissingReplaced2: [0, ∞]
        }

        private class DataPoint
        {
            [VectorType(3)]
            public float[] Features1 { get; set; }
            [VectorType(2)]
            public float[] Features2 { get; set; }
        }

        private sealed class SampleDataTransformed : DataPoint
        {
            [VectorType(3)]
            public float[] MissingReplaced1 { get; set; }
            [VectorType(2)]
            public float[] MissingReplaced2 { get; set; }
        }
    }
}
