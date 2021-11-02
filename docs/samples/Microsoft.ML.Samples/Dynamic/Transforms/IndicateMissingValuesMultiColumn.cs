using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public static class IndicateMissingValuesMultiColumn
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
                    Features2 = new float[2] {float.NaN, 1} },

                new DataPoint(){ Features1 = new float[3] {-1, float.NaN, -3},
                    Features2 = new float[2] {1, float.PositiveInfinity} },
            };
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // IndicateMissingValues is used to create a boolean containing 'true'
            // where the value in the input column is missing. For floats and
            // doubles, missing values are NaN. We can use an array of
            // InputOutputColumnPair to apply the MissingValueIndicatorEstimator
            // to multiple columns in one pass over the data.
            var pipeline = mlContext.Transforms.IndicateMissingValues(new[] {
                new InputOutputColumnPair("MissingIndicator1", "Features1"),
                new InputOutputColumnPair("MissingIndicator2", "Features2")
            });

            // Now we can transform the data and look at the output to confirm the
            // behavior of the estimator. This operation doesn't actually evaluate
            // data until we read the data below.
            var tansformer = pipeline.Fit(data);
            var transformedData = tansformer.Transform(data);

            // We can extract the newly created column as an IEnumerable of
            // SampleDataTransformed, the class we define below.
            var rowEnumerable = mlContext.Data.CreateEnumerable<
                SampleDataTransformed>(transformedData, reuseRowObject: false);

            // And finally, we can write out the rows of the dataset, looking at the
            // columns of interest.
            foreach (var row in rowEnumerable)
                Console.WriteLine("Features1: [" + string.Join(", ", row
                    .Features1) + "]\t MissingIndicator1: [" + string.Join(", ",
                    row.MissingIndicator1) + "]\t Features2: [" + string.Join(", ",
                    row.Features2) + "]\t MissingIndicator2: [" + string.Join(", ",
                    row.MissingIndicator2) + "]");

            // Expected output:
            // Features1: [1, 1, 0]     MissingIndicator1: [False, False, False]        Features2: [1, 1]       MissingIndicator2: [False, False]
            // Features1: [0, NaN, 1]   MissingIndicator1: [False, True, False]         Features2: [NaN, 1]     MissingIndicator2: [True, False]
            // Features1: [-1, NaN, -3]         MissingIndicator1: [False, True, False]         Features2: [1, ∞]       MissingIndicator2: [False, False]
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
            public bool[] MissingIndicator1 { get; set; }
            public bool[] MissingIndicator2 { get; set; }

        }
    }
}
