using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public static class IndicateMissingValues
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable and convert it to an IDataView.
            var samples = new List<DataPoint>()
            {
                new DataPoint(){ Features = new float[3] {1, 1, 0} },
                new DataPoint(){ Features = new float[3] {0, float.NaN, 1} },
                new DataPoint(){ Features = new float[3] {-1, float.NaN, -3} },
            };
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // IndicateMissingValues is used to create a boolean containing 'true'
            // where the value in the input column is missing. For floats and
            // doubles, missing values are represented as NaN.
            var pipeline = mlContext.Transforms.IndicateMissingValues(
                "MissingIndicator", "Features");

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
                Console.WriteLine("Features: [" + string.Join(", ", row.Features) +
                    "]\t MissingIndicator: [" + string.Join(", ", row
                    .MissingIndicator) + "]");

            // Expected output:
            // Features: [1, 1, 0]      MissingIndicator: [False, False, False]
            // Features: [0, NaN, 1]    MissingIndicator: [False, True, False]
            // Features: [-1, NaN, -3]  MissingIndicator: [False, True, False]
        }

        private class DataPoint
        {
            [VectorType(3)]
            public float[] Features { get; set; }
        }

        private sealed class SampleDataTransformed : DataPoint
        {
            public bool[] MissingIndicator { get; set; }
        }
    }
}
