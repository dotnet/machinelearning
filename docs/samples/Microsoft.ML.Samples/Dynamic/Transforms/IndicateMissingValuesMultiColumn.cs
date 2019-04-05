using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class IndicateMissingValuesMultiColumn
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            var samples = new List<DataPoint>()
            {
                new DataPoint(){ Label = 3, Features1 = new float[3] {1, 1, 0}, Features2 = new float[2] {1, 1} },
                new DataPoint(){ Label = 32, Features1 = new float[3] {0, float.NaN, 1}, Features2 = new float[2] {float.NaN, 1} },
                new DataPoint(){ Label = float.NaN, Features1 = new float[3] {-1, float.NaN, -3}, Features2 = new float[2] {1, float.PositiveInfinity} },
            };
            // Convert training data to IDataView, the general data type used in ML.NET.
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // IndicateMissingValues is used to create a boolean containing
            // 'true' where the value in the input column is NaN. This value can be used
            // to replace missing values with other values. We can use an array of InputOutputColumnPair
            // to apply the MissingValueIndicatorEstimator to multiple columns in one pass over the data.
            IEstimator<ITransformer> pipeline = mlContext.Transforms.IndicateMissingValues(new[] {
                new InputOutputColumnPair("MissingIndicator1", "Features1"),
                new InputOutputColumnPair("MissingIndicator2", "Features2")
            });

            // Now we can transform the data and look at the output to confirm the behavior of the estimator.
            // This operation doesn't actually evaluate data until we read the data below.
            var tansformer = pipeline.Fit(data);
            var transformedData = tansformer.Transform(data);

            // We can extract the newly created column as an IEnumerable of SampleDataTransformed, the class we define below.
            var rowEnumerable = mlContext.Data.CreateEnumerable<SampleDataTransformed>(transformedData, reuseRowObject: false);

            // a small printing utility
            Func<object[], string> vectorPrinter = (object[] vector) =>
            {
                string preview = "[";
                foreach (var slot in vector)
                    preview += $"{slot} ";
               return preview += "]";

            };

            // And finally, we can write out the rows of the dataset, looking at the columns of interest.
            foreach (var row in rowEnumerable)
            {
                Console.WriteLine($"Label: {row.Label} Features1: {vectorPrinter(row.Features1.Cast<object>().ToArray())} " +
                    $"Features2: {vectorPrinter(row.Features2.Cast<object>().ToArray())} " +
                    $"MissingIndicator1: {vectorPrinter(row.MissingIndicator1.Cast<object>().ToArray())} " +
                    $"MissingIndicator2: {vectorPrinter(row.MissingIndicator2.Cast<object>().ToArray())}");
            }

            // Expected output:
            // Label: 3 Features1: [1 1 0] Features2: [1 1] MissingIndicator1: [False False False] MissingIndicator2: [False False]
            // Label: 32 Features1: [0 NaN 1] Features2: [NaN 1] MissingIndicator1: [False True False] MissingIndicator2: [True False]
            // Label: NaN Features1: [-1 NaN -3 ] Features2: [1 ∞ ] MissingIndicator1: [False True False] MissingIndicator2: [False False]
        }

        private class DataPoint
        {
            public float Label { get; set; }
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