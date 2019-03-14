using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class IndicateMissingValues
    {

        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            var samples = new List<DataPoint>()
            {
                new DataPoint(){ Label = 3, Features = new float[3] {1, 1, 0} },
                new DataPoint(){ Label = 32, Features = new float[3] {0, float.NaN, 1} },
                new DataPoint(){ Label = float.NaN, Features = new float[3] {-1, float.NaN, -3} },
            };
            // Convert training data to IDataView, the general data type used in ML.NET.
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // IndicateMissingValues is used to create a boolean containing
            // 'true' where the value in the input column is NaN. This value can be used
            // to replace missing values with other values.
            IEstimator<ITransformer> pipeline = mlContext.Transforms.IndicateMissingValues("MissingIndicator", "Features");

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
                Console.WriteLine($"Label: {row.Label} Features: {vectorPrinter(row.Features.Cast<object>().ToArray())} MissingIndicator: {vectorPrinter(row.MissingIndicator.Cast<object>().ToArray())}");
            }

            // Expected output:
            // 
            // Label: 3 Features: [1 1 0] MissingIndicator: [False False False]
            // Label: 32 Features: [0 NaN 1] MissingIndicator: [False True False]
            // Label: NaN Features: [-1 NaN -3 ] MissingIndicator: [False True False]
        }

        private class DataPoint
        {
            public float Label { get; set; }
            [VectorType(3)]
            public float[] Features { get; set; }
        }

        private sealed class SampleDataTransformed : DataPoint
        {
            public bool[] MissingIndicator { get; set; }
        }
    }
}