using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.Samples.Dynamic
{
    class ReplaceMissingValues
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
                new DataPoint(){ Label = 5, Features = new float[3] {-1, 2, -3} },
                 new DataPoint(){ Label = 9, Features = new float[3] {-1, 6, -3} },
            };
            // Convert training data to IDataView, the general data type used in ML.NET.
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // ReplaceMissingValues is used to create a column where missing values are replaced according to the ReplacementMode.
            var meanPipeline = mlContext.Transforms.ReplaceMissingValues("MissingReplaced", "Features", MissingValueReplacingEstimator.ReplacementMode.Mean);

            // Now we can transform the data and look at the output to confirm the behavior of the estimator.
            // This operation doesn't actually evaluate data until we read the data below.
            var meanTransformer = meanPipeline.Fit(data);
            var meanTransformedData = meanTransformer.Transform(data);

            // We can extract the newly created column as an IEnumerable of SampleDataTransformed, the class we define below.
            var meanRowEnumerable = mlContext.Data.CreateEnumerable<SampleDataTransformed>(meanTransformedData, reuseRowObject: false);

            // ReplaceMissingValues is used to create a column where missing values are replaced according to the ReplacementMode.
            var defaultPipeline = mlContext.Transforms.ReplaceMissingValues("MissingReplaced", "Features", MissingValueReplacingEstimator.ReplacementMode.DefaultValue);

            // Now we can transform the data and look at the output to confirm the behavior of the estimator.
            // This operation doesn't actually evaluate data until we read the data below.
            var defaultTransformer = defaultPipeline.Fit(data);
            var defaultTransformedData = defaultTransformer.Transform(data);

            // We can extract the newly created column as an IEnumerable of SampleDataTransformed, the class we define below.
            var defaultRowEnumerable = mlContext.Data.CreateEnumerable<SampleDataTransformed>(defaultTransformedData, reuseRowObject: false);

            // a small printing utility
            Func<object[], string> vectorPrinter = (object[] vector) =>
            {
                string preview = "[";
                foreach (var slot in vector)
                    preview += $"{slot} ";
                return preview += "]";

            };

            // And finally, we can write out the rows of the dataset, looking at the columns of interest.
            foreach (var row in meanRowEnumerable)
            {
                Console.WriteLine($"Label: {row.Label} Features: {vectorPrinter(row.Features.Cast<object>().ToArray())} MissingReplaced: {vectorPrinter(row.MissingReplaced.Cast<object>().ToArray())}");
            }

            // Expected output:
            // Notice how the NaN of the Features column for the second row is replaced by the mean of (1, 2, 6) the values in that row
            // 
            //Label: 3  Features: [1 1    0] MissingReplaced: [1  1  0]
            //Label: 32 Features: [0 NaN  1] MissingReplaced: [0  3  1]
            //Label: 5  Features: [-1 2 - 3] MissingReplaced: [-1 2 -3]
            //Label: 9  Features: [-1 6 - 3] MissingReplaced: [-1 6 -3]

            // And finally, we can write out the rows of the dataset, looking at the columns of interest.
            foreach (var row in defaultRowEnumerable)
            {
                Console.WriteLine($"Label: {row.Label} Features: {vectorPrinter(row.Features.Cast<object>().ToArray())} MissingReplaced: {vectorPrinter(row.MissingReplaced.Cast<object>().ToArray())}");
            }

            // Expected output:
            // Notice how the NaN of the Features column for the second row is replaced by 0, the default value for floats.
            // 
            //Label: 3  Features: [1 1 0]    MissingReplaced: [1 1 0]
            //Label: 32 Features: [0 NaN 1]  MissingReplaced: [0 0 1]
            //Label: 5  Features: [-1 2 - 3] MissingReplaced: [-1 2 - 3]
            //Label: 9  Features: [-1 6 - 3] MissingReplaced: [-1 6 - 3]
        }

        private class DataPoint
        {
            public float Label { get; set; }

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
