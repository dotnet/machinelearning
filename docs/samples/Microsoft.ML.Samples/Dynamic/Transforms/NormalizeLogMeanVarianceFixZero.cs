using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.Transforms.NormalizingTransformer;

namespace Samples.Dynamic
{
    public class NormalizeLogMeanVarianceFixZero
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging,
            // as well as the source of randomness.
            var mlContext = new MLContext();
            var samples = new List<DataPoint>()
            {
                new DataPoint(){ Features = new float[5] { 1, 1, 3, 0, float.MaxValue } },
                new DataPoint(){ Features = new float[5] { 2, 2, 2, 0, float.MinValue } },
                new DataPoint(){ Features = new float[5] { 0, 0, 1, 0, 0} },
                new DataPoint(){ Features = new float[5] {-1,-1,-1, 1, 1} }
            };
            // Convert training data to IDataView, the general data type used in ML.NET.
            var data = mlContext.Data.LoadFromEnumerable(samples);
            // NormalizeLogMeanVariance normalizes the data based on the computed mean and variance of the logarithm of the data.
            // Uses Cumulative distribution function as output.
            var normalize = mlContext.Transforms.NormalizeLogMeanVariance("Features", true, useCdf: true);

            // NormalizeLogMeanVariance normalizes the data based on the computed mean and variance of the logarithm of the data.
            var normalizeNoCdf = mlContext.Transforms.NormalizeLogMeanVariance("Features", true, useCdf: false);

            // Now we can transform the data and look at the output to confirm the behavior of the estimator.
            // This operation doesn't actually evaluate data until we read the data below.
            var normalizeTransform = normalize.Fit(data);
            var transformedData = normalizeTransform.Transform(data);
            var normalizeNoCdfTransform = normalizeNoCdf.Fit(data);
            var noCdfData = normalizeNoCdfTransform.Transform(data);
            var column = transformedData.GetColumn<float[]>("Features").ToArray();
            foreach (var row in column)
                Console.WriteLine(string.Join(", ", row.Select(x => x.ToString("f4"))));
            // Expected output:
            //  0.1587, 0.1587, 0.8654, 0.0000, 0.8413
            //  0.8413, 0.8413, 0.5837, 0.0000, 0.0000
            //  0.0000, 0.0000, 0.0940, 0.0000, 0.0000
            //  0.0000, 0.0000, 0.0000, 0.0000, 0.1587

            var columnFixZero = noCdfData.GetColumn<float[]>("Features").ToArray();
            foreach (var row in columnFixZero)
                Console.WriteLine(string.Join(", ", row.Select(x => x.ToString("f4"))));
            // Expected output:
            //  2.0403, 2.0403, 4.0001, 0.0000, 5423991000000000000000000000000000000.0000
            //  4.0806, 4.0806, 2.6667, 0.0000,-5423991000000000000000000000000000000.0000
            //  0.0000, 0.0000, 1.3334, 0.0000, 0.0000
            // -2.0403,-2.0403,-1.3334, 0.0000, 0.0159

            // Let's get transformation parameters. Since we work with only one column we need to pass 0 as parameter for GetNormalizerModelParameters.
            // If we have multiple columns transformations we need to pass index of InputOutputColumnPair.
            var transformParams = normalizeTransform.GetNormalizerModelParameters(0) as CdfNormalizerModelParameters<ImmutableArray<float>>;
            Console.WriteLine("The values in the column with index 1 in the resulting array would be produced by:");
            Console.WriteLine($"y = 0.5* (1 + ERF((Math.Log(x)- {transformParams.Mean[1]}) / ({transformParams.StandardDeviation[1]} * sqrt(2)))");

            // ERF is https://en.wikipedia.org/wiki/Error_function.
            // Expected output:
            // The values in the column with index 1 in the resulting array would be produced by:
            // y = 0.5 * (1 + ERF((Math.Log(x) - 0.3465736) / (0.3465736 * sqrt(2)))
            var noCdfParams = normalizeNoCdfTransform.GetNormalizerModelParameters(0) as AffineNormalizerModelParameters<ImmutableArray<float>>;
            var offset = noCdfParams.Offset.Length == 0 ? 0 : noCdfParams.Offset[1];
            var scale = noCdfParams.Scale[1];
            Console.WriteLine($"The values in the column with index 1 in the resulting array would be produced by: y = (x - ({offset})) * {scale}");
            // Expected output:
            // The values in the column with index 1 in the resulting array would be produced by: y = (x - (0)) * 2.040279
        }

        private class DataPoint
        {
            [VectorType(5)]
            public float[] Features { get; set; }
        }
    }
}
