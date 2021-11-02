using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.Transforms.NormalizingTransformer;

namespace Samples.Dynamic
{
    public class NormalizeMeanVariance
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();
            var samples = new List<DataPoint>()
            {
                new DataPoint(){ Features = new float[4] { 1, 1, 3, 0} },
                new DataPoint(){ Features = new float[4] { 2, 2, 2, 0} },
                new DataPoint(){ Features = new float[4] { 0, 0, 1, 0} },
                new DataPoint(){ Features = new float[4] {-1,-1,-1, 1} }
            };
            // Convert training data to IDataView, the general data type used in
            // ML.NET.
            var data = mlContext.Data.LoadFromEnumerable(samples);
            // NormalizeMeanVariance normalizes the data based on the computed mean
            // and variance of the data. Uses Cumulative distribution function as
            // output.
            var normalize = mlContext.Transforms.NormalizeMeanVariance("Features",
                useCdf: true);

            // NormalizeMeanVariance normalizes the data based on the computed mean
            // and variance of the data.
            var normalizeNoCdf = mlContext.Transforms.NormalizeMeanVariance(
                "Features", useCdf: false);

            // Now we can transform the data and look at the output to confirm the
            // behavior of the estimator. This operation doesn't actually evaluate
            // data until we read the data below.
            var normalizeTransform = normalize.Fit(data);
            var transformedData = normalizeTransform.Transform(data);
            var normalizeNoCdfTransform = normalizeNoCdf.Fit(data);
            var noCdfData = normalizeNoCdfTransform.Transform(data);
            var column = transformedData.GetColumn<float[]>("Features").ToArray();
            foreach (var row in column)
                Console.WriteLine(string.Join(", ", row.Select(x => x.ToString(
                    "f4"))));
            // Expected output:
            //  0.6726, 0.6726, 0.8816, 0.2819
            //  0.9101, 0.9101, 0.6939, 0.2819
            //  0.3274, 0.3274, 0.4329, 0.2819
            //  0.0899, 0.0899, 0.0641, 0.9584


            var columnFixZero = noCdfData.GetColumn<float[]>("Features").ToArray();
            foreach (var row in columnFixZero)
                Console.WriteLine(string.Join(", ", row.Select(x => x.ToString(
                    "f4"))));
            // Expected output:
            //  0.8165, 0.8165, 1.5492, 0.0000
            //  1.6330, 1.6330, 1.0328, 0.0000
            //  0.0000, 0.0000, 0.5164, 0.0000
            // -0.8165,-0.8165,-0.5164, 2.0000

            // Let's get transformation parameters. Since we work with only one
            // column we need to pass 0 as parameter for
            // GetNormalizerModelParameters. If we have multiple columns
            // transformations we need to pass index of InputOutputColumnPair.
            var transformParams = normalizeTransform
                .GetNormalizerModelParameters(0) as CdfNormalizerModelParameters<
                ImmutableArray<float>>;

            Console.WriteLine($"The 1-index value in resulting array would " +
                $"be produce by:");

            Console.WriteLine(" y = 0.5* (1 + ERF((x- " + transformParams.Mean[1] +
                ") / (" + transformParams.StandardDeviation[1] + " * sqrt(2)))");
            // ERF is https://en.wikipedia.org/wiki/Error_function.
            // Expected output:
            //  The 1-index value in resulting array would be produce by:
            //  y = 0.5 * (1 + ERF((x - 0.5) / (1.118034 * sqrt(2)))

            var noCdfParams = normalizeNoCdfTransform
                .GetNormalizerModelParameters(0) as
                AffineNormalizerModelParameters<ImmutableArray<float>>;

            var offset = noCdfParams.Offset.Length == 0 ? 0 : noCdfParams.Offset[1];
            var scale = noCdfParams.Scale[1];
            Console.WriteLine($"Values for slot 1 would be transformed by " +
                $"applying y = (x - ({offset})) * {scale}");
            // Expected output:
            // The 1-index value in resulting array would be produce by: y = (x - (0)) * 0.8164966
        }

        private class DataPoint
        {
            [VectorType(4)]
            public float[] Features { get; set; }
        }
    }
}
