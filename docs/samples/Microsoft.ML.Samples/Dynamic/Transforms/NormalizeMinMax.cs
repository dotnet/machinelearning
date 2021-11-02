using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.Transforms.NormalizingTransformer;

namespace Samples.Dynamic
{
    public class NormalizeMinMax
    {
        public static void Example()
        {
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
            // NormalizeMinMax normalize rows by finding min and max values in each
            // row slot and setting projection of min value to 0 and max to 1 and
            // everything else to values in between.
            var normalize = mlContext.Transforms.NormalizeMinMax("Features",
                fixZero: false);

            // Normalize rows by finding min and max values in each row slot, but
            // make sure zero values remain zero after normalization. Helps
            // preserve sparsity. That is, to help maintain very little non-zero elements.
            var normalizeFixZero = mlContext.Transforms.NormalizeMinMax("Features",
                fixZero: true);

            // Now we can transform the data and look at the output to confirm the
            // behavior of the estimator. This operation doesn't actually evaluate
            // data until we read the data below.
            var normalizeTransform = normalize.Fit(data);
            var transformedData = normalizeTransform.Transform(data);
            var normalizeFixZeroTransform = normalizeFixZero.Fit(data);
            var fixZeroData = normalizeFixZeroTransform.Transform(data);
            var column = transformedData.GetColumn<float[]>("Features").ToArray();
            foreach (var row in column)
                Console.WriteLine(string.Join(", ", row.Select(x => x.ToString(
                    "f4"))));
            // Expected output:
            //  0.6667, 0.6667, 1.0000, 0.0000
            //  1.0000, 1.0000, 0.7500, 0.0000
            //  0.3333, 0.3333, 0.5000, 0.0000
            //  0.0000, 0.0000, 0.0000, 1.0000

            var columnFixZero = fixZeroData.GetColumn<float[]>("Features")
                .ToArray();

            foreach (var row in columnFixZero)
                Console.WriteLine(string.Join(", ", row.Select(x => x.ToString(
                    "f4"))));
            // Expected output:
            //  0.5000, 0.5000, 1.0000, 0.0000
            //  1.0000, 1.0000, 0.6667, 0.0000
            //  0.0000, 0.0000, 0.3333, 0.0000
            // -0.5000,-0.5000,-0.3333, 1.0000

            // Get transformation parameters. Since we work with only one
            // column we need to pass 0 as parameter for
            // GetNormalizerModelParameters. If we have multiple columns
            // transformations we need to pass index of InputOutputColumnPair.
            var transformParams = normalizeTransform.GetNormalizerModelParameters(0)
                as AffineNormalizerModelParameters<ImmutableArray<float>>;

            Console.WriteLine($"The 1-index value in resulting array would be " +
                $"produced by:");

            Console.WriteLine(" y = (x - (" + (transformParams.Offset.Length == 0 ?
                0 : transformParams.Offset[1]) + ")) * " + transformParams
                .Scale[1]);
            // Expected output:
            //  The 1-index value in resulting array would be produce by: 
            //  y = (x - (-1)) * 0.3333333
        }

        private class DataPoint
        {
            [VectorType(4)]
            public float[] Features { get; set; }
        }
    }
}
