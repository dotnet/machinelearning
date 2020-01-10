using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.Transforms.NormalizingTransformer;

namespace Samples.Dynamic
{
    class NormalizeMinMaxMulticolumn
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();
            var samples = new List<DataPoint>()
            {
                new DataPoint()
                { 
                    Features = new float[4] { 1, 1, 3, 0 },
                    Features2 = new float[3] { 1, 2, 3 }
                },
                new DataPoint()
                { 
                    Features = new float[4] { 2, 2, 2, 0 },
                    Features2 = new float[3] { 3, 4, 5 }
                },
                new DataPoint()
                { 
                    Features = new float[4] { 0, 0, 1, 0 },
                    Features2 = new float[3] { 6, 7, 8 }
                },
                new DataPoint()
                { 
                    Features = new float[4] {-1,-1,-1, 1 },
                    Features2 = new float[3] { 9, 0, 4 }
                }
            };

            // Convert training data to IDataView, the general data type used in
            // ML.NET.
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // NormalizeMinMax normalize rows by finding min and max values in each
            // row slot and setting projection of min value to 0 and max to 1 and
            // everything else to values in between.
            var normalize = mlContext.Transforms.NormalizeMinMax(new[] {
                    new InputOutputColumnPair("Features"),
                    new InputOutputColumnPair("Features2")
                },
                fixZero: false);

            // Normalize rows by finding min and max values in each row slot, but
            // make sure zero values would remain zero after normalization. Helps
            // preserve sparsity.
            var normalizeFixZero = mlContext.Transforms.NormalizeMinMax(new[] {
                    new InputOutputColumnPair("Features"),
                    new InputOutputColumnPair("Features2")
                },
                fixZero: true);

            // Now we can transform the data and look at the output to confirm the
            // behavior of the estimator. This operation doesn't actually evaluate
            // data until we read the data below.
            var normalizeTransform = normalize.Fit(data);
            var transformedData = normalizeTransform.Transform(data);
            var normalizeFixZeroTransform = normalizeFixZero.Fit(data);
            var fixZeroData = normalizeFixZeroTransform.Transform(data);
            var column = transformedData.GetColumn<float[]>("Features").ToArray();
            var column2 = transformedData.GetColumn<float[]>("Features2").ToArray();

            for (int i = 0; i < column.Length; i++)
                Console.WriteLine(string.Join(", ", column[i].Select(x => x
                .ToString("f4"))) + "\t\t" + 
                string.Join(", ", column[i].Select(x => x.ToString("f4"))));

            // Expected output:
            // Features                               Features2  
            //0.6667, 0.6667, 1.0000, 0.0000          0.6667, 0.6667, 1.0000, 0.0000
            //1.0000, 1.0000, 0.7500, 0.0000          1.0000, 1.0000, 0.7500, 0.0000
            //0.3333, 0.3333, 0.5000, 0.0000          0.3333, 0.3333, 0.5000, 0.0000
            //0.0000, 0.0000, 0.0000, 1.0000          0.0000, 0.0000, 0.0000, 1.0000
            var columnFixZero = fixZeroData.GetColumn<float[]>("Features")
                .ToArray();

            foreach (var row in columnFixZero)
                Console.WriteLine(string.Join(", ", row.Select(x => x.ToString(
                    "f4"))));

            // Expected output:

            // Let's get transformation parameters. Since we work with only one
            // column we need to pass 0 as parameter for
            // GetNormalizerModelParameters. If we have multiple columns
            // transformations we need to pass index of InputOutputColumnPair.
            var transformParams = normalizeTransform.GetNormalizerModelParameters(0)
                as AffineNormalizerModelParameters<ImmutableArray<float>>;

            Console.WriteLine($"The 1-index value in resulting array would be " +
                $"produce by:");

            Console.WriteLine(" y = (x - (" + (transformParams.Offset.Length == 0 ?
                0 : transformParams.Offset[1]) + ")) * " + transformParams
                .Scale[1]);

            // Expected output:
            //  The 1-index value in resulting array would be produce by: 
        }

        private class DataPoint
        {
            [VectorType(4)]
            public float[] Features { get; set; }

            [VectorType(3)]
            public float[] Features2 { get; set; }
        }
    }
}
