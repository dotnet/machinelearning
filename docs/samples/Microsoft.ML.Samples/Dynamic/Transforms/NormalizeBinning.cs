using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.Transforms.NormalizingTransformer;

namespace Samples.Dynamic
{
    public class NormalizeBinning
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();
            var samples = new List<DataPoint>()
            {
                new DataPoint(){ Features = new float[4] { 8, 1, 3, 0} },
                new DataPoint(){ Features = new float[4] { 6, 2, 2, 0} },
                new DataPoint(){ Features = new float[4] { 4, 0, 1, 0} },
                new DataPoint(){ Features = new float[4] { 2,-1,-1, 1} }
            };
            // Convert training data to IDataView, the general data type used in
            // ML.NET.
            var data = mlContext.Data.LoadFromEnumerable(samples);
            // NormalizeBinning normalizes the data by constructing equidensity bins
            // and produce output based on 
            // to which bin the original value belongs.
            var normalize = mlContext.Transforms.NormalizeBinning("Features",
                maximumBinCount: 4, fixZero: false);

            // NormalizeBinning normalizes the data by constructing equidensity bins
            // and produce output based on to which bin original value belong but
            // make sure zero values would remain zero after normalization. Helps
            // preserve sparsity.
            var normalizeFixZero = mlContext.Transforms.NormalizeBinning("Features",
                maximumBinCount: 4, fixZero: true);

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
            //  1.0000, 0.6667, 1.0000, 0.0000
            //  0.6667, 1.0000, 0.6667, 0.0000
            //  0.3333, 0.3333, 0.3333, 0.0000
            //  0.0000, 0.0000, 0.0000, 1.0000

            var columnFixZero = fixZeroData.GetColumn<float[]>("Features")
                .ToArray();

            foreach (var row in columnFixZero)
                Console.WriteLine(string.Join(", ", row.Select(x => x.ToString(
                    "f4"))));
            // Expected output:
            //  1.0000, 0.3333, 1.0000, 0.0000
            //  0.6667, 0.6667, 0.6667, 0.0000
            //  0.3333, 0.0000, 0.3333, 0.0000
            //  0.0000, -0.3333, 0.0000, 1.0000

            // Let's get transformation parameters. Since we work with only one
            // column we need to pass 0 as parameter for
            // GetNormalizerModelParameters. If we have multiple columns
            // transformations we need to pass index of InputOutputColumnPair.
            var transformParams = normalizeTransform.GetNormalizerModelParameters(0)
                as BinNormalizerModelParameters<ImmutableArray<float>>;

            var density = transformParams.Density[0];
            var offset = (transformParams.Offset.Length == 0 ? 0 : transformParams
                .Offset[0]);

            Console.WriteLine($"The 0-index value in resulting array would be " +
                $"produce by: y = (Index(x) / {density}) - {offset}");

            Console.WriteLine("Where Index(x) is the index of the bin to which " +
                "x belongs");

            Console.WriteLine("Bins upper bounds are: " + string.Join(" ",
                transformParams.UpperBounds[0]));
            // Expected output:
            //  The 0-index value in resulting array would be produce by: y = (Index(x) / 3) - 0
            //  Where Index(x) is the index of the bin to which x belongs
            //  Bins upper bounds are: 3 5 7 ∞

            var fixZeroParams = (normalizeFixZeroTransform
                .GetNormalizerModelParameters(0) as BinNormalizerModelParameters<
                ImmutableArray<float>>);

            density = fixZeroParams.Density[1];
            offset = (fixZeroParams.Offset.Length == 0 ? 0 : fixZeroParams
                .Offset[1]);

            Console.WriteLine($"The 0-index value in resulting array would be " +
                $"produce by: y = (Index(x) / {density}) - {offset}");

            Console.WriteLine("Where Index(x) is the index of the bin to which x " +
                "belongs");

            Console.WriteLine("Bins upper bounds are: " + string.Join(" ",
                fixZeroParams.UpperBounds[1]));
            // Expected output:
            //  The 0-index value in resulting array would be produce by: y = (Index(x) / 3) - 0.3333333
            //  Where Index(x) is the index of the bin to which x belongs
            //  Bins upper bounds are: -0.5 0.5 1.5 ∞
        }

        private class DataPoint
        {
            [VectorType(4)]
            public float[] Features { get; set; }
        }
    }
}
