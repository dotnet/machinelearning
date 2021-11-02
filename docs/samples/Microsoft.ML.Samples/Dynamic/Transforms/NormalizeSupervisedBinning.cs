using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.Transforms.NormalizingTransformer;

namespace Samples.Dynamic
{
    public class NormalizeSupervisedBinning
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();
            var samples = new List<DataPoint>()
            {
                new DataPoint(){ Features = new float[4] { 8, 1, 3, 0},
                    Bin ="Bin1" },

                new DataPoint(){ Features = new float[4] { 6, 2, 2, 1},
                    Bin ="Bin2" },

                new DataPoint(){ Features = new float[4] { 5, 3, 0, 2},
                    Bin ="Bin2" },

                new DataPoint(){ Features = new float[4] { 4,-8, 1, 3},
                    Bin ="Bin3" },

                new DataPoint(){ Features = new float[4] { 2,-5,-1, 4},
                    Bin ="Bin3" }
            };
            // Convert training data to IDataView, the general data type used in
            // ML.NET.
            var data = mlContext.Data.LoadFromEnumerable(samples);
            // Let's transform "Bin" column from string to key.
            data = mlContext.Transforms.Conversion.MapValueToKey("Bin").Fit(data)
                .Transform(data);
            // NormalizeSupervisedBinning normalizes the data by constructing bins
            // based on correlation with the label column and produce output based
            // on to which bin original value belong.
            var normalize = mlContext.Transforms.NormalizeSupervisedBinning(
                "Features", labelColumnName: "Bin", mininimumExamplesPerBin: 1,
                fixZero: false);

            // NormalizeSupervisedBinning normalizes the data by constructing bins
            // based on correlation with the label column and produce output based
            // on to which bin original value belong but make sure zero values would
            // remain zero after normalization. Helps preserve sparsity.
            var normalizeFixZero = mlContext.Transforms.NormalizeSupervisedBinning(
                "Features", labelColumnName: "Bin", mininimumExamplesPerBin: 1,
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
            //  1.0000, 0.5000, 1.0000, 0.0000
            //  0.5000, 1.0000, 0.0000, 0.5000
            //  0.5000, 1.0000, 0.0000, 0.5000
            //  0.0000, 0.0000, 0.0000, 1.0000
            //  0.0000, 0.0000, 0.0000, 1.0000

            var columnFixZero = fixZeroData.GetColumn<float[]>("Features")
                .ToArray();

            foreach (var row in columnFixZero)
                Console.WriteLine(string.Join(", ", row.Select(x => x.ToString(
                    "f4"))));
            // Expected output:
            //  1.0000, 0.0000, 1.0000, 0.0000
            //  0.5000, 0.5000, 0.0000, 0.5000
            //  0.5000, 0.5000, 0.0000, 0.5000
            //  0.0000,-0.5000, 0.0000, 1.0000
            //  0.0000,-0.5000, 0.0000, 1.0000

            // Let's get transformation parameters. Since we work with only one
            // column we need to pass 0 as parameter for
            // GetNormalizerModelParameters.
            // If we have multiple columns transformations we need to pass index of
            // InputOutputColumnPair.
            var transformParams = normalizeTransform.GetNormalizerModelParameters(0)
                as BinNormalizerModelParameters<ImmutableArray<float>>;

            Console.WriteLine($"The 1-index value in resulting array would be " +
                $"produce by:");

            Console.WriteLine("y = (Index(x) / " + transformParams.Density[0] +
                ") - " + (transformParams.Offset.Length == 0 ? 0 : transformParams
                .Offset[0]));

            Console.WriteLine("Where Index(x) is the index of the bin to which " +
                "x belongs");

            Console.WriteLine("Bins upper borders are: " + string.Join(" ",
                transformParams.UpperBounds[0]));
            // Expected output:
            //  The 1-index value in resulting array would be produce by:
            //  y = (Index(x) / 2) - 0
            //  Where Index(x) is the index of the bin to which x belongs
            //  Bins upper bounds are: 4.5 7 ∞

            var fixZeroParams = normalizeFixZeroTransform
                .GetNormalizerModelParameters(0) as BinNormalizerModelParameters<
                ImmutableArray<float>>;

            Console.WriteLine($"The 1-index value in resulting array would be " +
                $"produce by:");

            Console.WriteLine(" y = (Index(x) / " + fixZeroParams.Density[1] +
                ") - " + (fixZeroParams.Offset.Length == 0 ? 0 : fixZeroParams
                .Offset[1]));

            Console.WriteLine("Where Index(x) is the index of the bin to which x " +
                "belongs");

            Console.WriteLine("Bins upper borders are: " + string.Join(" ",
                fixZeroParams.UpperBounds[1]));
            // Expected output:
            //  The 1-index value in resulting array would be produce by:
            //  y = (Index(x) / 2) - 0.5
            //  Where Index(x) is the index of the bin to which x belongs
            //  Bins upper bounds are: -2 1.5 ∞
        }

        private class DataPoint
        {
            [VectorType(4)]
            public float[] Features { get; set; }

            public string Bin { get; set; }
        }
    }
}
