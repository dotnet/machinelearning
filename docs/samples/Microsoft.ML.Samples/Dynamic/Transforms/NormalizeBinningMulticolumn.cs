using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.Transforms.NormalizingTransformer;

namespace Samples.Dynamic
{
    public class NormalizeBinningMulticolumn
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();
            var samples = new List<DataPoint>()
            {
                new DataPoint(){ Features = new float[4] { 8, 1, 3, 0},
                    Features2 = 1 },

                new DataPoint(){ Features = new float[4] { 6, 2, 2, 0},
                    Features2 = 4 },

                new DataPoint(){ Features = new float[4] { 4, 0, 1, 0},
                    Features2 = 1 },

                new DataPoint(){ Features = new float[4] { 2,-1,-1, 1},
                    Features2 = 2 }
            };
            // Convert training data to IDataView, the general data type used in
            // ML.NET.
            var data = mlContext.Data.LoadFromEnumerable(samples);
            // NormalizeBinning normalizes the data by constructing equidensity bins
            // and produce output based on to which bin the original value belongs.
            var normalize = mlContext.Transforms.NormalizeBinning(new[]{
                new InputOutputColumnPair("Features"),
                new InputOutputColumnPair("Features2"),
                },
                maximumBinCount: 4, fixZero: false);

            // Now we can transform the data and look at the output to confirm the
            // behavior of the estimator. This operation doesn't actually evaluate
            // data until we read the data below.
            var normalizeTransform = normalize.Fit(data);
            var transformedData = normalizeTransform.Transform(data);
            var column = transformedData.GetColumn<float[]>("Features").ToArray();
            var column2 = transformedData.GetColumn<float>("Features2").ToArray();

            for (int i = 0; i < column.Length; i++)
                Console.WriteLine(string.Join(", ", column[i].Select(x => x
                .ToString("f4"))) + "\t\t" + column2[i]);
            // Expected output:
            //
            //  Features                            Feature2
            //  1.0000, 0.6667, 1.0000, 0.0000          0
            //  0.6667, 1.0000, 0.6667, 0.0000          1
            //  0.3333, 0.3333, 0.3333, 0.0000          0
            //  0.0000, 0.0000, 0.0000, 1.0000          0.5
        }

        private class DataPoint
        {
            [VectorType(4)]
            public float[] Features { get; set; }

            public float Features2 { get; set; }
        }
    }
}
