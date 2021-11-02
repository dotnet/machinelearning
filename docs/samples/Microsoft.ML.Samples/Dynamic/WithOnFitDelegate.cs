using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using static Microsoft.ML.Transforms.NormalizingTransformer;

namespace Samples.Dynamic
{
    public class WithOnFitDelegate
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();
            var samples = new List<DataPoint>()
            {
                new DataPoint(){ Features = new float[4] { 8, 1, 3, 0},
                    Label = true },

                new DataPoint(){ Features = new float[4] { 6, 2, 2, 0},
                    Label = true },

                new DataPoint(){ Features = new float[4] { 4, 0, 1, 0},
                    Label = false },

                new DataPoint(){ Features = new float[4] { 2,-1,-1, 1},
                    Label = false }

            };
            // Convert training data to IDataView, the general data type used in
            // ML.NET.
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // Create a pipeline to normalize the features and train a binary
            // classifier. We use WithOnFitDelegate for the intermediate binning
            // normalization step, so that we can inspect the properties of the
            // normalizer after fitting.
            NormalizingTransformer binningTransformer = null;
            var pipeline =
                mlContext.Transforms
                .NormalizeBinning("Features", maximumBinCount: 3)
                .WithOnFitDelegate(
                fittedTransformer => binningTransformer = fittedTransformer)
                .Append(mlContext.BinaryClassification.Trainers
                .LbfgsLogisticRegression());

            Console.WriteLine(binningTransformer == null);
            // Expected Output:
            //   True

            var model = pipeline.Fit(data);

            // During fitting binningTransformer will get assigned a new value
            Console.WriteLine(binningTransformer == null);
            // Expected Output:
            //   False

            // Inspect some of the properties of the binning transformer
            var binningParam = binningTransformer.GetNormalizerModelParameters(0) as
                BinNormalizerModelParameters<ImmutableArray<float>>;

            for (int i = 0; i < binningParam.UpperBounds.Length; i++)
            {
                var upperBounds = string.Join(", ", binningParam.UpperBounds[i]);
                Console.WriteLine(
                    $"Bin {i}: Density = {binningParam.Density[i]}, " +
                    $"Upper-bounds = {upperBounds}");

            }
            // Expected output:
            //   Bin 0: Density = 2, Upper-bounds = 3, 7, Infinity
            //   Bin 1: Density = 2, Upper-bounds = -0.5, 1.5, Infinity
            //   Bin 2: Density = 2, Upper-bounds = 0, 2.5, Infinity
            //   Bin 3: Density = 1, Upper-bounds = 0.5, Infinity
        }

        private class DataPoint
        {
            [VectorType(4)]
            public float[] Features { get; set; }
            public bool Label { get; set; }
        }
    }
}

