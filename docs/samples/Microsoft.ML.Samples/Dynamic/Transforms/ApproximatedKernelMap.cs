using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Samples.Dynamic
{
    public static class ApproximatedKernelMap
    {
        // Transform feature vector to another non-linear space. See
        // https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf.
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();
            var samples = new List<DataPoint>()
            {
                new DataPoint(){ Features = new float[7] { 1, 1, 0, 0, 1, 0, 1} },
                new DataPoint(){ Features = new float[7] { 0, 0, 1, 0, 0, 1, 1} },
                new DataPoint(){ Features = new float[7] {-1, 1, 0,-1,-1, 0,-1} },
                new DataPoint(){ Features = new float[7] { 0,-1, 0, 1, 0,-1,-1} }
            };
            // Convert training data to IDataView, the general data type used in
            // ML.NET.
            var data = mlContext.Data.LoadFromEnumerable(samples);
            // ApproximatedKernel map takes data and maps it's to a random
            // low -dimensional space.
            var approximation = mlContext.Transforms.ApproximatedKernelMap(
                "Features", rank: 4, generator: new GaussianKernel(gamma: 0.7f),
                seed: 1);

            // Now we can transform the data and look at the output to confirm the
            // behavior of the estimator. This operation doesn't actually evaluate
            // data until we read the data below.
            var tansformer = approximation.Fit(data);
            var transformedData = tansformer.Transform(data);

            var column = transformedData.GetColumn<float[]>("Features").ToArray();
            foreach (var row in column)
                Console.WriteLine(string.Join(", ", row.Select(x => x.ToString(
                    "f4"))));

            // Expected output:
            // -0.0119, 0.5867, 0.4942,  0.7041
            //  0.4720, 0.5639, 0.4346,  0.2671
            // -0.2243, 0.7071, 0.7053, -0.1681
            //  0.0846, 0.5836, 0.6575,  0.0581
        }

        private class DataPoint
        {
            [VectorType(7)]
            public float[] Features { get; set; }
        }

    }
}
