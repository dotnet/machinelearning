using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic.Trainers.AnomalyDetection
{
    public static class RandomizedPcaSample
    {
        /// <summary>
        /// Example with 3 feature values.
        /// </summary>
        private class DataPoint
        {
            [VectorType(3)]
            public float[] Features;
        }

        /// <summary>
        /// Class used to capture prediction of <see cref="DataPoint"/> in <see cref="Example"/>.
        /// </summary>
        // We disable this warning because complier doesn't realize those fields below are assigned somewhere.
#pragma warning disable 649
        private class Result
        {
            // Outlier gets false while inlier has true.
            public bool PredictedLabel;
            // Outlier gets smaller score.
            public float Score;
        }
#pragma warning restore 649

        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            // Setting the seed to a fixed number in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            var samples = new List<DataPoint>()
            {
                new DataPoint(){ Features= new float[3] {1, 0, 0} },
                new DataPoint(){ Features= new float[3] {0, 2, 1} },
                new DataPoint(){ Features= new float[3] {1, 2, 3} },
                new DataPoint(){ Features= new float[3] {0, 1, 0} },
                new DataPoint(){ Features= new float[3] {0, 2, 1} },
                new DataPoint(){ Features= new float[3] {-100, 50, -100} }
            };

            // Convert native C# class to IDataView, a consumble format to ML.NET functions.
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // Create an anomaly detector. Its underlying algorithm is randomized PCA.
            var pipeline = mlContext.AnomalyDetection.Trainers.RandomizedPca(featureColumnName: nameof(DataPoint.Features), rank: 1, center: false);

            // Train the anomaly detector.
            var model = pipeline.Fit(data);

            // Apply the trained model on the training data.
            var transformed = model.Transform(data);

            // Read ML.NET predictions into C# class.
            var results = mlContext.Data.CreateEnumerable<Result>(transformed, reuseRowObject: false).ToList();

            // Let's go through all predictions.
            for (int i = 0; i < samples.Count; ++i)
            {
                // The i-th example's prediction result.
                var result = results[i];

                // The i-th example's feature vector in text format.
                var featuresInText = string.Join(',', samples[i].Features);

                if (result.PredictedLabel)
                    // The i-th sample is predicted as an inlier.
                    Console.WriteLine("The {0}-th example with features [{1}] is an inlier with a score of being inlier {2}",
                        i, featuresInText, result.Score);
                else
                    // The i-th sample is predicted as an outlier.
                    Console.WriteLine("The {0}-th example with features [{1}] is an outlier with a score of being inlier {2}",
                        i, featuresInText, result.Score);
            }
        }
    }
}
