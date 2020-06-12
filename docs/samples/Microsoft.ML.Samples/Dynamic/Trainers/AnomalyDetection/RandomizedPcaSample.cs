﻿using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic.Trainers.AnomalyDetection
{
    public static class RandomizedPcaSample
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for except
            // ion tracking and logging, as a catalog of available operations and as
            // the source of randomness. Setting the seed to a fixed number in this
            // example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Training data.
            var samples = new List<DataPoint>()
            {
                new DataPoint(){ Features = new float[3] {0, 2, 1} },
                new DataPoint(){ Features = new float[3] {0, 2, 1} },
                new DataPoint(){ Features = new float[3] {0, 2, 1} },
                new DataPoint(){ Features = new float[3] {0, 1, 2} },
                new DataPoint(){ Features = new float[3] {0, 2, 1} },
                new DataPoint(){ Features = new float[3] {2, 0, 0} }
            };

            // Convert the List<DataPoint> to IDataView, a consumable format to
            // ML.NET functions.
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // Create an anomaly detector. Its underlying algorithm is randomized
            // PCA.
            var pipeline = mlContext.AnomalyDetection.Trainers.RandomizedPca(
                featureColumnName: nameof(DataPoint.Features), rank: 1,
                    ensureZeroMean: false);

            // Train the anomaly detector.
            var model = pipeline.Fit(data);

            // Apply the trained model on the training data.
            var transformed = model.Transform(data);

            // Read ML.NET predictions into IEnumerable<Result>.
            var results = mlContext.Data.CreateEnumerable<Result>(transformed,
                reuseRowObject: false).ToList();

            // Let's go through all predictions.
            for (int i = 0; i < samples.Count; ++i)
            {
                // The i-th example's prediction result.
                var result = results[i];

                // The i-th example's feature vector in text format.
                var featuresInText = string.Join(',', samples[i].Features);

                if (result.PredictedLabel)
                    // The i-th sample is predicted as an outlier.
                    Console.WriteLine("The {0}-th example with features [{1}] is " +
                        "an outlier with a score of being inlier {2}", i,
                            featuresInText, result.Score);
                else
                    // The i-th sample is predicted as an inlier.
                    Console.WriteLine("The {0}-th example with features [{1}] is " +
                        "an inlier with a score of being inlier {2}", i,
                        featuresInText, result.Score);
            }
            // Lines printed out should be
            // The 0 - th example with features[0, 2, 1] is an inlier with a score of being outlier 0.1101028
            // The 1 - th example with features[0, 2, 1] is an inlier with a score of being outlier 0.1101028
            // The 2 - th example with features[0, 2, 1] is an inlier with a score of being outlier 0.1101028
            // The 3 - th example with features[0, 1, 2] is an outlier with a score of being outlier 0.5082728
            // The 4 - th example with features[0, 2, 1] is an inlier with a score of being outlier 0.1101028
            // The 5 - th example with features[2, 0, 0] is an outlier with a score of being outlier 1
        }

        // Example with 3 feature values. A training data set is a collection of
        // such examples.
        private class DataPoint
        {
            [VectorType(3)]
            public float[] Features { get; set; }
        }

        // Class used to capture prediction of DataPoint.
        private class Result
        {
            // Outlier gets true while inlier has false.
            public bool PredictedLabel { get; set; }
            // Inlier gets smaller score. Score is between 0 and 1.
            public float Score { get; set; }
        }
    }
}
