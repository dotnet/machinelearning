// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Xunit;

namespace Microsoft.ML.Scenarios
{
    public partial class ScenariosTests
    {
        public class ClusteringPrediction
        {
            [ColumnName("PredictedLabel")]
            public uint SelectedClusterId;
            [ColumnName("Score")]
            public float[] Distance;
        }

        public class ClusteringData
        {
            [ColumnName("Features")]
            [VectorType(2)]
            public float[] Points;
        }

        [Fact]
        public void PredictClusters()
        {
            int n = 1000;
            int k = 4;
            var rand = new Random(1);
            var clusters = new ClusteringData[k];
            var data = new ClusteringData[n];
            for (int i = 0; i < k; i++)
            {
                //pick clusters as points on circle with angle to axis X equal to 360*i/k
                clusters[i] = new ClusteringData { Points = new float[2] { (float)Math.Cos(Math.PI * i * 2 / k), (float)Math.Sin(Math.PI * i * 2 / k) } };
            }
            // create data points by randomly picking cluster and shifting point slightly away from it.
            for (int i = 0; i < n; i++)
            {
                var index = rand.Next(0, k);
                var shift = (rand.NextDouble() - 0.5) / 10;
                data[i] = new ClusteringData
                {
                    Points = new float[2]
                    {
                        (float)(clusters[index].Points[0] + shift),
                        (float)(clusters[index].Points[1] + shift)
                    }
                };
            }

            var mlContext = new MLContext(seed: 1, conc: 1);

            // Turn the data into the ML.NET data view.
            // We can use CreateDataView or CreateStreamingDataView, depending on whether 'churnData' is an IList, 
            // or merely an IEnumerable.
            var trainData = mlContext.Data.ReadFromEnumerable(data);
            var testData = mlContext.Data.ReadFromEnumerable(clusters);

            // Create Estimator
            var pipe = mlContext.Clustering.Trainers.KMeans("Features", clustersCount: k);

            // Train the pipeline
            var trainedModel = pipe.Fit(trainData);

            // Validate that initial points we pick up as centers of cluster during data generation belong to different clusters.
            var labels = new HashSet<uint>();
            var predictFunction = trainedModel.CreatePredictionEngine<ClusteringData, ClusteringPrediction>(mlContext);

            for (int i = 0; i < k; i++)
            {
                var scores = predictFunction.Predict(clusters[i]);
                Assert.True(!labels.Contains(scores.SelectedClusterId));
                labels.Add(scores.SelectedClusterId);
            }

            // Evaluate the trained pipeline
            var predicted = trainedModel.Transform(testData);
            var metrics = mlContext.Clustering.Evaluate(predicted);

            //Label is not specified, so NMI would be equal to NaN
            Assert.Equal(metrics.Nmi, double.NaN);
            //Calculate dbi is false by default so Dbi would be 0
            Assert.Equal(metrics.Dbi, (double)0.0);
            Assert.Equal(metrics.AvgMinScore, (double)0.0, 5);
        }
    }
}
