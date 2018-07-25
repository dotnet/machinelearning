// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;

namespace Microsoft.ML.Models
{
    /// <summary>
    /// This class contains the overall metrics computed by cluster evaluators.
    /// </summary>
    public sealed class ClusterMetrics
    {
        private ClusterMetrics()
        {
        }

        internal static List<ClusterMetrics> FromOverallMetrics(IHostEnvironment env, IDataView overallMetrics)
        {
            Contracts.AssertValue(env);
            env.AssertValue(overallMetrics);

            var metricsEnumerable = overallMetrics.AsEnumerable<SerializationClass>(env, true, ignoreMissingColumns: true);
            if (!metricsEnumerable.GetEnumerator().MoveNext())
            {
                throw env.Except("The overall ClusteringMetrics didn't have any rows.");
            }

            var metrics = new List<ClusterMetrics>();
            foreach (var metric in metricsEnumerable)
            {
                metrics.Add(new ClusterMetrics()
                {
                    AvgMinScore = metric.AvgMinScore,
                    Nmi = metric.Nmi,
                    Dbi = metric.Dbi,
                });
            }

            return metrics;
        }

        /// <summary>
        /// Davies-Bouldin Index.
        /// </summary>
        /// <remarks>
        /// DBI is a measure of the how much scatter is in the cluster and the cluster separation.
        /// </remarks>
        public double Dbi { get; private set; }

        /// <summary>
        /// Normalized Mutual Information
        /// </summary>
        /// <remarks>
        /// NMI is a measure of the mutual dependence between the true and predicted cluster labels for instances in the dataset.
        /// NMI ranges between 0 and 1 where "0" indicates clustering is random and "1" indicates clustering is perfect w.r.t true labels.
        /// </remarks>
        public double Nmi { get; private set; }

        /// <summary>
        /// Average minimum score.
        /// </summary>
        /// <remarks>
        /// AvgMinScore is the average squared-distance of examples from the respective cluster centroids.
        /// It is defined as
        /// AvgMinScore  = (1/m) * sum ((xi - c(xi))^2)
        /// where m is the number of instances in the dataset.
        /// xi is the i'th instance and c(xi) is the centriod of the predicted cluster for xi.
        /// </remarks>
        public double AvgMinScore { get; private set; }

        /// <summary>
        /// This class contains the public fields necessary to deserialize from IDataView.
        /// </summary>
        private sealed class SerializationClass
        {
#pragma warning disable 649 // never assigned
            [ColumnName(Runtime.Data.ClusteringEvaluator.Dbi)]
            public Double Dbi;

            [ColumnName(Runtime.Data.ClusteringEvaluator.Nmi)]
            public Double Nmi;

            [ColumnName(Runtime.Data.ClusteringEvaluator.AvgMinScore)]
            public Double AvgMinScore;

#pragma warning restore 649 // never assigned
        }
    }
}
