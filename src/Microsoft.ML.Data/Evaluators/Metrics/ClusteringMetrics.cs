// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// The metrics generated after evaluating the clustering predictions.
    /// </summary>
    public sealed class ClusteringMetrics
    {
        /// <summary>
        /// Normalized Mutual Information
        /// NMI is a measure of the mutual dependence of the variables.
        /// <a href="http://en.wikipedia.org/wiki/Mutual_information#Normalized_variants">Normalized variants</a> work on data that already has cluster labels.
        /// Its value ranged from 0 to 1, where higher numbers are better.
        /// </summary>
        public double Nmi { get; }

        /// <summary>
        /// Average Score. For the K-Means algorithm, the &apos;score&apos; is the distance from the centroid to the example.
        /// The average score is, therefore, a measure of proximity of the examples to cluster centroids.
        /// In other words, it&apos;s the &apos;cluster tightness&apos; measure.
        /// Note however, that this metric will only decrease if the number of clusters is increased,
        /// and in the extreme case (where each distinct example is its own cluster) it will be equal to zero.
        /// </summary>
        public double AvgMinScore { get; }

        /// <summary>
        /// <a href="https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index">Davies-Bouldin Index</a>
        /// DBI is a measure of the how much scatter is in the cluster and the cluster separation.
        /// </summary>
        public double Dbi { get; }

        internal ClusteringMetrics(IExceptionContext ectx, DataViewRow overallResult, bool calculateDbi)
        {
            double Fetch(string name) => RowCursorUtils.Fetch<double>(ectx, overallResult, name);

            Nmi = Fetch(ClusteringEvaluator.Nmi);
            AvgMinScore = Fetch(ClusteringEvaluator.AvgMinScore);

            if (calculateDbi)
                Dbi = Fetch(ClusteringEvaluator.Dbi);
        }

        internal ClusteringMetrics(double nmi, double avgMinScore, double dbi)
        {
            Nmi = nmi;
            AvgMinScore = avgMinScore;
            Dbi = dbi;
        }
    }
}