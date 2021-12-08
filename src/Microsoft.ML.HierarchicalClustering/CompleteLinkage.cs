// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Trainers;

namespace Microsoft.ML
{
    /// <summary>
    ///     Implements the maximum or complete-linkage clustering method, i.e., returning the maximum value of all pairwise
    ///     distances between the elements in each cluster. The method is also known as farthest neighbor clustering.
    /// </summary>
    /// <remarks>
    ///     Complete linkage clustering avoids a drawback of <see cref="SingleLinkage" /> - the
    ///     so-called chaining phenomenon, where clusters formed via single linkage clustering may be forced together due to
    ///     single elements being close to each other, even though many of the elements in each cluster may be very distant to
    ///     each other. Complete linkage tends to find compact clusters of approximately equal diameter (
    ///     <see href="https://en.wikipedia.org/wiki/Complete-linkage_clustering" />).
    ///     However, complete-link clustering suffers from a different problem. It pays too much attention to outliers, points
    ///     that do not fit well into the global structure of the cluster (
    ///     <see href="https://nlp.stanford.edu/IR-book/html/htmledition/single-link-and-complete-link-clustering-1.html" />).
    /// </remarks>
    internal class CompleteLinkage : LinkageAlgorithmBase
    {
        public CompleteLinkage(IDictionary<int, ClusterInfo> clusters)
        {
            this.Clusters = clusters;
        }

        /// <inheritdoc />
        public override void CalculateSimilarity(int clusterIdx1, int clusterIdx2)
        {
            foreach (var kv in this.Clusters[clusterIdx1].Similarity)
            {
                this.Clusters[clusterIdx1].Similarity[kv.Key] =
                    Math.Max(this.Clusters[clusterIdx1].Similarity[kv.Key],
                    this.Clusters[clusterIdx2].Similarity[kv.Key]);
            }
        }

    }
}
