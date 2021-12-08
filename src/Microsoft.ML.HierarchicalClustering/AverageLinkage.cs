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
    ///     Implements the unweighted pair-group average method or UPGMA, i.e., returns the mean distance between the elements
    ///     in each cluster.
    /// </summary>
    /// <remarks>
    ///     Average linkage tries to strike a balance between <see cref="SingleLinkage" /> and
    ///     <see cref="CompleteLinkage" />. It uses average pairwise dissimilarity, so clusters tend to be
    ///     relatively compact and relatively far apart. However, it is not clear what properties the resulting clusters have
    ///     when we cut an average linkage tree at given distance. Single and complete linkage trees each had simple
    ///     interpretations [1].
    ///     References:
    ///     [1] - <see href="http://www.stat.cmu.edu/~ryantibs/datamining/lectures/05-clus2-marked.pdf" />.
    /// </remarks>
    internal class AverageLinkage : LinkageAlgorithmBase
    {
        public AverageLinkage(IDictionary<int, ClusterInfo> clusters)
        {
            this.Clusters = clusters;
        }

        /// <inheritdoc />
        public override void CalculateSimilarity(int clusterIdx1, int clusterIdx2)
        {
            var size1 = this.Clusters[clusterIdx1].Members.Count;
            var size2 = this.Clusters[clusterIdx2].Members.Count;
            var totalSize = size1 + size2;
            foreach (var kv in this.Clusters[clusterIdx1].Similarity)
            {
                this.Clusters[clusterIdx1].Similarity[kv.Key] =
                    (size1 * this.Clusters[clusterIdx1].Similarity[kv.Key] +
                    size2 * this.Clusters[clusterIdx2].Similarity[kv.Key]) / totalSize;
            }
        }

    }
}
