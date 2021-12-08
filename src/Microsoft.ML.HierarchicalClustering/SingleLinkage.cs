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
    ///     Implements the minimum or single-linkage clustering method, i.e., returns the minimum value of all pairwise
    ///     distances between the elements in each cluster. The method is also known as nearest neighbor clustering.
    /// </summary>
    /// <remarks>
    ///     A drawback of this method is that it tends to produce long thin clusters in which nearby elements of the same
    ///     cluster have small distances, but elements at opposite ends of a cluster may be much farther from each other than
    ///     two elements of other clusters [1].
    ///     Since the merge criterion is strictly local, a chain of points can be extended for long distances without regard to
    ///     the overall shape of the emerging cluster. This effect is called chaining [2].
    ///     References:
    ///     [1] - <see href="https://en.wikipedia.org/wiki/Single-linkage_clustering" />.
    ///     [2] -
    ///     <see href="https://nlp.stanford.edu/IR-book/html/htmledition/single-link-and-complete-link-clustering-1.html" />
    /// </remarks>
    internal class SingleLinkage : LinkageAlgorithmBase
    {
        public SingleLinkage(IDictionary<int, ClusterInfo> clusters)
        {
            this.Clusters = clusters;
        }

        /// <inheritdoc />
        public override void CalculateSimilarity(int clusterIdx1, int clusterIdx2)
        {
            foreach (var kv in this.Clusters[clusterIdx1].Similarity)
            {
                this.Clusters[clusterIdx1].Similarity[kv.Key] =
                    Math.Min(this.Clusters[clusterIdx1].Similarity[kv.Key],
                    this.Clusters[clusterIdx2].Similarity[kv.Key]);
            }
        }

    }
}
