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
    /// Base class for linkage alogirthm
    /// Finds best matching clusters.
    /// </summary>
    internal abstract class LinkageAlgorithmBase : ILinkageAlgorithm
    {
        /// <inheritdoc />
        public IList<DataSimilarity> Similarities { get; set; }

        /// <inheritdoc />
        public IDictionary<int, ClusterInfo> Clusters { get; set; }

        /// <inheritdoc />
        public abstract void CalculateSimilarity(int clusterIdx1, int clusterIdx2);

        /// <inheritdoc />
        public int GetBestMatch(Dictionary<int, float> similarity)
        {
            var clusterIdx2 = 0;
            var minDissimilarity = double.MaxValue;

            var clusterIndices = Clusters.Keys.ToList();
            for (int i = 0; i < clusterIndices.Count; i++)
            {
                foreach (var kv in this.Clusters[i].Similarity)
                {
                    if (kv.Value < minDissimilarity)
                    {
                        minDissimilarity = kv.Value;
                        clusterIdx2 = kv.Key;
                    }
                }
            }
            return clusterIdx2;
        }

        /// <inheritdoc />
        public void UpdateCluster(out int clusterIdx1, out int clusterIdx2)
        {
            clusterIdx1 = clusterIdx2 = 0;

            // for each pair of clusters that still exist
            var minDissimilarity = double.MaxValue;

            var clusterIndices = this.Clusters.Keys.ToList();
            for (int i = 0; i < clusterIndices.Count; i++)
            {
                var j = GetBestMatch(this.Clusters[i].Similarity);
                if (this.Clusters[i].Similarity[j] < minDissimilarity)
                {
                    minDissimilarity = this.Clusters[i].Similarity[j];
                    clusterIdx1 = i;
                    clusterIdx2 = j;
                }
            }

            if (this.Clusters[clusterIdx1].Members.Count < this.Clusters[clusterIdx2].Members.Count)
            {
                var temp = clusterIdx1;
                clusterIdx1 = clusterIdx2;
                clusterIdx2 = temp;
            }

            this.Clusters[clusterIdx1].Members.AddRange(this.Clusters[clusterIdx2].Members);
            this.CalculateSimilarity(clusterIdx1, clusterIdx2);

            this.Clusters.Remove(clusterIdx2);
            foreach (var kv in this.Clusters)
            {
                kv.Value.Similarity.Remove(clusterIdx2);
                kv.Value.Similarity[clusterIdx1] = this.Clusters[clusterIdx1].Similarity[kv.Key];
            }
        }

    }
}
