// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

namespace Microsoft.ML
{
    /// <summary>
    /// Interface for linkage alogirthm
    /// </summary>
    internal interface ILinkageAlgorithm
    {
        /// <summary>
        /// Training data along with squared l2 norm and euclidean distance with all other training data points
        /// </summary>
        IList<DataSimilarity> Similarities { get; set; }

        /// <summary>
        /// Cluster information. List of members of cluster. Similarity with other clusters.
        /// </summary>
        IDictionary<int, ClusterInfo> Clusters { get; set; }


        /// <summary>
        /// Calculate similarity scores for the cluster formed by merging clusters 
        /// represented by clusterIdx1 and clusterIdx2
        /// </summary>
        void CalculateSimilarity(int clusterIdx1, int clusterIdx2);

        /// <summary>
        /// Find the best matching cluster with the given cluster
        /// </summary>
        /// <param name="similarity">A dictionary of cluster index and similarity score</param>
        /// <returns>Best cluster index</returns>
        int GetBestMatch(Dictionary<int, float> similarity);

        /// <summary>
        /// Find the two clusters that needs to be merged in current iteration.
        /// Combines clusters represented by clusterIdx1 and clusterIdx2.
        /// Updates the list of Clusters with the new cluster.
        /// Updates similarity scores between all the clusters.
        /// </summary>
        void UpdateCluster(out int clusterIdx1, out int clusterIdx2);
    }
}
