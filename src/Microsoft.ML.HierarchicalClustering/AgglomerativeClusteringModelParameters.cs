// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

[assembly: LoadableClass(typeof(AgglomerativeClusteringModelParameters), null, typeof(SignatureLoadModel),
    "AgglormerativeClustering predictor", AgglomerativeClusteringModelParameters.LoaderSignature)]

namespace Microsoft.ML.Trainers
{
    public sealed class AgglomerativeClusteringModelParameters :
        ModelParametersBase<VBuffer<float>>
    {
        internal const string LoaderSignature = "AgglormerativeClusteringPredictor";

        private protected override PredictionKind PredictionKind => PredictionKind.Clustering;

        private readonly List<int> _dendrogram;
        private readonly Dictionary<int, ClusterInfo> _clusters;

        /// <summary>
        /// Initialize predictor with a trained model.
        /// </summary>
        /// <param name="env">The host environment</param>
        /// <param name="clusters">Cluster</param>
        /// <param name="dendrogram"></param>
        /// <param name="copyIn">If true then the <paramref name="clusters"/> and <paramref name="dendrogram"/>  vectors will be subject to
        /// a deep copy, if false then this constructor will take ownership of the passed in centroid vectors.
        /// If false then the caller must take care to not use or modify the input vectors once this object
        /// is constructed, and should probably remove all references.</param>
        internal AgglomerativeClusteringModelParameters(IHostEnvironment env, Dictionary<int, ClusterInfo> clusters, List<int> dendrogram, bool copyIn)
            : base(env, LoaderSignature)
        {
            _dendrogram = dendrogram;
            _clusters = clusters;
        }

        /// <summary>
        /// Returns a copy of clusters
        /// </summary>

        public List<List<int>> GetClusters()
        {
            List<List<int>> clusterSet = new List<List<int>>();
            foreach (var cluster in _clusters.Values)
            {
                clusterSet.Add(cluster.Members.ToList());
            }
            return clusterSet;
        }

        /// <summary>
        /// Returns a copy of dendrograms
        /// </summary>

        public List<int> GetDendrogram()
        {
            return _dendrogram.ToList();
        }
    }
}
