// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers;
using System;

namespace Microsoft.ML
{
    /// <summary>
    /// The trainer context extensions for the <see cref="KMeansPlusPlusTrainer"/>.
    /// </summary>
    public static class KMeansClusteringExtensions
    {
        /// <summary>
        /// Train a KMeans++ clustering algorithm.
        /// </summary>
        /// <param name="ctx">The regression context trainer object.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="clustersCount">The number of clusters to use for KMeans.</param>
        /// <param name="advancedSettings">Algorithm advanced settings.</param>
        public static KMeansPlusPlusTrainer KMeans(this ClusteringContext.ClusteringTrainers ctx,
           string features = DefaultColumnNames.Features,
           string weights = null,
           int clustersCount = KMeansPlusPlusTrainer.Defaults.K,
           Action<KMeansPlusPlusTrainer.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new KMeansPlusPlusTrainer(env, features, clustersCount, weights, advancedSettings);
        }
    }
}
