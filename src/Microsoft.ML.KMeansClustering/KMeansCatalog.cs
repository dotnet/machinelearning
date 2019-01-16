// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Trainers.KMeans;

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
        /// <param name="featureColumn">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="clustersCount">The number of clusters to use for KMeans.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/KMeans_example.cs)]
        /// ]]></format>
        /// </example>
        public static KMeansPlusPlusTrainer KMeans(this ClusteringContext.ClusteringTrainers ctx,
           string featureColumn = DefaultColumnNames.Features,
           string weights = null,
           int clustersCount = KMeansPlusPlusTrainer.Defaults.K)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            var options = new KMeansPlusPlusTrainer.Options
            {
                FeatureColumn = featureColumn,
                WeightColumn = weights != null ? Optional<string>.Explicit(weights) : Optional<string>.Implicit(DefaultColumnNames.Weight),
                K = clustersCount
            };
            return new KMeansPlusPlusTrainer(env, options);
        }

        /// <summary>
        /// Train a KMeans++ clustering algorithm.
        /// </summary>
        /// <param name="ctx">The regression context trainer object.</param>
        /// <param name="options">Algorithm advanced settings.</param>
        public static KMeansPlusPlusTrainer KMeans(this ClusteringContext.ClusteringTrainers ctx, KMeansPlusPlusTrainer.Options options = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new KMeansPlusPlusTrainer(env, options);
        }
    }
}
