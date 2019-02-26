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
        /// <param name="catalog">The clustering catalog trainer object.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="clustersCount">The number of clusters to use for KMeans.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[KMeans](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/KMeans.cs)]
        /// ]]></format>
        /// </example>
        public static KMeansPlusPlusTrainer KMeans(this ClusteringCatalog.ClusteringTrainers catalog,
           string featureColumnName = DefaultColumnNames.Features,
           string exampleWeightColumnName = null,
           int clustersCount = KMeansPlusPlusTrainer.Defaults.ClustersCount)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);

            var options = new KMeansPlusPlusTrainer.Options
            {
                FeatureColumn = featureColumnName,
                WeightColumn = exampleWeightColumnName,
                ClustersCount = clustersCount
            };
            return new KMeansPlusPlusTrainer(env, options);
        }

        /// <summary>
        /// Train a KMeans++ clustering algorithm.
        /// </summary>
        /// <param name="catalog">The clustering catalog trainer object.</param>
        /// <param name="options">Algorithm advanced options.</param>
        public static KMeansPlusPlusTrainer KMeans(this ClusteringCatalog.ClusteringTrainers catalog, KMeansPlusPlusTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new KMeansPlusPlusTrainer(env, options);
        }
    }
}
