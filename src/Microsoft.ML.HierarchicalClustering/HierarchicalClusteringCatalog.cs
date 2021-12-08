// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using static Microsoft.ML.Trainers.AgglomerativeClusteringTrainer;

namespace Microsoft.ML
{
    /// <summary>
    /// Collection of extension methods for the <see cref="ClusteringCatalog.ClusteringTrainers"/> to create
    /// instances of Hierarchical clustering trainers.
    /// </summary>
    public static class HierarchicalClusteringExtensions
    {
        /// <summary>
        /// Train a Hierarchical clustering algorithm using <see cref="AgglomerativeClusteringTrainer"/>.
        /// </summary>
        /// <param name="catalog">The clustering catalog trainer object.</param>
        /// <param name="numberOfClusters">The number of clusters to used.</param>
        /// <param name="linkageCriterion">The linkage criterion to used.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Clustering/)]
        /// ]]></format>
        /// </example>
        public static AgglomerativeClusteringTrainer AgglomerativeClustering(this ClusteringCatalog.ClusteringTrainers catalog,
           int numberOfClusters = AgglomerativeClusteringTrainer.Defaults.NumberOfClusters,
           LinkageCriterion linkageCriterion = AgglomerativeClusteringTrainer.Defaults.Linkage)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);

            var options = new AgglomerativeClusteringTrainer.Options
            {
                NumberOfClusters = numberOfClusters,
                LinkageCriterion = linkageCriterion,
            };
            return new AgglomerativeClusteringTrainer(env, options);
        }

        /// <summary>
        /// Train a Hierarchical clustering algorithm using <see cref="AgglomerativeClusteringTrainer"/>.
        /// </summary>
        /// <param name="catalog">The clustering catalog trainer object.</param>
        /// <param name="options">Algorithm advanced options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Clustering/)]
        /// ]]></format>
        /// </example>
        public static AgglomerativeClusteringTrainer AgglomerativeClustering(this ClusteringCatalog.ClusteringTrainers catalog, AgglomerativeClusteringTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new AgglomerativeClusteringTrainer(env, options);
        }
    }
}
