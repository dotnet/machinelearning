// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Trainers.PCA;
using Microsoft.ML.Transforms.Projections;
using static Microsoft.ML.Trainers.PCA.RandomizedPcaTrainer;

namespace Microsoft.ML
{
    public static class PcaCatalog
    {
        /// <summary>Initializes a new instance of <see cref="PrincipalComponentAnalysisEstimator"/>.</summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="weightColumn">The name of the weight column.</param>
        /// <param name="rank">The number of principal components.</param>
        /// <param name="overSampling">Oversampling parameter for randomized PrincipalComponentAnalysis training.</param>
        /// <param name="center">If enabled, data is centered to be zero mean.</param>
        /// <param name="seed">The seed for random number generation.</param>
        public static PrincipalComponentAnalysisEstimator ProjectToPrincipalComponents(this TransformsCatalog.ProjectionTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            string weightColumn = PrincipalComponentAnalysisEstimator.Defaults.WeightColumn,
            int rank = PrincipalComponentAnalysisEstimator.Defaults.Rank,
            int overSampling = PrincipalComponentAnalysisEstimator.Defaults.Oversampling,
            bool center = PrincipalComponentAnalysisEstimator.Defaults.Center,
            int? seed = null)
            => new PrincipalComponentAnalysisEstimator(CatalogUtils.GetEnvironment(catalog),
                outputColumnName, inputColumnName, weightColumn, rank, overSampling, center, seed);

        /// <summary>Initializes a new instance of <see cref="PrincipalComponentAnalysisEstimator"/>.</summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">Input columns to apply PrincipalComponentAnalysis on.</param>
        public static PrincipalComponentAnalysisEstimator ProjectToPrincipalComponents(this TransformsCatalog.ProjectionTransforms catalog, params PrincipalComponentAnalysisEstimator.ColumnInfo[] columns)
            => new PrincipalComponentAnalysisEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Trains an approximate PCA using Randomized SVD algorithm.
        /// </summary>
        /// <param name="catalog">The anomaly detection catalog trainer object.</param>
        /// <param name="featureColumn">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="rank">The number of components in the PCA.</param>
        /// <param name="oversampling">Oversampling parameter for randomized PCA training.</param>
        /// <param name="center">If enabled, data is centered to be zero mean.</param>
        /// <param name="seed">The seed for random number generation.</param>
        public static RandomizedPcaTrainer RandomizedPca(this AnomalyDetectionCatalog.AnomalyDetectionTrainers catalog,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            int rank = Options.Defaults.NumComponents,
            int oversampling = Options.Defaults.OversamplingParameters,
            bool center = Options.Defaults.IsCenteredZeroMean,
            int? seed = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new RandomizedPcaTrainer(env, featureColumn, weights, rank, oversampling, center, seed);
        }

        /// <summary>
        /// Trains an approximate PCA using Randomized SVD algorithm.
        /// </summary>
        /// <param name="catalog">The anomaly detection catalog trainer object.</param>
        /// <param name="options">Advanced options to the algorithm.</param>
        public static RandomizedPcaTrainer RandomizedPca(this AnomalyDetectionCatalog.AnomalyDetectionTrainers catalog, Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new RandomizedPcaTrainer(env, options);
        }
    }
}
