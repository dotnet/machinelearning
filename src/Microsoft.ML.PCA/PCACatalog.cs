﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using static Microsoft.ML.Trainers.RandomizedPcaTrainer;

namespace Microsoft.ML
{
    public static class PcaCatalog
    {
        /// <summary>Initializes a new instance of <see cref="PrincipalComponentAnalysisEstimator"/>.</summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="rank">The number of principal components.</param>
        /// <param name="overSampling">Oversampling parameter for randomized PrincipalComponentAnalysis training.</param>
        /// <param name="center">If enabled, data is centered to be zero mean.</param>
        /// <param name="seed">The seed for random number generation.</param>
        public static PrincipalComponentAnalysisEstimator ProjectToPrincipalComponents(this TransformsCatalog.ProjectionTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            string exampleWeightColumnName = null,
            int rank = PrincipalComponentAnalysisEstimator.Defaults.Rank,
            int overSampling = PrincipalComponentAnalysisEstimator.Defaults.Oversampling,
            bool center = PrincipalComponentAnalysisEstimator.Defaults.Center,
            int? seed = null)
            => new PrincipalComponentAnalysisEstimator(CatalogUtils.GetEnvironment(catalog),
                outputColumnName, inputColumnName, exampleWeightColumnName, rank, overSampling, center, seed);

        /// <summary>Initializes a new instance of <see cref="PrincipalComponentAnalysisEstimator"/>.</summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">Input columns to apply PrincipalComponentAnalysis on.</param>
        public static PrincipalComponentAnalysisEstimator ProjectToPrincipalComponents(this TransformsCatalog.ProjectionTransforms catalog, params PrincipalComponentAnalysisEstimator.ColumnOptions[] columns)
            => new PrincipalComponentAnalysisEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Trains an approximate PCA using Randomized SVD algorithm.
        /// </summary>
        /// <param name="catalog">The anomaly detection catalog trainer object.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="rank">The number of components in the PCA.</param>
        /// <param name="oversampling">Oversampling parameter for randomized PCA training.</param>
        /// <param name="center">If enabled, data is centered to be zero mean.</param>
        /// <param name="seed">The seed for random number generation.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[RPCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/AnomalyDetection/RandomizedPcaSample.cs)]
        /// ]]></format>
        /// </example>
        public static RandomizedPcaTrainer RandomizedPca(this AnomalyDetectionCatalog.AnomalyDetectionTrainers catalog,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int rank = Options.Defaults.NumComponents,
            int oversampling = Options.Defaults.OversamplingParameters,
            bool center = Options.Defaults.IsCenteredZeroMean,
            int? seed = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new RandomizedPcaTrainer(env, featureColumnName, exampleWeightColumnName, rank, oversampling, center, seed);
        }

        /// <summary>
        /// Trains an approximate PCA using Randomized SVD algorithm.
        /// </summary>
        /// <param name="catalog">The anomaly detection catalog trainer object.</param>
        /// <param name="options">Advanced options to the algorithm.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[RPCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/AnomalyDetection/RandomizedPcaSampleWithOptions.cs)]
        /// ]]></format>
        /// </example>
        public static RandomizedPcaTrainer RandomizedPca(this AnomalyDetectionCatalog.AnomalyDetectionTrainers catalog, Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new RandomizedPcaTrainer(env, options);
        }
    }
}
