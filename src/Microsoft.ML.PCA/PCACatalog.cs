﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using static Microsoft.ML.Trainers.RandomizedPcaTrainer;

namespace Microsoft.ML
{
    /// <summary>
    /// Collection of extension methods used by the <see cref="AnomalyDetectionCatalog.AnomalyDetectionTrainers"/>,
    /// and <see cref="TransformsCatalog"/> catalogs to create instances of Principal Component Analysis (PCA) components.
    /// </summary>
    public static class PcaCatalog
    {
        /// <summary>Initializes a new instance of <see cref="PrincipalComponentAnalyzer"/>.</summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="rank">The number of principal components.</param>
        /// <param name="overSampling">Oversampling parameter for randomized PrincipalComponentAnalysis training.</param>
        /// <param name="ensureZeroMean">If enabled, data is centered to be zero mean.</param>
        /// <param name="seed">The seed for random number generation.</param>
        public static PrincipalComponentAnalyzer ProjectToPrincipalComponents(this TransformsCatalog catalog,
            string outputColumnName,
            string inputColumnName = null,
            string exampleWeightColumnName = null,
            int rank = PrincipalComponentAnalyzer.Defaults.Rank,
            int overSampling = PrincipalComponentAnalyzer.Defaults.Oversampling,
            bool ensureZeroMean = PrincipalComponentAnalyzer.Defaults.EnsureZeroMean,
            int? seed = null)
            => new PrincipalComponentAnalyzer(CatalogUtils.GetEnvironment(catalog),
                outputColumnName, inputColumnName, exampleWeightColumnName, rank, overSampling, ensureZeroMean, seed);

        /// <summary>Initializes a new instance of <see cref="PrincipalComponentAnalyzer"/>.</summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">Input columns to apply PrincipalComponentAnalysis on.</param>
        [BestFriend]
        internal static PrincipalComponentAnalyzer ProjectToPrincipalComponents(this TransformsCatalog catalog, params PrincipalComponentAnalyzer.ColumnOptions[] columns)
            => new PrincipalComponentAnalyzer(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Create <see cref="RandomizedPcaTrainer"/>, which trains an approximate principal component analysis (PCA) model using randomized singular value decomposition (SVD) algorithm.
        /// </summary>
        /// <param name="catalog">The anomaly detection catalog trainer object.</param>
        /// <param name="featureColumnName">The name of the feature column. The column data must be a known-sized vector of <see cref="System.Single"/>.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional). To use the weight column, the column data
        /// must be of type <see cref="System.Single"/>.</param>
        /// <param name="rank">The number of components in the PCA.</param>
        /// <param name="oversampling">Oversampling parameter for randomized PCA training.</param>
        /// <param name="ensureZeroMean">If enabled, data is centered to be zero mean.</param>
        /// <param name="seed">The seed for random number generation.</param>
        /// <remarks>
        /// By default the threshold used to determine the label of a data point based on the predicted score is 0.5. Scores range from 0 to 1. A data point with predicted
        /// score higher than 0.5 is considered an outlier. Use <see cref="AnomalyDetectionCatalog.ChangeModelThreshold"/> to change this threshold.
        /// </remarks>
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
            bool ensureZeroMean = Options.Defaults.EnsureZeroMean,
            int? seed = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new RandomizedPcaTrainer(env, featureColumnName, exampleWeightColumnName, rank, oversampling, ensureZeroMean, seed);
        }

        /// <summary>
        /// Create <see cref="RandomizedPcaTrainer"/> with advanced options, which trains an approximate principal component analysis (PCA) model using randomized singular value decomposition (SVD) algorithm.
        /// </summary>
        /// <param name="catalog">The anomaly detection catalog trainer object.</param>
        /// <param name="options">Advanced options to the algorithm.</param>
        /// <remarks>
        /// By default the threshold used to determine the label of a data point based on the predicted score is 0.5. Scores range from 0 to 1. A data point with predicted
        /// score higher than 0.5 is considered an outlier. Use <see cref="AnomalyDetectionCatalog.ChangeModelThreshold"/> to change this threshold.
        /// </remarks>
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
