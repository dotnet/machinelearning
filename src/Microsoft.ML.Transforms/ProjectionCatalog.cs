// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    /// <summary>
    /// The catalog of projection transformations.
    /// </summary>
    public static class ProjectionCatalog
    {
        /// <summary>
        /// Takes column filled with a vector of floats and maps its to a random low-dimensional feature space.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="dimension">The number of random Fourier features to create.</param>
        /// <param name="useCosAndSinBases">If <see langword="true"/>, use both of cos and sin basis functions to create two features for every random Fourier frequency.
        /// Otherwise, only cos bases would be used.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[CreateRandomFourierFeatures](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/ProjectionTransforms.cs?range=1-6,12-112)]
        /// ]]>
        /// </format>
        /// </example>
        public static RandomFourierFeaturizingEstimator CreateRandomFourierFeatures(this TransformsCatalog.ProjectionTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            int dimension = RandomFourierFeaturizingEstimator.Defaults.Dimension,
            bool useCosAndSinBases = RandomFourierFeaturizingEstimator.Defaults.UseCosAndSinBases)
            => new RandomFourierFeaturizingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, dimension, useCosAndSinBases);

        /// <summary>
        /// Takes columns filled with a vector of floats and maps its to a random low-dimensional feature space.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">The input columns to use for the transformation.</param>
        public static RandomFourierFeaturizingEstimator CreateRandomFourierFeatures(this TransformsCatalog.ProjectionTransforms catalog, params RandomFourierFeaturizingEstimator.ColumnOptions[] columns)
            => new RandomFourierFeaturizingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Takes column filled with a vector of floats and normazlize its <paramref name="normKind"/> to one. By setting <paramref name="ensureZeroMean"/> to <see langword="true"/>,
        /// a pre-processing step would be applied to make the specified column's mean be a zero vector.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="normKind">Type of norm to use to normalize each sample. The indicated norm of the resulted vector will be normalized to one.</param>
        /// <param name="ensureZeroMean">If <see langword="true"/>, subtract mean from each value before normalizing and use the raw input otherwise.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[LpNormalize](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/ProjectionTransforms.cs?range=1-6,12-112)]
        /// ]]>
        /// </format>
        /// </example>
        public static LpNormalizingEstimator LpNormalize(this TransformsCatalog.ProjectionTransforms catalog, string outputColumnName, string inputColumnName = null,
            LpNormalizingEstimatorBase.NormKind normKind = LpNormalizingEstimatorBase.Defaults.Norm, bool ensureZeroMean = LpNormalizingEstimatorBase.Defaults.LpEnsureZeroMean)
            => new LpNormalizingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, normKind, ensureZeroMean);

        /// <summary>
        /// Takes column filled with a vector of floats and normazlize its norm to one. Note that the allowed norm functions are defined in <see cref="LpNormalizingEstimatorBase.NormKind"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns"> Describes the parameters of the lp-normalization process for each column pair.</param>
        public static LpNormalizingEstimator LpNormalize(this TransformsCatalog.ProjectionTransforms catalog, params LpNormalizingEstimator.LpNormColumnOptions[] columns)
            => new LpNormalizingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Takes column filled with a vector of floats and computes global contrast normalization of it. By setting <paramref name="ensureZeroMean"/> to <see langword="true"/>,
        /// a pre-processing step would be applied to make the specified column's mean be a zero vector.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="ensureZeroMean">If <see langword="true"/>, subtract mean from each value before normalizing and use the raw input otherwise.</param>
        /// <param name="ensureUnitStandardDeviation">If <see langword="true"/>, resulted vector's standard deviation would be one. Otherwise, resulted vector's L2-norm would be one.</param>
        /// <param name="scale">Scale features by this value.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[GlobalContrastNormalize](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/ProjectionTransforms.cs?range=1-6,12-112)]
        /// ]]>
        /// </format>
        /// </example>
        public static GlobalContrastNormalizingEstimator GlobalContrastNormalize(this TransformsCatalog.ProjectionTransforms catalog, string outputColumnName, string inputColumnName = null,
             bool ensureZeroMean = LpNormalizingEstimatorBase.Defaults.GcnEnsureZeroMean,
             bool ensureUnitStandardDeviation = LpNormalizingEstimatorBase.Defaults.EnsureUnitStdDev,
             float scale = LpNormalizingEstimatorBase.Defaults.Scale)
            => new GlobalContrastNormalizingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, ensureZeroMean, ensureUnitStandardDeviation, scale);

        /// <summary>
        /// Takes columns filled with a vector of floats and computes global contrast normalization of it.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns"> Describes the parameters of the gcn-normaliztion process for each column pair.</param>
        public static GlobalContrastNormalizingEstimator GlobalContrastNormalize(this TransformsCatalog.ProjectionTransforms catalog, params GlobalContrastNormalizingEstimator.GcnColumnOptions[] columns)
            => new GlobalContrastNormalizingEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
