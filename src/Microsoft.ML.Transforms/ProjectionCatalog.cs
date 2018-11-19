﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms.Projections;

namespace Microsoft.ML
{
    public static class ProjectionCatalog
    {
        /// <summary>
        /// Takes column filled with a vector of floats and maps its to a random low-dimensional feature space.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="inputColumn">Name of the column to be transformed.</param>
        /// <param name="outputColumn">Name of the output column. If this is null '<paramref name="inputColumn"/>' will be used.</param>
        /// <param name="newDim">The number of random Fourier features to create.</param>
        /// <param name="useSin">Create two features for every random Fourier frequency? (one for cos and one for sin).</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[CreateRandomFourierFeatures](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/ProjectionTransforms.cs?range=1-6,12-112)]
        /// ]]>
        /// </format>
        /// </example>
        public static RandomFourierFeaturizingEstimator CreateRandomFourierFeatures(this TransformsCatalog.ProjectionTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            int newDim = RandomFourierFeaturizingEstimator.Defaults.NewDim,
            bool useSin = RandomFourierFeaturizingEstimator.Defaults.UseSin)
            => new RandomFourierFeaturizingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, newDim, useSin);

        /// <summary>
        /// Takes columns filled with a vector of floats and maps its to a random low-dimensional feature space.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">The input columns to use for the transformation.</param>
        public static RandomFourierFeaturizingEstimator CreateRandomFourierFeatures(this TransformsCatalog.ProjectionTransforms catalog, params RandomFourierFeaturizingTransformer.ColumnInfo[] columns)
            => new RandomFourierFeaturizingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Takes column filled with a vector of floats and computes L-p norm of it.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        /// <param name="outputColumn">Name of the column resulting from the transformation of <paramref name="inputColumn"/>. Null means <paramref name="inputColumn"/> is replaced. </param>
        /// <param name="normKind">Type of norm to use to normalize each sample.</param>
        /// <param name="subMean">Subtract mean from each value before normalizing.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[LpNormalize](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/ProjectionTransforms.cs?range=1-6,12-112)]
        /// ]]>
        /// </format>
        /// </example>
        public static LpNormalizingEstimator LpNormalize(this TransformsCatalog.ProjectionTransforms catalog, string inputColumn, string outputColumn =null,
            LpNormalizingEstimatorBase.NormalizerKind normKind = LpNormalizingEstimatorBase.Defaults.NormKind, bool subMean = LpNormalizingEstimatorBase.Defaults.LpSubstractMean)
            => new LpNormalizingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, normKind, subMean);

        /// <summary>
        /// Takes columns filled with a vector of floats and computes L-p norm of it.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns"> Describes the parameters of the lp-normalization process for each column pair.</param>
        public static LpNormalizingEstimator LpNormalize(this TransformsCatalog.ProjectionTransforms catalog, params LpNormalizingTransformer.LpNormColumnInfo[] columns)
            => new LpNormalizingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Takes column filled with a vector of floats and computes global contrast normalization of it.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        /// <param name="outputColumn">Name of the column resulting from the transformation of <paramref name="inputColumn"/>. Null means <paramref name="inputColumn"/> is replaced. </param>
        /// <param name="substractMean">Subtract mean from each value before normalizing.</param>
        /// <param name="useStdDev">Normalize by standard deviation rather than L2 norm.</param>
        /// <param name="scale">Scale features by this value.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[GlobalContrastNormalize](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/ProjectionTransforms.cs?range=1-6,12-112)]
        /// ]]>
        /// </format>
        /// </example>
        public static GlobalContrastNormalizingEstimator GlobalContrastNormalize(this TransformsCatalog.ProjectionTransforms catalog, string inputColumn, string outputColumn = null,
             bool substractMean = LpNormalizingEstimatorBase.Defaults.GcnSubstractMean,
             bool useStdDev = LpNormalizingEstimatorBase.Defaults.UseStdDev,
             float scale = LpNormalizingEstimatorBase.Defaults.Scale)
            => new GlobalContrastNormalizingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, substractMean, useStdDev, scale);

        /// <summary>
        /// Takes columns filled with a vector of floats and computes global contrast normalization of it.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns"> Describes the parameters of the gcn-normaliztion process for each column pair.</param>
        public static GlobalContrastNormalizingEstimator GlobalContrastNormalize(this TransformsCatalog.ProjectionTransforms catalog, params LpNormalizingTransformer.GcnColumnInfo[] columns)
            => new GlobalContrastNormalizingEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
