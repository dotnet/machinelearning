﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Normalizers;

namespace Microsoft.ML
{
    /// <summary>
    /// Extensions for normalizer operations.
    /// </summary>
    public static class NormalizerCatalog
    {
        /// <summary>
        /// Normalize (rescale) the column according to the specified <paramref name="mode"/>.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="mode">The <see cref="NormalizingEstimator.NormalizerMode"/> used to map the old values in the new scale. </param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Normalize](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Normalizer.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static NormalizingEstimator Normalize(this TransformsCatalog catalog,
           string outputColumnName, string inputColumnName = null,
            NormalizingEstimator.NormalizerMode mode = NormalizingEstimator.NormalizerMode.MinMax)
            => new NormalizingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName ?? outputColumnName, mode);

        /// <summary>
        /// Normalize (rescale) several columns according to the specified <paramref name="mode"/>.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="mode">The <see cref="NormalizingEstimator.NormalizerMode"/> used to map the old values to the new ones. </param>
        /// <param name="columns">The pairs of input and output columns.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Normalize](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Normalizer.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static NormalizingEstimator Normalize(this TransformsCatalog catalog,
            NormalizingEstimator.NormalizerMode mode,
            params SimpleColumnInfo[] columns)
            => new NormalizingEstimator(CatalogUtils.GetEnvironment(catalog), mode, SimpleColumnInfo.ConvertToValueTuples(columns));

        /// <summary>
        /// Normalize (rescale) columns according to specified custom parameters.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="columns">The normalization settings for all the columns</param>
        public static NormalizingEstimator Normalize(this TransformsCatalog catalog,
            params NormalizingEstimator.ColumnBase[] columns)
            => new NormalizingEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
