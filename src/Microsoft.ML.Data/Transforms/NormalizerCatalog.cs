// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML
{
    /// <summary>
    /// Extensions for normalizer operations.
    /// </summary>
    public static class NormalizerCatalogExtensions
    {
        /// <summary>
        /// Normalize (rescale) the column according to the specified <paramref name="mode"/>.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="columnName">The column name</param>
        /// <param name="mode">The normalization mode (<see cref="Normalizer.NormalizerMode"/>). </param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ConcatWith] (](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/MinMaxNormalizer.cs?range=34-35 )]
        /// ]]>
        /// </format>
        /// </example>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ConcatWith] (](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/MinMaxNormalizer.cs?range =6-11,16-86)]
        /// ]]>
        /// </format>
        /// </example>
        public static Normalizer Normalizer(this TransformsCatalog catalog, string columnName, Normalizer.NormalizerMode mode = Runtime.Data.Normalizer.NormalizerMode.MinMax)
            => new Normalizer(CatalogUtils.GetEnvironment(catalog), columnName, mode);

        /// <summary>
        /// Normalize (rescale) several columns according to the specified <paramref name="mode"/>.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="mode">The normalization mode (<see cref="Normalizer.NormalizerMode"/>). </param>
        /// <param name="columns">The pairs of input and output columns.</param>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ConcatWith] (](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/MinMaxNormalizer.cs?range=57-60 )]
        /// ]]>
        /// </format>
        /// </example>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ConcatWith] (](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/MinMaxNormalizer.cs?range =6-11,16-86)]
        /// ]]>
        /// </format>
        /// </example>
        public static Normalizer Normalizer(this TransformsCatalog catalog, Normalizer.NormalizerMode mode, params (string input, string output)[] columns)
            => new Normalizer(CatalogUtils.GetEnvironment(catalog), mode, columns);

        /// <summary>
        /// Normalize (rescale) columns according to specified custom parameters.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="columns">The normalization settings for all the columns</param>
        public static Normalizer Normalizer(this TransformsCatalog catalog, params Normalizer.ColumnBase[] columns)
            => new Normalizer(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
