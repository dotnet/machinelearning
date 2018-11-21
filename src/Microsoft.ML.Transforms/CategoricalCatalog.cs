﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms.Categorical;

namespace Microsoft.ML
{
    /// <summary>
    /// Static extensions for categorical transforms.
    /// </summary>
    public static class CategoricalCatalog
    {
        /// <summary>
        /// Convert a text column into one-hot encoded vector.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="inputColumn">The input column</param>
        /// <param name="outputColumn">The output column. If <c>null</c>, <paramref name="inputColumn"/> is used.</param>
        /// <param name="outputKind">The conversion mode.</param>
        /// <returns></returns>
        public static OneHotEncodingEstimator OneHotEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                string inputColumn,
                string outputColumn = null,
                OneHotEncodingTransformer.OutputKind outputKind = OneHotEncodingTransformer.OutputKind.Ind)
            => new OneHotEncodingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, outputKind);

        /// <summary>
        /// Convert several text column into one-hot encoded vectors.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="columns">The column settings.</param>
        /// <returns></returns>
        public static OneHotEncodingEstimator OneHotEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                params OneHotEncodingEstimator.ColumnInfo[] columns)
            => new OneHotEncodingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Convert a text column into hash-based one-hot encoded vector.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="inputColumn">The input column</param>
        /// <param name="outputColumn">The output column. If <c>null</c>, <paramref name="inputColumn"/> is used.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="invertHash">Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.</param>
        /// <param name="outputKind">The conversion mode.</param>
        /// <returns></returns>
        public static OneHotHashEncodingEstimator OneHotHashEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                string inputColumn,
                string outputColumn = null,
                int hashBits = OneHotHashEncodingEstimator.Defaults.HashBits,
                int invertHash = OneHotHashEncodingEstimator.Defaults.InvertHash,
                OneHotEncodingTransformer.OutputKind outputKind = OneHotEncodingTransformer.OutputKind.Ind)
            => new OneHotHashEncodingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, hashBits, invertHash, outputKind);

        /// <summary>
        /// Convert several text column into hash-based one-hot encoded vectors.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="columns">The column settings.</param>
        /// <returns></returns>
        public static OneHotHashEncodingEstimator OneHotHashEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                params OneHotHashEncodingEstimator.ColumnInfo[] columns)
            => new OneHotHashEncodingEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
