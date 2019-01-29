// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
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
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="outputKind">The conversion mode.</param>
        /// <returns></returns>
        public static OneHotEncodingEstimator OneHotEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                string outputColumnName,
                string inputColumnName = null,
                OneHotEncodingTransformer.OutputKind outputKind = OneHotEncodingTransformer.OutputKind.Ind)
            => new OneHotEncodingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, outputKind);

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
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="invertHash">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="invertHash"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        /// <param name="outputKind">The conversion mode.</param>
        /// <returns></returns>
        public static OneHotHashEncodingEstimator OneHotHashEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                string outputColumnName,
                string inputColumnName = null,
                int hashBits = OneHotHashEncodingEstimator.Defaults.HashBits,
                int invertHash = OneHotHashEncodingEstimator.Defaults.InvertHash,
                OneHotEncodingTransformer.OutputKind outputKind = OneHotEncodingTransformer.OutputKind.Ind)
            => new OneHotHashEncodingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName ?? outputColumnName, hashBits, invertHash, outputKind);

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
