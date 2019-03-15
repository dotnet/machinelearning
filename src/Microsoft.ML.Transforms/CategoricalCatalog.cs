﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    /// <summary>
    /// The catalog of categorical transformations.
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
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[RPCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Categorical/OneHotEncoding.cs)]
        /// ]]></format>
        /// </example>
        public static OneHotEncodingEstimator OneHotEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                string outputColumnName,
                string inputColumnName = null,
                OneHotEncodingEstimator.OutputKind outputKind = OneHotEncodingEstimator.OutputKind.Indicator)
            => new OneHotEncodingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, outputKind);

        /// <summary>
        /// Convert several text column into one-hot encoded vectors.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="columns">The column settings.</param>
        public static OneHotEncodingEstimator OneHotEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                params OneHotEncodingEstimator.ColumnOptions[] columns)
            => new OneHotEncodingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Convert several text column into one-hot encoded vectors.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="columns">The column settings.</param>
        /// <param name="keyData">Specifies an ordering for the encoding. If specified, this should be a single column data view,
        /// and the key-values will be taken from that column. If unspecified, the ordering will be determined from the input data upon fitting.</param>
        public static OneHotEncodingEstimator OneHotEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                OneHotEncodingEstimator.ColumnOptions[] columns,
                IDataView keyData = null)
            => new OneHotEncodingEstimator(CatalogUtils.GetEnvironment(catalog), columns, keyData);

        /// <summary>
        /// Convert a text column into hash-based one-hot encoded vector.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="numberOfBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="maximumNumberOfInverts">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="maximumNumberOfInverts"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        /// <param name="outputKind">The conversion mode.</param>
        public static OneHotHashEncodingEstimator OneHotHashEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                string outputColumnName,
                string inputColumnName = null,
                int numberOfBits = OneHotHashEncodingEstimator.Defaults.NumberOfBits,
                int maximumNumberOfInverts = OneHotHashEncodingEstimator.Defaults.MaximumNumberOfInverts,
                OneHotEncodingEstimator.OutputKind outputKind = OneHotEncodingEstimator.OutputKind.Indicator)
            => new OneHotHashEncodingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName ?? outputColumnName, numberOfBits, maximumNumberOfInverts, outputKind);

        /// <summary>
        /// Convert several text column into hash-based one-hot encoded vectors.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="columns">The column settings.</param>
        public static OneHotHashEncodingEstimator OneHotHashEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                params OneHotHashEncodingEstimator.ColumnOptions[] columns)
            => new OneHotHashEncodingEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
