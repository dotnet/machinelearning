// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    /// <summary>
    /// Collection of extension methods for <see cref="TransformsCatalog.CategoricalTransforms"/> to create categorical
    /// transformer components.
    /// </summary>
    public static class CategoricalCatalog
    {
        /// <summary>
        /// Create a <see cref="OneHotEncodingEstimator"/>, which converts the input column specified by <paramref name="inputColumnName"/>
        /// into a column of one-hot encoded vectors named <paramref name="outputColumnName"/>.
        /// </summary>
        /// <param name="catalog">The transform catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// This column's data type will be a vector of <see cref="System.Single"/> if <paramref name="outputKind"/> is
        /// <see cref="OneHotEncodingEstimator.OutputKind.Bag"/>, <see cref="OneHotEncodingEstimator.OutputKind.Indicator"/>, and <see cref="OneHotEncodingEstimator.OutputKind.Binary"/>.
        /// If <paramref name="outputKind"/> is <see cref="OneHotEncodingEstimator.OutputKind.Key"/>, this column's data type will be a key in the case of a scalar input column
        /// or a vector of keys in the case of a vector input column.</param>
        /// <param name="inputColumnName">Name of column to convert to one-hot vectors. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/>
        /// will be used as source. This column's data type can be scalar or vector of numeric, text, boolean, <see cref="System.DateTime"/> or <see cref="System.DateTimeOffset"/>,</param>
        /// <param name="outputKind">Output kind: Bag (multi-set vector), Indicator (indicator vector), Key (index), or Binary encoded indicator vector.</param>
        /// <param name="maximumNumberOfKeys">Maximum number of terms to keep per column when auto-training.</param>
        /// <param name="keyOrdinality">How items should be ordered when vectorized. If <see cref="ValueToKeyMappingEstimator.KeyOrdinality.ByOccurrence"/>
        /// choosen they will be in the order encountered. If <see cref="ValueToKeyMappingEstimator.KeyOrdinality.ByValue"/>,
        /// items are sorted according to their default comparison, for example, text sorting will be case sensitive (for example, 'A' then 'Z' then 'a').</param>
        /// <param name="keyData">Specifies an ordering for the encoding. If specified, this should be a single column data view,
        /// and the key-values will be taken from that column. If unspecified, the ordering will be determined from the input data upon fitting.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[OneHotEncoding](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Categorical/OneHotEncoding.cs)]
        /// ]]></format>
        /// </example>
        public static OneHotEncodingEstimator OneHotEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                string outputColumnName,
                string inputColumnName = null,
                OneHotEncodingEstimator.OutputKind outputKind = OneHotEncodingEstimator.Defaults.OutKind,
                int maximumNumberOfKeys = ValueToKeyMappingEstimator.Defaults.MaximumNumberOfKeys,
                ValueToKeyMappingEstimator.KeyOrdinality keyOrdinality = ValueToKeyMappingEstimator.Defaults.Ordinality,
                IDataView keyData = null)
            => new OneHotEncodingEstimator(CatalogUtils.GetEnvironment(catalog),
                new[] { new OneHotEncodingEstimator.ColumnOptions(outputColumnName, inputColumnName, outputKind, maximumNumberOfKeys, keyOrdinality) }, keyData);

        /// <summary>
        /// Create a <see cref="OneHotEncodingEstimator"/>, which converts one or more input text columns specified in <paramref name="columns"/>
        /// into as many columns of one-hot encoded vectors.
        /// </summary>
        /// <remarks>If multiple columns are passed to the estimator, all of the columns will be processed in a single pass over the data.
        /// Therefore, it is more efficient to specify one estimator with many columns than it is to specify many estimators each with a single column.</remarks>
        /// <param name="catalog">The transform catalog.</param>
        /// <param name="columns">The pairs of input and output columns. The output columns' data type will be a vector of <see cref="System.Single"/> if <paramref name="outputKind"/> is
        /// <see cref="OneHotEncodingEstimator.OutputKind.Bag"/>, <see cref="OneHotEncodingEstimator.OutputKind.Indicator"/>, and <see cref="OneHotEncodingEstimator.OutputKind.Binary"/>.
        /// If <paramref name="outputKind"/> is <see cref="OneHotEncodingEstimator.OutputKind.Key"/>, the output columns' data type will be a key in the case of scalar input column
        /// or a vector of keys in the case of a vector input column.</param>
        /// <param name="outputKind">Output kind: Bag (multi-set vector), Ind (indicator vector), Key (index), or Binary encoded indicator vector.</param>
        /// <param name="maximumNumberOfKeys">Maximum number of terms to keep per column when auto-training.</param>
        /// <param name="keyOrdinality">How items should be ordered when vectorized. If <see cref="ValueToKeyMappingEstimator.KeyOrdinality.ByOccurrence"/>
        /// choosen they will be in the order encountered. If <see cref="ValueToKeyMappingEstimator.KeyOrdinality.ByValue"/>,
        /// items are sorted according to their default comparison, for example, text sorting will be case sensitive (for example, 'A' then 'Z' then 'a').</param>
        /// <param name="keyData">Specifies an ordering for the encoding. If specified, this should be a single column data view,
        /// and the key-values will be taken from that column. If unspecified, the ordering will be determined from the input data upon fitting.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[OneHotEncoding](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Categorical/OneHotEncodingMultiColumn.cs)]
        /// ]]></format>
        /// </example>
        public static OneHotEncodingEstimator OneHotEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                InputOutputColumnPair[] columns,
                OneHotEncodingEstimator.OutputKind outputKind = OneHotEncodingEstimator.Defaults.OutKind,
                int maximumNumberOfKeys = ValueToKeyMappingEstimator.Defaults.MaximumNumberOfKeys,
                ValueToKeyMappingEstimator.KeyOrdinality keyOrdinality = ValueToKeyMappingEstimator.Defaults.Ordinality,
                IDataView keyData = null)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            env.CheckValue(columns, nameof(columns));
            var columnOptions = columns.Select(x => new OneHotEncodingEstimator.ColumnOptions(x.OutputColumnName, x.InputColumnName, outputKind, maximumNumberOfKeys, keyOrdinality)).ToArray();
            return new OneHotEncodingEstimator(env, columnOptions, keyData);
        }

        /// <summary>
        /// Convert several text column into one-hot encoded vectors.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="columns">The column settings.</param>
        [BestFriend]
        internal static OneHotEncodingEstimator OneHotEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                params OneHotEncodingEstimator.ColumnOptions[] columns)
            => new OneHotEncodingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Convert several text column into one-hot encoded vectors.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="columns">The column settings.</param>
        /// <param name="keyData">Specifies an ordering for the encoding. If specified, this should be a single column data view,
        /// and the key-values will be taken from that column. If unspecified, the ordering will be determined from the input data upon fitting.</param>
        [BestFriend]
        internal static OneHotEncodingEstimator OneHotEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                OneHotEncodingEstimator.ColumnOptions[] columns,
                IDataView keyData = null)
            => new OneHotEncodingEstimator(CatalogUtils.GetEnvironment(catalog), columns, keyData);

        /// <summary>
        /// Create a <see cref="OneHotHashEncodingEstimator"/>, which converts a text column specified by <paramref name="inputColumnName"/>
        /// into a hash-based one-hot encoded vector column named <paramref name="outputColumnName"/>.
        /// </summary>
        /// <param name="catalog">The transform catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// This column's data type will be a vector of <see cref="System.Single"/> if <paramref name="outputKind"/> is
        /// <see cref="OneHotEncodingEstimator.OutputKind.Bag"/>, <see cref="OneHotEncodingEstimator.OutputKind.Indicator"/>, and <see cref="OneHotEncodingEstimator.OutputKind.Binary"/>.
        /// If <paramref name="outputKind"/> is <see cref="OneHotEncodingEstimator.OutputKind.Key"/>, this column's data type will be a key in the case of a scalar input column
        /// or a vector of keys in the case of a vector input column. </param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.
        /// This column's data type can be scalar or vector of numeric, text, boolean, <see cref="System.DateTime"/> or <see cref="System.DateTimeOffset"/>.</param>
        /// <param name="outputKind">The conversion mode.</param>
        /// <param name="numberOfBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="useOrderedHashing">Whether the position of each term should be included in the hash.</param>
        /// <param name="maximumNumberOfInverts">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="maximumNumberOfInverts"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[OneHotHashEncoding](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Categorical/OneHotHashEncoding.cs)]
        /// ]]></format>
        /// </example>
        public static OneHotHashEncodingEstimator OneHotHashEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                string outputColumnName,
                string inputColumnName = null,
                OneHotEncodingEstimator.OutputKind outputKind = OneHotEncodingEstimator.OutputKind.Indicator,
                int numberOfBits = OneHotHashEncodingEstimator.Defaults.NumberOfBits,
                uint seed = OneHotHashEncodingEstimator.Defaults.Seed,
                bool useOrderedHashing = OneHotHashEncodingEstimator.Defaults.UseOrderedHashing,
                int maximumNumberOfInverts = OneHotHashEncodingEstimator.Defaults.MaximumNumberOfInverts)
            => new OneHotHashEncodingEstimator(CatalogUtils.GetEnvironment(catalog),
                new[] { new OneHotHashEncodingEstimator.ColumnOptions(outputColumnName, inputColumnName, outputKind, numberOfBits, seed, useOrderedHashing, maximumNumberOfInverts) });

        /// <summary>
        /// Create a <see cref="OneHotHashEncodingEstimator"/>, which converts one or more input text columns specified by <paramref name="columns"/>
        /// into as many columns of hash-based one-hot encoded vectors.
        /// </summary>
        /// <remarks>If multiple columns are passed to the estimator, all of the columns will be processed in a single pass over the data.
        /// Therefore, it is more efficient to specify one estimator with many columns than it is to specify many estimators each with a single column.
        /// </remarks>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="columns">The pairs of input and output columns. The output columns' data type will be a vector of <see cref="System.Single"/> if <paramref name="outputKind"/> is
        /// <see cref="OneHotEncodingEstimator.OutputKind.Bag"/>, <see cref="OneHotEncodingEstimator.OutputKind.Indicator"/>, and <see cref="OneHotEncodingEstimator.OutputKind.Binary"/>.
        /// If <paramref name="outputKind"/> is <see cref="OneHotEncodingEstimator.OutputKind.Key"/>, the output columns' data type will be a key in the case of scalar input column
        /// or a vector of keys in the case of a vector input column.</param>
        /// <param name="outputKind">The conversion mode.</param>
        /// <param name="numberOfBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="useOrderedHashing">Whether the position of each term should be included in the hash.</param>
        /// <param name="maximumNumberOfInverts">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column. Hashing, as such, can map many initial values to one.
        /// <paramref name="maximumNumberOfInverts"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[OneHotHashEncoding](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Categorical/OneHotHashEncodingMultiColumn.cs)]
        /// ]]></format>
        /// </example>
        public static OneHotHashEncodingEstimator OneHotHashEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                InputOutputColumnPair[] columns,
                OneHotEncodingEstimator.OutputKind outputKind = OneHotEncodingEstimator.OutputKind.Indicator,
                int numberOfBits = OneHotHashEncodingEstimator.Defaults.NumberOfBits,
                uint seed = OneHotHashEncodingEstimator.Defaults.Seed,
                bool useOrderedHashing = OneHotHashEncodingEstimator.Defaults.UseOrderedHashing,
                int maximumNumberOfInverts = OneHotHashEncodingEstimator.Defaults.MaximumNumberOfInverts)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            env.CheckValue(columns, nameof(columns));
            var columnOptions = columns.Select(x => new OneHotHashEncodingEstimator.ColumnOptions(x.OutputColumnName, x.InputColumnName, outputKind, numberOfBits, seed, useOrderedHashing, maximumNumberOfInverts)).ToArray();
            return new OneHotHashEncodingEstimator(env, columnOptions);
        }

        /// <summary>
        /// Convert several text column into hash-based one-hot encoded vectors.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="columns">The column settings.</param>
        [BestFriend]
        internal static OneHotHashEncodingEstimator OneHotHashEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                params OneHotHashEncodingEstimator.ColumnOptions[] columns)
            => new OneHotHashEncodingEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
