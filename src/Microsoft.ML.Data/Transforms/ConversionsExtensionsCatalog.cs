// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Conversions;

namespace Microsoft.ML
{
    using ConvertDefaults = TypeConvertingEstimator.Defaults;
    using HashDefaults = HashingEstimator.Defaults;

    /// <summary>
    /// Extensions for the HashEstimator.
    /// </summary>
    public static class ConversionsExtensionsCatalog
    {
        /// <summary>
        /// Hashes the values in the input column.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 31, inclusive.</param>
        /// <param name="invertHash">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="invertHash"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        public static HashingEstimator Hash(this TransformsCatalog.ConversionTransforms catalog, string outputColumnName, string inputColumnName = null,
            int hashBits = HashDefaults.HashBits, int invertHash = HashDefaults.InvertHash)
            => new HashingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, hashBits, invertHash);

        /// <summary>
        /// Hashes the values in the input column.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">Description of dataset columns and how to process them.</param>
        public static HashingEstimator Hash(this TransformsCatalog.ConversionTransforms catalog, params HashingTransformer.ColumnInfo[] columns)
            => new HashingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Changes column type of the input column.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="outputKind">Number of bits to hash into. Must be between 1 and 31, inclusive.</param>
        public static TypeConvertingEstimator ConvertType(this TransformsCatalog.ConversionTransforms catalog, string outputColumnName, string inputColumnName = null,
            DataKind outputKind = ConvertDefaults.DefaultOutputKind)
            => new TypeConvertingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, outputKind);

        /// <summary>
        /// Changes column type of the input column.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">Description of dataset columns and how to process them.</param>
        public static TypeConvertingEstimator ConvertType(this TransformsCatalog.ConversionTransforms catalog, params TypeConvertingTransformer.ColumnInfo[] columns)
            => new TypeConvertingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Convert the key types back to their original values.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="inputColumnName">Name of the column to transform.</param>
        public static KeyToValueMappingEstimator MapKeyToValue(this TransformsCatalog.ConversionTransforms catalog, string inputColumnName)
            => new KeyToValueMappingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumnName);

        /// <summary>
        ///  Convert the key types (name of the column specified in the first item of the tuple) back to their original values
        ///  (named as specified in the second item of the tuple).
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog</param>
        /// <param name="columns">The pairs of input and output columns.</param>
        public static KeyToValueMappingEstimator MapKeyToValue(this TransformsCatalog.ConversionTransforms catalog, params (string outputColumnName, string inputColumnName)[] columns)
             => new KeyToValueMappingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Convert the key types back to their original vectors.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="columns">The input column to map back to vectors.</param>
        public static KeyToVectorMappingEstimator MapKeyToVector(this TransformsCatalog.ConversionTransforms catalog,
            params KeyToVectorMappingTransformer.ColumnInfo[] columns)
            => new KeyToVectorMappingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Convert the key types back to their original vectors.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="bag">Whether bagging is used for the conversion. </param>
        public static KeyToVectorMappingEstimator MapKeyToVector(this TransformsCatalog.ConversionTransforms catalog,
            string outputColumnName, string inputColumnName = null, bool bag = KeyToVectorMappingEstimator.Defaults.Bag)
            => new KeyToVectorMappingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, bag);

        /// <summary>
        /// Converts value types into <see cref="KeyType"/>.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="maxNumTerms">Maximum number of keys to keep per column when auto-training.</param>
        /// <param name="sort">How items should be ordered when vectorized. If <see cref="ValueToKeyMappingTransformer.SortOrder.Occurrence"/> choosen they will be in the order encountered.
        /// If <see cref="ValueToKeyMappingTransformer.SortOrder.Value"/>, items are sorted according to their default comparison, for example, text sorting will be case sensitive (for example, 'A' then 'Z' then 'a').</param>
        public static ValueToKeyMappingEstimator MapValueToKey(this TransformsCatalog.ConversionTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            int maxNumTerms = ValueToKeyMappingEstimator.Defaults.MaxNumTerms,
            ValueToKeyMappingTransformer.SortOrder sort = ValueToKeyMappingEstimator.Defaults.Sort)
           => new ValueToKeyMappingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, maxNumTerms, sort);

        /// <summary>
        /// Converts value types into <see cref="KeyType"/>, optionally loading the keys to use from <paramref name="keyData"/>.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="columns">The data columns to map to keys.</param>
        /// <param name="keyData">The data view containing the terms. If specified, this should be a single column data
        /// view, and the key-values will be taken from taht column. If unspecified, the key-values will be determined
        /// from the input data upon fitting.</param>
        public static ValueToKeyMappingEstimator MapValueToKey(this TransformsCatalog.ConversionTransforms catalog,
            ValueToKeyMappingTransformer.ColumnInfo[] columns, IDataView keyData = null)
            => new ValueToKeyMappingEstimator(CatalogUtils.GetEnvironment(catalog), columns, keyData);

        /// <summary>
        /// Maps specified keys to specified values
        /// </summary>
        /// <typeparam name="TInputType">The key type.</typeparam>
        /// <typeparam name="TOutputType">The value type.</typeparam>
        /// <param name="catalog">The categorical transform's catalog</param>
        /// <param name="keys">The list of keys to use for the mapping. The mapping is 1-1 with values. This list must be the same length as values and
        /// cannot contain duplicate keys.</param>
        /// <param name="values">The list of values to pair with the keys for the mapping. This list must be equal to the same length as keys.</param>
        /// <param name="columns">The columns to apply this transform on.</param>
        /// <returns></returns>
        public static ValueMappingEstimator<TInputType, TOutputType> ValueMap<TInputType, TOutputType>(
            this TransformsCatalog.ConversionTransforms catalog,
            IEnumerable<TInputType> keys,
            IEnumerable<TOutputType> values,
            params (string outputColumnName, string inputColumnName)[] columns)
            => new ValueMappingEstimator<TInputType, TOutputType>(CatalogUtils.GetEnvironment(catalog), keys, values, columns);

        /// <summary>
        /// Maps the <paramref name="columns.input"/> using the keys in the dictionary to the values of dictionary i.e.
        /// a value 'x' in the <paramref name="columns.input"/> would be mappped to a value stored in dictionary[x].
        /// In this case, the <paramref name="lookupMap"/> is used to build up the dictionary where <paramref name="keyColumn"/>
        /// and <paramref name="valueColumn"/> specify the keys and values of dictionary respectively.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog</param>
        /// <param name="lookupMap">An instance of <see cref="IDataView"/> that contains the key and value columns.</param>
        /// <param name="keyColumn">Name of the key column in <paramref name="lookupMap"/>.</param>
        /// <param name="valueColumn">Name of the value column in <paramref name="lookupMap"/>.</param>
        /// <param name="columns">The columns to apply this transform on.</param>
        /// <returns></returns>
        public static ValueMappingEstimator ValueMap(
            this TransformsCatalog.ConversionTransforms catalog,
            IDataView lookupMap, string keyColumn, string valueColumn, params (string outputColumnName, string inputColumnName)[] columns)
            => new ValueMappingEstimator(CatalogUtils.GetEnvironment(catalog), lookupMap, keyColumn, valueColumn, columns);
    }
}
