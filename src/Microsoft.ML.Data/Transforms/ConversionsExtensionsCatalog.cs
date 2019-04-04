// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    using ConvertDefaults = TypeConvertingEstimator.Defaults;
    using HashDefaults = HashingEstimator.Defaults;

    /// <summary>
    /// Extensions for the conversion transformations.
    /// </summary>
    public static class ConversionsExtensionsCatalog
    {
        /// <summary>
        /// Hashes the values in the input column.
        /// </summary>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="numberOfBits">Number of bits to hash into. Must be between 1 and 31, inclusive.</param>
        /// <param name="maximumNumberOfInverts">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="maximumNumberOfInverts"/>Specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        ///  <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[Hash](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Conversion/Hash.cs)]
        /// ]]></format>
        /// </example>

        public static HashingEstimator Hash(this TransformsCatalog.ConversionTransforms catalog, string outputColumnName, string inputColumnName = null,
            int numberOfBits = HashDefaults.NumberOfBits, int maximumNumberOfInverts = HashDefaults.MaximumNumberOfInverts)
            => new HashingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, numberOfBits, maximumNumberOfInverts);

        /// <summary>
        /// Hashes the values in the input column.
        /// </summary>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="columns">Description of dataset columns and how to process them.</param>
        [BestFriend]
        internal static HashingEstimator Hash(this TransformsCatalog.ConversionTransforms catalog, params HashingEstimator.ColumnOptions[] columns)
            => new HashingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Changes column type of the input column.
        /// </summary>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="outputKind">The expected kind of the output column.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[ConvertType](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Conversion/ConvertType.cs)]
        /// ]]></format>
        /// </example>
        public static TypeConvertingEstimator ConvertType(this TransformsCatalog.ConversionTransforms catalog, string outputColumnName, string inputColumnName = null,
            DataKind outputKind = ConvertDefaults.DefaultOutputKind)
            => new TypeConvertingEstimator(CatalogUtils.GetEnvironment(catalog), new[] { new TypeConvertingEstimator.ColumnOptions(outputColumnName, outputKind, inputColumnName) });

        /// <summary>
        /// Changes column type of the input columns.
        /// </summary>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="columns">Specifies the names of the columns on which to apply the transformation.</param>
        /// <param name="outputKind">The expected kind of the output column.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[ConvertType](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Conversion/ConvertTypeMultiColumn.cs)]
        /// ]]></format>
        /// </example>
        public static TypeConvertingEstimator ConvertType(this TransformsCatalog.ConversionTransforms catalog,
            InputOutputColumnPair[] columns,
            DataKind outputKind = ConvertDefaults.DefaultOutputKind)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            env.CheckValue(columns, nameof(columns));
            var columnOptions = columns.Select(x => new TypeConvertingEstimator.ColumnOptions(x.OutputColumnName, outputKind, x.InputColumnName)).ToArray();
            return new TypeConvertingEstimator(env, columnOptions);
        }

        /// <summary>
        /// Changes column type of the input column.
        /// </summary>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="columns">Description of dataset columns and how to process them.</param>
        [BestFriend]
        internal static TypeConvertingEstimator ConvertType(this TransformsCatalog.ConversionTransforms catalog, params TypeConvertingEstimator.ColumnOptions[] columns)
            => new TypeConvertingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Convert the key types back to their original values.
        /// </summary>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[MapKeyToValue](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Conversion/ValueMappingStringToKeyType.cs)]
        /// ]]></format>
        /// </example>
        public static KeyToValueMappingEstimator MapKeyToValue(this TransformsCatalog.ConversionTransforms catalog, string outputColumnName, string inputColumnName = null)
            => new KeyToValueMappingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName);

        /// <summary>
        /// Convert the key types back to their original values.
        /// </summary>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="columns">Specifies the names of the columns on which to apply the transformation.</param>
        public static KeyToValueMappingEstimator MapKeyToValue(this TransformsCatalog.ConversionTransforms catalog, InputOutputColumnPair[] columns)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            env.CheckValue(columns, nameof(columns));
            return new KeyToValueMappingEstimator(env, columns.Select(x => (x.OutputColumnName, x.InputColumnName)).ToArray());
        }

        /// <summary>
        /// Maps key types or key values into a floating point vector.
        /// </summary>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="columns">The input column to map back to vectors.</param>
        [BestFriend]
        internal static KeyToVectorMappingEstimator MapKeyToVector(this TransformsCatalog.ConversionTransforms catalog,
            params KeyToVectorMappingEstimator.ColumnOptions[] columns)
            => new KeyToVectorMappingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Maps key types or key values into a floating point vector.
        /// </summary>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="outputCountVector">Whether to combine multiple indicator vectors into a single vector of counts instead of concatenating them.
        /// This is only relevant when the input column is a vector of keys.</param>
        public static KeyToVectorMappingEstimator MapKeyToVector(this TransformsCatalog.ConversionTransforms catalog,
            string outputColumnName, string inputColumnName = null, bool outputCountVector = KeyToVectorMappingEstimator.Defaults.OutputCountVector)
            => new KeyToVectorMappingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, outputCountVector);

        /// <summary>
        /// Maps columns of key types or key values into columns of floating point vectors.
        /// </summary>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="columns">Specifies the names of the columns on which to apply the transformation.</param>
        /// <param name="outputCountVector">Whether to combine multiple indicator vectors into a single vector of counts instead of concatenating them.
        /// This is only relevant when the input column is a vector of keys.</param>
        public static KeyToVectorMappingEstimator MapKeyToVector(this TransformsCatalog.ConversionTransforms catalog,
            InputOutputColumnPair[] columns, bool outputCountVector = KeyToVectorMappingEstimator.Defaults.OutputCountVector)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            env.CheckValue(columns, nameof(columns));
            var columnOptions = columns.Select(x => new KeyToVectorMappingEstimator.ColumnOptions(x.OutputColumnName, x.InputColumnName, outputCountVector)).ToArray();
            return new KeyToVectorMappingEstimator(env, columnOptions);

        }

        /// <summary>
        /// Converts value types into <see cref="KeyDataViewType"/>.
        /// </summary>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="maximumNumberOfKeys">Maximum number of keys to keep per column when auto-training.</param>
        /// <param name="keyOrdinality">How items should be ordered when vectorized. If <see cref="ValueToKeyMappingEstimator.KeyOrdinality.ByOccurrence"/> choosen they will be in the order encountered.
        /// If <see cref="ValueToKeyMappingEstimator.KeyOrdinality.ByValue"/>, items are sorted according to their default comparison, for example, text sorting will be case sensitive (for example, 'A' then 'Z' then 'a').</param>
        /// <param name="addKeyValueAnnotationsAsText">Whether key value annotations should be text, regardless of the actual input type.</param>
        /// <param name="keyData">The data view containing the terms. If specified, this should be a single column data
        /// view, and the key-values will be taken from that column. If unspecified, the key-values will be determined
        /// from the input data upon fitting.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[MapKeyToValue](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Conversion/KeyToValueToKey.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static ValueToKeyMappingEstimator MapValueToKey(this TransformsCatalog.ConversionTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            int maximumNumberOfKeys = ValueToKeyMappingEstimator.Defaults.MaximumNumberOfKeys,
            ValueToKeyMappingEstimator.KeyOrdinality keyOrdinality = ValueToKeyMappingEstimator.Defaults.Ordinality,
            bool addKeyValueAnnotationsAsText = ValueToKeyMappingEstimator.Defaults.AddKeyValueAnnotationsAsText,
            IDataView keyData = null)
           => new ValueToKeyMappingEstimator(CatalogUtils.GetEnvironment(catalog),
               new[] { new ValueToKeyMappingEstimator.ColumnOptions(outputColumnName, inputColumnName, maximumNumberOfKeys, keyOrdinality, addKeyValueAnnotationsAsText) }, keyData);

        /// <summary>
        /// Converts value types into <see cref="KeyDataViewType"/>.
        /// </summary>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="columns">Specifies the names of the columns on which to apply the transformation.</param>
        /// <param name="maximumNumberOfKeys">Maximum number of keys to keep per column when auto-training.</param>
        /// <param name="keyOrdinality">How items should be ordered when vectorized. If <see cref="ValueToKeyMappingEstimator.KeyOrdinality.ByOccurrence"/> choosen they will be in the order encountered.
        /// If <see cref="ValueToKeyMappingEstimator.KeyOrdinality.ByValue"/>, items are sorted according to their default comparison, for example, text sorting will be case sensitive (for example, 'A' then 'Z' then 'a').</param>
        /// <param name="addKeyValueAnnotationsAsText">Whether key value annotations should be text, regardless of the actual input type.</param>
        /// <param name="keyData">The data view containing the terms. If specified, this should be a single column data
        /// view, and the key-values will be taken from that column. If unspecified, the key-values will be determined
        /// from the input data upon fitting.</param>
        public static ValueToKeyMappingEstimator MapValueToKey(this TransformsCatalog.ConversionTransforms catalog,
            InputOutputColumnPair[] columns,
            int maximumNumberOfKeys = ValueToKeyMappingEstimator.Defaults.MaximumNumberOfKeys,
            ValueToKeyMappingEstimator.KeyOrdinality keyOrdinality = ValueToKeyMappingEstimator.Defaults.Ordinality,
            bool addKeyValueAnnotationsAsText = ValueToKeyMappingEstimator.Defaults.AddKeyValueAnnotationsAsText,
            IDataView keyData = null)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            env.CheckValue(columns, nameof(columns));
            var columnOptions = columns.Select(x => new ValueToKeyMappingEstimator.ColumnOptions(x.OutputColumnName, x.InputColumnName, maximumNumberOfKeys, keyOrdinality, addKeyValueAnnotationsAsText)).ToArray();
            return new ValueToKeyMappingEstimator(env, columnOptions, keyData);
        }

        /// <summary>
        /// Converts value types into <see cref="KeyDataViewType"/>, optionally loading the keys to use from <paramref name="keyData"/>.
        /// </summary>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="columns">The data columns to map to keys.</param>
        /// <param name="keyData">The data view containing the terms. If specified, this should be a single column data
        /// view, and the key-values will be taken from that column. If unspecified, the key-values will be determined
        /// from the input data upon fitting.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[MapKeyToValue](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Conversion/KeyToValueValueToKey.cs)]
        /// ]]>
        /// </format>
        /// </example>
        [BestFriend]
        internal static ValueToKeyMappingEstimator MapValueToKey(this TransformsCatalog.ConversionTransforms catalog,
            ValueToKeyMappingEstimator.ColumnOptions[] columns, IDataView keyData = null)
            => new ValueToKeyMappingEstimator(CatalogUtils.GetEnvironment(catalog), columns, keyData);

        /// <summary>
        /// <see cref="ValueMappingEstimator"/>
        /// </summary>
        /// <typeparam name="TInputType">The key type.</typeparam>
        /// <typeparam name="TOutputType">The value type.</typeparam>
        /// <param name="catalog">The conversion transform's catalog</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="keyValuePairs">Specifies the mapping that will be perfomed. The keys will be mapped to the values as specified in the <paramref name="keyValuePairs"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="treatValuesAsKeyType">Whether to treat the values as a <see cref="KeyDataViewType"/>.</param>
        /// <returns>An instance of the <see cref="ValueMappingEstimator"/></returns>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[MapValue](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Conversion/MapValue.cs)]
        /// ]]></format>
        /// </example>
        public static ValueMappingEstimator<TInputType, TOutputType> MapValue<TInputType, TOutputType>(
            this TransformsCatalog.ConversionTransforms catalog,
            string outputColumnName,
            IEnumerable<KeyValuePair<TInputType, TOutputType>> keyValuePairs,
            string inputColumnName = null,
            bool treatValuesAsKeyType = false)
        {
            var keys = keyValuePairs.Select(pair => pair.Key);
            var values = keyValuePairs.Select(pair => pair.Value);

            var lookupMap = DataViewHelper.CreateDataView(catalog.GetEnvironment(), keys, values,
                ValueMappingTransformer.DefaultKeyColumnName,
                ValueMappingTransformer.DefaultValueColumnName, treatValuesAsKeyType);

            return new ValueMappingEstimator<TInputType, TOutputType>(catalog.GetEnvironment(), lookupMap,
                lookupMap.Schema[ValueMappingTransformer.DefaultKeyColumnName],
                lookupMap.Schema[ValueMappingTransformer.DefaultValueColumnName],
                new[] { (outputColumnName, inputColumnName ?? outputColumnName) });
        }

        /// <summary>
        /// <see cref="ValueMappingEstimator"/>
        /// </summary>
        /// <typeparam name="TInputType">The key type.</typeparam>
        /// <typeparam name="TOutputType">The value type.</typeparam>
        /// <param name="catalog">The conversion transform's catalog</param>
        /// <param name="keyValuePairs">Specifies the mapping that will be perfomed. The keys will be mapped to the values as specified in the <paramref name="keyValuePairs"/>.</param>
        /// <param name="columns">The columns to apply this transform on.</param>
        /// <returns>An instance of the <see cref="ValueMappingEstimator"/></returns>
        [BestFriend]
        internal static ValueMappingEstimator<TInputType, TOutputType> MapValue<TInputType, TOutputType>(
            this TransformsCatalog.ConversionTransforms catalog,
            IEnumerable<KeyValuePair<TInputType, TOutputType>> keyValuePairs,
            params InputOutputColumnPair[] columns)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            env.CheckValue(columns, nameof(columns));
            var keys = keyValuePairs.Select(pair => pair.Key);
            var values = keyValuePairs.Select(pair => pair.Value);

            var lookupMap = DataViewHelper.CreateDataView(catalog.GetEnvironment(), keys, values,
                ValueMappingTransformer.DefaultKeyColumnName,
                ValueMappingTransformer.DefaultValueColumnName, false);

            return new ValueMappingEstimator<TInputType, TOutputType>(catalog.GetEnvironment(), lookupMap,
                lookupMap.Schema[ValueMappingTransformer.DefaultKeyColumnName],
                lookupMap.Schema[ValueMappingTransformer.DefaultValueColumnName],
                InputOutputColumnPair.ConvertToValueTuples(columns));
        }

        /// <summary>
        /// <see cref="ValueMappingEstimator"/>
        /// </summary>
        /// <typeparam name="TInputType">The key type.</typeparam>
        /// <typeparam name="TOutputType">The value type.</typeparam>
        /// <param name="catalog">The conversion transform's catalog</param>
        /// <param name="keyValuePairs">Specifies the mapping that will be perfomed. The keys will be mapped to the values as specified in the <paramref name="keyValuePairs"/>.</param>
        /// <param name="treatValuesAsKeyType">Whether to treat the values as a <see cref="KeyDataViewType"/>.</param>
        /// <param name="columns">The columns to apply this transform on.</param>
        /// <returns>An instance of the <see cref="ValueMappingEstimator"/></returns>
        [BestFriend]
        internal static ValueMappingEstimator<TInputType, TOutputType> MapValue<TInputType, TOutputType>(
            this TransformsCatalog.ConversionTransforms catalog,
            IEnumerable<KeyValuePair<TInputType, TOutputType>> keyValuePairs,
            bool treatValuesAsKeyType,
            params InputOutputColumnPair[] columns)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            env.CheckValue(columns, nameof(columns));
            var keys = keyValuePairs.Select(pair => pair.Key);
            var values = keyValuePairs.Select(pair => pair.Value);

            var lookupMap = DataViewHelper.CreateDataView(catalog.GetEnvironment(), keys, values,
                ValueMappingTransformer.DefaultKeyColumnName,
                ValueMappingTransformer.DefaultValueColumnName, treatValuesAsKeyType);

            return new ValueMappingEstimator<TInputType, TOutputType>(catalog.GetEnvironment(), lookupMap,
                lookupMap.Schema[ValueMappingTransformer.DefaultKeyColumnName],
                lookupMap.Schema[ValueMappingTransformer.DefaultValueColumnName],
                InputOutputColumnPair.ConvertToValueTuples(columns));
        }

        /// <summary>
        /// <see cref="ValueMappingEstimator"/>
        /// </summary>
        /// <typeparam name="TInputType">The key type.</typeparam>
        /// <typeparam name="TOutputType">The value type.</typeparam>
        /// <param name="catalog">The conversion transform's catalog</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="keyValuePairs">Specifies the mapping that will be perfomed. The keys will be mapped to the values as specified in the <paramref name="keyValuePairs"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <returns>An instance of the <see cref="ValueMappingEstimator"/></returns>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[MapValue](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Conversion/MappValueToArray.cs)]
        /// ]]></format>
        /// </example>
        public static ValueMappingEstimator<TInputType, TOutputType> MapValue<TInputType, TOutputType>(
            this TransformsCatalog.ConversionTransforms catalog,
            string outputColumnName,
            IEnumerable<KeyValuePair<TInputType, TOutputType[]>> keyValuePairs,
            string inputColumnName = null)
        {
            var keys = keyValuePairs.Select(pair => pair.Key);
            var values = keyValuePairs.Select(pair => pair.Value);

            // Convert parallel key and value lists to IDataView with two columns, so that the underlying infra can use it.
            var lookupMap = DataViewHelper.CreateDataView(catalog.GetEnvironment(), keys, values,
                ValueMappingTransformer.DefaultKeyColumnName,
                ValueMappingTransformer.DefaultValueColumnName);

            return new ValueMappingEstimator<TInputType, TOutputType>(catalog.GetEnvironment(), lookupMap,
                lookupMap.Schema[ValueMappingTransformer.DefaultKeyColumnName],
                lookupMap.Schema[ValueMappingTransformer.DefaultValueColumnName],
                new[] { (outputColumnName, inputColumnName ?? outputColumnName) });
        }

        /// <summary>
        /// <see cref="ValueMappingEstimator"/>
        /// </summary>
        /// <typeparam name="TInputType">The key type.</typeparam>
        /// <typeparam name="TOutputType">The value type.</typeparam>
        /// <param name="catalog">The conversion transform's catalog</param>
        /// <param name="keyValuePairs">Specifies the mapping that will be perfomed. The keys will be mapped to the values as specified in the <paramref name="keyValuePairs"/>.</param>
        /// <param name="columns">The columns to apply this transform on.</param>
        /// <returns>An instance of the <see cref="ValueMappingEstimator"/></returns>
        [BestFriend]
        internal static ValueMappingEstimator<TInputType, TOutputType> MapValue<TInputType, TOutputType>(
            this TransformsCatalog.ConversionTransforms catalog,
            IEnumerable<KeyValuePair<TInputType, TOutputType[]>> keyValuePairs,
            params InputOutputColumnPair[] columns)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            env.CheckValue(columns, nameof(columns));
            var keys = keyValuePairs.Select(pair => pair.Key);
            var values = keyValuePairs.Select(pair => pair.Value);

            var lookupMap = DataViewHelper.CreateDataView(catalog.GetEnvironment(), keys, values,
                ValueMappingTransformer.DefaultKeyColumnName,
                ValueMappingTransformer.DefaultValueColumnName);

            return new ValueMappingEstimator<TInputType, TOutputType>(catalog.GetEnvironment(), lookupMap,
                lookupMap.Schema[ValueMappingTransformer.DefaultKeyColumnName],
                lookupMap.Schema[ValueMappingTransformer.DefaultValueColumnName],
                InputOutputColumnPair.ConvertToValueTuples(columns));
        }

        /// <summary>
        /// <see cref="ValueMappingEstimator"/>
        /// </summary>
        /// <param name="catalog">The conversion transform's catalog</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="lookupMap">An instance of <see cref="IDataView"/> that contains the <paramref name="keyColumn"/> and <paramref name="valueColumn"/> columns.</param>
        /// <param name="keyColumn">The key column in <paramref name="lookupMap"/>.</param>
        /// <param name="valueColumn">The value column in <paramref name="lookupMap"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <returns>A instance of the ValueMappingEstimator</returns>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[MapValue](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Conversion/MappValueIdvLookup.cs)]
        /// ]]></format>
        /// </example>
        public static ValueMappingEstimator MapValue(
            this TransformsCatalog.ConversionTransforms catalog,
            string outputColumnName, IDataView lookupMap, DataViewSchema.Column keyColumn, DataViewSchema.Column valueColumn, string inputColumnName = null)
        {
            return new ValueMappingEstimator(CatalogUtils.GetEnvironment(catalog), lookupMap, keyColumn, valueColumn,
              new[] { (outputColumnName, inputColumnName ?? outputColumnName) });
        }

        /// <summary>
        /// <see cref="ValueMappingEstimator"/>
        /// </summary>
        /// <param name="catalog">The conversion transform's catalog</param>
        /// <param name="lookupMap">An instance of <see cref="IDataView"/> that contains the <paramref name="keyColumn"/> and <paramref name="valueColumn"/> columns.</param>
        /// <param name="keyColumn">The key column in <paramref name="lookupMap"/>.</param>
        /// <param name="valueColumn">The value column in <paramref name="lookupMap"/>.</param>
        /// <param name="columns">The columns to apply this transform on.</param>
        /// <returns>A instance of the ValueMappingEstimator</returns>
        [BestFriend]
        internal static ValueMappingEstimator MapValue(
            this TransformsCatalog.ConversionTransforms catalog,
            IDataView lookupMap, DataViewSchema.Column keyColumn, DataViewSchema.Column valueColumn, params InputOutputColumnPair[] columns)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            env.CheckValue(columns, nameof(columns));
            return new ValueMappingEstimator(env, lookupMap, keyColumn, valueColumn, InputOutputColumnPair.ConvertToValueTuples(columns));
        }
    }
}
