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
    using static Microsoft.ML.Transforms.HashingEstimator;
    using ConvertDefaults = TypeConvertingEstimator.Defaults;
    using HashDefaults = HashingEstimator.Defaults;

    /// <summary>
    /// Collection of extension methods for <see cref="TransformsCatalog"/> to create instances of data conversion and mapping transformer components.
    /// </summary>
    public static class ConversionsExtensionsCatalog
    {
        /// <summary>
        /// Create a <see cref="HashingEstimator"/>, which hashes the data from the column specified in <paramref name="inputColumnName"/>
        /// to a new column: <paramref name="outputColumnName"/>.
        /// </summary>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// This column's data type will be a vector of keys, or a scalar of key based on whether the input column data types
        /// are vectors or scalars.</param>
        /// <param name="inputColumnName">Name of the column whose data will be hashed.
        /// If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.
        /// This estimator operates over vectors or scalars of text, numeric, boolean, key or <see cref="DataViewRowId"/> data types. </param>
        /// <param name="numberOfBits">Number of bits to hash into. Must be between 1 and 31, inclusive.</param>
        /// <param name="maximumNumberOfInverts">During hashing we construct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the annotations for the new column.Hashing, as such, can map many initial values to one.
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
        /// Create a <see cref="HashingEstimator"/>, which hashes the input column's data type <see cref="ColumnOptions.InputColumnName" />
        /// to a new column: <see cref="ColumnOptions.Name" />.
        /// </summary>
        /// <remarks>This transform can operate over several columns.</remarks>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">Advanced options for the estimator that also contain the input and output column names.
        /// This estimator operates over text, numeric, boolean, key and <see cref="DataViewRowId"/> data types.
        /// The new column's data type will be a vector of <see cref="System.UInt32"/>, or a <see cref="System.UInt32"/> based on whether the input column data types
        /// are vectors or scalars.</param>
        ///  <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[Hash](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Conversion/HashWithOptions.cs)]
        /// ]]></format>
        /// </example>
        public static HashingEstimator Hash(this TransformsCatalog.ConversionTransforms catalog, params ColumnOptions[] columns)
            => new HashingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Create a <see cref="TypeConvertingEstimator"/>, which converts the type of the data to the type specified in <paramref name="outputKind"/>.
        /// </summary>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.
        /// This transform operates over numeric, boolean, text, <see cref="System.DateTime"/> and key data types.</param>
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
        /// Create a <see cref="TypeConvertingEstimator"/>, which converts the type of the data to the type specified in <paramref name="outputKind"/>.
        /// </summary>
        /// <remarks>This transform can operate over several columns.</remarks>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="columns">The input and output columns.
        /// This transform operates over numeric, boolean, text, <see cref="System.DateTime"/> and key data types.</param>
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
        /// Create a <see cref="TypeConvertingEstimator"/>, which converts the type of the data in the <see cref="TypeConvertingEstimator.ColumnOptions.InputColumnName"/>
        /// to the type specified in the <see cref="TypeConvertingEstimator.ColumnOptions.OutputKind"/>
        /// </summary>
        /// <remarks>This transform can operate over several columns.</remarks>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="columns">The input and output columns.
        /// This transform operates over numeric, boolean, text, <see cref="System.DateTime"/> and keys.</param>
        [BestFriend]
        internal static TypeConvertingEstimator ConvertType(this TransformsCatalog.ConversionTransforms catalog, params TypeConvertingEstimator.ColumnOptions[] columns)
            => new TypeConvertingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Create a <see cref="KeyToValueMappingEstimator"/>, which converts the key types back to their original values.
        /// </summary>
        /// <remarks>This transform often is in the pipeline after one of the overloads of
        /// <see cref="MapValueToKey(TransformsCatalog.ConversionTransforms, InputOutputColumnPair[], int, ValueToKeyMappingEstimator.KeyOrdinality, bool, IDataView)"/></remarks>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// Its type will be the original value's type.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.
        /// This transform operates over keys.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[MapKeyToValue](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Conversion/KeyToValueToKey.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static KeyToValueMappingEstimator MapKeyToValue(this TransformsCatalog.ConversionTransforms catalog, string outputColumnName, string inputColumnName = null)
            => new KeyToValueMappingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName);

        /// <summary>
        /// Create a <see cref="KeyToValueMappingEstimator"/>, which converts the key types back to their original values.
        /// </summary>
        /// <remarks>This transform can operate over several columns.
        /// This transform often is in the pipeline after one of the overloads of
        /// <see cref="MapValueToKey(TransformsCatalog.ConversionTransforms, InputOutputColumnPair[], int, ValueToKeyMappingEstimator.KeyOrdinality, bool, IDataView)"/></remarks>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="columns">The input and output columns.
        /// This transform operates over keys.
        /// The new column's data type will be the original value's type.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[MapKeyToValue](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Conversion/MapKeyToValueMultiColumn.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static KeyToValueMappingEstimator MapKeyToValue(this TransformsCatalog.ConversionTransforms catalog, InputOutputColumnPair[] columns)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            env.CheckValue(columns, nameof(columns));
            return new KeyToValueMappingEstimator(env, columns.Select(x => (x.OutputColumnName, x.InputColumnName)).ToArray());
        }

        /// <summary>
        /// Create a <see cref="KeyToVectorMappingEstimator"/>, which maps the value of a key into a floating point vector representing the value.
        /// </summary>
        /// <remarks>This transform can operate over several columns.</remarks>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="columns">The input and output columns.</param>
        [BestFriend]
        internal static KeyToVectorMappingEstimator MapKeyToVector(this TransformsCatalog.ConversionTransforms catalog,
            params KeyToVectorMappingEstimator.ColumnOptions[] columns)
            => new KeyToVectorMappingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Create a <see cref="KeyToVectorMappingEstimator"/>, which maps the value of a key into a floating point vector representing the value.
        /// </summary>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// The data type is a vector of <see cref="System.Single"/> representing the input value.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.
        /// This transform operates over keys.</param>
        /// <param name="outputCountVector">Whether to combine multiple indicator vectors into a single vector of counts instead of concatenating them.
        /// This is only relevant when the input column is a vector of keys.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[MapKeyToVector](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Conversion/MapKeyToVector.cs)]
        /// ]]></format>
        /// </example>
        public static KeyToVectorMappingEstimator MapKeyToVector(this TransformsCatalog.ConversionTransforms catalog,
            string outputColumnName, string inputColumnName = null, bool outputCountVector = KeyToVectorMappingEstimator.Defaults.OutputCountVector)
            => new KeyToVectorMappingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, outputCountVector);

        /// <summary>
        /// Create a <see cref="KeyToVectorMappingEstimator"/>, which maps the value of a key into a floating point vector representing the value.
        /// </summary>
        /// <remarks>This transform can operate over several columns of keys.</remarks>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="columns">The input and output columns.
        /// The new column's data type is a vector of <see cref="System.Single"/> representing the original value.</param>
        /// <param name="outputCountVector">Whether to combine multiple indicator vectors into a single vector of counts instead of concatenating them.
        /// This is only relevant when the input column is a vector of keys.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[MapKeyToVector](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Conversion/MapKeyToVectorMultiColumn.cs)]
        /// ]]></format>
        /// </example>
        public static KeyToVectorMappingEstimator MapKeyToVector(this TransformsCatalog.ConversionTransforms catalog,
            InputOutputColumnPair[] columns, bool outputCountVector = KeyToVectorMappingEstimator.Defaults.OutputCountVector)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            env.CheckValue(columns, nameof(columns));
            var columnOptions = columns.Select(x => new KeyToVectorMappingEstimator.ColumnOptions(x.OutputColumnName, x.InputColumnName, outputCountVector)).ToArray();
            return new KeyToVectorMappingEstimator(env, columnOptions);

        }

        /// <summary>
        /// Create a <see cref="ValueToKeyMappingEstimator"/>, which converts categorical values into numerical keys.
        /// </summary>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column containing the keys.</param>
        /// <param name="inputColumnName">Name of the column containing the categorical values. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> is used.
        /// The input data types can be numeric, text, boolean, <see cref="System.DateTime"/> or <see cref="System.DateTimeOffset"/>.
        /// </param>
        /// <param name="maximumNumberOfKeys">Maximum number of keys to keep per column when training.</param>
        /// <param name="keyOrdinality">The order in which keys are assigned.
        /// If set to <see cref="ValueToKeyMappingEstimator.KeyOrdinality.ByOccurrence"/>, keys are assigned in the order encountered.
        /// If set to <see cref="ValueToKeyMappingEstimator.KeyOrdinality.ByValue"/>, values are sorted, and keys are assigned based on the sort order.</param>
        /// <param name="addKeyValueAnnotationsAsText">If set to true, use text type
        /// for values, regardless of the actual input type. When doing the reverse
        /// mapping, the values are text rather than the original input type.</param>
        /// <param name="keyData">Use a pre-defined mapping between values and keys, instead of building
        /// the mapping from the input data during training. If specified, this should be a single column <see cref="IDataView"/> containing the values.
        /// The keys are allocated based on the value of keyOrdinality.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[MapValueToKey](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Conversion/KeyToValueToKey.cs)]
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
        /// Create a <see cref="ValueToKeyMappingEstimator"/>, which converts categorical values into keys.
        /// </summary>
        /// <remarks>This transform can operate over multiple pairs of columns, creating a mapping for each pair.</remarks>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="columns">The input and output columns.
        /// The input data types can be numeric, text, boolean, <see cref="System.DateTime"/> or <see cref="System.DateTimeOffset"/>.
        /// </param>
        /// <param name="maximumNumberOfKeys">Maximum number of keys to keep per column when training.</param>
        /// <param name="keyOrdinality">The order in which keys are assigned.
        /// If set to <see cref="ValueToKeyMappingEstimator.KeyOrdinality.ByOccurrence"/>, keys are assigned in the order encountered.
        /// If set to <see cref="ValueToKeyMappingEstimator.KeyOrdinality.ByValue"/>, values are sorted, and keys are assigned based on the sort order.</param>
        /// <param name="addKeyValueAnnotationsAsText">If set to true, use text type
        /// for values, regardless of the actual input type. When doing the reverse
        /// mapping, the values are text rather than the original input type.</param>
        /// <param name="keyData">Use a pre-defined mapping between values and keys, instead of building
        /// the mapping from the input data during training. If specified, this should be a single column <see cref="IDataView"/> containing the values.
        /// The keys are allocated based on the value of keyOrdinality.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[MapValueToKey](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Conversion/MapValueToKeyMultiColumn.cs)]
        /// ]]>
        /// </format>
        /// </example>
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
        /// Create a <see cref="ValueToKeyMappingEstimator"/>, which converts value types into keys, optionally loading the keys to use from <paramref name="keyData"/>.
        /// </summary>
        /// <remarks>This transform can operate over several columns.</remarks>
        /// <param name="catalog">The conversion transform's catalog.</param>
        /// <param name="columns">The input and output columns.
        /// The input data types can be numeric, text, boolean, <see cref="System.DateTime"/> or <see cref="System.DateTimeOffset"/>.
        /// </param>
        /// <param name="keyData">The data view containing the terms. If specified, this should be a single column data
        /// view, and the key-values will be taken from that column. If unspecified, the key-values will be determined
        /// from the input data upon fitting.</param>
        [BestFriend]
        internal static ValueToKeyMappingEstimator MapValueToKey(this TransformsCatalog.ConversionTransforms catalog,
            ValueToKeyMappingEstimator.ColumnOptions[] columns, IDataView keyData = null)
            => new ValueToKeyMappingEstimator(CatalogUtils.GetEnvironment(catalog), columns, keyData);

        /// <summary>
        /// Create a <see cref="ValueMappingEstimator"/>, which converts value types into keys, loading the keys to use from <paramref name="keyValuePairs"/>.
        /// </summary>
        /// <typeparam name="TInputType">The key type.</typeparam>
        /// <typeparam name="TOutputType">The value type.</typeparam>
        /// <param name="catalog">The conversion transform's catalog</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// The output data types can be primitives or vectors of numeric, text, boolean, <see cref="System.DateTime"/>, <see cref="System.DateTimeOffset"/> or <see cref="DataViewRowId"/> types.</param>
        /// <param name="keyValuePairs">Specifies the mapping that will be performed. The keys will be mapped to the values as specified in the <paramref name="keyValuePairs"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform.
        /// If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.
        /// The input data types can be primitives or vectors of numeric, text, boolean, <see cref="System.DateTime"/>, <see cref="System.DateTimeOffset"/> or <see cref="DataViewRowId"/> types.
        /// </param>
        /// <param name="treatValuesAsKeyType">Whether to treat the values as a key.</param>
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
        /// Create a <see cref="ValueMappingEstimator"/>, which converts value types into keys, loading the keys to use from <paramref name="keyValuePairs"/>.
        /// </summary>
        /// <remarks>This transform can operate over several columns.</remarks>
        /// <typeparam name="TInputType">The key type.</typeparam>
        /// <typeparam name="TOutputType">The value type.</typeparam>
        /// <param name="catalog">The conversion transform's catalog</param>
        /// <param name="keyValuePairs">Specifies the mapping that will be performed. The keys will be mapped to the values as specified in the <paramref name="keyValuePairs"/>.</param>
        /// <param name="columns">The input and output columns.
        /// The data types can be primitives or vectors of numeric, text, boolean, <see cref="System.DateTime"/>, <see cref="System.DateTimeOffset"/> or <see cref="DataViewRowId"/> types.
        /// </param>
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
        /// Create a <see cref="ValueMappingEstimator"/>, which converts value types into keys, loading the keys to use from <paramref name="keyValuePairs"/>.
        /// </summary>
        /// <remarks>This transform can operate over several columns.</remarks>
        /// <typeparam name="TInputType">The key type.</typeparam>
        /// <typeparam name="TOutputType">The value type.</typeparam>
        /// <param name="catalog">The conversion transform's catalog</param>
        /// <param name="keyValuePairs">Specifies the mapping that will be performed. The keys will be mapped to the values as specified in the <paramref name="keyValuePairs"/>.</param>
        /// <param name="treatValuesAsKeyType">Whether to treat the values as a keys.</param>
        /// <param name="columns">The input and output columns.
        /// The data types can be primitives or vectors of numeric, text, boolean, <see cref="System.DateTime"/>, <see cref="System.DateTimeOffset"/> or <see cref="DataViewRowId"/> types.
        /// </param>
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
        /// Create a <see cref="ValueMappingEstimator"/>, which converts value types into keys, loading the keys to use from <paramref name="keyValuePairs"/>.
        /// </summary>
        /// <typeparam name="TInputType">The key type.</typeparam>
        /// <typeparam name="TOutputType">The value type.</typeparam>
        /// <param name="catalog">The conversion transform's catalog</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// The data types can be primitives or vectors of numeric, text, boolean, <see cref="System.DateTime"/>, <see cref="System.DateTimeOffset"/> or <see cref="DataViewRowId"/> types,
        /// as specified in the <typeparamref name="TOutputType"/>.</param>
        /// <param name="keyValuePairs">Specifies the mapping that will be performed. The keys will be mapped to the values as specified in the <paramref name="keyValuePairs"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.
        /// The data types can be primitives or vectors of numeric, text, boolean, <see cref="System.DateTime"/>, <see cref="System.DateTimeOffset"/> or <see cref="DataViewRowId"/> types,
        /// as specified in the <typeparamref name="TInputType"/>.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[MapValue](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Conversion/MapValueToArray.cs)]
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
        /// Create a <see cref="ValueMappingEstimator"/>, which converts value types into keys, loading the keys to use from <paramref name="keyValuePairs"/>.
        /// </summary>
        /// <remarks>This transform can operate over several columns.</remarks>
        /// <typeparam name="TInputType">The key type.</typeparam>
        /// <typeparam name="TOutputType">The value type.</typeparam>
        /// <param name="catalog">The conversion transform's catalog</param>
        /// <param name="keyValuePairs">Specifies the mapping that will be performed. The keys will be mapped to the values as specified in the <paramref name="keyValuePairs"/>.</param>
        /// <param name="columns">The input and output columns. The data types can be primitives or vectors of numeric, text, boolean, <see cref="System.DateTime"/>,
        /// <see cref="System.DateTimeOffset"/> or <see cref="DataViewRowId"/> types, as specified in the <typeparamref name="TInputType"/> and <typeparamref name="TOutputType"/>.</param>
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
        /// Create a <see cref="ValueMappingEstimator"/>, which converts value types into keys, loading the keys to use from the <paramref name="lookupMap"/> where the <paramref name="keyColumn"/>
        /// specifies the keys, and the <paramref name="valueColumn"/> the respective value.
        /// </summary>
        /// <param name="catalog">The conversion transform's catalog</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// The data types can be primitives or vectors of numeric, text, boolean, <see cref="System.DateTime"/>, <see cref="System.DateTimeOffset"/>
        /// or <see cref="DataViewRowId"/> types.</param>
        /// <param name="lookupMap">An instance of <see cref="IDataView"/> that contains the <paramref name="keyColumn"/> and <paramref name="valueColumn"/> columns.</param>
        /// <param name="keyColumn">The key column in <paramref name="lookupMap"/>.</param>
        /// <param name="valueColumn">The value column in <paramref name="lookupMap"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.
        /// The data types can be primitives or vectors of numeric, text, boolean, <see cref="System.DateTime"/>, <see cref="System.DateTimeOffset"/>
        /// or <see cref="DataViewRowId"/> types.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[MapValue](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Conversion/MapValueIdvLookup.cs)]
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
        /// Create a <see cref="ValueMappingEstimator"/>, which converts value types into keys,
        /// loading the keys to use from the <paramref name="lookupMap"/> where the <paramref name="keyColumn"/>
        /// specifies the keys, and the <paramref name="valueColumn"/> the respective value.
        /// </summary>
        /// <remarks>This transform can operate over several columns.</remarks>
        /// <param name="catalog">The conversion transform's catalog</param>
        /// <param name="lookupMap">An instance of <see cref="IDataView"/> that contains the <paramref name="keyColumn"/> and <paramref name="valueColumn"/> columns.</param>
        /// <param name="keyColumn">The key column in <paramref name="lookupMap"/>.</param>
        /// <param name="valueColumn">The value column in <paramref name="lookupMap"/>.</param>
        /// <param name="columns">The input and output columns.</param>
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
