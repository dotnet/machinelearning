// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(ValueMappingTransformer.Summary, typeof(IDataTransform), typeof(ValueMappingTransformer),
    typeof(ValueMappingTransformer.Options), typeof(SignatureDataTransform),
    ValueMappingTransformer.UserName, "ValueMapping", "ValueMappingTransformer", ValueMappingTransformer.ShortName,
    "TermLookup", "Lookup", "LookupTransform", DocName = "transform/ValueMappingTransformer.md")]

[assembly: LoadableClass(ValueMappingTransformer.Summary, typeof(IDataTransform), typeof(ValueMappingTransformer), null, typeof(SignatureLoadDataTransform),
    "Value Mapping Transform", ValueMappingTransformer.LoaderSignature, ValueMappingTransformer.TermLookupLoaderSignature)]

[assembly: LoadableClass(ValueMappingTransformer.Summary, typeof(ValueMappingTransformer), null, typeof(SignatureLoadModel),
    "Value Mapping Transform", ValueMappingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ValueMappingTransformer), null, typeof(SignatureLoadRowMapper),
    ValueMappingTransformer.UserName, ValueMappingTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    /// <include file='doc.xml' path='doc/members/member[@name="ValueMappingEstimator"]/*' />
    public class ValueMappingEstimator : TrivialEstimator<ValueMappingTransformer>
    {
        private readonly (string outputColumnName, string inputColumnName)[] _columns;

        /// <summary>
        /// Constructs the ValueMappingEstimator, key type -> value type mapping
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="lookupMap">An instance of <see cref="IDataView"/> that contains the key and value columns.</param>
        /// <param name="keyColumn">Name of the key column in <paramref name="lookupMap"/>.</param>
        /// <param name="valueColumn">Name of the value column in <paramref name="lookupMap"/>.</param>
        /// <param name="columns">The list of names of the input columns to apply the transformation, and the name of the resulting column.</param>
        internal ValueMappingEstimator(IHostEnvironment env, IDataView lookupMap, string keyColumn, string valueColumn, params (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ValueMappingEstimator)),
                    new ValueMappingTransformer(env, lookupMap, keyColumn, valueColumn, columns))
        {
            _columns = columns;
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public sealed override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            var resultDic = inputSchema.ToDictionary(x => x.Name);

            // Decide the output VectorKind for the columns
            // based on the value type of the dictionary
            var vectorKind = SchemaShape.Column.VectorKind.Scalar;
            if (Transformer.ValueColumnType is VectorType)
            {
                vectorKind = SchemaShape.Column.VectorKind.Vector;
                if (Transformer.ValueColumnType.GetVectorSize() == 0)
                    vectorKind = SchemaShape.Column.VectorKind.VariableVector;
            }

            // Set the data type of the output column
            // if the output VectorKind is a vector or variable vector then
            // this is the data type of items stored in the vector.
            var isKey = Transformer.ValueColumnType is KeyType;
            var columnType = (isKey) ? NumberDataViewType.UInt32 :
                                    Transformer.ValueColumnType.GetItemType();
            var metadataShape = SchemaShape.Create(Transformer.ValueColumnMetadata.Schema);
            foreach (var (outputColumnName, inputColumnName) in _columns)
            {
                if (!inputSchema.TryFindColumn(inputColumnName, out var originalColumn))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", inputColumnName);

                if (originalColumn.Kind == SchemaShape.Column.VectorKind.VariableVector ||
                   originalColumn.Kind == SchemaShape.Column.VectorKind.Vector)
                {
                    if (Transformer.ValueColumnType is VectorType)
                        throw Host.ExceptNotSupp("Column '{0}' cannot be mapped to values when the column and the map values are both vector type.", inputColumnName);
                    // if input to the estimator is of vector type then output should always be vector.
                    // The transformer maps each item in input vector to the values in the dictionary
                    // producing a vector as output.
                    vectorKind = originalColumn.Kind;
                }
                // Create the Value column
                var col = new SchemaShape.Column(outputColumnName, vectorKind, columnType, isKey, metadataShape);
                resultDic[outputColumnName] = col;
            }
            return new SchemaShape(resultDic.Values);
        }
    }

    /// <include file='doc.xml' path='doc/members/member[@name="ValueMappingEstimator"]/*' />
    /// <typeparam name="TKey">Specifies the key type.</typeparam>
    /// <typeparam name="TValue">Specifies the value type.</typeparam>
    public sealed class ValueMappingEstimator<TKey, TValue> : ValueMappingEstimator
    {
        private (string outputColumnName, string inputColumnName)[] _columns;

        /// <summary>
        /// Constructs the ValueMappingEstimator, key type -> value type mapping
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="keys">The list of keys of TKey.</param>
        /// <param name="values">The list of values of TValue.</param>
        /// <param name="columns">The list of columns to apply.</param>
        internal ValueMappingEstimator(IHostEnvironment env, IEnumerable<TKey> keys, IEnumerable<TValue> values, params (string outputColumnName, string inputColumnName)[] columns)
            : base(env, DataViewHelper.CreateDataView(env, keys, values, ValueMappingTransformer.KeyColumnName, ValueMappingTransformer.ValueColumnName, false), ValueMappingTransformer.KeyColumnName, ValueMappingTransformer.ValueColumnName, columns)
        {
            _columns = columns;
        }

        /// <summary>
        /// Constructs the ValueMappingEstimator, key type -> value type mapping
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="keys">The list of keys of TKey.</param>
        /// <param name="values">The list of values of TValue.</param>
        /// <param name="treatValuesAsKeyType">Specifies to treat the values as a <see cref="KeyType"/>.</param>
        /// <param name="columns">The list of columns to apply.</param>
        internal ValueMappingEstimator(IHostEnvironment env, IEnumerable<TKey> keys, IEnumerable<TValue> values, bool treatValuesAsKeyType, params (string outputColumnName, string inputColumnName)[] columns)
            : base(env, DataViewHelper.CreateDataView(env, keys, values, ValueMappingTransformer.KeyColumnName, ValueMappingTransformer.ValueColumnName, treatValuesAsKeyType), ValueMappingTransformer.KeyColumnName, ValueMappingTransformer.ValueColumnName, columns)
        {
            _columns = columns;
        }

        /// <summary>
        /// Constructs the ValueMappingEstimator, key type -> value array type mapping
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="keys">The list of keys of TKey.</param>
        /// <param name="values">The list of values of TValue[].</param>
        /// <param name="columns">The list of columns to apply.</param>
        internal ValueMappingEstimator(IHostEnvironment env, IEnumerable<TKey> keys, IEnumerable<TValue[]> values, params (string outputColumnName, string inputColumnName)[] columns)
            : base(env, DataViewHelper.CreateDataView(env, keys, values, ValueMappingTransformer.KeyColumnName, ValueMappingTransformer.ValueColumnName), ValueMappingTransformer.KeyColumnName, ValueMappingTransformer.ValueColumnName, columns)
        {
            _columns = columns;
        }
    }

    /// <summary>
    /// The DataViewHelper provides a set of static functions to create a DataView given a list of keys and values.
    /// </summary>
    internal class DataViewHelper
    {
        /// <summary>
        /// Helper function to retrieve the Primitie type given a Type
        /// </summary>
        internal static PrimitiveDataViewType GetPrimitiveType(Type rawType, out bool isVectorType)
        {
            Type type = rawType;
            isVectorType = false;
            if (type.IsArray)
            {
                type = rawType.GetElementType();
                isVectorType = true;
            }

            if (!type.TryGetDataKind(out InternalDataKind kind))
                throw new InvalidOperationException($"Unsupported type {type} used in mapping.");

            return ColumnTypeExtensions.PrimitiveTypeFromKind(kind);
        }

        /// <summary>
        /// Helper function for a reverse lookup given value. This is used for generating the metadata of the value column.
        /// </summary>

        private static ValueGetter<VBuffer<ReadOnlyMemory<char>>> GetKeyValueGetter<TKey>(TKey[] keys)
        {
            return
                (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                {
                    var editor = VBufferEditor.Create(ref dst, keys.Length);
                    for (int i = 0; i < keys.Length; i++)
                        editor.Values[i] = keys[i].ToString().AsMemory();
                    dst = editor.Commit();
                };
        }

        /// <summary>
        /// Helper function to add a column to an ArrayDataViewBuilder. This handles the case if the type is a string.
        /// </summary>
        internal static void AddColumnWrapper<T>(ArrayDataViewBuilder builder, string columnName, PrimitiveDataViewType primitiveType, T[] values)
        {
            if (typeof(T) == typeof(string))
                builder.AddColumn(columnName, values.Select(x => x.ToString()).ToArray());
            else
                builder.AddColumn(columnName, primitiveType, values);
        }

        /// <summary>
        /// Helper function to add a column to an ArrayDataViewBuilder. This handles the case if the type is an array of strings.
        /// </summary>
        internal static void AddColumnWrapper<T>(ArrayDataViewBuilder builder, string columnName, PrimitiveDataViewType primitiveType, T[][] values)
        {
            if (typeof(T) == typeof(string))
            {
                var convertedValues = new List<ReadOnlyMemory<char>[]>();
                foreach (var value in values)
                {
                    var converted = value.Select(x => x.ToString().AsMemory());
                    convertedValues.Add(converted.ToArray());
                }

                builder.AddColumn(columnName, primitiveType, convertedValues.ToArray());
            }
            else
                builder.AddColumn(columnName, primitiveType, values);
        }

        /// <summary>
        /// Helper function to create an IDataView given a list of key and vector-based values
        /// </summary>
        internal static IDataView CreateDataView<TKey, TValue>(IHostEnvironment env,
                                                                IEnumerable<TKey> keys,
                                                                IEnumerable<TValue[]> values,
                                                                string keyColumnName,
                                                                string valueColumnName)
        {
            var keyType = GetPrimitiveType(typeof(TKey), out bool isKeyVectorType);
            var valueType = GetPrimitiveType(typeof(TValue), out bool isValueVectorType);
            var dataViewBuilder = new ArrayDataViewBuilder(env);
            AddColumnWrapper(dataViewBuilder, keyColumnName, keyType, keys.ToArray());
            AddColumnWrapper(dataViewBuilder, valueColumnName, valueType, values.ToArray());
            return dataViewBuilder.GetDataView();
        }

        /// <summary>
        /// Helper function that builds the IDataView given a list of keys and non-vector values
        /// </summary>
        internal static IDataView CreateDataView<TKey, TValue>(IHostEnvironment env,
                                                             IEnumerable<TKey> keys,
                                                             IEnumerable<TValue> values,
                                                             string keyColumnName,
                                                             string valueColumnName,
                                                             bool treatValuesAsKeyTypes)
        {
            var keyType = GetPrimitiveType(typeof(TKey), out bool isKeyVectorType);
            var valueType = GetPrimitiveType(typeof(TValue), out bool isValueVectorType);

            var dataViewBuilder = new ArrayDataViewBuilder(env);
            AddColumnWrapper(dataViewBuilder, keyColumnName, keyType, keys.ToArray());
            if (treatValuesAsKeyTypes)
            {
                // When treating the values as KeyTypes, generate the unique
                // set of values. This is used for generating the metadata of
                // the column.
                HashSet<TValue> valueSet = new HashSet<TValue>();
                foreach (var v in values)
                {
                    if (valueSet.Contains(v))
                        continue;
                    valueSet.Add(v);
                }

                var metaKeys = valueSet.ToArray();

                // Key Values are treated in one of two ways:
                // If the values are of type uint or ulong, these values are used directly as the keys types and no new keys are created.
                // If the values are not of uint or ulong, then key values are generated as uints starting from 1, since 0 is missing key.
                if (valueType.RawType == typeof(uint))
                {
                    uint[] indices = values.Select((x) => Convert.ToUInt32(x)).ToArray();
                    dataViewBuilder.AddColumn(valueColumnName, GetKeyValueGetter(metaKeys), (ulong)metaKeys.Length, indices);
                }
                else if (valueType.RawType == typeof(ulong))
                {
                    ulong[] indices = values.Select((x) => Convert.ToUInt64(x)).ToArray();
                    dataViewBuilder.AddColumn(valueColumnName, GetKeyValueGetter(metaKeys), (ulong)metaKeys.Length, indices);
                }
                else
                {
                    // When generating the indices, treat each value as being unique, i.e. two values that are the same will
                    // be assigned the same index. The dictionary is used to maintain uniqueness, indices will contain
                    // the full list of indices (equal to the same length of values).
                    Dictionary<TValue, uint> keyTypeValueMapping = new Dictionary<TValue, uint>();
                    uint[] indices = new uint[values.Count()];
                    // Start the index at 1
                    uint index = 1;
                    for (int i = 0; i < values.Count(); ++i)
                    {
                        TValue value = values.ElementAt(i);
                        if (!keyTypeValueMapping.ContainsKey(value))
                        {
                            keyTypeValueMapping.Add(value, index);
                            index++;
                        }

                        var keyValue = keyTypeValueMapping[value];
                        indices[i] = keyValue;
                    }

                    dataViewBuilder.AddColumn(valueColumnName, GetKeyValueGetter(metaKeys), (ulong)metaKeys.Count(), indices);
                }
            }
            else
                AddColumnWrapper(dataViewBuilder, valueColumnName, valueType, values.ToArray());

            return dataViewBuilder.GetDataView();
        }
    }

    /// <include file='doc.xml' path='doc/members/member[@name="ValueMappingEstimator"]/*' />
    public class ValueMappingTransformer : OneToOneTransformerBase
    {
        internal const string Summary = "Maps text values columns to new columns using a map dataset.";
        internal const string LoaderSignature = "ValueMappingTransformer";
        internal const string UserName = "Value Mapping Transform";
        internal const string ShortName = "ValueMap";

        internal const string TermLookupLoaderSignature = "TermLookupTransform";

        // Stream names for the binary idv streams.
        private const string DefaultMapName = "DefaultMap.idv";
        internal static string KeyColumnName = "Key";
        internal static string ValueColumnName = "Value";
        private ValueMap _valueMap;
        private readonly byte[] _dataView;

        internal DataViewType ValueColumnType => _valueMap.ValueType;
        internal DataViewSchema.Annotations ValueColumnMetadata { get; }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "VALUMAPG",
                verWrittenCur: 0x00010001, // Initial.
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ValueMappingTransformer).Assembly.FullName);
        }

        private static VersionInfo GetTermLookupVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TXTLOOKT",
                // verWrittenCur: 0x00010001, // Initial.
                verWrittenCur: 0x00010002, // Dropped sizeof(Float).
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ValueMappingTransformer).Assembly.FullName);
        }

        internal sealed class Column : OneToOneColumn
        {
            internal static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        internal sealed class Options
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "The data file containing the terms", ShortName = "data", SortOrder = 2)]
            public string DataFile;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the column containing the keys", ShortName = "keyCol, term, TermColumn")]
            public string KeyColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the column containing the values", ShortName = "valueCol, value")]
            public string ValueColumn;

            [Argument(ArgumentType.Multiple, HelpText = "The data loader", NullName = "<Auto>", SignatureType = typeof(SignatureDataLoader))]
            public IComponentFactory<IMultiStreamSource, ILegacyDataLoader> Loader;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Specifies whether the values are key values or numeric, only valid when loader is not specified and the type of data is not an idv.",
                ShortName = "key")]
            public bool ValuesAsKeyType = true;
        }

        internal ValueMappingTransformer(IHostEnvironment env, IDataView lookupMap,
            string keyColumn, string valueColumn, (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ValueMappingTransformer)), columns)
        {
            Host.CheckNonEmpty(keyColumn, nameof(keyColumn), "A key column must be specified when passing in an IDataView for the value mapping");
            Host.CheckNonEmpty(valueColumn, nameof(valueColumn), "A value column must be specified when passing in an IDataView for the value mapping");
            _valueMap = CreateValueMapFromDataView(lookupMap, keyColumn, valueColumn);
            int valueColumnIdx = 0;
            Host.Check(lookupMap.Schema.TryGetColumnIndex(valueColumn, out valueColumnIdx));
            ValueColumnMetadata = lookupMap.Schema[valueColumnIdx].Annotations;

            // Create the byte array of the original IDataView, this is used for saving out the data.
            _dataView = GetBytesFromDataView(Host, lookupMap, keyColumn, valueColumn);
        }

        private ValueMap CreateValueMapFromDataView(IDataView dataView, string keyColumn, string valueColumn)
        {
            // Confirm that the key and value columns exist in the dataView
            Host.Check(dataView.Schema.TryGetColumnIndex(keyColumn, out int keyIdx), "Key column " + keyColumn + " does not exist in the given dataview");
            Host.Check(dataView.Schema.TryGetColumnIndex(valueColumn, out int valueIdx), "Value column " + valueColumn + " does not exist in the given dataview");
            var keyType = dataView.Schema[keyIdx].Type;
            var valueType = dataView.Schema[valueIdx].Type;
            var valueMap = ValueMap.Create(keyType, valueType, ValueColumnMetadata);
            using (var cursor = dataView.GetRowCursor(dataView.Schema[keyIdx], dataView.Schema[valueIdx]))
                valueMap.Train(Host, cursor);
            return valueMap;
        }

        private static TextLoader.Column GenerateValueColumn(IHostEnvironment env,
                                                  IDataView loader,
                                                  string valueColumnName,
                                                  int keyIdx,
                                                  int valueIdx,
                                                  string fileName)
        {
            // Scan the source to determine the min max of the column
            ulong keyMin = ulong.MaxValue;
            ulong keyMax = ulong.MinValue;

            // scan the input to create convert the values as key types
            using (var cursor = loader.GetRowCursorForAllColumns())
            {
                using (var ch = env.Start($"Processing key values from file {fileName}"))
                {
                    var getKey = cursor.GetGetter<ReadOnlyMemory<char>>(cursor.Schema[keyIdx]);
                    var getValue = cursor.GetGetter<ReadOnlyMemory<char>>(cursor.Schema[valueIdx]);
                    int countNonKeys = 0;

                    ReadOnlyMemory<char> key = default;
                    ReadOnlyMemory<char> value = default;
                    while (cursor.MoveNext())
                    {
                        getKey(ref key);
                        getValue(ref value);

                        ulong res;
                        // Try to parse the text as a key value between 1 and ulong.MaxValue. If this succeeds and res>0,
                        // we update max and min accordingly. If res==0 it means the value is missing, in which case we ignore it for
                        // computing max and min.
                        if (Data.Conversion.Conversions.Instance.TryParseKey(in value, ulong.MaxValue - 1, out res))
                        {
                            if (res < keyMin && res != 0)
                                keyMin = res;
                            if (res > keyMax)
                                keyMax = res;
                        }
                        // If parsing as key did not succeed, the value can still be 0, so we try parsing it as a ulong. If it succeeds,
                        // then the value is 0, and we update min accordingly.
                        else if (Microsoft.ML.Data.Conversion.Conversions.Instance.TryParse(in value, out res))
                        {
                            keyMin = 0;
                        }
                        //If parsing as a ulong fails, we increment the counter for the non-key values.
                        else
                        {
                            if (countNonKeys < 5)
                                ch.Warning($"Key '{key}' in mapping file is mapped to non key value '{value}'");
                            countNonKeys++;
                        }
                    }

                    if (countNonKeys > 0)
                        ch.Warning($"Found {countNonKeys} non key values in the file '{fileName}'");
                    if (keyMin > keyMax)
                    {
                        keyMin = 0;
                        keyMax = uint.MaxValue - 1;
                        ch.Warning($"Did not find any valid key values in the file '{fileName}'");
                    }
                    else
                        ch.Info($"Found key values in the range {keyMin} to {keyMax} in the file '{fileName}'");
                }
            }

            TextLoader.Column valueColumn = new TextLoader.Column(valueColumnName, DataKind.UInt32, 1);
            if (keyMax < int.MaxValue)
                valueColumn.KeyCount = new KeyCount(keyMax + 1);
            else if (keyMax < uint.MaxValue)
                valueColumn.KeyCount = new KeyCount();
            else
            {
                valueColumn.Type = DataKind.UInt64.ToInternalDataKind();
                valueColumn.KeyCount = new KeyCount();
            }

            return valueColumn;
        }

        private static ValueMappingTransformer CreateTransformInvoke<TKey, TValue>(IHostEnvironment env,
                                                                                        IDataView idv,
                                                                                        string keyColumnName,
                                                                                        string valueColumnName,
                                                                                        bool treatValuesAsKeyTypes,
                                                                                        (string outputColumnName, string inputColumnName)[] columns)
        {
            // Read in the data
            // scan the input to create convert the values as key types
            List<TKey> keys = new List<TKey>();
            List<TValue> values = new List<TValue>();

            var keyColumn = idv.Schema[keyColumnName];
            var valueColumn = idv.Schema[valueColumnName];
            using (var cursor = idv.GetRowCursorForAllColumns())
            {
                using (var ch = env.Start("Processing key values"))
                {
                    TKey key = default;
                    TValue value = default;
                    var getKey = cursor.GetGetter<TKey>(keyColumn);
                    var getValue = cursor.GetGetter<TValue>(valueColumn);
                    while (cursor.MoveNext())
                    {
                        try
                        {
                            getKey(ref key);
                        }
                        catch (InvalidOperationException)
                        {
                            ch.Warning("Invalid key parsed, row will be skipped.");
                            continue;
                        }

                        try
                        {
                            getValue(ref value);
                        }
                        catch (InvalidOperationException)
                        {
                            ch.Warning("Invalid value parsed for key {key}, row will be skipped.");
                            continue;
                        }

                        keys.Add(key);
                        values.Add(value);
                    }
                }
            }

            var lookupMap = DataViewHelper.CreateDataView(env, keys, values, keyColumnName, valueColumnName, treatValuesAsKeyTypes);
            return new ValueMappingTransformer(env, lookupMap, keyColumnName, valueColumnName, columns);
        }

        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckUserArg(!string.IsNullOrWhiteSpace(options.DataFile), nameof(options.DataFile));
            env.CheckValueOrNull(options.KeyColumn);
            env.CheckValueOrNull(options.ValueColumn);

            var keyColumnName = (string.IsNullOrEmpty(options.KeyColumn)) ? KeyColumnName : options.KeyColumn;
            var valueColumnName = (string.IsNullOrEmpty(options.ValueColumn)) ? ValueColumnName : options.ValueColumn;

            IMultiStreamSource fileSource = new MultiFileSource(options.DataFile);
            IDataView loader;
            if (options.Loader != null)
            {
                loader = options.Loader.CreateComponent(env, fileSource);
            }
            else
            {
                var extension = Path.GetExtension(options.DataFile);
                if (extension.Equals(".idv", StringComparison.OrdinalIgnoreCase))
                    loader = new BinaryLoader(env, new BinaryLoader.Arguments(), fileSource);
                else if (extension.Equals(".tdv"))
                    loader = new TransposeLoader(env, new TransposeLoader.Arguments(), fileSource);
                else
                {
                    // The user has not specified how to load this file. This will attempt to load the
                    // data file as two text columns. If the user has also specified ValuesAsKeyTypes,
                    // this will default to the key column as a text column and the value column as a uint column

                    // Set the keyColumnName and valueColumnName to the default values.
                    keyColumnName = KeyColumnName;
                    valueColumnName = ValueColumnName;
                    TextLoader.Column keyColumn = default;
                    TextLoader.Column valueColumn = default;

                    // Default to a text loader. KeyType and ValueType are assumed to be string
                    // types unless ValueAsKeyType is specified.
                    if (options.ValuesAsKeyType)
                    {
                        keyColumn = new TextLoader.Column(keyColumnName, DataKind.String, 0);
                        valueColumn = new TextLoader.Column(valueColumnName, DataKind.String, 1);
                        var txtArgs = new TextLoader.Options()
                        {
                            Columns = new TextLoader.Column[]
                            {
                                keyColumn,
                                valueColumn
                            }
                        };

                        try
                        {
                            var textLoader = TextLoader.LoadFile(env, txtArgs, fileSource);
                            valueColumn = GenerateValueColumn(env, textLoader, valueColumnName, 0, 1, options.DataFile);
                        }
                        catch (Exception ex)
                        {
                            throw env.Except(ex, $"Failed to parse the lookup file '{options.DataFile}' in ValueMappingTransformerer");
                        }
                    }
                    else
                    {
                        keyColumn = new TextLoader.Column(keyColumnName, DataKind.String, 0);
                        valueColumn = new TextLoader.Column(valueColumnName, DataKind.Single, 1);
                    }

                    loader = TextLoader.Create(
                        env,
                        new TextLoader.Options()
                        {
                            Columns = new TextLoader.Column[]
                            {
                                keyColumn,
                                valueColumn
                            }
                        },
                        fileSource);
                }
            }

            env.AssertValue(loader);
            env.Assert(loader.Schema.TryGetColumnIndex(keyColumnName, out int keyColumnIndex));
            env.Assert(loader.Schema.TryGetColumnIndex(valueColumnName, out int valueColumnIndex));

            ValueMappingTransformer transformer = null;
            (string outputColumnName, string inputColumnName)[] columns = options.Columns.Select(x => (x.Name, x.Source)).ToArray();
            transformer = new ValueMappingTransformer(env, loader, keyColumnName, valueColumnName, columns);
            return transformer.MakeDataTransform(input);
        }

        /// <summary>
        /// Helper function to determine the model version that is being loaded.
        /// </summary>
        private static bool CheckModelVersion(ModelLoadContext ctx, VersionInfo versionInfo)
        {
            try
            {
                ctx.CheckVersionInfo(versionInfo);
                return true;
            }
            catch (Exception)
            {
                //consume
                return false;
            }
        }

        private protected static ValueMappingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));

            // Checks for both the TermLookup for backwards compatibility
            var termLookupModel = CheckModelVersion(ctx, GetTermLookupVersionInfo());
            env.Check(termLookupModel || CheckModelVersion(ctx, GetVersionInfo()));

            // *** Binary format ***
            // int: number of added columns
            // for each added column
            //   string: output column name
            //   string: input column name
            // Binary stream of mapping

            var length = ctx.Reader.ReadInt32();
            var columns = new (string outputColumnName, string inputColumnName)[length];
            for (int i = 0; i < length; i++)
            {
                columns[i].outputColumnName = ctx.LoadNonEmptyString();
                columns[i].inputColumnName = ctx.LoadNonEmptyString();
            }

            byte[] rgb = null;
            Action<BinaryReader> fn = r => rgb = ReadAllBytes(env, r);

            if (!ctx.TryLoadBinaryStream(DefaultMapName, fn))
                throw env.ExceptDecode();

            var binaryLoader = GetLoader(env, rgb);
            var keyColumnName = (termLookupModel) ? "Term" : KeyColumnName;
            return new ValueMappingTransformer(env, binaryLoader, keyColumnName, ValueColumnName, columns);
        }

        private static byte[] ReadAllBytes(IExceptionContext ectx, BinaryReader rdr)
        {
            Contracts.AssertValue(ectx);
            ectx.AssertValue(rdr);
            ectx.Assert(rdr.BaseStream.CanSeek);

            long size = rdr.BaseStream.Length;
            ectx.CheckDecode(size <= int.MaxValue);

            var rgb = new byte[(int)size];
            int cb = rdr.Read(rgb, 0, rgb.Length);
            ectx.CheckDecode(cb == rgb.Length);

            return rgb;
        }

        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected static PrimitiveDataViewType GetPrimitiveType(Type rawType, out bool isVectorType)
        {
            Type type = rawType;
            isVectorType = false;
            if (type.IsArray)
            {
                type = rawType.GetElementType();
                isVectorType = true;
            }

            if (!type.TryGetDataKind(out InternalDataKind kind))
                throw Contracts.Except($"Unsupported type {type} used in mapping.");

            return ColumnTypeExtensions.PrimitiveTypeFromKind(kind);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.SetVersionInfo(GetVersionInfo());
            SaveColumns(ctx);

            // Save out the byte stream of the IDataView of the data source
            ctx.SaveBinaryStream(DefaultMapName, w => w.Write(_dataView));
        }

        /// <summary>
        /// Base class that contains the mapping of keys to values.
        /// </summary>
        private abstract class ValueMap
        {
            public readonly DataViewType KeyType;
            public readonly DataViewType ValueType;

            public ValueMap(DataViewType keyType, DataViewType valueType)
            {
                KeyType = keyType;
                ValueType = valueType;
            }

            public static ValueMap Create(DataViewType keyType, DataViewType valueType, DataViewSchema.Annotations valueMetadata)
            {
                Func<DataViewType, DataViewType, DataViewSchema.Annotations, ValueMap> del = CreateValueMapInvoke<int, int>;
                var meth = del.Method.GetGenericMethodDefinition().MakeGenericMethod(keyType.RawType, valueType.RawType);
                return (ValueMap)meth.Invoke(null, new object[] { keyType, valueType, valueMetadata });
            }

            private static ValueMap CreateValueMapInvoke<TKey, TValue>(DataViewType keyType,
                                                                                DataViewType valueType,
                                                                                DataViewSchema.Annotations valueMetadata)
                => new ValueMap<TKey, TValue>(keyType, valueType, valueMetadata);

            public abstract void Train(IHostEnvironment env, DataViewRowCursor cursor);

            public abstract Delegate GetGetter(DataViewRow input, int index);

            public abstract IDataView GetDataView(IHostEnvironment env);
        }

        /// <summary>
        /// Implementation mapping class that maps a key of TKey to a specified value of TValue.
        /// </summary>
        private class ValueMap<TKey, TValue> : ValueMap
        {
            private Dictionary<TKey, TValue> _mapping;
            private TValue _missingValue;
            private readonly DataViewSchema.Annotations _valueMetadata;

            private Dictionary<TKey, TValue> CreateDictionary()
            {
                if (typeof(TKey) == typeof(ReadOnlyMemory<char>))
                    return new Dictionary<ReadOnlyMemory<char>, TValue>(new ReadOnlyMemoryUtils.ReadonlyMemoryCharComparer()) as Dictionary<TKey, TValue>;
                return new Dictionary<TKey, TValue>();
            }

            public ValueMap(DataViewType keyType, DataViewType valueType, DataViewSchema.Annotations valueMetadata)
                : base(keyType, valueType)
            {
                _mapping = CreateDictionary();
                _valueMetadata = valueMetadata;
            }

            /// <summary>
            /// Generates the mapping based on the IDataView
            /// </summary>
            public override void Train(IHostEnvironment env, DataViewRowCursor cursor)
            {
                // Validate that the conversion is supported for non-vector types
                bool identity;
                ValueMapper<ReadOnlyMemory<char>, TValue> conv;

                // For keys that are not in the mapping, the missingValue will be returned.
                _missingValue = default;
                if (!(ValueType is VectorType))
                {
                    // For handling missing values, this follows how a missing value is handled when loading from a text source.
                    // First check if there is a String->ValueType conversion method. If so, call the conversion method with an
                    // empty string, the returned value will be the new missing value.
                    // NOTE this will return NA for R4 and R8 types.
                    if (Data.Conversion.Conversions.Instance.TryGetStandardConversion<ReadOnlyMemory<char>, TValue>(
                                                                        TextDataViewType.Instance,
                                                                        ValueType,
                                                                        out conv,
                                                                        out identity))
                    {
                        TValue value = default;
                        conv(string.Empty.AsMemory(), ref value);
                        _missingValue = value;
                    }
                }

                var keyGetter = cursor.GetGetter<TKey>(cursor.Schema[0]);
                var valueGetter = cursor.GetGetter<TValue>(cursor.Schema[1]);
                while (cursor.MoveNext())
                {
                    TKey key = default;
                    TValue value = default;
                    keyGetter(ref key);
                    valueGetter(ref value);
                    if (_mapping.ContainsKey(key))
                        throw env.Except($"Duplicate keys in data '{key}'");

                    _mapping.Add(key, value);
                }
            }

            private TValue MapValue(TKey key)
            {
                if (_mapping.ContainsKey(key))
                {
                    if (ValueType is VectorType vectorType)
                        return Utils.MarshalInvoke(GetVector<int>, vectorType.ItemType.RawType, _mapping[key]);
                    else
                        return Utils.MarshalInvoke(GetValue<int>, ValueType.RawType, _mapping[key]);
                }
                else
                    return _missingValue;
            }

            public override Delegate GetGetter(DataViewRow input, int index)
            {
                var column = input.Schema[index];
                if (column.Type is VectorType)
                {
                    var src = default(VBuffer<TKey>);
                    var getSrc = input.GetGetter<VBuffer<TKey>>(column);

                    ValueGetter<VBuffer<TValue>> retVal =
                        (ref VBuffer<TValue> dst) =>
                        {
                            getSrc(ref src);
                            var editor = VBufferEditor.Create(ref dst, src.Length);
                            var values = src.GetValues();
                            src.GetIndices().CopyTo(editor.Indices);
                            for (int ich = 0; ich < values.Length; ich++)
                            {
                                editor.Values[ich] = MapValue(values[ich]);
                            }
                            dst = editor.Commit();
                        };
                    return retVal;
                }
                else
                {
                    var src = default(TKey);
                    var getSrc = input.GetGetter<TKey>(column);
                    ValueGetter<TValue> retVal =
                        (ref TValue dst) =>
                        {
                            getSrc(ref src);
                            dst = MapValue(src);
                        };
                    return retVal;
                }
            }

            public override IDataView GetDataView(IHostEnvironment env)
                => DataViewHelper.CreateDataView(env,
                                                 _mapping.Keys,
                                                 _mapping.Values,
                                                 ValueMappingTransformer.KeyColumnName,
                                                 ValueMappingTransformer.ValueColumnName,
                                                 ValueType is KeyType);

            private static TValue GetVector<T>(TValue value)
            {
                if (value is VBuffer<T> valueRef)
                {
                    VBuffer<T> dest = default;
                    valueRef.CopyTo(ref dest);
                    if (dest is TValue destRef)
                        return destRef;
                }

                return default;
            }

            private static TValue GetValue<T>(TValue value) => value;
        }

        /// <summary>
        /// Retrieves the byte array given a dataview and columns
        /// </summary>
        private static byte[] GetBytesFromDataView(IHost host, IDataView lookup, string keyColumn, string valueColumn)
        {
            Contracts.AssertValue(host);
            host.AssertValue(lookup);
            host.AssertNonEmpty(keyColumn);
            host.AssertNonEmpty(valueColumn);

            var schema = lookup.Schema;

            if (!schema.GetColumnOrNull(keyColumn).HasValue)
                throw host.ExceptUserArg(nameof(Options.KeyColumn), $"Key column not found: '{keyColumn}'");
            if (!schema.GetColumnOrNull(valueColumn).HasValue)
                throw host.ExceptUserArg(nameof(Options.ValueColumn), $"Value column not found: '{valueColumn}'");

            var cols = new List<(string outputColumnName, string inputColumnName)>()
            {
                (KeyColumnName, keyColumn),
                (ValueColumnName, valueColumn)
            };

            var view = new ColumnCopyingTransformer(host, cols.ToArray()).Transform(lookup);
            view = ColumnSelectingTransformer.CreateKeep(host, view, cols.Select(x => x.outputColumnName).ToArray());

            var saver = new BinarySaver(host, new BinarySaver.Arguments());
            using (var strm = new MemoryStream())
            {
                saver.SaveData(strm, view, 0, 1);
                return strm.ToArray();
            }
        }

        private static BinaryLoader GetLoader(IHostEnvironment env, byte[] bytes)
        {
            env.AssertValue(env);
            env.AssertValue(bytes);

            var strm = new MemoryStream(bytes, writable: false);
            return new BinaryLoader(env, new BinaryLoader.Arguments(), strm);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema)
        {
            return new Mapper(this, schema, _valueMap, ValueColumnMetadata, ColumnPairs);
        }

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly DataViewSchema _inputSchema;
            private readonly ValueMap _valueMap;
            private readonly DataViewSchema.Annotations _valueMetadata;
            private readonly (string outputColumnName, string inputColumnName)[] _columns;
            private readonly ValueMappingTransformer _parent;

            internal Mapper(ValueMappingTransformer transform,
                            DataViewSchema inputSchema,
                            ValueMap valueMap,
                            DataViewSchema.Annotations valueMetadata,
                            (string outputColumnName, string inputColumnName)[] columns)
                : base(transform.Host.Register(nameof(Mapper)), transform, inputSchema)
            {
                _inputSchema = inputSchema;
                _valueMetadata = valueMetadata;
                _valueMap = valueMap;
                _columns = columns;
                _parent = transform;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _columns.Length);
                disposer = null;

                return _valueMap.GetGetter(input, ColMapNewToOld[iinfo]);
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_columns.Length];
                for (int i = 0; i < _columns.Length; i++)
                {
                    if (_inputSchema[_columns[i].inputColumnName].Type is VectorType && _valueMap.ValueType is VectorType)
                        throw _parent.Host.ExceptNotSupp("Column '{0}' cannot be mapped to values when the column and the map values are both vector type.", _columns[i].inputColumnName);
                    var colType = _valueMap.ValueType;
                    if (_inputSchema[_columns[i].inputColumnName].Type is VectorType)
                        colType = new VectorType(ColumnTypeExtensions.PrimitiveTypeFromType(_valueMap.ValueType.GetItemType().RawType));
                    result[i] = new DataViewSchema.DetachedColumn(_columns[i].outputColumnName, colType, _valueMetadata);
                }
                return result;
            }
        }
    }
}
