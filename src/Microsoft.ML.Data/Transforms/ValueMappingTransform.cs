// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

[assembly: LoadableClass(ValueMappingTransform.Summary, typeof(IDataTransform), typeof(ValueMappingTransform),
    typeof(ValueMappingTransform.Arguments), typeof(SignatureDataTransform),
    ValueMappingTransform.UserName, "ValueMapping", "ValueMappingTransform", ValueMappingTransform.ShortName,
    DocName = "transform/ValueMappingTransform.md")]

[assembly: LoadableClass(ValueMappingTransform.Summary, typeof(IDataTransform), typeof(ValueMappingTransform), null, typeof(SignatureLoadDataTransform),
    "Value Mapping Transform", ValueMappingTransform.LoaderSignature)]

[assembly: LoadableClass(ValueMappingTransform.Summary, typeof(ValueMappingTransform), null, typeof(SignatureLoadModel),
    "Value Mapping Transform", ValueMappingTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ValueMappingTransform), null, typeof(SignatureLoadRowMapper),
    ValueMappingTransform.UserName, ValueMappingTransform.LoaderSignature)]

[assembly: LoadableClass("", typeof(IDataTransform), typeof(ValueMappingTransform), null, typeof(SignatureLoadDataTransform),
    "", ValueMappingTransform.TermLookupLoaderSignature)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// The ValueMappingEstimator is a 1-1 mapping from a key to value. The key type and value type are specified
    /// through TKeyType and TValueType. Arrays are supported for vector types which can be used as either a key or a value
    /// or both. The mapping is specified, not trained by providiing a list of keys and a list of values.
    /// </summary>
    /// <typeparam name="TKeyType">Specifies the key type</typeparam>
    /// <typeparam name="TValueType">Specifies the value type</typeparam>
    public sealed class ValueMappingEstimator<TKeyType, TValueType> : TrivialEstimator<ValueMappingTransform<TKeyType, TValueType>>
    {
        private (string input, string output)[] _columns;

        /// <summary>
        /// Constructs the ValueMappingEstimator, key type -> value type mapping
        /// </summary>
        /// <param name="env">Instance of the host environment</param>
        /// <param name="keys">The list of keys of TKeyType</param>
        /// <param name="values">The list of values of TValueType</param>
        /// <param name="columns">The list of columns to apply</param>
        public ValueMappingEstimator(IHostEnvironment env, IEnumerable<TKeyType> keys, IEnumerable<TValueType> values, params (string input, string output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ValueMappingEstimator<TKeyType, TValueType>)),
                    new ValueMappingTransform<TKeyType, TValueType>(env, keys, values, false, columns))
        {
            _columns = columns;
        }

        /// <summary>
        /// Constructs the ValueMappingEstimator, key type -> value type mapping
        /// </summary>
        /// <param name="env">Instance of the host environment</param>
        /// <param name="keys">The list of keys of TKeyType</param>
        /// <param name="values">The list of values of TValueType</param>
        /// <param name="treatValuesAsKeyType">Specifies to treat the values as a <see cref="KeyType"/></param>
        /// <param name="columns">The list of columns to apply</param>
        public ValueMappingEstimator(IHostEnvironment env, IEnumerable<TKeyType> keys, IEnumerable<TValueType> values, bool treatValuesAsKeyType, params (string input, string output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ValueMappingEstimator<TKeyType, TValueType>)),
                    new ValueMappingTransform<TKeyType, TValueType>(env, keys, values, treatValuesAsKeyType, columns))
        {
            _columns = columns;
        }

        /// <summary>
        /// Constructs the ValueMappingEstimator, key type -> value array type mapping
        /// </summary>
        /// <param name="env">Instance of the host environment</param>
        /// <param name="keys">The list of keys of TKeyType</param>
        /// <param name="values">The list of values of TValueType[]</param>
        /// <param name="columns">The list of columns to apply</param>
        public ValueMappingEstimator(IHostEnvironment env, IEnumerable<TKeyType> keys, IEnumerable<TValueType[]> values, params (string input, string output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ValueMappingEstimator<TKeyType, TValueType>)),
                    new ValueMappingTransform<TKeyType, TValueType>(env, keys, values, columns))
        {
            _columns = columns;
        }

        /// <summary>
        /// Retrieves the output schema given the input schema
        /// </summary>
        /// <param name="inputSchema">Input schema</param>
        /// <returns>Returns the generated output schema</returns>
        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            var resultDic = inputSchema.Columns.ToDictionary(x => x.Name);
            var vectorKind = Transformer.ValueColumnType.IsVector ? SchemaShape.Column.VectorKind.Vector : SchemaShape.Column.VectorKind.Scalar;
            var isKey = Transformer.ValueColumnType.IsKey;
            var columnType = (isKey) ? PrimitiveType.FromKind(DataKind.U4) :
                                    Transformer.ValueColumnType;
            foreach (var (Input, Output) in _columns)
            {
                if (!inputSchema.TryFindColumn(Input, out var originalColumn))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", Input);

                // Get the type from TOutputType
                var col = new SchemaShape.Column(Output, vectorKind, columnType, isKey, originalColumn.Metadata);
                resultDic[Output] = col;
            }
            return new SchemaShape(resultDic.Values);
        }
    }

    internal class DataViewHelper
    {
        public static PrimitiveType GetPrimitiveType(Type rawType, out bool isVectorType)
        {
            Type type = rawType;
            isVectorType = false;
            if (type.IsArray)
            {
                type = rawType.GetElementType();
                isVectorType = true;
            }

            if  (!type.TryGetDataKind(out DataKind kind))
            {
                throw new InvalidOperationException($"Unsupported type {type} used in mapping.");
            }

            return PrimitiveType.FromKind(kind);
        }

        private static ValueGetter<VBuffer<ReadOnlyMemory<char>>> GetKeyValueGetter<TValue>(TValue[] values)
        {
            return
                (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                {
                    var editor = VBufferEditor.Create(ref dst, values.Length);
                    for (int i = 0; i < values.Length; i++)
                        editor.Values[i] = values[i].ToString().AsMemory();
                    dst = editor.Commit();
                };
        }

        public static IDataView CreateDataView<TKey, TValue>(IHostEnvironment env,
                                                                IEnumerable<TKey> keys,
                                                                IEnumerable<TValue[]> values,
                                                                string keyColumnName,
                                                                string valueColumnName)
        {
            // Build DataView from the mapping
            var keyType = GetPrimitiveType(typeof(TKey), out bool isKeyVectorType);
            var valueType = GetPrimitiveType(typeof(TValue), out bool isValueVectorType);
            var dataViewBuilder = new ArrayDataViewBuilder(env);
            dataViewBuilder.AddColumn(keyColumnName, keyType, keys.ToArray());
            dataViewBuilder.AddColumn(valueColumnName, valueType, values.ToArray());
            return dataViewBuilder.GetDataView();
        }

        public static IDataView CreateDataView<TKey, TValue>(IHostEnvironment env,
                                                             IEnumerable<TKey> keys,
                                                             IEnumerable<TValue> values,
                                                             string keyColumnName,
                                                             string valueColumnName,
                                                             bool treatValuesAsKeyTypes)
        {
            // Build DataView from the mapping
            var keyType = GetPrimitiveType(typeof(TKey), out bool isKeyVectorType);
            var valueType = GetPrimitiveType(typeof(TValue), out bool isValueVectorType);

            var dataViewBuilder = new ArrayDataViewBuilder(env);
            dataViewBuilder.AddColumn(keyColumnName, keyType, keys.ToArray());
            //var valuesArray = values.ToArray();
            if (treatValuesAsKeyTypes)
            {
                // If the values are key values, there are two different ways in which they are handled:
                // 1) If the values are of type uint, then it is assumed that these values are the
                // key values. In this case, the values are used for the key values.
                // 2) If the values are not of type uint. Then key type values are generated as a number range starting at 0.
                if (valueType.RawKind == DataKind.U4)
                {
                    uint[] indices = values.Select((x) => Convert.ToUInt32(x) - 1).ToArray();
                    var min = indices.Min();
                    var max = indices.Max();
                    int count = (int)(max - min + 1);
                    dataViewBuilder.AddColumn(valueColumnName, GetKeyValueGetter(indices), min, count, indices);
                }
                else if (valueType.RawKind == DataKind.U8)
                {
                    ulong[] indices = values.Select((x) => Convert.ToUInt64(x) - 1).ToArray();
                    var min = indices.Min();
                    var max = indices.Max();
                    int count = (int)(max - min + 1);
                    dataViewBuilder.AddColumn(valueColumnName, GetKeyValueGetter(indices), min, count, indices);
                }
                else
                {
                    // When generating the indices, treat each value as being unique, i.e. two values that are the same will
                    // be assigned the same index. The dictionary is used to maintain uniqueness, indices will contain
                    // the full list of indices (equal to the same length of values).
                    Dictionary<TValue, uint> keyTypeValueMapping = new Dictionary<TValue, uint>();
                    uint[] indices  = new uint[values.Count()];
                    // Start the index at 1 since key types start at 1, 0 is invalid
                    uint index = 1;
                    for(int i = 0; i < values.Count(); ++i)
                    {
                        TValue value = values.ElementAt(i);
                        if(!keyTypeValueMapping.ContainsKey(value))
                        {
                            keyTypeValueMapping.Add(value, index);
                            index++;
                        }

                        var keyValue = keyTypeValueMapping[value];
                        indices[i] = keyValue;
                    }

                    dataViewBuilder.AddColumn(valueColumnName, GetKeyValueGetter(values.ToArray()), 0, indices.Count(), indices);
                }
            }
            else
            {
                dataViewBuilder.AddColumn(valueColumnName, valueType, values.ToArray());
            }

            return dataViewBuilder.GetDataView();
        }
    }

    /// <summary>
    /// The ValueMappingTransform is a 1-1 mapping from a key to value. The key type and value type are specified
    /// through TKeyType and TValueType. Arrays are supported for vector types which can be used as either a key or a value
    /// or both. The mapping is specified, not trained by providiing a list of keys and a list of values.
    /// </summary>
    /// <typeparam name="TKeyType">Specifies the key type</typeparam>
    /// <typeparam name="TValueType">Specifies the value type</typeparam>
    public sealed class ValueMappingTransform<TKeyType, TValueType> : ValueMappingTransform
    {
        /// <summary>
        /// Constructs a ValueMappingTransform with a key type to value type
        /// </summary>
        /// <param name="env">Instance of the host environment</param>
        /// <param name="keys">The list of keys that are TKeyType</param>
        /// <param name="values">The list of values that are TValueType</param>
        /// <param name="treatValuesAsKeyTypes">Specifies to treat the values as a <see cref="KeyType"/></param>
        /// <param name="columns">The specified columns to apply</param>
        public ValueMappingTransform(IHostEnvironment env, IEnumerable<TKeyType> keys, IEnumerable<TValueType> values, bool treatValuesAsKeyTypes, (string Input, string Output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ValueMappingTransform<TKeyType, TValueType>)),
                  ConvertToDataView(env, keys, values, treatValuesAsKeyTypes), KeyColumnName, ValueColumnName, columns)
        { }

        /// <summary>
        /// Constructs a ValueMappingTransform with a key type to value array type
        /// </summary>
        /// <param name="env">Instance of the host environment</param>
        /// <param name="keys">The list of keys that are TKeyType</param>
        /// <param name="values">The list of values that are TValueType[]</param>
        /// <param name="columns">The specified columns to apply</param>
        public ValueMappingTransform(IHostEnvironment env, IEnumerable<TKeyType> keys, IEnumerable<TValueType[]> values, (string Input, string Output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ValueMappingTransform<TKeyType, TValueType>)),
                ConvertToDataView(env, keys, values), KeyColumnName, ValueColumnName, columns)
        { }

        private static IDataView ConvertToDataView(IHostEnvironment env, IEnumerable<TKeyType> keys, IEnumerable<TValueType> values, bool treatValuesAsKeyValue)
            => DataViewHelper.CreateDataView(env,
                    keys,
                    values,
                    ValueMappingTransform.KeyColumnName,
                    ValueMappingTransform.ValueColumnName,
                    treatValuesAsKeyValue);

        // Handler for vector value types
        private static IDataView ConvertToDataView(IHostEnvironment env, IEnumerable<TKeyType> keys, IEnumerable<TValueType[]> values)
            => DataViewHelper.CreateDataView(env, keys, values, ValueMappingTransform.KeyColumnName, ValueMappingTransform.ValueColumnName);
    }

    public class ValueMappingTransform : OneToOneTransformerBase
    {
        internal const string Summary = "Maps text values columns to new columns using a map dataset.";
        internal const string LoaderSignature = "ValueMappingTransform";
        internal const string UserName = "Value Mapping Transform";
        internal const string ShortName = "ValueMap";

        internal const string TermLookupLoaderSignature = "TermLookupTransform";

        // Stream names for the binary idv streams.
        private const string DefaultMapName = "DefaultMap.idv";
        protected static string KeyColumnName = "Key";
        protected static string ValueColumnName = "Value";
        private ValueMap _valueMap;
        private Schema.Metadata _valueMetadata;

        public ColumnType ValueColumnType => _valueMap.ValueType;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "VALUMAPG",
                verWrittenCur: 0x00010001, // Initial.
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ValueMappingTransform).Assembly.FullName);
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
                loaderAssemblyName: typeof(ValueMappingTransform).Assembly.FullName);
        }

        public sealed class Column : OneToOneColumn
        {
            public static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the column containing the keys", ShortName = "key")]
            public string KeyColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the column containing the values", ShortName = "value")]
            public string ValueColumn;

            [Argument(ArgumentType.Multiple, HelpText = "The data loader", NullName = "<Auto>", SignatureType = typeof(SignatureDataLoader))]
            public IComponentFactory<IMultiStreamSource, IDataLoader> Loader;

            [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "The data file containing the terms", ShortName = "data", SortOrder = 2)]
            public string DataFile;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Specifies whether the values are key values or numeric, only valid when loader is not specified and the type of data is not an idv.")]
            public bool ValuesAsKeyType = true;
        }

        protected ValueMappingTransform(IHostEnvironment env, IDataView lookupMap,
            string keyColumn, string valueColumn, (string Input, string Output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ValueMappingTransform)), columns)
        {
            env.CheckNonEmpty(keyColumn, nameof(keyColumn), "A key column must be specified when passing in an IDataView for the value mapping");
            env.CheckNonEmpty(valueColumn, nameof(valueColumn), "A value column must be specified when passing in an IDataView for the value mapping");
            _valueMap = CreateValueMapFromDataView(lookupMap, keyColumn, valueColumn);
            env.Assert(lookupMap.Schema.TryGetColumnIndex(valueColumn, out int valueColumnIdx));
            _valueMetadata = CopyMetadata(lookupMap.Schema[valueColumnIdx].Metadata);
        }

        private Schema.Metadata CopyMetadata(Schema.Metadata metadata)
        {
            var meta = new MetadataBuilder();
            meta.Add(metadata, x=> true);
            return meta.GetMetadata();
        }

        private ValueMap CreateValueMapFromDataView(IDataView dataView, string keyColumn, string valueColumn)
        {
            // Confirm that the key and value columns exist in the dataView
            Host.Check(dataView.Schema.TryGetColumnIndex(keyColumn, out int keyIdx), "Key column " + keyColumn + " does not exist in the given dataview");
            Host.Check(dataView.Schema.TryGetColumnIndex(valueColumn, out int valueIdx), "Value column " + valueColumn + " does not exist in the given dataview");
            var keyType = dataView.Schema.GetColumnType(keyIdx);
            var valueType = dataView.Schema.GetColumnType(valueIdx);
            var valueMap = ValueMap.Create(keyType, valueType, _valueMetadata);
            using (var cursor = dataView.GetRowCursor(c=> c == keyIdx || c == valueIdx))
                valueMap.Train(Host, cursor);
            return valueMap;
        }

        private static TextLoader.Column GenerateValueColumn(IHostEnvironment env,
                                                  IDataView loader,
                                                  string valueColumnName,
                                                  int keyIdx,
                                                  int valueIdx)
        {
            // Scan the source to determine the min max of the column
            ulong keyMin = ulong.MinValue;
            ulong keyMax = ulong.MinValue;

            // scan the input to create convert the values as key types
            using (var cursor = loader.GetRowCursor(c => true))
            {
                using(var ch = env.Start("Processing key values"))
                {
                    var getKey = cursor.GetGetter<ReadOnlyMemory<char>>(keyIdx);
                    var getValue = cursor.GetGetter<ReadOnlyMemory<char>>(valueIdx);
                    int countNonKeys = 0;

                    ReadOnlyMemory<char> key = default;
                    ReadOnlyMemory<char> value = default;
                    while(cursor.MoveNext())
                    {
                        getKey(ref key);
                        getValue(ref value);

                        ulong res;
                        // Try to parse the text as a key value between 1 and ulong.MaxValue. If this succeeds and res>0,
                        // we update max and min accordingly. If res==0 it means the value is missing, in which case we ignore it for
                        // computing max and min.
                        if (Microsoft.ML.Runtime.Data.Conversion.Conversions.Instance.TryParseKey(in value, 1, ulong.MaxValue, out res))
                        {
                            if (res < keyMin && res != 0)
                                keyMin = res;
                            if (res > keyMax)
                                keyMax = res;
                        }
                        // If parsing as key did not succeed, the value can still be 0, so we try parsing it as a ulong. If it succeeds,
                        // then the value is 0, and we update min accordingly.
                        else if (Microsoft.ML.Runtime.Data.Conversion.Conversions.Instance.TryParse(in value, out res))
                        {
                            keyMin = 0;
                        }
                        //If parsing as a ulong fails, we increment the counter for the non-key values.
                        else
                        {
                            if (countNonKeys < 5)
                                ch.Warning("Key '{0}' in mapping file is mapped to non key value '{1}'", key, value);
                            countNonKeys++;
                        }
                    }
                }
            }

            TextLoader.Column valueColumn = new TextLoader.Column(valueColumnName, DataKind.U4, 1);
            if (keyMax - keyMin < (ulong)int.MaxValue)
            {
                valueColumn.KeyRange = new KeyRange(keyMin, keyMax);
            }
            else if (keyMax - keyMin < (ulong)uint.MaxValue)
            {
                valueColumn.KeyRange = new KeyRange(keyMin);
            }
            else
            {
                valueColumn.Type = DataKind.U8;
                valueColumn.KeyRange = new KeyRange(keyMin);
            }

            return valueColumn;
        }

        private static ValueMappingTransform CreateTransformInvoke<TKeyType, TValueType>(IHostEnvironment env,
                                                                                        IDataView idv,
                                                                                        string keyColumnName,
                                                                                        string valueColumnName,
                                                                                        bool treatValuesAsKeyTypes,
                                                                                        (string Input, string Output)[] columns)
        {
            // Read in the data
            // scan the input to create convert the values as key types
            List<TKeyType> keys = new List<TKeyType>();
            List<TValueType> values = new List<TValueType>();

            idv.Schema.TryGetColumnIndex(keyColumnName, out int keyIdx);
            idv.Schema.TryGetColumnIndex(valueColumnName, out int valueIdx);
            using (var cursor = idv.GetRowCursor(c => true))
            {
                using(var ch = env.Start("Processing key values"))
                {
                    TKeyType key = default;
                    TValueType value = default;
                    var getKey = cursor.GetGetter<TKeyType>(keyIdx);
                    var getValue = cursor.GetGetter<TValueType>(valueIdx);
                    while(cursor.MoveNext())
                    {
                        try
                        {
                            getKey(ref key);
                        }
                        catch(InvalidOperationException)
                        {
                            ch.Warning("Invalid key parsed, row will be skipped.");
                            continue;
                        }

                        try
                        {
                            getValue(ref value);
                        }
                        catch(InvalidOperationException)
                        {
                            ch.Warning("Invalid value parsed for key {key}, row will be skipped.");
                            continue;
                        }

                        keys.Add(key);
                        values.Add(value);
                    }
                }
            }

            return new ValueMappingTransform<TKeyType, TValueType>(env, keys, values, treatValuesAsKeyTypes, columns);
        }

        private static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.Assert(!string.IsNullOrWhiteSpace(args.DataFile));
            env.CheckValueOrNull(args.KeyColumn);
            env.CheckValueOrNull(args.ValueColumn);

            var keyColumnName = (string.IsNullOrEmpty(args.KeyColumn)) ? KeyColumnName : args.KeyColumn;
            var valueColumnName = (string.IsNullOrEmpty(args.ValueColumn)) ? ValueColumnName : args.ValueColumn;

            IMultiStreamSource fileSource = new MultiFileSource(args.DataFile);
            IDataView loader;
            if (args.Loader != null)
            {
                loader = args.Loader.CreateComponent(env, fileSource);
            }
            else
            {
                var extension = Path.GetExtension(args.DataFile);
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

                    TextLoader.Column keyColumn = new TextLoader.Column(keyColumnName, DataKind.TXT, 0);
                    TextLoader.Column valueColumn = new TextLoader.Column(valueColumnName, DataKind.TXT, 1);
                    /*
                    if (args.ValuesAsKeyType)
                    {
                        valueColumn = new TextLoader.Column(valueColumnName, DataKind.U8, 1);
                    }*/

                    var txtArgs = new TextLoader.Arguments()
                    {
                        Column=new TextLoader.Column[]
                        {
                            keyColumn,
                            valueColumn
                        }
                    };

                    //loader = TextLoader.ReadFile(env, txtArgs, fileSource);
                    var textLoader = TextLoader.ReadFile(env, txtArgs, fileSource);
                    //env.Assert(textLoader.Schema.TryGetColumnIndex(keyColumnName, out int keyColumnIndex));
                    //env.Assert(textLoader.Schema.TryGetColumnIndex(valueColumnName, out int valueColumnIndex));

                    // Default to a text loader. KeyType and ValueType are assumed to be string
                    // types unless ValueAsKeyType is specified.
                    //TextLoader.Column keyColumn = new TextLoader.Column(keyColumnName, DataKind.TXT, keyColumnIndex);
                    //TextLoader.Column valueColumn = new TextLoader.Column(valueColumnName, DataKind.TXT, valueColumnIndex);
                    if (args.ValuesAsKeyType)
                    {
                        valueColumn = GenerateValueColumn(env, textLoader, valueColumnName, 0, 1);
                        // Change ValueColumn to be of type U4
                        //valueColumn = new TextLoader.Column(valueColumnName, DataKind.U4, valueColumnIndex);
                        //GenerateKeyRangeAndMinFromValues(env, textLoader, keyColumnIndex, valueColumnIndex, out ulong min, out ulong max);
                        //valueColumn.KeyRange = new KeyRange(min, max);
                    }

                    loader = TextLoader.Create(
                        env,
                        new TextLoader.Arguments()
                        {
                            Column = new TextLoader.Column[]
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

            ValueMappingTransform transformer = null;
            (string Source, string Name)[] columns = args.Column.Select(x => (x.Source, x.Name)).ToArray();
        /*
            Func<IHostEnvironment, IDataView, string, string, bool, (string Input, string Output)[], ValueMappingTransform> del = CreateTransformInvoke<int, int>;
            var meth = del.Method.GetGenericMethodDefinition().MakeGenericMethod(loader.Schema[keyColumnIndex].Type.RawType,
                                                                                  loader.Schema[valueColumnIndex].Type.RawType);
            transformer = (ValueMappingTransform)meth.Invoke(null, new object[] { env,
                                                                                      loader,
                                                                                      keyColumnName,
                                                                                      valueColumnName,
                                                                                      args.ValuesAsKeyType,
                                                                                      columns
                                                                                 });
            /*
            if (args.ValuesAsKeyType)
            {
                Func<IHostEnvironment, IDataView, string, string, bool, (string Input, string Output)[], ValueMappingTransform> del = CreateTransformInvoke<int, int>;
                var meth = del.Method.GetGenericMethodDefinition().MakeGenericMethod(loader.Schema[keyColumnIndex].Type.RawType,
                                                                                      loader.Schema[valueColumnIndex].Type.RawType);
                transformer = (ValueMappingTransform)meth.Invoke(null, new object[] { env,
                                                                                          loader,
                                                                                          keyColumnName,
                                                                                          valueColumnName,
                                                                                          args.ValuesAsKeyType,
                                                                                          columns
                                                                                     });
            }
            else
            */
            transformer =  new ValueMappingTransform(env, loader, keyColumnName, valueColumnName, columns);
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

        protected static ValueMappingTransform Create(IHostEnvironment env, ModelLoadContext ctx)
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
            var columns = new (string Source, string Name)[length];
            for (int i = 0; i < length; i++)
            {
                columns[i].Name = ctx.LoadNonEmptyString();
                columns[i].Source = ctx.LoadNonEmptyString();
            }

            byte[] rgb = null;
            Action<BinaryReader> fn = r => rgb = ReadAllBytes(env, r);

            if (!ctx.TryLoadBinaryStream(DefaultMapName, fn))
                throw env.ExceptDecode();

            var binaryLoader = GetLoader(env, rgb);
            var keyColumnName = (termLookupModel) ? "Term" : KeyColumnName;
            return new ValueMappingTransform(env, binaryLoader, keyColumnName, ValueColumnName, columns);
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

        protected static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(Schema.Create(inputSchema));

        protected static PrimitiveType GetPrimitiveType(Type rawType, out bool isVectorType)
        {
            Type type = rawType;
            isVectorType = false;
            if (type.IsArray)
            {
                type = rawType.GetElementType();
                isVectorType = true;
            }

            if  (!type.TryGetDataKind(out DataKind kind))
            {
                throw new InvalidOperationException($"Unsupported type {type} used in mapping.");
            }

            return PrimitiveType.FromKind(kind);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.SetVersionInfo(GetVersionInfo());
            SaveColumns(ctx);

            // convert value map to a dataview and serialize as bytes
            var bytes = GetBytesFromDataView(Host, _valueMap.GetDataView(Host), KeyColumnName, ValueColumnName);
            ctx.SaveBinaryStream(DefaultMapName, w => w.Write(bytes));
        }

        /// <summary>
        /// Base class that contains the mapping of keys to values.
        /// </summary>
        private abstract class ValueMap
        {
            public readonly ColumnType KeyType;
            public readonly ColumnType ValueType;

            public ValueMap(ColumnType keyType, ColumnType valueType)
            {
                KeyType = keyType;
                ValueType = valueType;
            }

            public static ValueMap Create(ColumnType keyType, ColumnType valueType, Schema.Metadata valueMetadata)
            {
               Func<ColumnType, ColumnType, Schema.Metadata, ValueMap> del = CreateValueMapInvoke<int, int>;
               var meth = del.Method.GetGenericMethodDefinition().MakeGenericMethod(keyType.RawType, valueType.RawType);
               return (ValueMap)meth.Invoke(null, new object[] { keyType, valueType, valueMetadata });
            }

            private static ValueMap CreateValueMapInvoke<TKeyType, TValueType>(ColumnType keyType,
                                                                                ColumnType valueType,
                                                                                Schema.Metadata valueMetadata)
            {
                return new ValueMap<TKeyType, TValueType>(keyType, valueType, valueMetadata);
            }

            public abstract void Train(IHostEnvironment env, IRowCursor cursor);

            public abstract Delegate GetGetter(IRow input, int index);

            public abstract IDataView GetDataView(IHostEnvironment env);
        }

        /// <summary>
        /// Implementation mapping class that maps a key of TKeyType to a specified value of TValueType.
        /// </summary>
        private class ValueMap<TKeyType, TValueType> : ValueMap
        {
            private Dictionary<TKeyType, TValueType> _mapping;
            private TValueType _missingValue;
            private Schema.Metadata _valueMetadata;

            private Dictionary<TKeyType, TValueType> CreateDictionary()
            {
                if (typeof(TKeyType) == typeof(ReadOnlyMemory<char>))
                    return new Dictionary<ReadOnlyMemory<char>, TValueType>(new ReadOnlyMemoryUtils.ReadonlyMemoryCharComparer()) as Dictionary<TKeyType, TValueType>;
                return new Dictionary<TKeyType, TValueType>();
            }

            public ValueMap(ColumnType keyType, ColumnType valueType, Schema.Metadata valueMetadata)
                : base(keyType, valueType)
            {
                _mapping = CreateDictionary();
                _valueMetadata = valueMetadata;
            }

            public override void Train(IHostEnvironment env, IRowCursor cursor)
            {
                // Validate that the conversion is supported for non-vector types
                bool identity;
                ValueMapper<ReadOnlyMemory<char>, TValueType> conv;

                // For keys that are not in the mapping, the missingValue will be returned.
                _missingValue  = default;
                if (!ValueType.IsVector)
                {
                    // For handling missing values, this follows how a missing value is handled when loading from a text source.
                    // First check if there is a String->ValueType conversion method. If so, call the conversion method with an
                    // empty string, the returned value will be the new missing value.
                    // NOTE this will return NA for R4 and R8 types.
                    if (Microsoft.ML.Runtime.Data.Conversion.Conversions.Instance.TryGetStandardConversion<ReadOnlyMemory<char>, TValueType>(
                                                                        TextType.Instance,
                                                                        ValueType,
                                                                        out conv,
                                                                        out identity))
                    {
                        TValueType value = default;
                        conv(string.Empty.AsMemory(), ref value);
                        _missingValue = value;
                    }
                }

                var keyGetter = cursor.GetGetter<TKeyType>(0);
                var valueGetter = cursor.GetGetter<TValueType>(1);
                while(cursor.MoveNext())
                {
                    TKeyType key = default;
                    TValueType value = default;
                    keyGetter(ref key);
                    valueGetter(ref value);
                    if (_mapping.ContainsKey(key))
                    {
                        throw env.Except($"Duplicate keys in data '{key}'");
                    }
                    _mapping.Add(key, value);
                }
            }

            public override Delegate GetGetter(IRow input, int index)
            {
                var src = default(TKeyType);
                ValueGetter<TKeyType> getSrc = input.GetGetter<TKeyType>(index);
                ValueGetter<TValueType> retVal =
                (ref TValueType dst) =>
                {
                    getSrc(ref src);
                    if (_mapping.ContainsKey(src))
                    {
                        if (ValueType.IsVector)
                            dst = Utils.MarshalInvoke(GetVector<int>, ValueType.ItemType.RawType, _mapping[src]);
                        else
                            dst = Utils.MarshalInvoke(GetValue<int>, ValueType.RawType, _mapping[src]);
                    }
                    else
                       dst = _missingValue;
                };
                return retVal;
            }

            public override IDataView GetDataView(IHostEnvironment env)
                => DataViewHelper.CreateDataView(env,
                                                 _mapping.Keys,
                                                 _mapping.Values,
                                                 ValueMappingTransform.KeyColumnName,
                                                 ValueMappingTransform.ValueColumnName,
                                                 ValueType.IsKey);

            private static TValueType GetVector<T>(TValueType value)
            {
                if (value is VBuffer<T> valueRef)
                {
                    VBuffer<T> dest = default;
                    valueRef.CopyTo(ref dest);
                    if (dest is TValueType destRef)
                        return destRef;
                }

                return default;
            }

            private static TValueType GetValue<T>(TValueType value)
                => value;
        }

        private static byte[] GetBytesFromDataView(IHost host, IDataView lookup, string keyColumn, string valueColumn)
        {
            Contracts.AssertValue(host);
            host.AssertValue(lookup);
            host.AssertNonEmpty(keyColumn);
            host.AssertNonEmpty(valueColumn);

            var schema = lookup.Schema;

            if (!schema.TryGetColumnIndex(keyColumn, out int colKey))
                throw host.ExceptUserArg(nameof(Arguments.KeyColumn), "column not found: '{0}'", keyColumn);
            if (!schema.TryGetColumnIndex(valueColumn, out int colValue))
                throw host.ExceptUserArg(nameof(Arguments.ValueColumn), "column not found: '{0}'", valueColumn);

            var cols = new List<(string Source, string Name)>()
            {
                (keyColumn, KeyColumnName),
                (valueColumn, ValueColumnName)
            };

            var view = new ColumnCopyingTransformer(host, cols.ToArray()).Transform(lookup);
            view = ColumnSelectingTransformer.CreateKeep(host, view, cols.Select(x=>x.Name).ToArray());

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

        protected override IRowMapper MakeRowMapper(Schema schema)
        {
            return new Mapper(this, Schema.Create(schema), _valueMap,  _valueMetadata, ColumnPairs);
        }

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly Schema _inputSchema;
            private readonly ValueMap _valueMap;
            private readonly Schema.Metadata _valueMetadata;
            private readonly (string Source, string Name)[] _columns;
            private readonly ValueMappingTransform _parent;

            internal Mapper(ValueMappingTransform transform,
                            Schema inputSchema,
                            ValueMap valueMap,
                            Schema.Metadata valueMetadata,
                            (string input, string output)[] columns)
                : base(transform.Host.Register(nameof(Mapper)), transform, inputSchema)
            {
                _inputSchema = inputSchema;
                _valueMetadata = valueMetadata;
                _valueMap = valueMap;
                _columns = columns;
                _parent = transform;
            }

            protected override Delegate MakeGetter(IRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _columns.Length);
                disposer = null;

                return _valueMap.GetGetter(input, ColMapNewToOld[iinfo]);
            }

            protected override Schema.DetachedColumn[] GetOutputColumnsCore()
            {
                var md = new MetadataBuilder();

                var result = new Schema.DetachedColumn[_columns.Length];
                for (int i = 0; i < _columns.Length; i++)
                {
                    var srcCol = _inputSchema[_columns[i].Source];
                    result[i] = new Schema.DetachedColumn(_columns[i].Name, _valueMap.ValueType, md.GetMetadata());
                }
                return result;
            }
        }
    }
}
