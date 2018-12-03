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
            var columnType = (isKey) ? Transformer.ValueColumnType.ItemType :
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

        private static ValueGetter<VBuffer<ReadOnlyMemory<char>>> GetKeyValueGetter(TValueType[] values)
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

        private static IDataView ConvertToDataView(IHostEnvironment env, IEnumerable<TKeyType> keys, IEnumerable<TValueType> values, bool treatValuesAsKeyValue)
        {
            // Build DataView from the mapping
            var keyType = ValueMappingTransform.GetPrimitiveType(typeof(TKeyType), out bool isKeyVectorType);
            var valueType = ValueMappingTransform.GetPrimitiveType(typeof(TValueType), out bool isValueVectorType);

            // If treatValuesAsKeyValues can only be used with non-vector types
            env.Check(!(treatValuesAsKeyValue && valueType.IsVector), "Treating values as key value types can only be used on non-vector types.");

            var dataViewBuilder = new ArrayDataViewBuilder(env);
            dataViewBuilder.AddColumn(ValueMappingTransform.KeyColumnName, keyType, keys.ToArray());
            var valuesArr = values.ToArray();
            if (treatValuesAsKeyValue)
            {
                uint[] indices = Enumerable.Range(0, count: values.Count()).Select(i => (uint)i).ToArray();
                dataViewBuilder.AddColumn(ValueMappingTransform.ValueColumnName, GetKeyValueGetter(valuesArr), 0, indices.Length, indices);
            }
            else
            {
                dataViewBuilder.AddColumn(ValueMappingTransform.ValueColumnName, valueType, values.ToArray());
            }

            return dataViewBuilder.GetDataView();
        }

        private static IDataView ConvertToDataView(IHostEnvironment env, IEnumerable<TKeyType> keys, IEnumerable<TValueType[]> values)
        {
            // Build DataView from the mapping
            var keyType = ValueMappingTransform.GetPrimitiveType(typeof(TKeyType), out bool isKeyVectorType);
            var valueType = ValueMappingTransform.GetPrimitiveType(typeof(TValueType), out bool isValueVectorType);
            var dataViewBuilder = new ArrayDataViewBuilder(env);
            dataViewBuilder.AddColumn(ValueMappingTransform.KeyColumnName, keyType, keys.ToArray());
            dataViewBuilder.AddColumn(ValueMappingTransform.ValueColumnName, valueType, values.ToArray());
            return dataViewBuilder.GetDataView();
        }
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
        }

        protected ValueMappingTransform(IHostEnvironment env, IDataView lookupMap, string keyColumn, string valueColumn, (string Input, string Output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ValueMappingTransform)), columns)
        {
            env.CheckNonEmpty(keyColumn, nameof(keyColumn), "A key column must be specified when passing in an IDataView for the value mapping");
            env.CheckNonEmpty(valueColumn, nameof(valueColumn), "A value column must be specified when passing in an IDataView for the value mapping");
            _valueMap = CreateValueMapFromDataView(lookupMap, keyColumn, valueColumn);
        }

        private ValueMap CreateValueMapFromDataView(IDataView dataView, string keyColumn, string valueColumn)
        {
            // Confirm that the key and value columns exist in the dataView
            Host.Check(dataView.Schema.TryGetColumnIndex(keyColumn, out int keyIdx), "Key column " + keyColumn + " does not exist in the given dataview");
            Host.Check(dataView.Schema.TryGetColumnIndex(valueColumn, out int valueIdx), "Value column " + valueColumn + " does not exist in the given dataview");
            var keyType = dataView.Schema.GetColumnType(keyIdx);
            var valueType = dataView.Schema.GetColumnType(valueIdx);
            var valueMap = ValueMap.Create(keyType, valueType);
            using (var cursor = dataView.GetRowCursor(c=> c == keyIdx || c == valueIdx))
                valueMap.Train(Host, cursor);
            return valueMap;
        }

        private static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.Assert(!string.IsNullOrWhiteSpace(args.DataFile));
            env.AssertNonEmpty(args.KeyColumn);
            env.AssertNonEmpty(args.ValueColumn);

            IMultiStreamSource fileSource = new MultiFileSource(args.DataFile);
            IDataView loader;
            if (args.Loader != null)
            {
                loader = args.Loader.CreateComponent(env, fileSource);
            }
            else
            {
                loader = new BinaryLoader(env, new BinaryLoader.Arguments(), fileSource);
            }

            var transformer = new ValueMappingTransform(env, loader, args.KeyColumn, args.ValueColumn, args.Column.Select(x => (x.Source, x.Name)).ToArray());
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
            env.Check(CheckModelVersion(ctx, GetVersionInfo()) ||
                        CheckModelVersion(ctx, GetTermLookupVersionInfo()));

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
            return new ValueMappingTransform(env, binaryLoader, KeyColumnName, ValueColumnName, columns);
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
        /// Holds the values that the terms map to.
        /// </summary>
        protected abstract class ValueMap
        {
            public readonly ColumnType KeyType;
            public readonly ColumnType ValueType;

            public ValueMap(ColumnType keyType, ColumnType valueType)
            {
                KeyType = keyType;
                ValueType = valueType;
            }

            public static ValueMap Create(ColumnType keyType, ColumnType valueType)
            {
               Func<ColumnType, ColumnType, ValueMap> del = CreateValueMapInvoke<int, int>;
               var meth = del.Method.GetGenericMethodDefinition().MakeGenericMethod(keyType.RawType, valueType.RawType);
               return (ValueMap)meth.Invoke(null, new object[] { keyType, valueType });
            }

            private static ValueMap CreateValueMapInvoke<TKeyType, TValueType>(ColumnType keyType, ColumnType valueType)
            {
                return new ValueMap<TKeyType, TValueType>(keyType, valueType);
            }

            public abstract void Train(IHostEnvironment env, IRowCursor cursor);

            public abstract Delegate GetGetter(IRow input, int index);

            public abstract IDataView GetDataView(IHostEnvironment env);
        }

        private class ValueMap<TKeyType, TValueType> : ValueMap
        {
            private Dictionary<TKeyType, TValueType> _mapping;
            private TValueType _missingValue;

            public ValueMap(ColumnType keyType, ColumnType valueType)
                : base(keyType, valueType)
            {
                _mapping = new Dictionary<TKeyType, TValueType>();
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
                    if (Runtime.Data.Conversion.Conversions.Instance.TryGetStandardConversion<ReadOnlyMemory<char>, TValueType>(
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
            {
                var dataViewBuilder = new ArrayDataViewBuilder(env);
                var keyType = ValueMappingTransform.GetPrimitiveType(typeof(TKeyType), out bool isKeyVectorType);
                var valueType = ValueMappingTransform.GetPrimitiveType(typeof(TValueType), out bool isValueVectorType);
                dataViewBuilder.AddColumn(ValueMappingTransform.KeyColumnName, keyType, _mapping.Keys.ToArray());
                dataViewBuilder.AddColumn(ValueMappingTransform.ValueColumnName,valueType, _mapping.Values.ToArray());
                return dataViewBuilder.GetDataView();
            }

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
            return new Mapper(this, Schema.Create(schema), _valueMap, ColumnPairs);
        }

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly Schema _inputSchema;
            private readonly ValueMap _valueMap;
            private readonly (string Source, string Name)[] _columns;
            private readonly ValueMappingTransform _parent;

            internal Mapper(ValueMappingTransform transform,
                            Schema inputSchema,
                            ValueMap valueMap,
                            (string input, string output)[] columns)
                : base(transform.Host.Register(nameof(Mapper)), transform, inputSchema)
            {
                _inputSchema = inputSchema;
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
                var result = new Schema.DetachedColumn[_columns.Length];
                for (int i = 0; i < _columns.Length; i++)
                {
                    var srcCol = _inputSchema[_columns[i].Source];
                    result[i] = new Schema.DetachedColumn(_columns[i].Name, _valueMap.ValueType, srcCol.Metadata);
                }
                return result;
            }
        }
    }
}
