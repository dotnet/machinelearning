// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
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

[assembly: LoadableClass(ValueMappingTransform.Summary, typeof(IDataTransform), typeof(ValueMappingTransform), null, typeof(SignatureLoadDataTransform),
    "Value Mapping Transform", ValueMappingTransform.LoaderSignature)]

[assembly: LoadableClass(ValueMappingTransform.Summary, typeof(ValueMappingTransform), null, typeof(SignatureLoadModel),
    "Value Mapping Transform", ValueMappingTransform.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    public sealed class ValueMappingEstimator<TKeyType, TValueType> : TrivialEstimator<ValueMappingTransform<TKeyType, TValueType>>
    {
        private (string input, string output)[] _columns;

        public ValueMappingEstimator(IHostEnvironment env, IEnumerable<TKeyType> keys, IEnumerable<TValueType> values, params (string input, string output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ValueMappingEstimator<TKeyType, TValueType>)),
                    new ValueMappingTransform<TKeyType, TValueType>(env, keys, values, columns))
        {
            _columns = columns;
        }

        public ValueMappingEstimator(IHostEnvironment env, IEnumerable<TKeyType[]> keys, IEnumerable<TValueType> values, params (string input, string output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ValueMappingEstimator<TKeyType, TValueType>)),
                    new ValueMappingTransform<TKeyType, TValueType>(env, keys, values, columns))
        {
            _columns = columns;
        }

        public ValueMappingEstimator(IHostEnvironment env, IEnumerable<TKeyType> keys, IEnumerable<TValueType[]> values, params (string input, string output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ValueMappingEstimator<TKeyType, TValueType>)),
                    new ValueMappingTransform<TKeyType, TValueType>(env, keys, values, columns))
        {
            _columns = columns;
        }

        public ValueMappingEstimator(IHostEnvironment env, IEnumerable<TKeyType[]> keys, IEnumerable<TValueType[]> values, params (string input, string output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ValueMappingEstimator<TKeyType, TValueType>)),
                    new ValueMappingTransform<TKeyType, TValueType>(env, keys, values, columns))
        {
            _columns = columns;
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            var resultDic = inputSchema.Columns.ToDictionary(x => x.Name);

            var outputType = typeof(TValueType);
            ColumnType outputColumnType = default;
            if (outputType.IsGenericEx(typeof(VBuffer<>)))
            {
                Type vBufferType = outputType.GetGenericArguments()[0];
                vBufferType.TryGetDataKind(out DataKind kind);
                outputColumnType = new VectorType(PrimitiveType.FromKind(kind));
            }
            else
            {
                outputType.TryGetDataKind(out DataKind kind);
                outputColumnType = PrimitiveType.FromKind(kind);
            }

            foreach (var (Input, Output) in _columns)
            {
                if (!inputSchema.TryFindColumn(Input, out var originalColumn))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", Input);

                // Get the type from TOutputType
                var col = new SchemaShape.Column(Output, originalColumn.Kind, outputColumnType, originalColumn.IsKey, originalColumn.Metadata);
                resultDic[Output] = col;
            }
            return new SchemaShape(resultDic.Values);
        }
    }

    public sealed class ValueMappingTransform<TKeyType, TValueType> : ValueMappingTransform
    {
        public ValueMappingTransform(IHostEnvironment env, IEnumerable<TKeyType> keys, IEnumerable<TValueType> values, (string Input, string Output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ValueMappingTransform<TKeyType, TValueType>)),
                  ConvertToDataView(env, keys, values), columns)
        { }

        public ValueMappingTransform(IHostEnvironment env, IEnumerable<TKeyType[]> keys, IEnumerable<TValueType> values, (string Input, string Output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ValueMappingTransform<TKeyType, TValueType>)),
                ConvertToDataView(env, keys, values), columns)
        { }

        public ValueMappingTransform(IHostEnvironment env, IEnumerable<TKeyType> keys, IEnumerable<TValueType[]> values, (string Input, string Output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ValueMappingTransform<TKeyType, TValueType>)),
                ConvertToDataView(env, keys, values), columns)
        { }

        public ValueMappingTransform(IHostEnvironment env, IEnumerable<TKeyType[]> keys, IEnumerable<TValueType[]> values, (string Input, string Output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ValueMappingTransform<TKeyType, TValueType>)),
                  ConvertToDataView(env, keys, values), columns)
        { }

        private static IDataView ConvertToDataView(IHostEnvironment env, IEnumerable<TKeyType> keys, IEnumerable<TValueType> values)
        {
            // Build DataView from the mapping
            var keyType = ValueMappingTransform.GetPrimitiveType(typeof(TKeyType), out bool isKeyVectorType);
            var valueType = ValueMappingTransform.GetPrimitiveType(typeof(TValueType), out bool isValueVectorType);
            var dataViewBuilder = new ArrayDataViewBuilder(env);
            dataViewBuilder.AddColumn(ValueMappingTransform.KeyColumnName, keyType, keys.ToArray());
            dataViewBuilder.AddColumn(ValueMappingTransform.ValueColumnName, valueType, values.ToArray());
            return dataViewBuilder.GetDataView();
        }

        private static IDataView ConvertToDataView(IHostEnvironment env, IEnumerable<TKeyType[]> keys, IEnumerable<TValueType> values)
        {
            // Build DataView from the mapping
            var keyType = ValueMappingTransform.GetPrimitiveType(typeof(TKeyType), out bool isKeyVectorType);
            var valueType = ValueMappingTransform.GetPrimitiveType(typeof(TValueType), out bool isValueVectorType);
            var dataViewBuilder = new ArrayDataViewBuilder(env);
            dataViewBuilder.AddColumn(ValueMappingTransform.KeyColumnName, keyType, keys.ToArray());
            dataViewBuilder.AddColumn(ValueMappingTransform.ValueColumnName, valueType, values.ToArray());
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

        private static IDataView ConvertToDataView(IHostEnvironment env, IEnumerable<TKeyType[]> keys, IEnumerable<TValueType[]> values)
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

        // Stream names for the binary idv streams.
        private const string DefaultMapName = "DefaultMap.idv";
        protected static string KeyColumnName = "Key";
        protected static string ValueColumnName = "Value";
        private ValueMap _valueMap;

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

            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the text column containing the terms", ShortName = "term")]
            public string TermColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the column containing the values", ShortName = "value")]
            public string ValueColumn;

            [Argument(ArgumentType.Multiple, HelpText = "The data loader", NullName = "<Auto>", SignatureType = typeof(SignatureDataLoader))]
            public IComponentFactory<IMultiStreamSource, IDataLoader> Loader;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "If term and value columns are unspecified, specifies whether the values are key values or numeric.", ShortName = "key")]
            public bool KeyValues = true;

            [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "The data file containing the terms", ShortName = "data", SortOrder = 2)]
            public string DataFile;
        }

        protected ValueMappingTransform(IHostEnvironment env, IDataView lookupMap, (string Input, string Output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ValueMappingTransform)), columns)
        {
            _valueMap = CreateValueMapFromDataView(lookupMap);
        }

        private ValueMap CreateValueMapFromDataView(IDataView dataView)
        {
            Contracts.Check(dataView.Schema.GetColumns().Count() == 2);
            Contracts.Check(dataView.GetRowCount() > 0);
            var keyType = dataView.Schema.GetColumnType(0);
            var valueType = dataView.Schema.GetColumnType(1);
            var valueMap = ValueMap.Create(keyType, valueType);
            using (var cursor = dataView.GetRowCursor(c=> true))
                valueMap.Train(Host, cursor);
            return valueMap;
        }

        protected static ValueMappingTransform Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

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
            return new ValueMappingTransform(env, binaryLoader, columns);
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

            type.TryGetDataKind(out DataKind kind);
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

            public ValueMap(ColumnType keyType, ColumnType valueType)
                : base(keyType, valueType)
            {
                _mapping = new Dictionary<TKeyType, TValueType>();
            }

            public override void Train(IHostEnvironment env, IRowCursor cursor)
            {
                while(cursor.MoveNext())
                {
                    var keyGetter = cursor.GetGetter<TKeyType>(0);
                    var valueGetter = cursor.GetGetter<TValueType>(1);
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
                ValueGetter<TKeyType> getSrc = input.GetGetter<TKeyType>(index);;
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
                        dst = default;
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

        private static byte[] GetBytesFromDataView(IHost host, IDataView lookup, string termColumn, string valueColumn)
        {
            Contracts.AssertValue(host);
            host.AssertValue(lookup);
            host.AssertNonEmpty(termColumn);
            host.AssertNonEmpty(valueColumn);

            int colTerm;
            int colValue;
            var schema = lookup.Schema;

            if (!schema.TryGetColumnIndex(termColumn, out colTerm))
                throw host.ExceptUserArg(nameof(Arguments.TermColumn), "column not found: '{0}'", termColumn);
            if (!schema.TryGetColumnIndex(valueColumn, out colValue))
                throw host.ExceptUserArg(nameof(Arguments.ValueColumn), "column not found: '{0}'", valueColumn);

            // REVIEW: Should we allow term to be a vector of text (each term in the vector
            // would map to the same value)?
            var typeTerm = schema.GetColumnType(colTerm);
            host.CheckUserArg(typeTerm.IsText, nameof(Arguments.TermColumn), "term column must contain text");
            var typeValue = schema.GetColumnType(colValue);
            var cols = new List<(string Source, string Name)>()
            {
                (termColumn, KeyColumnName),
                (valueColumn, ValueColumnName)
            };

            var view = new ColumnsCopyingTransformer(host, cols.ToArray()).Transform(lookup);
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
            => new Mapper(this, Schema.Create(schema), _valueMap, ColumnPairs);

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

            protected override Schema.Column[] GetOutputColumnsCore()
            {
                var result = new Schema.Column[_columns.Length];
                for (int i = 0; i < _columns.Length; i++)
                {
                    var srcCol = _inputSchema[_columns[i].Source];
                    result[i] = new Schema.Column(_columns[i].Name, _valueMap.ValueType, srcCol.Metadata);
                }
                return result;
            }
        }
    }
}
