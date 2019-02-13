// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Model.Onnx;
using Microsoft.ML.Model.Pfa;
using Microsoft.ML.Transforms.Conversions;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(KeyToVectorMappingTransformer.Summary, typeof(IDataTransform), typeof(KeyToVectorMappingTransformer), typeof(KeyToVectorMappingTransformer.Options), typeof(SignatureDataTransform),
    "Key To Vector Transform", KeyToVectorMappingTransformer.UserName, "KeyToVector", "ToVector", DocName = "transform/KeyToVectorTransform.md")]

[assembly: LoadableClass(KeyToVectorMappingTransformer.Summary, typeof(IDataTransform), typeof(KeyToVectorMappingTransformer), null, typeof(SignatureLoadDataTransform),
    "Key To Vector Transform", KeyToVectorMappingTransformer.LoaderSignature)]

[assembly: LoadableClass(KeyToVectorMappingTransformer.Summary, typeof(KeyToVectorMappingTransformer), null, typeof(SignatureLoadModel),
    KeyToVectorMappingTransformer.UserName, KeyToVectorMappingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(KeyToVectorMappingTransformer), null, typeof(SignatureLoadRowMapper),
   KeyToVectorMappingTransformer.UserName, KeyToVectorMappingTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms.Conversions
{
    /// <summary>
    /// Converts the key types back to their original vectors.
    /// </summary>
    public sealed class KeyToVectorMappingTransformer : OneToOneTransformerBase
    {
        internal abstract class ColumnBase : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Whether to combine multiple indicator vectors into a single bag vector instead of concatenating them. This is only relevant when the input is a vector.")]
            public bool? Bag;

            private protected ColumnBase()
            {
            }

            private protected override bool TryUnparseCore(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (Bag != null)
                    return false;
                return base.TryUnparseCore(sb);
            }

            private protected override bool TryUnparseCore(StringBuilder sb, string extra)
            {
                Contracts.AssertValue(sb);
                Contracts.AssertNonEmpty(extra);
                if (Bag != null)
                    return false;
                return base.TryUnparseCore(sb, extra);
            }
        }

        [BestFriend]
        internal sealed class Column : ColumnBase
        {
            internal static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

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
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:src)",
                Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Whether to combine multiple indicator vectors into a single bag vector instead of concatenating them. This is only relevant when the input is a vector.")]
            public bool Bag = KeyToVectorMappingEstimator.Defaults.Bag;
        }

        private const string RegistrationName = "KeyToVector";

        public IReadOnlyCollection<KeyToVectorMappingEstimator.ColumnInfo> Columns => _columns.AsReadOnly();
        private readonly KeyToVectorMappingEstimator.ColumnInfo[] _columns;

        private static (string outputColumnName, string inputColumnName)[] GetColumnPairs(KeyToVectorMappingEstimator.ColumnInfo[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Name, x.InputColumnName)).ToArray();
        }

        private string TestIsKey(ColumnType type)
        {
            if (type.GetItemType().GetKeyCount() > 0)
                return null;
            return "key type of known cardinality";
        }

        protected override void CheckInputColumn(Schema inputSchema, int col, int srcCol)
        {
            var type = inputSchema[srcCol].Type;
            string reason = TestIsKey(type);
            if (reason != null)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].inputColumnName, reason, type.ToString());
        }

        internal KeyToVectorMappingTransformer(IHostEnvironment env, params KeyToVectorMappingEstimator.ColumnInfo[] columns) :
            base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            _columns = columns.ToArray();
        }

        internal const string LoaderSignature = "KeyToVectorTransform";
        internal const string UserName = "KeyToVectorTransform";
        internal const string Summary = "Converts a key column to an indicator vector.";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "KEY2VECT",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Get rid of writing float size in model context
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(KeyToVectorMappingTransformer).Assembly.FullName);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            // *** Binary format ***
            // <base>
            // for each added column
            //   byte: bag as 0/1
            SaveColumns(ctx);

            Host.Assert(_columns.Length == ColumnPairs.Length);
            for (int i = 0; i < _columns.Length; i++)
                ctx.Writer.WriteBoolByte(_columns[i].Bag);
        }

        // Factory method for SignatureLoadModel.
        private static KeyToVectorMappingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            if (ctx.Header.ModelVerWritten == 0x00010001)
            {
                int cbFloat = ctx.Reader.ReadInt32();
                env.CheckDecode(cbFloat == sizeof(float));
            }
            return new KeyToVectorMappingTransformer(host, ctx);
        }

        private KeyToVectorMappingTransformer(IHost host, ModelLoadContext ctx)
          : base(host, ctx)
        {
            var columnsLength = ColumnPairs.Length;
            // *** Binary format ***
            // <base>
            // for each added column
            //   byte: bag as 0/1
            var bags = new bool[columnsLength];
            bags = ctx.Reader.ReadBoolArray(columnsLength);

            _columns = new KeyToVectorMappingEstimator.ColumnInfo[columnsLength];
            for (int i = 0; i < columnsLength; i++)
                _columns[i] = new KeyToVectorMappingEstimator.ColumnInfo(ColumnPairs[i].outputColumnName, ColumnPairs[i].inputColumnName, bags[i]);
        }

        // Factory method for SignatureDataTransform.
        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            env.CheckValue(options.Columns, nameof(options.Columns));
            var cols = new KeyToVectorMappingEstimator.ColumnInfo[options.Columns.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                var item = options.Columns[i];

                cols[i] = new KeyToVectorMappingEstimator.ColumnInfo(
                    item.Name,
                    item.Source ?? item.Name,
                    item.Bag ?? options.Bag);
            };
            return new KeyToVectorMappingTransformer(env, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, Schema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(Schema schema) => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase, ISaveAsOnnx, ISaveAsPfa
        {
            private sealed class ColInfo
            {
                public readonly string Name;
                public readonly string InputColumnName;
                public readonly ColumnType TypeSrc;

                public ColInfo(string outputColumnName, string inputColumnName, ColumnType type)
                {
                    Name = outputColumnName;
                    InputColumnName = inputColumnName;
                    TypeSrc = type;
                }
            }

            private readonly KeyToVectorMappingTransformer _parent;
            private readonly ColInfo[] _infos;
            private readonly VectorType[] _types;

            public Mapper(KeyToVectorMappingTransformer parent, Schema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _infos = CreateInfos(inputSchema);
                _types = new VectorType[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    int valueCount = _infos[i].TypeSrc.GetValueCount();
                    int keyCount = _infos[i].TypeSrc.GetItemType().GetKeyCountAsInt32(Host);
                    if (_parent._columns[i].Bag || valueCount == 1)
                        _types[i] = new VectorType(NumberType.Float, keyCount);
                    else
                        _types[i] = new VectorType(NumberType.Float, valueCount, keyCount);
                }
            }

            private ColInfo[] CreateInfos(Schema inputSchema)
            {
                Host.AssertValue(inputSchema);
                var infos = new ColInfo[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    if (!inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out int colSrc))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.ColumnPairs[i].inputColumnName);
                    var type = inputSchema[colSrc].Type;
                    _parent.CheckInputColumn(inputSchema, i, colSrc);
                    infos[i] = new ColInfo(_parent.ColumnPairs[i].outputColumnName, _parent.ColumnPairs[i].inputColumnName, type);
                }
                return infos;
            }

            protected override Schema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new Schema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out int colIndex);
                    Host.Assert(colIndex >= 0);
                    var builder = new MetadataBuilder();
                    AddMetadata(i, builder);
                    result[i] = new Schema.DetachedColumn(_parent.ColumnPairs[i].outputColumnName, _types[i], builder.GetMetadata());
                }
                return result;
            }

            private void AddMetadata(int iinfo, MetadataBuilder builder)
            {
                InputSchema.TryGetColumnIndex(_infos[iinfo].InputColumnName, out int srcCol);
                var inputMetadata = InputSchema[srcCol].Metadata;

                var srcType = _infos[iinfo].TypeSrc;
                int srcValueCount = srcType.GetValueCount();

                VectorType typeNames = null;
                int metaKeyValuesCol = 0;
                if (inputMetadata.Schema.TryGetColumnIndex(MetadataUtils.Kinds.KeyValues, out metaKeyValuesCol))
                    typeNames = inputMetadata.Schema[metaKeyValuesCol].Type as VectorType;
                if (typeNames == null || !typeNames.IsKnownSize || !(typeNames.ItemType is TextType) ||
                    typeNames.Size != srcType.GetItemType().GetKeyCountAsInt32(Host))
                {
                    typeNames = null;
                }

                if (_parent._columns[iinfo].Bag || srcValueCount == 1)
                {
                    if (typeNames != null)
                    {
                        var getter = inputMetadata.GetGetter<VBuffer<ReadOnlyMemory<char>>>(metaKeyValuesCol);
                        var slotNamesType = new VectorType(TextType.Instance, _types[iinfo]);
                        builder.AddSlotNames(slotNamesType.Size, getter);
                    }
                }
                else
                {
                    if (typeNames != null && _types[iinfo].IsKnownSize)
                    {
                        ValueGetter<VBuffer<ReadOnlyMemory<char>>> getter = (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                        {
                            GetSlotNames(iinfo, ref dst);
                        };
                        var slotNamesType = new VectorType(TextType.Instance, _types[iinfo]);
                        builder.Add(MetadataUtils.Kinds.SlotNames, slotNamesType, getter);
                    }
                }

                if (!_parent._columns[iinfo].Bag && srcValueCount > 0)
                {
                    ValueGetter<VBuffer<int>> getter = (ref VBuffer<int> dst) =>
                    {
                        GetCategoricalSlotRanges(iinfo, ref dst);
                    };
                    builder.Add(MetadataUtils.Kinds.CategoricalSlotRanges, MetadataUtils.GetCategoricalType(srcValueCount), getter);
                }

                if (!_parent._columns[iinfo].Bag || srcValueCount == 1)
                {
                    ValueGetter<bool> getter = (ref bool dst) =>
                    {
                        dst = true;
                    };
                    builder.Add(MetadataUtils.Kinds.IsNormalized, BoolType.Instance, getter);
                }
            }

            // Combines source key names and slot names to produce final slot names.
            private void GetSlotNames(int iinfo, ref VBuffer<ReadOnlyMemory<char>> dst)
            {
                Host.Assert(0 <= iinfo && iinfo < _infos.Length);
                var typeSrc = _infos[iinfo].TypeSrc as VectorType;
                Host.Assert(typeSrc != null && typeSrc.IsKnownSize);

                // Size one should have been treated the same as Bag (by the caller).
                // Variable size should have thrown (by the caller).
                Host.Assert(typeSrc.Size > 1);

                // Get the source slot names, defaulting to empty text.
                var namesSlotSrc = default(VBuffer<ReadOnlyMemory<char>>);

                var inputMetadata = InputSchema[_infos[iinfo].InputColumnName].Metadata;
                Contracts.AssertValue(inputMetadata);
                var typeSlotSrc = inputMetadata.Schema.GetColumnOrNull(MetadataUtils.Kinds.SlotNames)?.Type as VectorType;
                if (typeSlotSrc != null && typeSlotSrc.Size == typeSrc.Size && typeSlotSrc.ItemType is TextType)
                {
                    inputMetadata.GetValue(MetadataUtils.Kinds.SlotNames, ref namesSlotSrc);
                    Host.Check(namesSlotSrc.Length == typeSrc.Size);
                }
                else
                    namesSlotSrc = VBufferUtils.CreateEmpty<ReadOnlyMemory<char>>(typeSrc.Size);

                int keyCount = typeSrc.ItemType.GetKeyCountAsInt32(Host);
                int slotLim = _types[iinfo].Size;
                Host.Assert(slotLim == (long)typeSrc.Size * keyCount);

                // Get the source key names, in an array (since we will use them multiple times).
                var namesKeySrc = default(VBuffer<ReadOnlyMemory<char>>);
                inputMetadata.GetValue(MetadataUtils.Kinds.KeyValues, ref namesKeySrc);
                Host.Check(namesKeySrc.Length == keyCount);
                var keys = new ReadOnlyMemory<char>[keyCount];
                namesKeySrc.CopyTo(keys);

                var editor = VBufferEditor.Create(ref dst, slotLim);

                var sb = new StringBuilder();
                int slot = 0;
                foreach (var kvpSlot in namesSlotSrc.Items(all: true))
                {
                    Contracts.Assert(slot == (long)kvpSlot.Key * keyCount);
                    sb.Clear();
                    if (!kvpSlot.Value.IsEmpty)
                        sb.AppendMemory(kvpSlot.Value);
                    else
                        sb.Append('[').Append(kvpSlot.Key).Append(']');
                    sb.Append('.');

                    int len = sb.Length;
                    foreach (var key in keys)
                    {
                        sb.Length = len;
                        sb.AppendMemory(key);
                        editor.Values[slot++] = sb.ToString().AsMemory();
                    }
                }
                Host.Assert(slot == slotLim);

                dst = editor.Commit();
            }

            private void GetCategoricalSlotRanges(int iinfo, ref VBuffer<int> dst)
            {
                Host.Assert(0 <= iinfo && iinfo < _infos.Length);

                var info = _infos[iinfo];
                int valueCount = info.TypeSrc.GetValueCount();

                Host.Assert(valueCount > 0);

                int[] ranges = new int[valueCount * 2];
                int size = info.TypeSrc.GetItemType().GetKeyCountAsInt32(Host);

                ranges[0] = 0;
                ranges[1] = size - 1;
                for (int i = 2; i < ranges.Length; i += 2)
                {
                    ranges[i] = ranges[i - 1] + 1;
                    ranges[i + 1] = ranges[i] + size - 1;
                }

                dst = new VBuffer<int>(ranges.Length, ranges);
            }

            protected override Delegate MakeGetter(Row input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _infos.Length);
                disposer = null;

                var info = _infos[iinfo];
                if (!(info.TypeSrc is VectorType))
                    return MakeGetterOne(input, iinfo);
                if (_parent._columns[iinfo].Bag)
                    return MakeGetterBag(input, iinfo);
                return MakeGetterInd(input, iinfo);
            }

            /// <summary>
            /// This is for the singleton case. This should be equivalent to both Bag and Ord over
            /// a vector of size one.
            /// </summary>
            private ValueGetter<VBuffer<float>> MakeGetterOne(Row input, int iinfo)
            {
                Host.AssertValue(input);
                KeyType keyTypeSrc = _infos[iinfo].TypeSrc as KeyType;
                Host.Assert(keyTypeSrc != null);
                int size = keyTypeSrc.GetCountAsInt32(Host);
                Host.Assert(size == _types[iinfo].Size);
                Host.Assert(size > 0);
                input.Schema.TryGetColumnIndex(_infos[iinfo].InputColumnName, out int srcCol);
                Host.Assert(srcCol >= 0);
                var getSrc = RowCursorUtils.GetGetterAs<uint>(NumberType.U4, input, srcCol);
                var src = default(uint);
                return
                    (ref VBuffer<float> dst) =>
                    {
                        getSrc(ref src);
                        if (src == 0 || src > size)
                        {
                            VBufferUtils.Resize(ref dst, size, 0);
                            return;
                        }

                        var editor = VBufferEditor.Create(ref dst, size, 1, requireIndicesOnDense: true);
                        editor.Values[0] = 1;
                        editor.Indices[0] = (int)src - 1;

                        dst = editor.Commit();
                    };
            }

            /// <summary>
            /// This is for the bagging case - vector input and outputs should be added.
            /// </summary>
            private ValueGetter<VBuffer<float>> MakeGetterBag(Row input, int iinfo)
            {
                Host.AssertValue(input);
                var info = _infos[iinfo];
                VectorType srcVectorType = info.TypeSrc as VectorType;
                Host.Assert(srcVectorType != null);

                KeyType keyTypeSrc = srcVectorType.ItemType as KeyType;
                Host.Assert(keyTypeSrc != null);
                Host.Assert(_parent._columns[iinfo].Bag);
                int size = keyTypeSrc.GetCountAsInt32(Host);
                Host.Assert(size == _types[iinfo].Size);
                Host.Assert(size > 0);

                int cv = srcVectorType.Size;
                Host.Assert(cv >= 0);
                input.Schema.TryGetColumnIndex(info.InputColumnName, out int srcCol);
                Host.Assert(srcCol >= 0);
                var getSrc = RowCursorUtils.GetVecGetterAs<uint>(NumberType.U4, input, srcCol);
                var src = default(VBuffer<uint>);
                var bldr = BufferBuilder<float>.CreateDefault();
                return
                    (ref VBuffer<float> dst) =>
                    {
                        bldr.Reset(size, false);

                        getSrc(ref src);
                        Host.Check(cv == 0 || src.Length == cv);

                        // The indices are irrelevant in the bagging case.
                        var values = src.GetValues();
                        int count = values.Length;
                        for (int slot = 0; slot < count; slot++)
                        {
                            uint key = values[slot] - 1;
                            if (key < size)
                                bldr.AddFeature((int)key, 1);
                        }

                        bldr.GetResult(ref dst);
                    };
            }

            /// <summary>
            /// This is for the indicator (non-bagging) case - vector input and outputs should be concatenated.
            /// </summary>
            private ValueGetter<VBuffer<float>> MakeGetterInd(Row input, int iinfo)
            {
                Host.AssertValue(input);
                var info = _infos[iinfo];
                VectorType srcVectorType = info.TypeSrc as VectorType;
                Host.Assert(srcVectorType != null);

                KeyType keyTypeSrc = srcVectorType.ItemType as KeyType;
                Host.Assert(keyTypeSrc != null);
                Host.Assert(!_parent._columns[iinfo].Bag);

                int size = keyTypeSrc.GetCountAsInt32(Host);
                Host.Assert(size > 0);

                int cv = srcVectorType.Size;
                Host.Assert(cv >= 0);
                Host.Assert(_types[iinfo].Size == size * cv);
                input.Schema.TryGetColumnIndex(info.InputColumnName, out int srcCol);
                var getSrc = RowCursorUtils.GetVecGetterAs<uint>(NumberType.U4, input, srcCol);
                var src = default(VBuffer<uint>);
                return
                    (ref VBuffer<float> dst) =>
                    {
                        getSrc(ref src);
                        int lenSrc = src.Length;
                        Host.Check(lenSrc == cv || cv == 0);

                        // Since we generate values in order, no need for a builder.
                        int lenDst = checked(size * lenSrc);
                        var values = src.GetValues();
                        int cntSrc = values.Length;
                        var editor = VBufferEditor.Create(ref dst, lenDst, cntSrc);

                        int count = 0;
                        if (src.IsDense)
                        {
                            Host.Assert(lenSrc == cntSrc);
                            for (int slot = 0; slot < cntSrc; slot++)
                            {
                                Host.Assert(count < cntSrc);
                                uint key = values[slot] - 1;
                                if (key >= (uint)size)
                                    continue;
                                editor.Values[count] = 1;
                                editor.Indices[count++] = slot * size + (int)key;
                            }
                        }
                        else
                        {
                            var indices = src.GetIndices();
                            for (int islot = 0; islot < cntSrc; islot++)
                            {
                                Host.Assert(count < cntSrc);
                                uint key = values[islot] - 1;
                                if (key >= (uint)size)
                                    continue;
                                editor.Values[count] = 1;
                                editor.Indices[count++] = indices[islot] * size + (int)key;
                            }
                        }
                        dst = editor.CommitTruncated(count);
                    };
            }

            public bool CanSaveOnnx(OnnxContext ctx) => true;

            public bool CanSavePfa => true;

            public void SaveAsOnnx(OnnxContext ctx)
            {
                Host.CheckValue(ctx, nameof(ctx));

                for (int iinfo = 0; iinfo < _infos.Length; ++iinfo)
                {
                    ColInfo info = _infos[iinfo];
                    string inputColumnName = info.InputColumnName;
                    if (!ctx.ContainsColumn(inputColumnName))
                    {
                        ctx.RemoveColumn(info.Name, false);
                        continue;
                    }

                    if (!SaveAsOnnxCore(ctx, iinfo, info, ctx.GetVariableName(inputColumnName),
                        ctx.AddIntermediateVariable(_types[iinfo], info.Name)))
                    {
                        ctx.RemoveColumn(info.Name, true);
                    }
                }
            }

            public void SaveAsPfa(BoundPfaContext ctx)
            {
                Host.CheckValue(ctx, nameof(ctx));

                var toHide = new List<string>();
                var toDeclare = new List<KeyValuePair<string, JToken>>();

                for (int iinfo = 0; iinfo < _infos.Length; ++iinfo)
                {
                    var info = _infos[iinfo];
                    var srcName = info.InputColumnName;
                    string srcToken = ctx.TokenOrNullForName(srcName);
                    if (srcToken == null)
                    {
                        toHide.Add(info.Name);
                        continue;
                    }
                    var result = SaveAsPfaCore(ctx, iinfo, info, srcToken);
                    if (result == null)
                    {
                        toHide.Add(info.Name);
                        continue;
                    }
                    toDeclare.Add(new KeyValuePair<string, JToken>(info.Name, result));
                }
                ctx.Hide(toHide.ToArray());
                ctx.DeclareVar(toDeclare.ToArray());
            }

            private JToken SaveAsPfaCore(BoundPfaContext ctx, int iinfo, ColInfo info, JToken srcToken)
            {
                Contracts.AssertValue(ctx);
                Contracts.Assert(0 <= iinfo && iinfo < _infos.Length);
                Contracts.Assert(_infos[iinfo] == info);
                Contracts.AssertValue(srcToken);
                Contracts.Assert(CanSavePfa);

                ColumnType srcType = info.TypeSrc;
                ColumnType srcItemType = srcType.GetItemType();
                int keyCount = srcItemType.GetKeyCountAsInt32(Host);
                Host.Assert(keyCount > 0);
                // If the input type is scalar, we can just use the fanout function.
                if (!(srcType is VectorType srcVectorType))
                    return PfaUtils.Call("cast.fanoutDouble", srcToken, 0, keyCount, false);

                JToken arrType = PfaUtils.Type.Array(PfaUtils.Type.Double);
                if (!(_parent._columns[iinfo].Bag || srcVectorType.Size == 1))
                {
                    // The concatenation case. We can still use fanout, but we just append them all together.
                    return PfaUtils.Call("a.flatMap", srcToken,
                        PfaContext.CreateFuncBlock(new JArray() { PfaUtils.Param("k", PfaUtils.Type.Int) },
                        arrType, PfaUtils.Call("cast.fanoutDouble", "k", 0, keyCount, false)));
                }

                // The bag case, while the most useful, is the most elaborate and difficult: we create
                // an all-zero array and then add items to it.
                const string funcName = "keyToVecUpdate";
                if (!ctx.Pfa.ContainsFunc(funcName))
                {
                    var toFunc = PfaContext.CreateFuncBlock(
                        new JArray() { PfaUtils.Param("v", PfaUtils.Type.Double) }, PfaUtils.Type.Double,
                        PfaUtils.Call("+", "v", 1));

                    ctx.Pfa.AddFunc(funcName,
                        new JArray(PfaUtils.Param("a", arrType), PfaUtils.Param("i", PfaUtils.Type.Int)),
                        arrType, PfaUtils.If(PfaUtils.Call(">=", "i", 0),
                        PfaUtils.Index("a", "i").AddReturn("to", toFunc), "a"));
                }

                return PfaUtils.Call("a.fold", srcToken,
                    PfaUtils.Call("cast.fanoutDouble", -1, 0, keyCount, false), PfaUtils.FuncRef("u." + funcName));
            }

            private bool SaveAsOnnxCore(OnnxContext ctx, int iinfo, ColInfo info, string srcVariableName, string dstVariableName)
            {
                var shape = ctx.RetrieveShapeOrNull(srcVariableName);
                // Make sure that shape must present for calculating the reduction axes. The shape here is generally not null
                // because inputs and outputs of a transform are declared with shapes.
                Contracts.CheckValue(shape, nameof(shape));

                // If Bag is true, the output of ONNX LabelEncoder needs to be fed into ONNX ReduceSum because
                // default ONNX LabelEncoder just matches the behavior of Bag=false.
                var encodedVariableName = _parent._columns[iinfo].Bag ? ctx.AddIntermediateVariable(null, "encoded", true) : dstVariableName;

                string opType = "OneHotEncoder";
                var node = ctx.CreateNode(opType, srcVariableName, encodedVariableName, ctx.GetNodeName(opType));
                node.AddAttribute("cats_int64s", Enumerable.Range(0, info.TypeSrc.GetItemType().GetKeyCountAsInt32(Host)).Select(x => (long)x));
                node.AddAttribute("zeros", true);
                if (_parent._columns[iinfo].Bag)
                {
                    // If input shape is [1, 3], then OneHotEncoder may produce a 3-D tensor. Thus, we need to do a
                    // reduction along the second last axis to merge the one-hot vectors produced by all input features.
                    // Note that one input feature got expended to an one-hot vector.
                    opType = "ReduceSum";
                    var reduceNode = ctx.CreateNode(opType, encodedVariableName, dstVariableName, ctx.GetNodeName(opType), "");
                    reduceNode.AddAttribute("axes", new long[] { shape.Count - 1 });
                    reduceNode.AddAttribute("keepdims", 0);
                }
                return true;
            }
        }
    }

    /// <summary>
    /// Estimator for <see cref="KeyToVectorMappingTransformer"/>. Converts the key types back to their original vectors.
    /// </summary>
    public sealed class KeyToVectorMappingEstimator : TrivialEstimator<KeyToVectorMappingTransformer>
    {
        internal static class Defaults
        {
            public const bool Bag = false;
        }

        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        public sealed class ColumnInfo
        {
            /// <summary> Name of the column resulting from the transformation of <cref see="InputColumnName"/>.</summary>
            public readonly string Name;
            /// <summary> Name of column to transform.</summary>
            public readonly string InputColumnName;
            /// <summary>
            /// Whether to combine multiple indicator vectors into a single bag vector instead of concatenating them.
            /// This is only relevant when the input column is a vector.
            /// </summary>
            public readonly bool Bag;

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="name"/> will be used as source.</param>
            /// <param name="bag">Whether to combine multiple indicator vectors into a single bag vector instead of concatenating them. This is only relevant when the input column is a vector.</param>
            public ColumnInfo(string name, string inputColumnName = null, bool bag = Defaults.Bag)
            {
                Contracts.CheckNonWhiteSpace(name, nameof(name));
                Name = name;
                InputColumnName = inputColumnName ?? name;
                Bag = bag;
            }
        }

        internal KeyToVectorMappingEstimator(IHostEnvironment env, params ColumnInfo[] columns)
            : this(env, new KeyToVectorMappingTransformer(env, columns))
        {
        }

        internal KeyToVectorMappingEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null, bool bag = Defaults.Bag)
            : this(env, new KeyToVectorMappingTransformer(env, new ColumnInfo(outputColumnName, inputColumnName ?? outputColumnName, bag)))
        {
        }

        private KeyToVectorMappingEstimator(IHostEnvironment env, KeyToVectorMappingTransformer transformer)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(KeyToVectorMappingEstimator)), transformer)
        {
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colInfo in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.InputColumnName, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName);
                if (!col.ItemType.IsStandardScalar())
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName);

                var metadata = new List<SchemaShape.Column>();
                if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.KeyValues, out var keyMeta))
                    if (col.Kind != SchemaShape.Column.VectorKind.VariableVector && keyMeta.ItemType is TextType)
                        metadata.Add(new SchemaShape.Column(MetadataUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, keyMeta.ItemType, false));
                if (!colInfo.Bag && (col.Kind == SchemaShape.Column.VectorKind.Scalar || col.Kind == SchemaShape.Column.VectorKind.Vector))
                    metadata.Add(new SchemaShape.Column(MetadataUtils.Kinds.CategoricalSlotRanges, SchemaShape.Column.VectorKind.Vector, NumberType.I4, false));
                if (!colInfo.Bag || (col.Kind == SchemaShape.Column.VectorKind.Scalar))
                    metadata.Add(new SchemaShape.Column(MetadataUtils.Kinds.IsNormalized, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false));

                result[colInfo.Name] = new SchemaShape.Column(colInfo.Name, SchemaShape.Column.VectorKind.Vector, NumberType.R4, false, new SchemaShape(metadata));
            }

            return new SchemaShape(result.Values);
        }
    }
}
