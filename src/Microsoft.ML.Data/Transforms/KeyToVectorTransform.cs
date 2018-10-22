// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.Model.Pfa;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
using Microsoft.ML.Transforms.CategoricalTransforms;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

[assembly: LoadableClass(KeyToVectorTransform.Summary, typeof(IDataTransform), typeof(KeyToVectorTransform), typeof(KeyToVectorTransform.Arguments), typeof(SignatureDataTransform),
    "Key To Vector Transform", KeyToVectorTransform.UserName, "KeyToVector", "ToVector", DocName = "transform/KeyToVectorTransform.md")]

[assembly: LoadableClass(KeyToVectorTransform.Summary, typeof(IDataTransform), typeof(KeyToVectorTransform), null, typeof(SignatureLoadDataTransform),
    "Key To Vector Transform", KeyToVectorTransform.LoaderSignature)]

[assembly: LoadableClass(KeyToVectorTransform.Summary, typeof(KeyToVectorTransform), null, typeof(SignatureLoadModel),
    KeyToVectorTransform.UserName, KeyToVectorTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(KeyToVectorTransform), null, typeof(SignatureLoadRowMapper),
   KeyToVectorTransform.UserName, KeyToVectorTransform.LoaderSignature)]

namespace Microsoft.ML.Transforms.CategoricalTransforms
{
    public sealed class KeyToVectorTransform : OneToOneTransformerBase
    {
        public abstract class ColumnBase : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Whether to combine multiple indicator vectors into a single bag vector instead of concatenating them. This is only relevant when the input is a vector.")]
            public bool? Bag;

            protected override bool TryUnparseCore(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (Bag != null)
                    return false;
                return base.TryUnparseCore(sb);
            }

            protected override bool TryUnparseCore(StringBuilder sb, string extra)
            {
                Contracts.AssertValue(sb);
                Contracts.AssertNonEmpty(extra);
                if (Bag != null)
                    return false;
                return base.TryUnparseCore(sb, extra);
            }
        }

        public sealed class Column : ColumnBase
        {
            public static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

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
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Whether to combine multiple indicator vectors into a single bag vector instead of concatenating them. This is only relevant when the input is a vector.")]
            public bool Bag = KeyToVectorEstimator.Defaults.Bag;
        }

        public class ColumnInfo
        {
            public readonly string Input;
            public readonly string Output;
            public readonly bool Bag;

            public ColumnInfo(string input, string output, bool bag = KeyToVectorEstimator.Defaults.Bag)
            {
                Input = input;
                Output = output;
                Bag = bag;
            }
        }

        private const string RegistrationName = "KeyToVector";

        public IReadOnlyCollection<ColumnInfo> Columns => _columns.AsReadOnly();
        private readonly ColumnInfo[] _columns;

        private static (string input, string output)[] GetColumnPairs(ColumnInfo[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Input, x.Output)).ToArray();
        }

        private string TestIsKey(ColumnType type)
        {
            if (type.ItemType.KeyCount > 0)
                return null;
            return "key type of known cardinality";
        }

        protected override void CheckInputColumn(ISchema inputSchema, int col, int srcCol)
        {
            var type = inputSchema.GetColumnType(srcCol);
            string reason = TestIsKey(type);
            if (reason != null)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].input, reason, type.ToString());
        }

        public KeyToVectorTransform(IHostEnvironment env, params ColumnInfo[] columns) :
            base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            _columns = columns.ToArray();
        }

        public const string LoaderSignature = "KeyToVectorTransform";
        public const string UserName = "KeyToVectorTransform";
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
                loaderAssemblyName: typeof(KeyToVectorTransform).Assembly.FullName);
        }

        public override void Save(ModelSaveContext ctx)
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
        private static KeyToVectorTransform Create(IHostEnvironment env, ModelLoadContext ctx)
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
            return new KeyToVectorTransform(host, ctx);
        }

        private KeyToVectorTransform(IHost host, ModelLoadContext ctx)
          : base(host, ctx)
        {
            var columnsLength = ColumnPairs.Length;
            // *** Binary format ***
            // <base>
            // for each added column
            //   byte: bag as 0/1
            var bags = new bool[columnsLength];
            bags = ctx.Reader.ReadBoolArray(columnsLength);

            _columns = new ColumnInfo[columnsLength];
            for (int i = 0; i < columnsLength; i++)
                _columns[i] = new ColumnInfo(ColumnPairs[i].input, ColumnPairs[i].output, bags[i]);
        }

        public static IDataTransform Create(IHostEnvironment env, IDataView input, params ColumnInfo[] columns) =>
             new KeyToVectorTransform(env, columns).MakeDataTransform(input);

        // Factory method for SignatureDataTransform.
        private static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            env.CheckValue(args.Column, nameof(args.Column));
            var cols = new ColumnInfo[args.Column.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                var item = args.Column[i];

                cols[i] = new ColumnInfo(item.Source ?? item.Name,
                    item.Name,
                    item.Bag ?? args.Bag);
            };
            return new KeyToVectorTransform(env, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        protected override IRowMapper MakeRowMapper(ISchema schema) => new Mapper(this, Schema.Create(schema));

        private sealed class Mapper : MapperBase, ISaveAsOnnx, ISaveAsPfa
        {
            private sealed class ColInfo
            {
                public readonly string Name;
                public readonly string Source;
                public readonly ColumnType TypeSrc;

                public ColInfo(string name, string source, ColumnType type)
                {
                    Name = name;
                    Source = source;
                    TypeSrc = type;
                }
            }

            private readonly KeyToVectorTransform _parent;
            private readonly ColInfo[] _infos;
            private readonly VectorType[] _types;

            public Mapper(KeyToVectorTransform parent, Schema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _infos = CreateInfos(inputSchema);
                _types = new VectorType[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    if (_parent._columns[i].Bag || _infos[i].TypeSrc.ValueCount == 1)
                        _types[i] = new VectorType(NumberType.Float, _infos[i].TypeSrc.ItemType.KeyCount);
                    else
                        _types[i] = new VectorType(NumberType.Float, _infos[i].TypeSrc.ValueCount, _infos[i].TypeSrc.ItemType.KeyCount);
                }
            }

            private ColInfo[] CreateInfos(ISchema inputSchema)
            {
                Host.AssertValue(inputSchema);
                var infos = new ColInfo[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    if (!inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out int colSrc))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.ColumnPairs[i].input);
                    var type = inputSchema.GetColumnType(colSrc);
                    _parent.CheckInputColumn(inputSchema, i, colSrc);
                    infos[i] = new ColInfo(_parent.ColumnPairs[i].output, _parent.ColumnPairs[i].input, type);
                }
                return infos;
            }

            public override Schema.Column[] GetOutputColumns()
            {
                var result = new Schema.Column[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out int colIndex);
                    Host.Assert(colIndex >= 0);
                    var builder = new Schema.Metadata.Builder();
                    AddMetadata(i, builder);
                    result[i] = new Schema.Column(_parent.ColumnPairs[i].output, _types[i], builder.GetMetadata());
                }
                return result;
            }

            private void AddMetadata(int iinfo, Schema.Metadata.Builder builder)
            {
                InputSchema.TryGetColumnIndex(_infos[iinfo].Source, out int srcCol);
                var inputMetadata = InputSchema[srcCol].Metadata;

                var srcType = _infos[iinfo].TypeSrc;

                ColumnType typeNames = null;
                int metaKeyValuesCol = 0;
                if (inputMetadata.Schema.TryGetColumnIndex(MetadataUtils.Kinds.KeyValues, out metaKeyValuesCol))
                    typeNames = inputMetadata.Schema[metaKeyValuesCol].Type;
                if (typeNames == null || !typeNames.IsKnownSizeVector || !typeNames.ItemType.IsText ||
                    typeNames.VectorSize != _infos[iinfo].TypeSrc.ItemType.KeyCount)
                {
                    typeNames = null;
                }

                if (_parent._columns[iinfo].Bag || _infos[iinfo].TypeSrc.ValueCount == 1)
                {
                    if (typeNames != null)
                    {
                        var getter = inputMetadata.GetGetter<VBuffer<ReadOnlyMemory<char>>>(metaKeyValuesCol);
                        var slotNamesType = new VectorType(TextType.Instance, _types[iinfo]);
                        builder.AddSlotNames(slotNamesType.VectorSize, getter);
                    }
                }
                else
                {
                    if (typeNames != null && _types[iinfo].IsKnownSizeVector)
                    {
                        ValueGetter<VBuffer<ReadOnlyMemory<char>>> getter = (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                        {
                            GetSlotNames(iinfo, ref dst);
                        };
                        var slotNamesType = new VectorType(TextType.Instance, _types[iinfo]);
                        builder.Add(new Schema.Column(MetadataUtils.Kinds.SlotNames, slotNamesType, null), getter);
                    }
                }

                if (!_parent._columns[iinfo].Bag && srcType.ValueCount > 0)
                {
                    ValueGetter<VBuffer<int>> getter = (ref VBuffer<int> dst) =>
                    {
                        GetCategoricalSlotRanges(iinfo, ref dst);
                    };
                    builder.Add(new Schema.Column(MetadataUtils.Kinds.CategoricalSlotRanges, MetadataUtils.GetCategoricalType(_infos[iinfo].TypeSrc.ValueCount), null), getter);
                }

                if (!_parent._columns[iinfo].Bag || srcType.ValueCount == 1)
                {
                    ValueGetter<bool> getter = (ref bool dst) =>
                    {
                        dst = true;
                    };
                    builder.Add(new Schema.Column(MetadataUtils.Kinds.IsNormalized, BoolType.Instance, null), getter);
                }
            }

            // Combines source key names and slot names to produce final slot names.
            private void GetSlotNames(int iinfo, ref VBuffer<ReadOnlyMemory<char>> dst)
            {
                Host.Assert(0 <= iinfo && iinfo < _infos.Length);
                Host.Assert(_types[iinfo].IsKnownSizeVector);

                // Size one should have been treated the same as Bag (by the caller).
                // Variable size should have thrown (by the caller).
                var typeSrc = _infos[iinfo].TypeSrc;
                Host.Assert(typeSrc.VectorSize > 1);

                // Get the source slot names, defaulting to empty text.
                var namesSlotSrc = default(VBuffer<ReadOnlyMemory<char>>);

                var inputMetadata = InputSchema[_infos[iinfo].Source].Metadata;
                Contracts.AssertValue(inputMetadata);
                var typeSlotSrc = inputMetadata.Schema.GetColumnOrNull(MetadataUtils.Kinds.SlotNames)?.Type;
                if (typeSlotSrc != null && typeSlotSrc.VectorSize == typeSrc.VectorSize && typeSlotSrc.ItemType.IsText)
                {
                    inputMetadata.GetValue(MetadataUtils.Kinds.SlotNames, ref namesSlotSrc);
                    Host.Check(namesSlotSrc.Length == typeSrc.VectorSize);
                }
                else
                    namesSlotSrc = VBufferUtils.CreateEmpty<ReadOnlyMemory<char>>(typeSrc.VectorSize);

                int keyCount = typeSrc.ItemType.ItemType.KeyCount;
                int slotLim = _types[iinfo].VectorSize;
                Host.Assert(slotLim == (long)typeSrc.VectorSize * keyCount);

                // Get the source key names, in an array (since we will use them multiple times).
                var namesKeySrc = default(VBuffer<ReadOnlyMemory<char>>);
                inputMetadata.GetValue(MetadataUtils.Kinds.KeyValues, ref namesKeySrc);
                Host.Check(namesKeySrc.Length == keyCount);
                var keys = new ReadOnlyMemory<char>[keyCount];
                namesKeySrc.CopyTo(keys);

                var values = dst.Values;
                if (Utils.Size(values) < slotLim)
                    values = new ReadOnlyMemory<char>[slotLim];

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
                        values[slot++] = sb.ToString().AsMemory();
                    }
                }
                Host.Assert(slot == slotLim);

                dst = new VBuffer<ReadOnlyMemory<char>>(slotLim, values, dst.Indices);
            }

            private void GetCategoricalSlotRanges(int iinfo, ref VBuffer<int> dst)
            {
                Host.Assert(0 <= iinfo && iinfo < _infos.Length);

                var info = _infos[iinfo];

                Host.Assert(info.TypeSrc.ValueCount > 0);

                int[] ranges = new int[info.TypeSrc.ValueCount * 2];
                int size = info.TypeSrc.ItemType.KeyCount;

                ranges[0] = 0;
                ranges[1] = size - 1;
                for (int i = 2; i < ranges.Length; i += 2)
                {
                    ranges[i] = ranges[i - 1] + 1;
                    ranges[i + 1] = ranges[i] + size - 1;
                }

                dst = new VBuffer<int>(ranges.Length, ranges);
            }

            protected override Delegate MakeGetter(IRow input, int iinfo, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _infos.Length);
                disposer = null;

                var info = _infos[iinfo];
                if (!info.TypeSrc.IsVector)
                    return MakeGetterOne(input, iinfo);
                if (_parent._columns[iinfo].Bag)
                    return MakeGetterBag(input, iinfo);
                return MakeGetterInd(input, iinfo);
            }

            /// <summary>
            /// This is for the singleton case. This should be equivalent to both Bag and Ord over
            /// a vector of size one.
            /// </summary>
            private ValueGetter<VBuffer<float>> MakeGetterOne(IRow input, int iinfo)
            {
                Host.AssertValue(input);
                Host.Assert(_infos[iinfo].TypeSrc.IsKey);
                Host.Assert(_infos[iinfo].TypeSrc.KeyCount == _types[iinfo].VectorSize);

                int size = _infos[iinfo].TypeSrc.KeyCount;
                Host.Assert(size > 0);
                input.Schema.TryGetColumnIndex(_infos[iinfo].Source, out int srcCol);
                Host.Assert(srcCol >= 0);
                var getSrc = RowCursorUtils.GetGetterAs<uint>(NumberType.U4, input, srcCol);
                var src = default(uint);
                return
                    (ref VBuffer<float> dst) =>
                    {
                        getSrc(ref src);
                        if (src == 0 || src > size)
                        {
                            dst = new VBuffer<float>(size, 0, dst.Values, dst.Indices);
                            return;
                        }

                        var values = dst.Values;
                        var indices = dst.Indices;
                        if (Utils.Size(values) < 1)
                            values = new float[1];
                        if (Utils.Size(indices) < 1)
                            indices = new int[1];
                        values[0] = 1;
                        indices[0] = (int)src - 1;

                        dst = new VBuffer<float>(size, 1, values, indices);
                    };
            }

            /// <summary>
            /// This is for the bagging case - vector input and outputs should be added.
            /// </summary>
            private ValueGetter<VBuffer<float>> MakeGetterBag(IRow input, int iinfo)
            {
                Host.AssertValue(input);
                Host.Assert(_infos[iinfo].TypeSrc.IsVector);
                Host.Assert(_infos[iinfo].TypeSrc.ItemType.IsKey);
                Host.Assert(_parent._columns[iinfo].Bag);
                Host.Assert(_infos[iinfo].TypeSrc.ItemType.KeyCount == _types[iinfo].VectorSize);

                var info = _infos[iinfo];
                int size = info.TypeSrc.ItemType.KeyCount;
                Host.Assert(size > 0);

                int cv = info.TypeSrc.VectorSize;
                Host.Assert(cv >= 0);
                input.Schema.TryGetColumnIndex(info.Source, out int srcCol);
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
                        var values = src.Values;
                        int count = src.Count;
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
            private ValueGetter<VBuffer<float>> MakeGetterInd(IRow input, int iinfo)
            {
                Host.AssertValue(input);
                Host.Assert(_infos[iinfo].TypeSrc.IsVector);
                Host.Assert(_infos[iinfo].TypeSrc.ItemType.IsKey);
                Host.Assert(!_parent._columns[iinfo].Bag);

                var info = _infos[iinfo];
                int size = info.TypeSrc.ItemType.KeyCount;
                Host.Assert(size > 0);

                int cv = info.TypeSrc.VectorSize;
                Host.Assert(cv >= 0);
                Host.Assert(_types[iinfo].VectorSize == size * cv);
                input.Schema.TryGetColumnIndex(info.Source, out int srcCol);
                var getSrc = RowCursorUtils.GetVecGetterAs<uint>(NumberType.U4, input, srcCol);
                var src = default(VBuffer<uint>);
                return
                    (ref VBuffer<float> dst) =>
                    {
                        getSrc(ref src);
                        int lenSrc = src.Length;
                        Host.Check(lenSrc == cv || cv == 0);

                        // Since we generate values in order, no need for a builder.
                        var valuesDst = dst.Values;
                        var indicesDst = dst.Indices;

                        int lenDst = checked(size * lenSrc);
                        int cntSrc = src.Count;
                        if (Utils.Size(valuesDst) < cntSrc)
                            valuesDst = new float[cntSrc];
                        if (Utils.Size(indicesDst) < cntSrc)
                            indicesDst = new int[cntSrc];

                        var values = src.Values;
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
                                valuesDst[count] = 1;
                                indicesDst[count++] = slot * size + (int)key;
                            }
                        }
                        else
                        {
                            var indices = src.Indices;
                            for (int islot = 0; islot < cntSrc; islot++)
                            {
                                Host.Assert(count < cntSrc);
                                uint key = values[islot] - 1;
                                if (key >= (uint)size)
                                    continue;
                                valuesDst[count] = 1;
                                indicesDst[count++] = indices[islot] * size + (int)key;
                            }
                        }
                        dst = new VBuffer<float>(lenDst, count, valuesDst, indicesDst);
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
                    string sourceColumnName = info.Source;
                    if (!ctx.ContainsColumn(sourceColumnName))
                    {
                        ctx.RemoveColumn(info.Name, false);
                        continue;
                    }

                    if (!SaveAsOnnxCore(ctx, iinfo, info, ctx.GetVariableName(sourceColumnName),
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
                    var srcName = info.Source;
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

                int keyCount = info.TypeSrc.ItemType.KeyCount;
                Host.Assert(keyCount > 0);
                // If the input type is scalar, we can just use the fanout function.
                if (!info.TypeSrc.IsVector)
                    return PfaUtils.Call("cast.fanoutDouble", srcToken, 0, keyCount, false);

                JToken arrType = PfaUtils.Type.Array(PfaUtils.Type.Double);
                if (_parent._columns[iinfo].Bag || info.TypeSrc.ValueCount == 1)
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
                node.AddAttribute("cats_int64s", Enumerable.Range(0, info.TypeSrc.ItemType.KeyCount).Select(x => (long)x));
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

    public sealed class KeyToVectorEstimator : TrivialEstimator<KeyToVectorTransform>
    {
        internal static class Defaults
        {
            public const bool Bag = false;
        }

        public KeyToVectorEstimator(IHostEnvironment env, params KeyToVectorTransform.ColumnInfo[] columns)
            : this(env, new KeyToVectorTransform(env, columns))
        {
        }

        public KeyToVectorEstimator(IHostEnvironment env, string name, string source = null, bool bag = Defaults.Bag)
            : this(env, new KeyToVectorTransform(env, new KeyToVectorTransform.ColumnInfo(source ?? name, name, bag)))
        {
        }

        private KeyToVectorEstimator(IHostEnvironment env, KeyToVectorTransform transformer)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(KeyToVectorEstimator)), transformer)
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var colInfo in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.Input, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);
                if ((col.ItemType.ItemType.RawKind == default) || !(col.ItemType.IsVector || col.ItemType.IsPrimitive))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);

                var metadata = new List<SchemaShape.Column>();
                if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.KeyValues, out var keyMeta))
                    if (col.Kind != SchemaShape.Column.VectorKind.VariableVector && keyMeta.ItemType.IsText)
                        metadata.Add(new SchemaShape.Column(MetadataUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, keyMeta.ItemType, false));
                if (!colInfo.Bag && (col.Kind == SchemaShape.Column.VectorKind.Scalar || col.Kind == SchemaShape.Column.VectorKind.Vector))
                    metadata.Add(new SchemaShape.Column(MetadataUtils.Kinds.CategoricalSlotRanges, SchemaShape.Column.VectorKind.Vector, NumberType.I4, false));
                if (!colInfo.Bag || (col.Kind == SchemaShape.Column.VectorKind.Scalar))
                    metadata.Add(new SchemaShape.Column(MetadataUtils.Kinds.IsNormalized, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false));

                result[colInfo.Output] = new SchemaShape.Column(colInfo.Output, SchemaShape.Column.VectorKind.Vector, NumberType.R4, false, new SchemaShape(metadata));
            }

            return new SchemaShape(result.Values);
        }
    }

    /// <summary>
    /// Extension methods for the static-pipeline over <see cref="PipelineColumn"/> objects.
    /// </summary>
    public static class KeyToVectorExtensions
    {
        private interface IColInput
        {
            PipelineColumn Input { get; }
            bool Bag { get; }
        }

        private sealed class OutVectorColumn<TKey, TValue> : Vector<float>, IColInput
        {
            public PipelineColumn Input { get; }
            public bool Bag { get; }

            public OutVectorColumn(Key<TKey, TValue> input)
             : base(Reconciler.Inst, input)
            {
                Input = input;
                Bag = false;
            }

            public OutVectorColumn(Vector<Key<TKey, TValue>> input, bool bag)
                : base(Reconciler.Inst, input)
            {
                Input = input;
                Bag = bag;
            }

            public OutVectorColumn(VarVector<Key<TKey, TValue>> input)
             : base(Reconciler.Inst, input)
            {
                Input = input;
                Bag = true;
            }
        }

        private sealed class OutVarVectorColumn<TKey, TValue> : VarVector<float>, IColInput
        {
            public PipelineColumn Input { get; }
            public bool Bag { get; }

            public OutVarVectorColumn(VarVector<Key<TKey, TValue>> input)
            : base(Reconciler.Inst, input)
            {
                Input = input;
                Bag = false;
            }
        }

        private sealed class OutVectorColumn<TKey> : Vector<float>, IColInput
        {
            public PipelineColumn Input { get; }
            public bool Bag { get; }

            public OutVectorColumn(Key<TKey> input)
             : base(Reconciler.Inst, input)
            {
                Input = input;
                Bag = false;
            }

            public OutVectorColumn(Vector<Key<TKey>> input, bool bag)
                : base(Reconciler.Inst, input)
            {
                Input = input;
                Bag = bag;
            }

            public OutVectorColumn(VarVector<Key<TKey>> input)
             : base(Reconciler.Inst, input)
            {
                Input = input;
                Bag = true;
            }
        }

        private sealed class OutVarVectorColumn<TKey> : VarVector<float>, IColInput
        {
            public PipelineColumn Input { get; }
            public bool Bag { get; }

            public OutVarVectorColumn(VarVector<Key<TKey>> input)
            : base(Reconciler.Inst, input)
            {
                Input = input;
                Bag = false;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            public static Reconciler Inst = new Reconciler();

            private Reconciler() { }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                var infos = new KeyToVectorTransform.ColumnInfo[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var col = (IColInput)toOutput[i];
                    infos[i] = new KeyToVectorTransform.ColumnInfo(inputNames[col.Input], outputNames[toOutput[i]], col.Bag);
                }
                return new KeyToVectorEstimator(env, infos);
            }
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces an indicator vector of floats.
        /// Each key value of the input is used to create an indicator vector: the indicator vector is the length of the key cardinality,
        /// where all values are 0, except for the entry corresponding to the value of the key, which is 1.
        /// If the key value is missing, then all values are 0. Naturally this tends to generate very sparse vectors.
        /// </summary>
        public static Vector<float> ToVector<TKey, TValue>(this Key<TKey, TValue> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey, TValue>(input);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces an indicator vector of floats.
        /// Each key value of the input is used to create an indicator vector: the indicator vector is the length of the key cardinality,
        /// where all values are 0, except for the entry corresponding to the value of the key, which is 1.
        /// If the key value is missing, then all values are 0. Naturally this tends to generate very sparse vectors.
        /// </summary>
        public static Vector<float> ToVector<TKey, TValue>(this Vector<Key<TKey, TValue>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey, TValue>(input, false);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces an indicator vector of floats.
        /// Each key value of the input is used to create an indicator vector: the indicator vector is the length of the key cardinality,
        /// where all values are 0, except for the entry corresponding to the value of the key, which is 1.
        /// If the key value is missing, then all values are 0. Naturally this tends to generate very sparse vectors.
        /// In this case then the indicator vectors for all values in the column will be simply added together,
        /// to produce the final vector with type equal to the key cardinality; so, in all cases, whether vector or scalar,
        /// the output column will be a vector type of length equal to that cardinality.
        /// </summary>
        public static VarVector<float> ToVector<TKey, TValue>(this VarVector<Key<TKey, TValue>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVarVectorColumn<TKey, TValue>(input);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces an indicator vector of floats.
        /// Each key value of the input is used to create an indicator vector: the indicator vector is the length of the key cardinality,
        /// where all values are 0, except for the entry corresponding to the value of the key, which is 1.
        /// If the key value is missing, then all values are 0. Naturally this tends to generate very sparse vectors.
        /// In this case then the indicator vectors for all values in the column will be simply added together,
        /// to produce the final vector with type equal to the key cardinality; so, in all cases, whether vector or scalar,
        /// the output column will be a vector type of length equal to that cardinality.
        /// </summary>
        public static Vector<float> ToBaggedVector<TKey, TValue>(this Vector<Key<TKey, TValue>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey, TValue>(input, true);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces an indicator vector of floats.
        /// Each key value of the input is used to create an indicator vector: the indicator vector is the length of the key cardinality,
        /// where all values are 0, except for the entry corresponding to the value of the key, which is 1.
        /// If the key value is missing, then all values are 0. Naturally this tends to generate very sparse vectors.
        /// In this case then the indicator vectors for all values in the column will be simply added together,
        /// to produce the final vector with type equal to the key cardinality; so, in all cases, whether vector or scalar,
        /// the output column will be a vector type of length equal to that cardinality.
        /// </summary>
        public static Vector<float> ToBaggedVector<TKey, TValue>(this VarVector<Key<TKey, TValue>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey, TValue>(input);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces an indicator vector of floats.
        /// Each key value of the input is used to create an indicator vector: the indicator vector is the length of the key cardinality,
        /// where all values are 0, except for the entry corresponding to the value of the key, which is 1.
        /// If the key value is missing, then all values are 0. Naturally this tends to generate very sparse vectors.
        /// </summary>
        public static Vector<float> ToVector<TKey>(this Key<TKey> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey>(input);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces an indicator vector of floats.
        /// Each key value of the input is used to create an indicator vector: the indicator vector is the length of the key cardinality,
        /// where all values are 0, except for the entry corresponding to the value of the key, which is 1.
        /// If the key value is missing, then all values are 0. Naturally this tends to generate very sparse vectors.
        /// </summary>
        public static Vector<float> ToVector<TKey>(this Vector<Key<TKey>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey>(input, false);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces an indicator vector of floats.
        /// Each key value of the input is used to create an indicator vector: the indicator vector is the length of the key cardinality,
        /// where all values are 0, except for the entry corresponding to the value of the key, which is 1.
        /// If the key value is missing, then all values are 0. Naturally this tends to generate very sparse vectors.
        /// In this case then the indicator vectors for all values in the column will be simply added together,
        /// to produce the final vector with type equal to the key cardinality; so, in all cases, whether vector or scalar,
        /// the output column will be a vector type of length equal to that cardinality.
        /// </summary>
        public static VarVector<float> ToVector<TKey>(this VarVector<Key<TKey>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVarVectorColumn<TKey>(input);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces an indicator vector of floats.
        /// Each key value of the input is used to create an indicator vector: the indicator vector is the length of the key cardinality,
        /// where all values are 0, except for the entry corresponding to the value of the key, which is 1.
        /// If the key value is missing, then all values are 0. Naturally this tends to generate very sparse vectors.
        /// In this case then the indicator vectors for all values in the column will be simply added together,
        /// to produce the final vector with type equal to the key cardinality; so, in all cases, whether vector or scalar,
        /// the output column will be a vector type of length equal to that cardinality.
        /// </summary>
        public static Vector<float> ToBaggedVector<TKey>(this Vector<Key<TKey>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey>(input, true);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces an indicator vector of floats.
        /// Each key value of the input is used to create an indicator vector: the indicator vector is the length of the key cardinality,
        /// where all values are 0, except for the entry corresponding to the value of the key, which is 1.
        /// If the key value is missing, then all values are 0. Naturally this tends to generate very sparse vectors.
        /// In this case then the indicator vectors for all values in the column will be simply added together,
        /// to produce the final vector with type equal to the key cardinality; so, in all cases, whether vector or scalar,
        /// the output column will be a vector type of length equal to that cardinality.
        /// </summary>
        public static Vector<float> ToBaggedVector<TKey>(this VarVector<Key<TKey>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey>(input);
        }
    }
}
