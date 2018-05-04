// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.Model.Pfa;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(KeyToVectorTransform.Summary, typeof(KeyToVectorTransform), typeof(KeyToVectorTransform.Arguments), typeof(SignatureDataTransform),
    "Key To Vector Transform", "KeyToVectorTransform", "KeyToVector", "ToVector", DocName = "transform/KeyToVectorTransform.md")]

[assembly: LoadableClass(KeyToVectorTransform.Summary, typeof(KeyToVectorTransform), null, typeof(SignatureLoadDataTransform),
    "Key To Vector Transform", KeyToVectorTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    public sealed class KeyToVectorTransform : OneToOneTransformBase
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
            public bool Bag;
        }

        internal const string Summary = "Converts a key column to an indicator vector.";

        public const string LoaderSignature = "KeyToVectorTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "KEY2VECT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private const string RegistrationName = "KeyToVector";

        // These arrays are parallel to Infos.
        // * _bag indicates whether vector inputs should have their output indicator vectors added
        //   (instead of concatenated). This is faithful to what the user specified in the Arguments
        //   and is persisted.
        // * _concat is whether, given the current input, there are multiple output instance vectors
        //   to concatenate. If _bag[i] is true, then _concat[i] will be false. If _bag[i] is false,
        //   _concat[i] will be true iff the input is a vector with either unknown length or length
        //   bigger than one. In the other cases (non-vector input and vector of length one), there
        //   is only one resulting indicator vector so no need to concatenate anything.
        // * _types contains the output column types.
        // * _slotNamesTypes contains the metadata types for slot name metadata. _slotNamesTypes[i] will
        //   be null if slot names are not available for the given column (eg, in the variable size case,
        //   or when the source doesn't have key names).
        private readonly bool[] _bag;
        private readonly bool[] _concat;
        private readonly VectorType[] _types;

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public KeyToVectorTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, Contracts.CheckRef(args, nameof(args)).Column,
                input, TestIsKey)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));

            _bag = new bool[Infos.Length];
            _concat = new bool[Infos.Length];
            _types = new VectorType[Infos.Length];
            for (int i = 0; i < Infos.Length; i++)
            {
                var item = args.Column[i];
                _bag[i] = item.Bag ?? args.Bag;
                ComputeType(this, Source.Schema, i, Infos[i], _bag[i], Metadata,
                    out _types[i], out _concat[i]);
            }
            Metadata.Seal();
        }

        private KeyToVectorTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestIsKey)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            // for each added column
            //   byte: bag as 0/1
            Host.AssertNonEmpty(Infos);
            int size = Infos.Length;
            _bag = new bool[size];
            _concat = new bool[Infos.Length];
            _types = new VectorType[size];
            for (int i = 0; i < size; i++)
            {
                _bag[i] = ctx.Reader.ReadBoolByte();
                ComputeType(this, Source.Schema, i, Infos[i], _bag[i], Metadata,
                    out _types[i], out _concat[i]);
            }
            Metadata.Seal();
        }

        public static KeyToVectorTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            return h.Apply("Loading Model",
                ch =>
                {
                    // *** Binary format ***
                    // int: sizeof(Float)
                    // <remainder handled in ctors>
                    int cbFloat = ctx.Reader.ReadInt32();
                    ch.CheckDecode(cbFloat == sizeof(Float));
                    return new KeyToVectorTransform(h, ctx, input);
                });
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // <base>
            // for each added column
            //   byte: bag as 0/1
            ctx.Writer.Write(sizeof(Float));
            SaveBase(ctx);

            Host.Assert(_bag.Length == Infos.Length);
            for (int i = 0; i < _bag.Length; i++)
                ctx.Writer.WriteBoolByte(_bag[i]);
        }

        public override bool CanSavePfa => true;
        public override bool CanSaveOnnx => true;

        protected override JToken SaveAsPfaCore(BoundPfaContext ctx, int iinfo, ColInfo info, JToken srcToken)
        {
            Contracts.AssertValue(ctx);
            Contracts.Assert(0 <= iinfo && iinfo < Infos.Length);
            Contracts.Assert(Infos[iinfo] == info);
            Contracts.AssertValue(srcToken);
            Contracts.Assert(CanSavePfa);

            int keyCount = info.TypeSrc.ItemType.KeyCount;
            Host.Assert(keyCount > 0);
            // If the input type is scalar, we can just use the fanout function.
            if (!info.TypeSrc.IsVector)
                return PfaUtils.Call("cast.fanoutDouble", srcToken, 0, keyCount, false);

            JToken arrType = PfaUtils.Type.Array(PfaUtils.Type.Double);
            if (_concat[iinfo])
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

        protected override bool SaveAsOnnxCore(OnnxContext ctx, int iinfo, ColInfo info, string srcVariableName, string dstVariableName)
        {
            string opType = "OneHotEncoder";
            var node = OnnxUtils.MakeNode(opType, srcVariableName, dstVariableName, ctx.GetNodeName(opType));
            OnnxUtils.NodeAddAttributes(node, "cats_int64s", Enumerable.Range(1, info.TypeSrc.ItemType.KeyCount).Select(x => (long)x));
            OnnxUtils.NodeAddAttributes(node, "zeros", true);
            ctx.AddNode(node);
            return true;
        }

        // Computes the column type and whether multiple indicator vectors need to be concatenated.
        // Also populates the metadata.
        private static void ComputeType(KeyToVectorTransform trans, ISchema input, int iinfo, ColInfo info, bool bag,
            MetadataDispatcher md, out VectorType type, out bool concat)
        {
            Contracts.AssertValue(trans);
            Contracts.AssertValue(input);
            Contracts.AssertValue(info);
            Contracts.Assert(info.TypeSrc.ItemType.IsKey);
            Contracts.AssertValue(md);

            int size = info.TypeSrc.ItemType.KeyCount;
            Contracts.Assert(size > 0);

            // See if the source has key names.
            var typeNames = input.GetMetadataTypeOrNull(MetadataUtils.Kinds.KeyValues, info.Source);
            if (typeNames == null || !typeNames.IsKnownSizeVector || !typeNames.ItemType.IsText ||
                typeNames.VectorSize != size)
            {
                typeNames = null;
            }

            // Don't pass through any source column metadata.
            using (var bldr = md.BuildMetadata(iinfo))
            {
                if (bag || info.TypeSrc.ValueCount == 1)
                {
                    // Output is a single vector computed as the sum of the output indicator vectors.
                    concat = false;
                    type = new VectorType(NumberType.Float, size);
                    if (typeNames != null)
                        bldr.AddGetter<VBuffer<DvText>>(MetadataUtils.Kinds.SlotNames, typeNames, trans.GetKeyNames);
                }
                else
                {
                    // Output is the concatenation of the multiple output indicator vectors.
                    concat = true;
                    type = new VectorType(NumberType.Float, info.TypeSrc.ValueCount, size);
                    if (typeNames != null && type.VectorSize > 0)
                    {
                        bldr.AddGetter<VBuffer<DvText>>(MetadataUtils.Kinds.SlotNames,
                            new VectorType(TextType.Instance, type), trans.GetSlotNames);
                    }
                }

                if (!bag && info.TypeSrc.ValueCount > 0)
                {
                    bldr.AddGetter<VBuffer<DvInt4>>(MetadataUtils.Kinds.CategoricalSlotRanges,
                        MetadataUtils.GetCategoricalType(info.TypeSrc.ValueCount), trans.GetCategoricalSlotRanges);
                }

                if (!bag || info.TypeSrc.ValueCount == 1)
                    bldr.AddPrimitive(MetadataUtils.Kinds.IsNormalized, BoolType.Instance, DvBool.True);
            }
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo & iinfo < _types.Length);
            return _types[iinfo];
        }

        private void GetCategoricalSlotRanges(int iinfo, ref VBuffer<DvInt4> dst)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);

            var info = Infos[iinfo];

            Host.Assert(info.TypeSrc.ValueCount > 0);

            DvInt4[] ranges = new DvInt4[info.TypeSrc.ValueCount * 2];
            int size = info.TypeSrc.ItemType.KeyCount;

            ranges[0] = 0;
            ranges[1] = size - 1;
            for (int i = 2; i < ranges.Length; i += 2)
            {
                ranges[i] = ranges[i - 1] + 1;
                ranges[i + 1] = ranges[i] + size - 1;
            }

            dst = new VBuffer<DvInt4>(ranges.Length, ranges);
        }

        // Used for slot names when appropriate.
        private void GetKeyNames(int iinfo, ref VBuffer<DvText> dst)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            Host.Assert(!_concat[iinfo]);

            // Slot names are just the key value names.
            Source.Schema.GetMetadata(MetadataUtils.Kinds.KeyValues, Infos[iinfo].Source, ref dst);
        }

        // Combines source key names and slot names to produce final slot names.
        private void GetSlotNames(int iinfo, ref VBuffer<DvText> dst)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            Host.Assert(_concat[iinfo]);
            Host.Assert(_types[iinfo].IsKnownSizeVector);

            // Size one should have been treated the same as Bag (by the caller).
            // Variable size should have thrown (by the caller).
            var typeSrc = Infos[iinfo].TypeSrc;
            Host.Assert(typeSrc.VectorSize > 1);

            // Get the source slot names, defaulting to empty text.
            var namesSlotSrc = default(VBuffer<DvText>);
            var typeSlotSrc = Source.Schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.SlotNames, Infos[iinfo].Source);
            if (typeSlotSrc != null && typeSlotSrc.VectorSize == typeSrc.VectorSize && typeSlotSrc.ItemType.IsText)
            {
                Source.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, Infos[iinfo].Source, ref namesSlotSrc);
                Host.Check(namesSlotSrc.Length == typeSrc.VectorSize);
            }
            else
                namesSlotSrc = VBufferUtils.CreateEmpty<DvText>(typeSrc.VectorSize);

            int keyCount = typeSrc.ItemType.KeyCount;
            int slotLim = _types[iinfo].VectorSize;
            Host.Assert(slotLim == (long)typeSrc.VectorSize * keyCount);

            // Get the source key names, in an array (since we will use them multiple times).
            var namesKeySrc = default(VBuffer<DvText>);
            Source.Schema.GetMetadata(MetadataUtils.Kinds.KeyValues, Infos[iinfo].Source, ref namesKeySrc);
            Host.Check(namesKeySrc.Length == keyCount);
            var keys = new DvText[keyCount];
            namesKeySrc.CopyTo(keys);

            var values = dst.Values;
            if (Utils.Size(values) < slotLim)
                values = new DvText[slotLim];

            var sb = new StringBuilder();
            int slot = 0;
            foreach (var kvpSlot in namesSlotSrc.Items(all: true))
            {
                Contracts.Assert(slot == (long)kvpSlot.Key * keyCount);
                sb.Clear();
                if (kvpSlot.Value.HasChars)
                    kvpSlot.Value.AddToStringBuilder(sb);
                else
                    sb.Append('[').Append(kvpSlot.Key).Append(']');
                sb.Append('.');

                int len = sb.Length;
                foreach (var key in keys)
                {
                    sb.Length = len;
                    key.AddToStringBuilder(sb);
                    values[slot++] = new DvText(sb.ToString());
                }
            }
            Host.Assert(slot == slotLim);

            dst = new VBuffer<DvText>(slotLim, values, dst.Indices);
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            var info = Infos[iinfo];
            if (!info.TypeSrc.IsVector)
                return MakeGetterOne(input, iinfo);
            if (_bag[iinfo])
                return MakeGetterBag(input, iinfo);
            return MakeGetterInd(input, iinfo);
        }

        /// <summary>
        /// This is for the singleton case. This should be equivalent to both Bag and Ord over
        /// a vector of size one.
        /// </summary>
        private ValueGetter<VBuffer<Float>> MakeGetterOne(IRow input, int iinfo)
        {
            Host.AssertValue(input);
            Host.Assert(Infos[iinfo].TypeSrc.IsKey);
            Host.Assert(Infos[iinfo].TypeSrc.KeyCount == _types[iinfo].VectorSize);

            int size = Infos[iinfo].TypeSrc.KeyCount;
            Host.Assert(size > 0);

            var getSrc = RowCursorUtils.GetGetterAs<uint>(NumberType.U4, input, Infos[iinfo].Source);
            var src = default(uint);
            return
                (ref VBuffer<Float> dst) =>
                {
                    getSrc(ref src);
                    if (src == 0 || src > size)
                    {
                        dst = new VBuffer<Float>(size, 0, dst.Values, dst.Indices);
                        return;
                    }

                    var values = dst.Values;
                    var indices = dst.Indices;
                    if (Utils.Size(values) < 1)
                        values = new Float[1];
                    if (Utils.Size(indices) < 1)
                        indices = new int[1];
                    values[0] = 1;
                    indices[0] = (int)src - 1;

                    dst = new VBuffer<Float>(size, 1, values, indices);
                };
        }

        /// <summary>
        /// This is for the bagging case - vector input and outputs should be added.
        /// </summary>
        private ValueGetter<VBuffer<Float>> MakeGetterBag(IRow input, int iinfo)
        {
            Host.AssertValue(input);
            Host.Assert(Infos[iinfo].TypeSrc.IsVector);
            Host.Assert(Infos[iinfo].TypeSrc.ItemType.IsKey);
            Host.Assert(_bag[iinfo]);
            Host.Assert(Infos[iinfo].TypeSrc.ItemType.KeyCount == _types[iinfo].VectorSize);

            var info = Infos[iinfo];
            int size = info.TypeSrc.ItemType.KeyCount;
            Host.Assert(size > 0);

            int cv = info.TypeSrc.VectorSize;
            Host.Assert(cv >= 0);

            var getSrc = RowCursorUtils.GetVecGetterAs<uint>(NumberType.U4, input, info.Source);
            var src = default(VBuffer<uint>);
            var bldr = BufferBuilder<float>.CreateDefault();
            return
                (ref VBuffer<Float> dst) =>
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
        private ValueGetter<VBuffer<Float>> MakeGetterInd(IRow input, int iinfo)
        {
            Host.AssertValue(input);
            Host.Assert(Infos[iinfo].TypeSrc.IsVector);
            Host.Assert(Infos[iinfo].TypeSrc.ItemType.IsKey);
            Host.Assert(!_bag[iinfo]);

            var info = Infos[iinfo];
            int size = info.TypeSrc.ItemType.KeyCount;
            Host.Assert(size > 0);

            int cv = info.TypeSrc.VectorSize;
            Host.Assert(cv >= 0);
            Host.Assert(_types[iinfo].VectorSize == size * cv);

            var getSrc = RowCursorUtils.GetVecGetterAs<uint>(NumberType.U4, input, info.Source);
            var src = default(VBuffer<uint>);
            return
                (ref VBuffer<Float> dst) =>
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
                        valuesDst = new Float[cntSrc];
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
                    dst = new VBuffer<Float>(lenDst, count, valuesDst, indicesDst);
                };
        }
    }
}
