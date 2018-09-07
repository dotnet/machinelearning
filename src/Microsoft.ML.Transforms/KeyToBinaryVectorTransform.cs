// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(KeyToBinaryVectorTransform.Summary, typeof(KeyToBinaryVectorTransform),
    typeof(KeyToBinaryVectorTransform.Arguments), typeof(SignatureDataTransform),
    "Key To Binary Vector Transform", "KeyToBinaryVectorTransform", "KeyToBinary",
    DocName = "transform/KeyToBinaryVectorTransform.md")]

[assembly: LoadableClass(KeyToBinaryVectorTransform.Summary, typeof(KeyToBinaryVectorTransform),
    null, typeof(SignatureLoadDataTransform), "Key To Binary Vector Transform", KeyToBinaryVectorTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    public sealed class KeyToBinaryVectorTransform : OneToOneTransformBase
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)",
                ShortName = "col", SortOrder = 1)]
            public KeyToVectorTransform.Column[] Column;
        }

        internal const string Summary = "Converts a key column to a binary encoded vector.";

        public const string LoaderSignature = "KeyToBinaryTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "KEY2BINR",
                verWrittenCur: 0x00000001, // Initial
                verReadableCur: 0x00000001,
                verWeCanReadBack: 0x00000001,
                loaderSignature: LoaderSignature);
        }

        private const string RegistrationName = "KeyToBinary";

        // These arrays are parallel to Infos.
        // * _concat is whether, given the current input, there are multiple output instance vectors
        //   to concatenate.
        // * _types contains the output column types.
        private readonly bool[] _concat;

        private readonly int[] _bitsPerKey;

        private readonly VectorType[] _types;

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="source">Name of the column to be transformed. If this is null '<paramref name="name"/>' will be used.</param>
        public KeyToBinaryVectorTransform(IHostEnvironment env, IDataView input, string name, string source = null)
            : this(env, new Arguments() { Column = new[] { new KeyToVectorTransform.Column() { Source = source ?? name, Name = name } } }, input)
        {
        }

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public KeyToBinaryVectorTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, Contracts.CheckRef(args, nameof(args)).Column,
                input, TestIsKey)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));

            Init(out _concat, out _types, out _bitsPerKey);
        }

        private KeyToBinaryVectorTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestIsKey)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            Host.AssertNonEmpty(Infos);

            Init(out _concat, out _types, out _bitsPerKey);
        }

        private void Init(out bool[] concat, out VectorType[] types, out int[] bitsPerKey)
        {
            concat = new bool[Infos.Length];
            types = new VectorType[Infos.Length];
            bitsPerKey = new int[Infos.Length];

            for (int i = 0; i < Infos.Length; i++)
                ComputeType(this, Source.Schema, i, Infos[i], Metadata,
                    out _types[i], out _concat[i], out _bitsPerKey[i]);

            Metadata.Seal();
        }

        public static KeyToBinaryVectorTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            return h.Apply("Loading Model", ch => new KeyToBinaryVectorTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));

            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            SaveBase(ctx);
        }

        /// <summary>
        /// Computes the column type and whether multiple indicator vectors need to be concatenated.
        /// Also populates the metadata.
        /// </summary>
        private static void ComputeType(KeyToBinaryVectorTransform trans, ISchema input, int iinfo,
            ColInfo info, MetadataDispatcher md, out VectorType type, out bool concat, out int bitsPerColumn)
        {
            Contracts.AssertValue(trans);
            Contracts.AssertValue(input);
            Contracts.AssertValue(info);
            Contracts.Assert(info.TypeSrc.ItemType.IsKey);
            Contracts.Assert(info.TypeSrc.ItemType.KeyCount > 0);

            //Add an additional bit for all 1s to represent missing values.
            bitsPerColumn = Utils.IbitHigh((uint)info.TypeSrc.ItemType.KeyCount) + 2;

            Contracts.Assert(bitsPerColumn > 0);

            // See if the source has key names.
            var typeNames = input.GetMetadataTypeOrNull(MetadataUtils.Kinds.KeyValues, info.Source);
            if (typeNames == null || !typeNames.IsKnownSizeVector || !typeNames.ItemType.IsText ||
                typeNames.VectorSize != info.TypeSrc.ItemType.KeyCount)
            {
                typeNames = null;
            }

            // Don't pass through any source column metadata.
            using (var bldr = md.BuildMetadata(iinfo))
            {
                if (info.TypeSrc.ValueCount == 1)
                {
                    // Output is a single vector computed as the sum of the output indicator vectors.
                    concat = false;
                    type = new VectorType(NumberType.Float, bitsPerColumn);
                    if (typeNames != null)
                    {
                        bldr.AddGetter<VBuffer<DvText>>(MetadataUtils.Kinds.SlotNames,
                            new VectorType(TextType.Instance, type), trans.GetKeyNames);
                    }

                    bldr.AddPrimitive(MetadataUtils.Kinds.IsNormalized, BoolType.Instance, true);
                }
                else
                {
                    // Output is the concatenation of the multiple output indicator vectors.
                    concat = true;
                    type = new VectorType(NumberType.Float, info.TypeSrc.ValueCount, bitsPerColumn);
                    if (typeNames != null && type.VectorSize > 0)
                    {
                        bldr.AddGetter<VBuffer<DvText>>(MetadataUtils.Kinds.SlotNames,
                            new VectorType(TextType.Instance, type), trans.GetSlotNames);
                    }
                }
            }
        }

        private void GenerateBitSlotName(int iinfo, ref VBuffer<DvText> dst)
        {
            const string slotNamePrefix = "Bit";
            var bldr = new BufferBuilder<DvText>(TextCombiner.Instance);
            bldr.Reset(_bitsPerKey[iinfo], true);
            for (int i = 0; i < _bitsPerKey[iinfo]; i++)
                bldr.AddFeature(i, new DvText(slotNamePrefix + (_bitsPerKey[iinfo] - i - 1)));

            bldr.GetResult(ref dst);
        }

        private void GetKeyNames(int iinfo, ref VBuffer<DvText> dst)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            Host.Assert(!_concat[iinfo]);

            GenerateBitSlotName(iinfo, ref dst);
        }

        private void GetSlotNames(int iinfo, ref VBuffer<DvText> dst)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            Host.Assert(_concat[iinfo]);
            Host.Assert(_types[iinfo].IsKnownSizeVector);

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

            int slotLim = _types[iinfo].VectorSize;
            Host.Assert(slotLim == (long)typeSrc.VectorSize * _bitsPerKey[iinfo]);

            var values = dst.Values;
            if (Utils.Size(values) < slotLim)
                values = new DvText[slotLim];

            var sb = new StringBuilder();
            int slot = 0;
            VBuffer<DvText> bits = default(VBuffer<DvText>);
            GenerateBitSlotName(iinfo, ref bits);
            foreach (var kvpSlot in namesSlotSrc.Items(all: true))
            {
                Contracts.Assert(slot == (long)kvpSlot.Key * _bitsPerKey[iinfo]);
                sb.Clear();
                if (kvpSlot.Value.HasChars)
                    kvpSlot.Value.AddToStringBuilder(sb);
                else
                    sb.Append('[').Append(kvpSlot.Key).Append(']');
                sb.Append('.');

                int len = sb.Length;
                foreach (var key in bits.Values)
                {
                    sb.Length = len;
                    key.AddToStringBuilder(sb);
                    values[slot++] = new DvText(sb.ToString());
                }
            }
            Host.Assert(slot == slotLim);

            dst = new VBuffer<DvText>(slotLim, values, dst.Indices);
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo & iinfo < _types.Length);
            return _types[iinfo];
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
            return MakeGetterInd(input, iinfo);
        }

        /// <summary>
        /// This is for the scalar case.
        /// </summary>
        private ValueGetter<VBuffer<float>> MakeGetterOne(IRow input, int iinfo)
        {
            Host.AssertValue(input);
            Host.Assert(Infos[iinfo].TypeSrc.IsKey);

            int bitsPerKey = _bitsPerKey[iinfo];
            Host.Assert(bitsPerKey == _types[iinfo].VectorSize);

            int dstLength = _types[iinfo].VectorSize;
            Host.Assert(dstLength > 0);

            var getSrc = RowCursorUtils.GetGetterAs<uint>(NumberType.U4, input, Infos[iinfo].Source);
            var src = default(uint);
            var bldr = new BufferBuilder<float>(R4Adder.Instance);
            return
                (ref VBuffer<float> dst) =>
                {
                    getSrc(ref src);
                    bldr.Reset(bitsPerKey, false);
                    EncodeValueToBinary(bldr, src, bitsPerKey, 0);
                    bldr.GetResult(ref dst);

                    Contracts.Assert(dst.Length == bitsPerKey);
                };
        }

        /// <summary>
        /// This is for the indicator case - vector input and outputs should be concatenated.
        /// </summary>
        private ValueGetter<VBuffer<float>> MakeGetterInd(IRow input, int iinfo)
        {
            Host.AssertValue(input);
            Host.Assert(Infos[iinfo].TypeSrc.IsVector);
            Host.Assert(Infos[iinfo].TypeSrc.ItemType.IsKey);

            int cv = Infos[iinfo].TypeSrc.VectorSize;
            Host.Assert(cv >= 0);

            var getSrc = RowCursorUtils.GetVecGetterAs<uint>(NumberType.U4, input, Infos[iinfo].Source);
            var src = default(VBuffer<uint>);
            var bldr = new BufferBuilder<float>(R4Adder.Instance);
            int bitsPerKey = _bitsPerKey[iinfo];
            return
                (ref VBuffer<float> dst) =>
                {
                    getSrc(ref src);
                    Host.Check(src.Length == cv || cv == 0);
                    bldr.Reset(src.Length * bitsPerKey, false);

                    int index = 0;
                    foreach (uint value in src.DenseValues())
                    {
                        EncodeValueToBinary(bldr, value, bitsPerKey, index * bitsPerKey);
                        index++;
                    }

                    bldr.GetResult(ref dst);

                    Contracts.Assert(dst.Length == src.Length * bitsPerKey);
                };
        }

        private void EncodeValueToBinary(BufferBuilder<float> bldr, uint value, int bitsToConsider, int startIndex)
        {
            Contracts.Assert(0 < bitsToConsider && bitsToConsider <= sizeof(uint) * 8);
            Contracts.Assert(startIndex >= 0);

            //Treat missing values, zero, as a special value of all 1s.
            value--;
            while (bitsToConsider > 0)
                bldr.AddFeature(startIndex++, (value >> --bitsToConsider) & 1U);
        }
    }
}
