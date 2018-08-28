// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(HashTransform.Summary, typeof(HashTransform), typeof(HashTransform.Arguments), typeof(SignatureDataTransform),
    "Hash Transform", "HashTransform", "Hash", DocName = "transform/HashTransform.md")]

[assembly: LoadableClass(HashTransform.Summary, typeof(HashTransform), null, typeof(SignatureLoadDataTransform),
    "Hash Transform", HashTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    using Conditional = System.Diagnostics.ConditionalAttribute;

    /// <summary>
    /// This transform can hash either single valued columns or vector columns. For vector columns,
    /// it hashes each slot separately.
    /// It can hash either text values or key values.
    /// </summary>
    public sealed class HashTransform : OneToOneTransformBase, ITransformTemplate
    {
        public const int NumBitsMin = 1;
        public const int NumBitsLim = 32;

        private static class Defaults
        {
            public const int HashBits = NumBitsLim - 1;
            public const uint Seed = 314489979;
            public const bool Ordered = false;
            public const int InvertHash = 0;
        }

        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col",
                SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of bits to hash into. Must be between 1 and 31, inclusive",
                ShortName = "bits", SortOrder = 2)]
            public int HashBits = Defaults.HashBits;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
            public uint Seed = Defaults.Seed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the position of each term should be included in the hash",
                ShortName = "ord")]
            public bool Ordered = Defaults.Ordered;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.",
                ShortName = "ih")]
            public int InvertHash = Defaults.InvertHash;
        }

        public sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of bits to hash into. Must be between 1 and 31, inclusive", ShortName = "bits")]
            public int? HashBits;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
            public uint? Seed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the position of each term should be included in the hash",
                ShortName = "ord")]
            public bool? Ordered;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.",
                ShortName = "ih")]
            public int? InvertHash;

            public static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            protected override bool TryParse(string str)
            {
                Contracts.AssertNonEmpty(str);

                // We accept N:B:S where N is the new column name, B is the number of bits,
                // and S is source column names.
                string extra;
                if (!base.TryParse(str, out extra))
                    return false;
                if (extra == null)
                    return true;

                int bits;
                if (!int.TryParse(extra, out bits))
                    return false;
                HashBits = bits;
                return true;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (Seed != null || Ordered != null || InvertHash != null)
                    return false;
                if (HashBits == null)
                    return TryUnparseCore(sb);
                string extra = HashBits.Value.ToString();
                return TryUnparseCore(sb, extra);
            }
        }

        private sealed class ColInfoEx
        {
            public readonly int HashBits;
            public readonly uint HashSeed;
            public readonly bool Ordered;

            public ColInfoEx(Arguments args, Column col)
            {
                HashBits = col.HashBits ?? args.HashBits;
                if (HashBits < NumBitsMin || HashBits >= NumBitsLim)
                    throw Contracts.ExceptUserArg(nameof(args.HashBits), "Should be between {0} and {1} inclusive", NumBitsMin, NumBitsLim - 1);
                HashSeed = col.Seed ?? args.Seed;
                Ordered = col.Ordered ?? args.Ordered;
            }

            public ColInfoEx(ModelLoadContext ctx)
            {
                // *** Binary format ***
                // int: HashBits
                // uint: HashSeed
                // byte: Ordered

                HashBits = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(NumBitsMin <= HashBits && HashBits < NumBitsLim);

                HashSeed = ctx.Reader.ReadUInt32();
                Ordered = ctx.Reader.ReadBoolByte();
            }

            public void Save(ModelSaveContext ctx)
            {
                // *** Binary format ***
                // int: HashBits
                // uint: HashSeed
                // byte: Ordered

                Contracts.Assert(NumBitsMin <= HashBits && HashBits < NumBitsLim);
                ctx.Writer.Write(HashBits);

                ctx.Writer.Write(HashSeed);
                ctx.Writer.WriteBoolByte(Ordered);
            }
        }

        private static string TestType(ColumnType type)
        {
            if (type.ItemType.IsText || type.ItemType.IsKey || type.ItemType == NumberType.R4 || type.ItemType == NumberType.R8)
                return null;
            return "Expected Text, Key, Single or Double item type";
        }

        private const string RegistrationName = "Hash";

        internal const string Summary = "Converts column values into hashes. This transform accepts text and keys as inputs. It works on single- and vector-valued columns, "
            + "and hashes each slot in a vector separately.";

        public const string LoaderSignature = "HashTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "HASHTRNS",
                // verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Invert hash key values, hash fix
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature);
        }

        private readonly ColInfoEx[] _exes;
        private readonly ColumnType[] _types;

        private readonly VBuffer<DvText>[] _keyValues;
        private readonly ColumnType[] _kvTypes;

        public static HashTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new HashTransform(h, ctx, input));
        }

        private HashTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestType)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <base>
            // Exes

            Host.AssertNonEmpty(Infos);
            _exes = new ColInfoEx[Infos.Length];
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
                _exes[iinfo] = new ColInfoEx(ctx);

            _types = InitColumnTypes();

            TextModelHelper.LoadAll(Host, ctx, Infos.Length, out _keyValues, out _kvTypes);
            SetMetadata();
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // <prefix handled in static Create method>
            // <base>
            // Exes

            SaveBase(ctx);
            Host.Assert(_exes.Length == Infos.Length);
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
                _exes[iinfo].Save(ctx);

            TextModelHelper.SaveAll(Host, ctx, Infos.Length, _keyValues);
        }

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="source">Name of the column to be transformed. If this is null '<paramref name="name"/>' will be used.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 31, inclusive.</param>
        /// <param name="invertHash">Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.</param>
        public HashTransform(IHostEnvironment env,
            IDataView input,
            string name,
            string source = null,
            int hashBits = Defaults.HashBits,
            int invertHash = Defaults.InvertHash)
            : this(env, new Arguments() {
                Column = new[] { new Column() { Source = source ?? name, Name = name } },
                HashBits = hashBits, InvertHash = invertHash }, input)
        {
        }

        public HashTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(Contracts.CheckRef(env, nameof(env)), RegistrationName, env.CheckRef(args, nameof(args)).Column,
                input, TestType)
        {
            if (args.HashBits < NumBitsMin || args.HashBits >= NumBitsLim)
                throw Host.ExceptUserArg(nameof(args.HashBits), "hashBits should be between {0} and {1} inclusive", NumBitsMin, NumBitsLim - 1);

            _exes = new ColInfoEx[Infos.Length];
            List<int> invertIinfos = null;
            List<int> invertHashMaxCounts = null;
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
            {
                _exes[iinfo] = new ColInfoEx(args, args.Column[iinfo]);
                int invertHashMaxCount = GetAndVerifyInvertHashMaxCount(args, args.Column[iinfo], _exes[iinfo]);
                if (invertHashMaxCount > 0)
                {
                    Utils.Add(ref invertIinfos, iinfo);
                    Utils.Add(ref invertHashMaxCounts, invertHashMaxCount);
                }
            }

            _types = InitColumnTypes();

            if (Utils.Size(invertIinfos) > 0)
            {
                // Build the invert hashes for all columns for which it was requested.
                var srcs = new HashSet<int>(invertIinfos.Select(i => Infos[i].Source));
                using (IRowCursor srcCursor = input.GetRowCursor(srcs.Contains))
                {
                    using (var ch = Host.Start("Invert hash building"))
                    {
                        InvertHashHelper[] helpers = new InvertHashHelper[invertIinfos.Count];
                        Action disposer = null;
                        for (int i = 0; i < helpers.Length; ++i)
                        {
                            int iinfo = invertIinfos[i];
                            Host.Assert(_types[iinfo].ItemType.KeyCount > 0);
                            var dstGetter = GetGetterCore(ch, srcCursor, iinfo, out disposer);
                            Host.Assert(disposer == null);
                            var ex = _exes[iinfo];
                            var maxCount = invertHashMaxCounts[i];
                            helpers[i] = InvertHashHelper.Create(srcCursor, Infos[iinfo], ex, maxCount, dstGetter);
                        }
                        while (srcCursor.MoveNext())
                        {
                            for (int i = 0; i < helpers.Length; ++i)
                                helpers[i].Process();
                        }
                        _keyValues = new VBuffer<DvText>[_exes.Length];
                        _kvTypes = new ColumnType[_exes.Length];
                        for (int i = 0; i < helpers.Length; ++i)
                        {
                            _keyValues[invertIinfos[i]] = helpers[i].GetKeyValuesMetadata();
                            Host.Assert(_keyValues[invertIinfos[i]].Length == _types[invertIinfos[i]].ItemType.KeyCount);
                            _kvTypes[invertIinfos[i]] = new VectorType(TextType.Instance, _keyValues[invertIinfos[i]].Length);
                        }
                        ch.Done();
                    }
                }
            }
            SetMetadata();
        }

        /// <summary>
        /// Re-apply constructor.
        /// </summary>
        private HashTransform(IHostEnvironment env, HashTransform transform, IDataView newSource)
            : base(env, RegistrationName, transform, newSource, TestType)
        {
            _exes = transform._exes;
            _types = InitColumnTypes();
            _keyValues = transform._keyValues;
            _kvTypes = transform._kvTypes;
            SetMetadata();
        }

        public IDataTransform ApplyToData(IHostEnvironment env, IDataView newSource)
        {
            return new HashTransform(env, this, newSource);
        }

        private static int GetAndVerifyInvertHashMaxCount(Arguments args, Column col, ColInfoEx ex)
        {
            var invertHashMaxCount = col.InvertHash ?? args.InvertHash;
            if (invertHashMaxCount != 0)
            {
                if (invertHashMaxCount == -1)
                    invertHashMaxCount = int.MaxValue;
                Contracts.CheckUserArg(invertHashMaxCount > 0, nameof(args.InvertHash), "Value too small, must be -1 or larger");
                // If the bits is 31 or higher, we can't declare a KeyValues of the appropriate length,
                // this requiring a VBuffer of length 1u << 31 which exceeds int.MaxValue.
                if (ex.HashBits >= 31)
                    throw Contracts.ExceptUserArg(nameof(args.InvertHash), "Cannot support invertHash for a {0} bit hash. 30 is the maximum possible.", ex.HashBits);
            }
            return invertHashMaxCount;
        }

        private ColumnType[] InitColumnTypes()
        {
            var types = new ColumnType[Infos.Length];
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
            {
                var keyCount = _exes[iinfo].HashBits < 31 ? 1 << _exes[iinfo].HashBits : 0;
                var itemType = new KeyType(DataKind.U4, 0, keyCount, keyCount > 0);
                if (!Infos[iinfo].TypeSrc.IsVector)
                    types[iinfo] = itemType;
                else
                    types[iinfo] = new VectorType(itemType, Infos[iinfo].TypeSrc.VectorSize);
            }
            return types;
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Check(0 <= iinfo & iinfo < Infos.Length);
            return _types[iinfo];
        }

        private void SetMetadata()
        {
            var md = Metadata;
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
            {
                using (var bldr = md.BuildMetadata(iinfo, Source.Schema, Infos[iinfo].Source,
                    MetadataUtils.Kinds.SlotNames))
                {
                    if (_kvTypes != null && _kvTypes[iinfo] != null)
                        bldr.AddGetter<VBuffer<DvText>>(MetadataUtils.Kinds.KeyValues, _kvTypes[iinfo], GetTerms);
                }
            }
            md.Seal();
        }

        private void GetTerms(int iinfo, ref VBuffer<DvText> dst)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            Host.Assert(Utils.Size(_keyValues) == Infos.Length);
            Host.Assert(_keyValues[iinfo].Length > 0);
            _keyValues[iinfo].CopyTo(ref dst);
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            if (!Infos[iinfo].TypeSrc.IsVector)
                return ComposeGetterOne(input, iinfo);
            return ComposeGetterVec(input, iinfo);
        }

        /// <summary>
        /// Getter generator for single valued inputs
        /// </summary>
        private ValueGetter<uint> ComposeGetterOne(IRow input, int iinfo)
        {
            var colType = Infos[iinfo].TypeSrc;
            Host.Assert(colType.IsText || colType.IsKey || colType == NumberType.R4 || colType == NumberType.R8);

            var mask = (1U << _exes[iinfo].HashBits) - 1;
            uint seed = _exes[iinfo].HashSeed;
            // In case of single valued input column, hash in 0 for the slot index.
            if (_exes[iinfo].Ordered)
                seed = Hashing.MurmurRound(seed, 0);

            switch (colType.RawKind)
            {
            case DataKind.Text:
                return ComposeGetterOneCore(GetSrcGetter<DvText>(input, iinfo), seed, mask);
            case DataKind.U1:
                return ComposeGetterOneCore(GetSrcGetter<byte>(input, iinfo), seed, mask);
            case DataKind.U2:
                return ComposeGetterOneCore(GetSrcGetter<ushort>(input, iinfo), seed, mask);
            case DataKind.U4:
                return ComposeGetterOneCore(GetSrcGetter<uint>(input, iinfo), seed, mask);
            case DataKind.R4:
                return ComposeGetterOneCore(GetSrcGetter<float>(input, iinfo), seed, mask);
            case DataKind.R8:
                return ComposeGetterOneCore(GetSrcGetter<double>(input, iinfo), seed, mask);
            default:
                Host.Assert(colType.RawKind == DataKind.U8);
                return ComposeGetterOneCore(GetSrcGetter<ulong>(input, iinfo), seed, mask);
            }
        }

        private ValueGetter<uint> ComposeGetterOneCore(ValueGetter<DvText> getSrc, uint seed, uint mask)
        {
            DvText src = default(DvText);
            return
                (ref uint dst) =>
                {
                    // REVIEW: Should we verify that the key value is within the advertised KeyCount?
                    // Values greater than KeyCount should be treated as zeros.
                    getSrc(ref src);
                    dst = HashCore(seed, ref src, mask);
                };
        }

        private ValueGetter<uint> ComposeGetterOneCore(ValueGetter<byte> getSrc, uint seed, uint mask)
        {
            byte src = 0;
            return
                (ref uint dst) =>
                {
                    getSrc(ref src);
                    dst = HashCore(seed, src, mask);
                };
        }

        private ValueGetter<uint> ComposeGetterOneCore(ValueGetter<ushort> getSrc, uint seed, uint mask)
        {
            ushort src = 0;
            return
                (ref uint dst) =>
                {
                    getSrc(ref src);
                    dst = HashCore(seed, src, mask);
                };
        }

        private ValueGetter<uint> ComposeGetterOneCore(ValueGetter<uint> getSrc, uint seed, uint mask)
        {
            uint src = 0;
            return
                (ref uint dst) =>
                {
                    getSrc(ref src);
                    dst = HashCore(seed, src, mask);
                };
        }

        private ValueGetter<uint> ComposeGetterOneCore(ValueGetter<ulong> getSrc, uint seed, uint mask)
        {
            ulong src = 0;
            return
                (ref uint dst) =>
                {
                    getSrc(ref src);
                    dst = HashCore(seed, src, mask);
                };
        }

        private ValueGetter<uint> ComposeGetterOneCore(ValueGetter<float> getSrc, uint seed, uint mask)
        {
            float src = 0;
            return
                (ref uint dst) =>
                {
                    getSrc(ref src);
                    dst = HashCore(seed, ref src, mask);
                };
        }

        private ValueGetter<uint> ComposeGetterOneCore(ValueGetter<double> getSrc, uint seed, uint mask)
        {
            double src = 0;
            return
                (ref uint dst) =>
                {
                    getSrc(ref src);
                    dst = HashCore(seed, ref src, mask);
                };
        }

        // This is a delegate for a function that loops over the first count elements of src, and hashes
        // them (either with their index or without) into dst.
        private delegate void HashLoop<TSrc>(int count, int[] indices, TSrc[] src, uint[] dst, uint seed, uint mask);

        private ValueGetter<VBuffer<uint>> ComposeGetterVec(IRow input, int iinfo)
        {
            var colType = Infos[iinfo].TypeSrc;
            Host.Assert(colType.IsVector);
            Host.Assert(colType.ItemType.IsText || colType.ItemType.IsKey || colType.ItemType == NumberType.R4 || colType.ItemType == NumberType.R8);

            switch (colType.ItemType.RawKind)
            {
            case DataKind.Text:
                return ComposeGetterVecCore<DvText>(input, iinfo, HashUnord, HashDense, HashSparse);
            case DataKind.U1:
                return ComposeGetterVecCore<byte>(input, iinfo, HashUnord, HashDense, HashSparse);
            case DataKind.U2:
                return ComposeGetterVecCore<ushort>(input, iinfo, HashUnord, HashDense, HashSparse);
            case DataKind.U4:
                return ComposeGetterVecCore<uint>(input, iinfo, HashUnord, HashDense, HashSparse);
            case DataKind.R4:
                return ComposeGetterVecCoreFloat<float>(input, iinfo, HashSparseUnord, HashUnord, HashDense);
            case DataKind.R8:
                return ComposeGetterVecCoreFloat<double>(input, iinfo, HashSparseUnord, HashUnord, HashDense);
            default:
                Host.Assert(colType.ItemType.RawKind == DataKind.U8);
                return ComposeGetterVecCore<ulong>(input, iinfo, HashUnord, HashDense, HashSparse);
            }
        }

        private ValueGetter<VBuffer<uint>> ComposeGetterVecCore<T>(IRow input, int iinfo,
             HashLoop<T> hasherUnord, HashLoop<T> hasherDense, HashLoop<T> hasherSparse)
        {
            Host.Assert(Infos[iinfo].TypeSrc.IsVector);
            Host.Assert(Infos[iinfo].TypeSrc.ItemType.RawType == typeof(T));

            var getSrc = GetSrcGetter<VBuffer<T>>(input, iinfo);
            var ex = _exes[iinfo];
            var mask = (1U << ex.HashBits) - 1;
            var seed = ex.HashSeed;
            var len = Infos[iinfo].TypeSrc.VectorSize;
            var src = default(VBuffer<T>);

            if (!ex.Ordered)
            {
                hasherDense = hasherUnord;
                hasherSparse = hasherUnord;
            }

            return
                (ref VBuffer<uint> dst) =>
                {
                    getSrc(ref src);
                    if (len > 0 && src.Length != len)
                        throw Host.Except("Hash transform expected {0} slots, but got {1}", len, src.Length);

                    var hashes = dst.Values;
                    if (Utils.Size(hashes) < src.Count)
                        hashes = new uint[src.Count];

                    if (src.IsDense)
                    {
                        hasherDense(src.Count, null, src.Values, hashes, seed, mask);
                        dst = new VBuffer<uint>(src.Length, hashes, dst.Indices);
                        return;
                    }

                    hasherSparse(src.Count, src.Indices, src.Values, hashes, seed, mask);
                    var indices = dst.Indices;
                    if (src.Count > 0)
                    {
                        if (Utils.Size(indices) < src.Count)
                            indices = new int[src.Count];
                        Array.Copy(src.Indices, indices, src.Count);
                    }
                    dst = new VBuffer<uint>(src.Length, src.Count, hashes, indices);
                };
        }

        private ValueGetter<VBuffer<uint>> ComposeGetterVecCoreFloat<T>(IRow input, int iinfo,
            HashLoop<T> hasherSparseUnord, HashLoop<T> hasherDenseUnord, HashLoop<T> hasherDenseOrdered)
        {
            Host.Assert(Infos[iinfo].TypeSrc.IsVector);
            Host.Assert(Infos[iinfo].TypeSrc.ItemType.RawType == typeof(T));

            var getSrc = GetSrcGetter<VBuffer<T>>(input, iinfo);
            var ex = _exes[iinfo];
            var mask = (1U << ex.HashBits) - 1;
            var seed = ex.HashSeed;
            var len = Infos[iinfo].TypeSrc.VectorSize;
            var src = default(VBuffer<T>);
            T[] denseValues = null;
            int expectedSrcLength = Infos[iinfo].TypeSrc.VectorSize;
            HashLoop<T> hasherDense = ex.Ordered ? hasherDenseOrdered : hasherDenseUnord;

            return
                (ref VBuffer<uint> dst) =>
                {
                    getSrc(ref src);
                    if (len > 0 && src.Length != len)
                        throw Host.Except("Hash transform expected {0} slots, but got {1}", len, src.Length);

                    T[] values = src.Values;
                    var srcIsDense = src.IsDense;
                    var hashes = dst.Values;

                    // force-densify the input in case of ordered hash.
                    if (!srcIsDense && ex.Ordered)
                    {
                        if (denseValues == null)
                            denseValues = new T[expectedSrcLength];
                        values = denseValues;
                        src.CopyTo(values);
                        srcIsDense = true;
                    }

                    if (srcIsDense)
                    {
                        if (Utils.Size(hashes) < values.Length)
                            hashes = new uint[values.Length];
                        hasherDense(values.Length, null, values, hashes, seed, mask);
                        dst = new VBuffer<uint>(values.Length, hashes, dst.Indices);
                        return;
                    }

                    // source is sparse at this point and hash is unordered
                    if (Utils.Size(hashes) < expectedSrcLength)
                        hashes = new uint[expectedSrcLength];
                    hasherSparseUnord(expectedSrcLength, src.Indices, values, hashes, seed, mask);
                    dst = new VBuffer<uint>(expectedSrcLength, hashes, dst.Indices);
                };
        }

        #region Core Hash functions, with and without index
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint HashCore(uint seed, ref DvText value, uint mask)
        {
            Contracts.Assert(Utils.IsPowerOfTwo(mask + 1));
            if (!value.HasChars)
                return 0;
            return (value.Trim().Hash(seed) & mask) + 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint HashCore(uint seed, ref DvText value, int i, uint mask)
        {
            Contracts.Assert(Utils.IsPowerOfTwo(mask + 1));
            if (!value.HasChars)
                return 0;
            return (value.Trim().Hash(Hashing.MurmurRound(seed, (uint)i)) & mask) + 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint HashCore(uint seed, ref float value, uint mask)
        {
            Contracts.Assert(Utils.IsPowerOfTwo(mask + 1));
            if (value.IsNA())
                return 0;
            // (value == 0 ? 0 : value) takes care of negative 0, its equal to positive 0 according to the IEEE 754 standard
            return (Hashing.MixHash(Hashing.MurmurRound(seed, FloatUtils.GetBits(value == 0 ? 0 : value))) & mask) + 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint HashCore(uint seed, ref float value, int i, uint mask)
        {
            Contracts.Assert(Utils.IsPowerOfTwo(mask + 1));
            if (value.IsNA())
                return 0;
            return (Hashing.MixHash(Hashing.MurmurRound(Hashing.MurmurRound(seed, (uint)i),
                FloatUtils.GetBits(value == 0 ? 0: value))) & mask) + 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint HashCore(uint seed, ref double value, uint mask)
        {
            Contracts.Assert(Utils.IsPowerOfTwo(mask + 1));
            if (value.IsNA())
                return 0;

            ulong v = FloatUtils.GetBits(value == 0 ? 0 : value);
            var hash = Hashing.MurmurRound(seed, Utils.GetLo(v));
            var hi = Utils.GetHi(v);
            if (hi != 0)
                hash = Hashing.MurmurRound(hash, hi);
            return (Hashing.MixHash(hash) & mask) + 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint HashCore(uint seed, ref double value, int i, uint mask)
        {
            // If the high word is zero, this should produce the same value as the uint version.
            Contracts.Assert(Utils.IsPowerOfTwo(mask + 1));
            if (value.IsNA())
                return 0;

            ulong v = FloatUtils.GetBits(value == 0 ? 0 : value);
            var lo = Utils.GetLo(v);
            var hi = Utils.GetHi(v);
            var hash = Hashing.MurmurRound(Hashing.MurmurRound(seed, (uint)i), lo);
            if (hi != 0)
                hash = Hashing.MurmurRound(hash, hi);
            return (Hashing.MixHash(hash) & mask) + 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint HashCore(uint seed, uint value, uint mask)
        {
            Contracts.Assert(Utils.IsPowerOfTwo(mask + 1));
            if (value == 0)
                return 0;
            return (Hashing.MixHash(Hashing.MurmurRound(seed, value)) & mask) + 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint HashCore(uint seed, uint value, int i, uint mask)
        {
            Contracts.Assert(Utils.IsPowerOfTwo(mask + 1));
            if (value == 0)
                return 0;
            return (Hashing.MixHash(Hashing.MurmurRound(Hashing.MurmurRound(seed, (uint)i), value)) & mask) + 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint HashCore(uint seed, ulong value, uint mask)
        {
            // If the high word is zero, this should produce the same value as the uint version.
            Contracts.Assert(Utils.IsPowerOfTwo(mask + 1));

            if (value == 0)
                return 0;

            var hash = Hashing.MurmurRound(seed, Utils.GetLo(value));
            var hi = Utils.GetHi(value);
            if (hi != 0)
                hash = Hashing.MurmurRound(hash, hi);
            return (Hashing.MixHash(hash) & mask) + 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint HashCore(uint seed, ulong value, int i, uint mask)
        {
            // If the high word is zero, this should produce the same value as the uint version.
            Contracts.Assert(Utils.IsPowerOfTwo(mask + 1));
            var lo = Utils.GetLo(value);
            var hi = Utils.GetHi(value);
            if (lo == 0 && hi == 0)
                return 0;
            var hash = Hashing.MurmurRound(Hashing.MurmurRound(seed, (uint)i), lo);
            if (hi != 0)
                hash = Hashing.MurmurRound(hash, hi);
            return (Hashing.MixHash(hash) & mask) + 1;
        }
#endregion Core Hash functions, with and without index

#region Unordered Loop: ignore indices
        private static void HashUnord(int count, int[] indices, DvText[] src, uint[] dst, uint seed, uint mask)
        {
            AssertValid(count, src, dst);

            for (int i = 0; i < count; i++)
                dst[i] = HashCore(seed, ref src[i], mask);
        }

        private static void HashUnord(int count, int[] indices, byte[] src, uint[] dst, uint seed, uint mask)
        {
            AssertValid(count, src, dst);

            for (int i = 0; i < count; i++)
                dst[i] = HashCore(seed, src[i], mask);
        }

        private static void HashUnord(int count, int[] indices, ushort[] src, uint[] dst, uint seed, uint mask)
        {
            AssertValid(count, src, dst);

            for (int i = 0; i < count; i++)
                dst[i] = HashCore(seed, src[i], mask);
        }

        private static void HashUnord(int count, int[] indices, uint[] src, uint[] dst, uint seed, uint mask)
        {
            AssertValid(count, src, dst);

            for (int i = 0; i < count; i++)
                dst[i] = HashCore(seed, src[i], mask);
        }

        private static void HashUnord(int count, int[] indices, ulong[] src, uint[] dst, uint seed, uint mask)
        {
            AssertValid(count, src, dst);

            for (int i = 0; i < count; i++)
                dst[i] = HashCore(seed, src[i], mask);
        }

        private static void HashUnord(int count, int[] indices, float[] src, uint[] dst, uint seed, uint mask)
        {
            AssertValid(count, src, dst);

            for (int i = 0; i < count; i++)
                dst[i] = HashCore(seed, ref src[i], mask);
        }
        private static void HashUnord(int count, int[] indices, double[] src, uint[] dst, uint seed, uint mask)
        {
            AssertValid(count, src, dst);

            for (int i = 0; i < count; i++)
                dst[i] = HashCore(seed, ref src[i], mask);
        }

        #endregion Unordered Loop: ignore indices

#region Dense Loop: ignore indices
        private static void HashDense(int count, int[] indices, DvText[] src, uint[] dst, uint seed, uint mask)
        {
            AssertValid(count, src, dst);

            for (int i = 0; i < count; i++)
                dst[i] = HashCore(seed, ref src[i], i, mask);
        }

        private static void HashDense(int count, int[] indices, byte[] src, uint[] dst, uint seed, uint mask)
        {
            AssertValid(count, src, dst);

            for (int i = 0; i < count; i++)
                dst[i] = HashCore(seed, src[i], i, mask);
        }

        private static void HashDense(int count, int[] indices, ushort[] src, uint[] dst, uint seed, uint mask)
        {
            AssertValid(count, src, dst);

            for (int i = 0; i < count; i++)
                dst[i] = HashCore(seed, src[i], i, mask);
        }

        private static void HashDense(int count, int[] indices, uint[] src, uint[] dst, uint seed, uint mask)
        {
            AssertValid(count, src, dst);

            for (int i = 0; i < count; i++)
                dst[i] = HashCore(seed, src[i], i, mask);
        }

        private static void HashDense(int count, int[] indices, ulong[] src, uint[] dst, uint seed, uint mask)
        {
            AssertValid(count, src, dst);

            for (int i = 0; i < count; i++)
                dst[i] = HashCore(seed, src[i], i, mask);
        }

        private static void HashDense(int count, int[] indices, float[] src, uint[] dst, uint seed, uint mask)
        {
            AssertValid(count, src, dst);

            for (int i = 0; i < count; i++)
                dst[i] = HashCore(seed, ref src[i], i, mask);
        }
        private static void HashDense(int count, int[] indices, double[] src, uint[] dst, uint seed, uint mask)
        {
            AssertValid(count, src, dst);

            for (int i = 0; i < count; i++)
                dst[i] = HashCore(seed, ref src[i], i, mask);
        }
#endregion Dense Loop: ignore indices

#region Sparse Loop: use indices
        private static void HashSparse(int count, int[] indices, DvText[] src, uint[] dst, uint seed, uint mask)
        {
            AssertValid(count, src, dst);
            Contracts.Assert(count <= Utils.Size(indices));

            for (int i = 0; i < count; i++)
                dst[i] = HashCore(seed, ref src[i], indices[i], mask);
        }

        private static void HashSparse(int count, int[] indices, byte[] src, uint[] dst, uint seed, uint mask)
        {
            AssertValid(count, src, dst);
            Contracts.Assert(count <= Utils.Size(indices));

            for (int i = 0; i < count; i++)
                dst[i] = HashCore(seed, src[i], indices[i], mask);
        }

        private static void HashSparse(int count, int[] indices, ushort[] src, uint[] dst, uint seed, uint mask)
        {
            AssertValid(count, src, dst);
            Contracts.Assert(count <= Utils.Size(indices));

            for (int i = 0; i < count; i++)
                dst[i] = HashCore(seed, src[i], indices[i], mask);
        }

        private static void HashSparse(int count, int[] indices, uint[] src, uint[] dst, uint seed, uint mask)
        {
            AssertValid(count, src, dst);
            Contracts.Assert(count <= Utils.Size(indices));

            for (int i = 0; i < count; i++)
                dst[i] = HashCore(seed, src[i], indices[i], mask);
        }

        private static void HashSparse(int count, int[] indices, ulong[] src, uint[] dst, uint seed, uint mask)
        {
            AssertValid(count, src, dst);
            Contracts.Assert(count <= Utils.Size(indices));

            for (int i = 0; i < count; i++)
                dst[i] = HashCore(seed, src[i], indices[i], mask);
        }

        private static void HashSparseUnord(int count, int[] indices, float[] src, uint[] dst, uint seed, uint mask)
        {
            Contracts.Assert(count >= 0);
            Contracts.Assert(count >= Utils.Size(indices));
            Contracts.Assert(count == Utils.Size(dst));

            float zero = 0.0f;
            uint zeroHash = HashCore(seed, ref zero, mask);

            int j = 0;
            for (int i = 0; i < count; i++)
            {
                if (Utils.Size(indices) <= j || indices[j] > i)
                    dst[i] = zeroHash;
                else if (indices[j] == i)
                    dst[i] = HashCore(seed, ref src[j++], mask);
                else
                    Contracts.Assert(false, "this should have never happened.");
            }
        }

        private static void HashSparseUnord(int count, int[] indices, double[] src, uint[] dst, uint seed, uint mask)
        {
            Contracts.Assert(count >= 0);
            Contracts.Assert(count >= Utils.Size(indices));
            Contracts.Assert(count == Utils.Size(dst));

            double zero = 0.0;
            uint zeroHash = HashCore(seed, ref zero, mask);

            int j = 0;
            for (int i = 0; i < count; i++)
            {
                if (Utils.Size(indices) <= j || indices[j] > i)
                    dst[i] = zeroHash;
                else if (indices[j] == i)
                    dst[i] = HashCore(seed, ref src[j++], mask);
                else
                    Contracts.Assert(false, "this should have never happened.");
            }
        }

#endregion Sparse Loop: use indices

        [Conditional("DEBUG")]
        private static void AssertValid<T>(int count, T[] src, uint[] dst)
        {
            Contracts.Assert(count >= 0);
            Contracts.Assert(count <= Utils.Size(src));
            Contracts.Assert(count <= Utils.Size(dst));
        }

        /// <summary>
        /// This is a utility class to acquire and build the inverse hash to populate
        /// KeyValues metadata.
        /// </summary>
        private abstract class InvertHashHelper
        {
            protected readonly IRow Row;
            private readonly bool _includeSlot;
            private readonly ColInfo _info;
            private readonly ColInfoEx _ex;

            private InvertHashHelper(IRow row, ColInfo info, ColInfoEx ex)
            {
                Contracts.AssertValue(row);
                Contracts.AssertValue(info);

                Row = row;
                _info = info;
                _ex = ex;
                // If this is a vector and ordered, then we must include the slot as part of the representation.
                _includeSlot = _info.TypeSrc.IsVector && _ex.Ordered;
            }

            /// <summary>
            /// Constructs an <see cref="InvertHashHelper"/> instance to accumulate hash/value pairs
            /// from a single column as parameterized by this transform, with values fetched from
            /// the row.
            /// </summary>
            /// <param name="row">The input source row, from which the hashed values can be fetched</param>
            /// <param name="info">The column info, describing the source</param>
            /// <param name="ex">The extra column info</param>
            /// <param name="invertHashMaxCount">The number of input hashed values to accumulate per output hash value</param>
            /// <param name="dstGetter">A hash getter, built on top of <paramref name="row"/>.</param>
            public static InvertHashHelper Create(IRow row, ColInfo info, ColInfoEx ex, int invertHashMaxCount, Delegate dstGetter)
            {
                ColumnType typeSrc = info.TypeSrc;
                Type t = typeSrc.IsVector ? (ex.Ordered ? typeof(ImplVecOrdered<>) : typeof(ImplVec<>)) : typeof(ImplOne<>);
                t = t.MakeGenericType(typeSrc.ItemType.RawType);
                var consTypes = new Type[] { typeof(IRow), typeof(OneToOneTransformBase.ColInfo), typeof(ColInfoEx), typeof(int), typeof(Delegate) };
                var constructorInfo = t.GetConstructor(consTypes);
                return (InvertHashHelper)constructorInfo.Invoke(new object[] { row, info, ex, invertHashMaxCount, dstGetter });
            }

            /// <summary>
            /// This calculates the hash/value pair from the current value of the column, and does
            /// appropriate processing of them to build the invert hash map.
            /// </summary>
            public abstract void Process();

            public abstract VBuffer<DvText> GetKeyValuesMetadata();

            private sealed class TextEqualityComparer : IEqualityComparer<DvText>
            {
                // REVIEW: Is this sufficiently useful? Should we be using term map, instead?
                private readonly uint _seed;

                public TextEqualityComparer(uint seed)
                {
                    _seed = seed;
                }

                public bool Equals(DvText x, DvText y)
                {
                    return x.Equals(y);
                }

                public int GetHashCode(DvText obj)
                {
                    if (!obj.HasChars)
                        return 0;
                    return (int)obj.Trim().Hash(_seed) + 1;
                }
            }

            private sealed class KeyValueComparer<T> : IEqualityComparer<KeyValuePair<int, T>>
            {
                private readonly IEqualityComparer<T> _tComparer;

                public KeyValueComparer(IEqualityComparer<T> tComparer)
                {
                    _tComparer = tComparer;
                }

                public bool Equals(KeyValuePair<int, T> x, KeyValuePair<int, T> y)
                {
                    return (x.Key == y.Key) && _tComparer.Equals(x.Value, y.Value);
                }

                public int GetHashCode(KeyValuePair<int, T> obj)
                {
                    return Hashing.CombineHash(obj.Key, _tComparer.GetHashCode(obj.Value));
                }
            }

            private IEqualityComparer<T> GetSimpleComparer<T>()
            {
                Contracts.Assert(_info.TypeSrc.ItemType.RawType == typeof(T));
                if (typeof(T) == typeof(DvText))
                {
                    // We are hashing twice, once to assign to the slot, and then again,
                    // to build a set of encountered elements. Obviously we cannot use the
                    // same seed used to assign to a slot, or otherwise this per-slot hash
                    // would have a lot of collisions. We ensure that we have different
                    // hash function by inverting the seed's bits.
                    var c = new TextEqualityComparer(~_ex.HashSeed);
                    return c as IEqualityComparer<T>;
                }
                // I assume I hope correctly that the default .NET hash function for uint
                // is sufficiently different from our implementation. The current
                // implementation at the time of this writing is to just return the
                // value itself cast to an int, so we should be fine.
                return EqualityComparer<T>.Default;
            }

            private abstract class Impl<T> : InvertHashHelper
            {
                protected readonly InvertHashCollector<T> Collector;

                protected Impl(IRow row, ColInfo info, ColInfoEx ex, int invertHashMaxCount)
                    : base(row, info, ex)
                {
                    Contracts.AssertValue(row);
                    Contracts.AssertValue(info);
                    Contracts.AssertValue(ex);

                    Collector = new InvertHashCollector<T>(1 << ex.HashBits, invertHashMaxCount, GetTextMap(), GetComparer());
                }

                protected virtual ValueMapper<T, StringBuilder> GetTextMap()
                {
                    return InvertHashUtils.GetSimpleMapper<T>(Row.Schema, _info.Source);
                }

                protected virtual IEqualityComparer<T> GetComparer()
                {
                    return GetSimpleComparer<T>();
                }

                public override VBuffer<DvText> GetKeyValuesMetadata()
                {
                    return Collector.GetMetadata();
                }
            }

            private sealed class ImplOne<T> : Impl<T>
            {
                private readonly ValueGetter<T> _srcGetter;
                private readonly ValueGetter<uint> _dstGetter;

                private T _value;
                private uint _hash;

                public ImplOne(IRow row, OneToOneTransformBase.ColInfo info, ColInfoEx ex, int invertHashMaxCount, Delegate dstGetter)
                    : base(row, info, ex, invertHashMaxCount)
                {
                    _srcGetter = Row.GetGetter<T>(_info.Source);
                    _dstGetter = dstGetter as ValueGetter<uint>;
                    Contracts.AssertValue(_dstGetter);
                }

                public override void Process()
                {
                    // REVIEW: This is suboptimal. We're essentially getting the source value
                    // twice, since the hash function will also do this. On the other hand, the obvious
                    // refactoring of changing the hash getter to be a value mapper and using that, I
                    // think has a disadvantage of incurring an *additional* delegate call within the
                    // getter, and possibly that's worse than this (since that would always be a cost
                    // for every application of the transform, and this is a cost only at the start,
                    // in some very specific situations I suspect will be a minority case).

                    // We don't get the source until we're certain we want to retain that value.
                    _dstGetter(ref _hash);
                    // Missing values do not get KeyValues. The first entry in the KeyValues metadata
                    // array corresponds to the first valid key, that is, where dstSlot is 1.
                    Collector.Add(_hash, _srcGetter, ref _value);
                }
            }

            private sealed class ImplVec<T> : Impl<T>
            {
                private readonly ValueGetter<VBuffer<T>> _srcGetter;
                private readonly ValueGetter<VBuffer<uint>> _dstGetter;

                private VBuffer<T> _value;
                private VBuffer<uint> _hash;

                public ImplVec(IRow row, OneToOneTransformBase.ColInfo info, ColInfoEx ex, int invertHashMaxCount, Delegate dstGetter)
                    : base(row, info, ex, invertHashMaxCount)
                {
                    _srcGetter = Row.GetGetter<VBuffer<T>>(_info.Source);
                    _dstGetter = dstGetter as ValueGetter<VBuffer<uint>>;
                    Contracts.AssertValue(_dstGetter);
                }

                public override void Process()
                {
                    _srcGetter(ref _value);
                    _dstGetter(ref _hash);
                    // The two arrays should be consistent in their density, length, count, etc.
                    Contracts.Assert(_value.IsDense == _hash.IsDense);
                    Contracts.Assert(_value.Length == _hash.Length);
                    Contracts.Assert(_value.Count == _hash.Count);
                    for (int i = 0; i < _value.Count; ++i)
                        Collector.Add(_hash.Values[i], _value.Values[i]);
                }
            }

            private sealed class ImplVecOrdered<T> : Impl<KeyValuePair<int, T>>
            {
                private readonly ValueGetter<VBuffer<T>> _srcGetter;
                private readonly ValueGetter<VBuffer<uint>> _dstGetter;

                private VBuffer<T> _value;
                private VBuffer<uint> _hash;

                public ImplVecOrdered(IRow row, OneToOneTransformBase.ColInfo info, ColInfoEx ex, int invertHashMaxCount, Delegate dstGetter)
                    : base(row, info, ex, invertHashMaxCount)
                {
                    _srcGetter = Row.GetGetter<VBuffer<T>>(_info.Source);
                    _dstGetter = dstGetter as ValueGetter<VBuffer<uint>>;
                    Contracts.AssertValue(_dstGetter);
                }

                protected override ValueMapper<KeyValuePair<int, T>, StringBuilder> GetTextMap()
                {
                    var simple = InvertHashUtils.GetSimpleMapper<T>(Row.Schema, _info.Source);
                    return InvertHashUtils.GetPairMapper(simple);
                }

                protected override IEqualityComparer<KeyValuePair<int, T>> GetComparer()
                {
                    return new KeyValueComparer<T>(GetSimpleComparer<T>());
                }

                public override void Process()
                {
                    _srcGetter(ref _value);
                    _dstGetter(ref _hash);
                    // The two arrays should be consistent in their density, length, count, etc.
                    Contracts.Assert(_value.IsDense == _hash.IsDense);
                    Contracts.Assert(_value.Length == _hash.Length);
                    Contracts.Assert(_value.Count == _hash.Count);
                    if (_hash.IsDense)
                    {
                        for (int i = 0; i < _value.Count; ++i)
                            Collector.Add(_hash.Values[i], new KeyValuePair<int, T>(i, _value.Values[i]));
                    }
                    else
                    {
                        for (int i = 0; i < _value.Count; ++i)
                            Collector.Add(_hash.Values[i], new KeyValuePair<int, T>(_hash.Indices[i], _value.Values[i]));
                    }
                }
            }
        }
    }
}

