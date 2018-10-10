// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(typeof(NgramHashTransform), typeof(NgramHashTransform.Arguments), typeof(SignatureDataTransform),
    "Ngram Hash Transform", "NgramHashTransform", "NgramHash")]

[assembly: LoadableClass(typeof(NgramHashTransform), null, typeof(SignatureLoadDataTransform),
    "Ngram Hash Transform", NgramHashTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    using Conditional = System.Diagnostics.ConditionalAttribute;

    public sealed class NgramHashTransform : RowToRowMapperTransformBase
    {
        public sealed class Column : ManyToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum ngram length", ShortName = "ngram")]
            public int? NgramLength;

            [Argument(ArgumentType.AtMostOnce, HelpText =
                "Whether to include all ngram lengths up to " + nameof(NgramLength) + " or only " + nameof(NgramLength), ShortName = "all")]
            public bool? AllLengths;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Maximum number of tokens to skip when constructing an ngram",
                ShortName = "skips")]
            public int? SkipLength;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Number of bits to hash into. Must be between 1 and 30, inclusive.",
                ShortName = "bits")]
            public int? HashBits;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
            public uint? Seed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to rehash unigrams", ShortName = "rehash")]
            public bool? RehashUnigrams;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the position of each source column should be included in the hash (when there are multiple source columns).", ShortName = "ord")]
            public bool? Ordered;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.",
                ShortName = "ih")]
            public int? InvertHash;

            // For all source columns, use these friendly names for the source
            // column names instead of the real column names.
            internal string[] FriendlyNames;

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
                if (!base.TryParse(str, out string extra))
                    return false;
                if (extra == null)
                    return true;

                if (!int.TryParse(extra, out int bits))
                    return false;
                HashBits = bits;
                return true;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (NgramLength != null || AllLengths != null || SkipLength != null || Seed != null ||
                    RehashUnigrams != null || Ordered != null || InvertHash != null)
                {
                    return false;
                }
                if (HashBits == null)
                    return TryUnparseCore(sb);

                string extra = HashBits.Value.ToString();
                return TryUnparseCore(sb, extra);
            }
        }

        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:hashBits:src)",
                ShortName = "col",
                SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum ngram length", ShortName = "ngram", SortOrder = 3)]
            public int NgramLength = 2;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Whether to include all ngram lengths up to " + nameof(NgramLength) + " or only " + nameof(NgramLength),
                ShortName = "all", SortOrder = 4)]
            public bool AllLengths = true;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Maximum number of tokens to skip when constructing an ngram",
                ShortName = "skips", SortOrder = 3)]
            public int SkipLength = 0;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Number of bits to hash into. Must be between 1 and 30, inclusive.",
                ShortName = "bits", SortOrder = 2)]
            public int HashBits = 16;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
            public uint Seed = 314489979;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to rehash unigrams", ShortName = "rehash")]
            public bool RehashUnigrams;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Whether the position of each source column should be included in the hash (when there are multiple source columns).",
                ShortName = "ord", SortOrder = 6)]
            public bool Ordered = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.",
                ShortName = "ih")]
            public int InvertHash;
        }

        private sealed class Bindings : ManyToOneColumnBindingsBase
        {
            public readonly VectorType[] Types;
            private readonly NgramHashTransform _parent;

            public Bindings(Arguments args, ISchema schemaInput, NgramHashTransform parent)
                : base(args.Column, schemaInput, TestTypes)
            {
                Types = new VectorType[args.Column.Length];
                _parent = parent;
            }

            public Bindings(ModelLoadContext ctx, ISchema schemaInput, NgramHashTransform parent)
                : base(ctx, schemaInput, TestTypes)
            {
                Types = new VectorType[Infos.Length];
                _parent = parent;
            }

            private static string TestTypes(ColumnType[] types)
            {
                const string reason = "Expected vector of Key type, and Key is convertable to U4";
                Contracts.AssertValue(types);
                for (int i = 0; i < types.Length; i++)
                {
                    var type = types[i];
                    if (!type.IsVector)
                        return reason;
                    if (!type.ItemType.IsKey)
                        return reason;
                    // Can only accept key types that can be converted to U4.
                    if (type.ItemType.KeyCount == 0 && type.ItemType.RawKind > DataKind.U4)
                        return reason;
                }

                return null;
            }

            protected override ColumnType GetColumnTypeCore(int iinfo)
            {
                Contracts.Assert(0 <= iinfo & iinfo < Infos.Length);
                Contracts.Assert(Types[iinfo] != null);
                return Types[iinfo];
            }

            protected override ColumnType GetMetadataTypeCore(string kind, int iinfo)
            {
                Contracts.AssertNonEmpty(kind);
                Contracts.Assert(0 <= iinfo && iinfo < Infos.Length);

                if (kind == MetadataUtils.Kinds.SlotNames && _parent._slotNamesTypes != null)
                    return _parent._slotNamesTypes[iinfo];
                return base.GetMetadataTypeCore(kind, iinfo);
            }

            protected override IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypesCore(int iinfo)
            {
                if (_parent._slotNamesTypes != null && _parent._slotNamesTypes[iinfo] != null)
                    return base.GetMetadataTypesCore(iinfo).Prepend(_parent._slotNamesTypes[iinfo].GetPair(MetadataUtils.Kinds.SlotNames));
                return base.GetMetadataTypesCore(iinfo);
            }

            protected override void GetMetadataCore<TValue>(string kind, int iinfo, ref TValue value)
            {
                if (kind == MetadataUtils.Kinds.SlotNames && _parent._slotNames != null && _parent._slotNames[iinfo].Length > 0)
                {
                    MetadataUtils.MetadataGetter<VBuffer<ReadOnlyMemory<char>>> getTerms = _parent.GetTerms;
                    getTerms.Marshal(iinfo, ref value);
                    return;
                }
                base.GetMetadataCore(kind, iinfo, ref value);
            }
        }

        private sealed class ColInfoEx
        {
            public readonly int NgramLength;
            public readonly int SkipLength;

            public readonly int HashBits;
            public readonly uint Seed;
            public readonly bool Rehash;
            public readonly bool Ordered;
            public readonly bool AllLengths;

            public ColInfoEx(Column item, Arguments args)
            {
                NgramLength = item.NgramLength ?? args.NgramLength;
                Contracts.CheckUserArg(0 < NgramLength && NgramLength <= NgramBufferBuilder.MaxSkipNgramLength, nameof(item.NgramLength));
                SkipLength = item.SkipLength ?? args.SkipLength;
                Contracts.CheckUserArg(0 <= SkipLength && SkipLength <= NgramBufferBuilder.MaxSkipNgramLength, nameof(item.SkipLength));
                if (NgramLength + SkipLength > NgramBufferBuilder.MaxSkipNgramLength)
                {
                    throw Contracts.ExceptUserArg(nameof(item.SkipLength),
                        "The sum of skipLength and ngramLength must be less than or equal to {0}",
                        NgramBufferBuilder.MaxSkipNgramLength);
                }

                HashBits = item.HashBits ?? args.HashBits;
                Contracts.CheckUserArg(1 <= HashBits && HashBits <= 30, nameof(item.HashBits));
                Seed = item.Seed ?? args.Seed;
                Rehash = item.RehashUnigrams ?? args.RehashUnigrams;
                Ordered = item.Ordered ?? args.Ordered;
                AllLengths = item.AllLengths ?? args.AllLengths;
            }

            public ColInfoEx(ModelLoadContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int: NgramLength
                // int: SkipLength
                // int: HashBits
                // uint: Seed
                // byte: Rehash
                // byte: Ordered
                // byte: AllLengths

                NgramLength = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(0 < NgramLength && NgramLength <= NgramBufferBuilder.MaxSkipNgramLength);
                SkipLength = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(0 <= SkipLength && SkipLength <= NgramBufferBuilder.MaxSkipNgramLength);
                Contracts.CheckDecode(SkipLength <= NgramBufferBuilder.MaxSkipNgramLength - NgramLength);
                HashBits = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(1 <= HashBits && HashBits <= 30);
                Seed = ctx.Reader.ReadUInt32();
                Rehash = ctx.Reader.ReadBoolByte();
                Ordered = ctx.Reader.ReadBoolByte();
                AllLengths = ctx.Reader.ReadBoolByte();
            }

            public void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int: NgramLength
                // int: SkipLength
                // int: HashBits
                // uint: Seed
                // byte: Rehash
                // byte: Ordered
                // byte: AllLengths

                Contracts.Assert(NgramLength > 0);
                ctx.Writer.Write(NgramLength);
                Contracts.Assert(SkipLength >= 0);
                ctx.Writer.Write(SkipLength);
                Contracts.Assert(1 <= HashBits && HashBits <= 30);
                ctx.Writer.Write(HashBits);
                ctx.Writer.Write(Seed);
                ctx.Writer.WriteBoolByte(Rehash);
                ctx.Writer.WriteBoolByte(Ordered);
                ctx.Writer.WriteBoolByte(AllLengths);
            }
        }

        internal const string Summary = "Produces a bag of counts of ngrams (sequences of consecutive values of length 1-n) in a given vector of keys. "
            + "It does so by hashing each ngram and using the hash value as the index in the bag.";

        public const string LoaderSignature = "NgramHashTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "HASHGRAM",
                // verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Invert hash key values, hash fix
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(NgramHashTransform).Assembly.FullName);
        }

        private readonly Bindings _bindings;
        private readonly ColInfoEx[] _exes;

        private readonly VBuffer<ReadOnlyMemory<char>>[] _slotNames;
        private readonly ColumnType[] _slotNamesTypes;

        private const string RegistrationName = "NgramHash";

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public NgramHashTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckUserArg(Utils.Size(args.Column) > 0, nameof(args.Column));

            _bindings = new Bindings(args, Source.Schema, this);
            _exes = new ColInfoEx[args.Column.Length];
            List<int> invertIinfos = null;
            int[] invertHashMaxCounts = new int[args.Column.Length];
            for (int iinfo = 0; iinfo < _exes.Length; iinfo++)
            {
                _exes[iinfo] = new ColInfoEx(args.Column[iinfo], args);
                var invertHashMaxCount = GetAndVerifyInvertHashMaxCount(args, args.Column[iinfo], _exes[iinfo]);
                if (invertHashMaxCount > 0)
                {
                    Utils.Add(ref invertIinfos, iinfo);
                    invertHashMaxCounts[iinfo] = invertHashMaxCount;
                }
            }

            InitColumnTypes();

            if (Utils.Size(invertIinfos) > 0)
            {
                // Build the invert hashes if we actually had any.
                var dstSrcs = new HashSet<int>(invertIinfos.Select(i => _bindings.MapIinfoToCol(i)));
                var inputPred = _bindings.GetDependencies(dstSrcs.Contains);
                var active = _bindings.GetActive(dstSrcs.Contains);
                string[][] friendlyNames = args.Column.Select(c => c.FriendlyNames).ToArray();
                var helper = new InvertHashHelper(this, friendlyNames, inputPred, invertHashMaxCounts);

                using (IRowCursor srcCursor = input.GetRowCursor(inputPred))
                using (var dstCursor = new RowCursor(this, srcCursor, active, helper.Decorate))
                {
                    var allGetters = InvertHashHelper.CallAllGetters(dstCursor);
                    while (dstCursor.MoveNext())
                        allGetters();
                }
                _slotNames = helper.SlotNamesMetadata(out _slotNamesTypes);
            }
        }

        private NgramHashTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, input)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // int: sizeof(Float)
            // bindings
            // for each added column
            //   ColInfoEx

            int cbFloat = ctx.Reader.ReadInt32();
            Host.CheckDecode(cbFloat == sizeof(Float));
            _bindings = new Bindings(ctx, Source.Schema, this);
            _exes = new ColInfoEx[_bindings.Infos.Length];

            for (int iinfo = 0; iinfo < _bindings.Infos.Length; iinfo++)
                _exes[iinfo] = new ColInfoEx(ctx);
            InitColumnTypes();
            TextModelHelper.LoadAll(Host, ctx, _exes.Length, out _slotNames, out _slotNamesTypes);
        }

        public static NgramHashTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new NgramHashTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // bindings
            // for each added column
            //   ColInfoEx

            ctx.Writer.Write(sizeof(Float));
            _bindings.Save(ctx);
            for (int iinfo = 0; iinfo < _exes.Length; iinfo++)
                _exes[iinfo].Save(ctx);
            TextModelHelper.SaveAll(Host, ctx, _exes.Length, _slotNames);
        }

        private void InitColumnTypes()
        {
            for (int iinfo = 0; iinfo < _exes.Length; iinfo++)
                _bindings.Types[iinfo] = new VectorType(NumberType.Float, 1 << _exes[iinfo].HashBits);
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

        private void GetTerms(int iinfo, ref VBuffer<ReadOnlyMemory<char>> dst)
        {
            Host.Assert(0 <= iinfo && iinfo < _exes.Length);
            Host.Assert(_slotNames[iinfo].Length > 0);
            _slotNames[iinfo].CopyTo(ref dst);
        }

        private NgramIdFinder GetNgramIdFinder(int iinfo)
        {
            uint mask = (1U << _exes[iinfo].HashBits) - 1;
            int ngramLength = _exes[iinfo].NgramLength;
            bool rehash = _exes[iinfo].Rehash;
            bool ordered = _exes[iinfo].Ordered;
            bool all = _exes[iinfo].AllLengths;
            uint seed = _exes[iinfo].Seed;

            // REVIEW: Consider the case when:
            // * The source key type has count == 2^n, where n is the number of hash bits
            // * rehash == false
            // * ngramLength == 1
            // * ordered == false
            // Then one might expect that this produces the same result as KeyToVector with bagging.
            // However, this shifts everything by one slot, and both the NA values and the original
            // last hash slot get mapped to the first slot.
            // One possible solution:
            // * KeyToVector should have an option to add a slot (the zeroth slot) for NA
            // * NgramHash should, when rehash is false (and perhaps when ordered is false?),
            //   add an extra slot at the beginning for the count of unigram missings.
            // Alternatively, this could drop unigram NA values.

            if (!all && ngramLength > 1)
            {
                if (ordered)
                {
                    // !allLengths, ordered, value of rehash doesn't matter.
                    return
                        (uint[] ngram, int lim, int icol, ref bool more) =>
                        {
                            AssertValid(ngram, ngramLength, lim, icol);
                            if (lim < ngramLength)
                                return -1;
                            var hash = Hashing.MurmurHash(seed, ngram, 0, lim);
                            if (icol > 0)
                                hash = Hashing.MurmurRound(hash, (uint)icol);
                            return (int)(Hashing.MixHash(hash) & mask);
                        };
                }
                else
                {
                    // !allLengths, !ordered, value of rehash doesn't matter.
                    return
                        (uint[] ngram, int lim, int icol, ref bool more) =>
                        {
                            AssertValid(ngram, ngramLength, lim, icol);
                            if (lim < ngramLength)
                                return -1;
                            return (int)(Hashing.MurmurHash(seed, ngram, 0, lim) & mask);
                        };
                }
            }
            else if (rehash)
            {
                if (ordered)
                {
                    // allLengths, rehash, ordered
                    return
                        (uint[] ngram, int lim, int icol, ref bool more) =>
                        {
                            AssertValid(ngram, ngramLength, lim, icol);
                            var hash = Hashing.MurmurHash(seed, ngram, 0, lim);
                            if (icol > 0)
                                hash = Hashing.MurmurRound(hash, (uint)icol);
                            return (int)(Hashing.MixHash(hash) & mask);
                        };
                }
                else
                {
                    // allLengths, rehash, !ordered
                    return
                        (uint[] ngram, int lim, int icol, ref bool more) =>
                        {
                            AssertValid(ngram, ngramLength, lim, icol);
                            return (int)(Hashing.MurmurHash(seed, ngram, 0, lim) & mask);
                        };
                }
            }
            else if (ngramLength > 1)
            {
                if (ordered)
                {
                    // allLengths, !rehash, ordered
                    return
                        (uint[] ngram, int lim, int icol, ref bool more) =>
                        {
                            AssertValid(ngram, ngramLength, lim, icol);
                            uint hash;
                            if (lim == 1)
                                hash = ngram[0];
                            else
                                hash = Hashing.MurmurHash(seed, ngram, 0, lim);
                            if (icol > 0)
                                hash = Hashing.MurmurRound(hash, (uint)icol);
                            return (int)(Hashing.MixHash(hash) & mask);
                        };
                }
                else
                {
                    // allLengths, !rehash, !ordered
                    return
                        (uint[] ngram, int lim, int icol, ref bool more) =>
                        {
                            AssertValid(ngram, ngramLength, lim, icol);
                            if (lim == 1)
                                return (int)(ngram[0] & mask);
                            return (int)(Hashing.MurmurHash(seed, ngram, 0, lim) & mask);
                        };
                }
            }
            else
            {
                if (ordered)
                {
                    // ngramLength==1, !rehash, ordered
                    return
                        (uint[] ngram, int lim, int icol, ref bool more) =>
                        {
                            AssertValid(ngram, ngramLength, lim, icol);
                            uint hash = ngram[0];
                            if (icol > 0)
                                hash = Hashing.MurmurRound(hash, (uint)icol);
                            return (int)(Hashing.MixHash(hash) & mask);
                        };
                }
                else
                {
                    // ngramLength==1, !rehash, !ordered
                    return
                        (uint[] ngram, int lim, int icol, ref bool more) =>
                        {
                            AssertValid(ngram, ngramLength, lim, icol);
                            return (int)(ngram[0] & mask);
                        };
                }
            }
        }

        [Conditional("DEBUG")]
        private void AssertValid(uint[] ngram, int ngramLength, int lim, int icol)
        {
            Host.Assert(0 <= lim && lim <= Utils.Size(ngram));
            Host.Assert(Utils.Size(ngram) == ngramLength);
            Host.Assert(icol >= 0);
        }

        public override ISchema Schema { get { return _bindings; } }

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate, "predicate");

            // Prefer parallel cursors iff some of our columns are active, otherwise, don't care.
            if (_bindings.AnyNewColumnsActive(predicate))
                return true;
            return null;
        }

        protected override IRowCursor GetRowCursorCore(Func<int, bool> predicate, IRandom rand = null)
        {
            Host.AssertValue(predicate, "predicate");
            Host.AssertValueOrNull(rand);

            var inputPred = _bindings.GetDependencies(predicate);
            var active = _bindings.GetActive(predicate);
            var input = Source.GetRowCursor(inputPred, rand);
            return new RowCursor(this, input, active);
        }

        public sealed override IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator,
            Func<int, bool> predicate, int n, IRandom rand = null)
        {
            Host.CheckValue(predicate, nameof(predicate));
            Host.CheckValueOrNull(rand);

            var inputPred = _bindings.GetDependencies(predicate);
            var active = _bindings.GetActive(predicate);
            var inputs = Source.GetRowCursorSet(out consolidator, inputPred, n, rand);
            Host.AssertNonEmpty(inputs);

            if (inputs.Length == 1 && n > 1 && _bindings.AnyNewColumnsActive(predicate))
                inputs = DataViewUtils.CreateSplitCursors(out consolidator, Host, inputs[0], n);
            Host.AssertNonEmpty(inputs);

            var cursors = new IRowCursor[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                cursors[i] = new RowCursor(this, inputs[i], active);
            return cursors;
        }

        protected override Func<int, bool> GetDependenciesCore(Func<int, bool> predicate)
        {
            return _bindings.GetDependencies(predicate);
        }

        protected override Delegate[] CreateGetters(IRow input, Func<int, bool> active, out Action disp)
        {
            Func<int, bool> activeInfos =
                iinfo =>
                {
                    int col = _bindings.MapIinfoToCol(iinfo);
                    return active(col);
                };

            var getters = new Delegate[_bindings.InfoCount];
            disp = null;
            using (var ch = Host.Start("CreateGetters"))
            {
                for (int iinfo = 0; iinfo < _bindings.InfoCount; iinfo++)
                {
                    if (!activeInfos(iinfo))
                        continue;
                    getters[iinfo] = MakeGetter(ch, input, iinfo);
                }
                return getters;
            }
        }

        protected override int MapColumnIndex(out bool isSrc, int col)
        {
            return _bindings.MapColumnIndex(out isSrc, col);
        }

        private Delegate MakeGetter(IChannel ch, IRow input, int iinfo, FinderDecorator decorator = null)
        {
            ch.Assert(_bindings.Infos[iinfo].SrcTypes.All(t => t.IsVector && t.ItemType.IsKey));

            var info = _bindings.Infos[iinfo];
            int srcCount = info.SrcIndices.Length;
            ValueGetter<VBuffer<uint>>[] getSrc = new ValueGetter<VBuffer<uint>>[srcCount];
            for (int isrc = 0; isrc < srcCount; isrc++)
                getSrc[isrc] = RowCursorUtils.GetVecGetterAs<uint>(NumberType.U4, input, info.SrcIndices[isrc]);
            var src = default(VBuffer<uint>);
            var ngramIdFinder = GetNgramIdFinder(iinfo);
            if (decorator != null)
                ngramIdFinder = decorator(iinfo, ngramIdFinder);
            var bldr = new NgramBufferBuilder(_exes[iinfo].NgramLength, _exes[iinfo].SkipLength,
                _bindings.Types[iinfo].ValueCount, ngramIdFinder);
            var keyCounts = _bindings.Infos[iinfo].SrcTypes.Select(
                t => t.ItemType.KeyCount > 0 ? (uint)t.ItemType.KeyCount : uint.MaxValue).ToArray();

            // REVIEW: Special casing the srcCount==1 case could potentially improve perf.
            ValueGetter<VBuffer<Float>> del =
                (ref VBuffer<Float> dst) =>
                {
                    bldr.Reset();
                    for (int i = 0; i < srcCount; i++)
                    {
                        getSrc[i](ref src);
                        bldr.AddNgrams(ref src, i, keyCounts[i]);
                    }
                    bldr.GetResult(ref dst);
                };
            return del;
        }

        private delegate NgramIdFinder FinderDecorator(int iinfo, NgramIdFinder finder);

        private sealed class RowCursor : SynchronizedCursorBase<IRowCursor>, IRowCursor
        {
            private readonly Bindings _bindings;
            private readonly bool[] _active;
            private readonly Delegate[] _getters;

            public ISchema Schema { get { return _bindings; } }

            public RowCursor(NgramHashTransform parent, IRowCursor input, bool[] active, FinderDecorator decorator = null)
                : base(parent.Host, input)
            {
                Ch.AssertValue(parent);
                Ch.Assert(active == null || active.Length == parent._bindings.ColumnCount);
                Ch.AssertValueOrNull(decorator);

                _bindings = parent._bindings;
                _active = active;

                _getters = new Delegate[_bindings.Infos.Length];
                for (int iinfo = 0; iinfo < _bindings.Infos.Length; iinfo++)
                {
                    if (IsIndexActive(iinfo))
                        _getters[iinfo] = parent.MakeGetter(Ch, Input, iinfo, decorator);
                }
            }

            private bool IsIndexActive(int iinfo)
            {
                Ch.Assert(0 <= iinfo & iinfo < _bindings.Infos.Length);
                return _active == null || _active[_bindings.MapIinfoToCol(iinfo)];
            }

            public bool IsColumnActive(int col)
            {
                Ch.Check(0 <= col && col < _bindings.ColumnCount);
                return _active == null || _active[col];
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.Check(IsColumnActive(col));

                int index = _bindings.MapColumnIndex(out bool isSrc, col);
                if (isSrc)
                    return Input.GetGetter<TValue>(index);

                Ch.Assert(_getters[index] != null);
                var fn = _getters[index] as ValueGetter<TValue>;
                if (fn == null)
                    throw Ch.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                return fn;
            }

            private ValueGetter<T> GetSrcGetter<T>(int iinfo, int isrc)
            {
                Ch.Assert(IsIndexActive(iinfo));
                return Input.GetGetter<T>(_bindings.Infos[iinfo].SrcIndices[isrc]);
            }
        }

        private sealed class InvertHashHelper
        {
            private readonly NgramHashTransform _parent;
            // One per output column (will be null if invert hashing is not specified for
            // this column).
            private readonly InvertHashCollector<NGram>[] _iinfoToCollector;
            // One per source column that we want to convert (will be null if we don't hash
            // them in a column where we've said we want invert hashing).
            private readonly ValueMapper<uint, StringBuilder>[] _srcTextGetters;
            // If null, or specific element is null, then just use the input column name.
            private readonly string[][] _friendlyNames;
            private readonly int[] _invertHashMaxCounts;

            public InvertHashHelper(NgramHashTransform parent, string[][] friendlyNames, Func<int, bool> inputPred, int[] invertHashMaxCounts)
            {
                Contracts.AssertValue(parent);
                Contracts.AssertValue(friendlyNames);
                Contracts.Assert(friendlyNames.Length == parent._bindings.InfoCount);
                Contracts.AssertValue(inputPred);
                Contracts.AssertValue(invertHashMaxCounts);
                Contracts.Assert(invertHashMaxCounts.Length == parent._bindings.InfoCount);
                _parent = parent;
                // One per iinfo (some may be null).
                _iinfoToCollector = new InvertHashCollector<NGram>[_parent._bindings.InfoCount];
                // One per source column (some may be null).
                _srcTextGetters = new ValueMapper<uint, StringBuilder>[_parent.Source.Schema.ColumnCount];
                _invertHashMaxCounts = invertHashMaxCounts;
                for (int i = 0; i < _srcTextGetters.Length; ++i)
                {
                    if (inputPred(i))
                        _srcTextGetters[i] = InvertHashUtils.GetSimpleMapper<uint>(_parent.Source.Schema, i);
                }
                _friendlyNames = friendlyNames;
            }

            /// <summary>
            /// Construct an action that calls all the getters for a row, so as to easily force computation
            /// of lazily computed values. This will have the side effect of calling the decorator.
            /// </summary>
            public static Action CallAllGetters(IRow row)
            {
                var colCount = row.Schema.ColumnCount;
                List<Action> getters = new List<Action>();
                for (int c = 0; c < colCount; ++c)
                {
                    if (row.IsColumnActive(c))
                        getters.Add(GetNoOpGetter(row, c));
                }
                var gettersArray = getters.ToArray();
                return
                    () =>
                    {
                        for (int i = 0; i < gettersArray.Length; ++i)
                            gettersArray[i]();
                    };
            }

            private static Action GetNoOpGetter(IRow row, int col)
            {
                Func<IRow, int, Action> func = GetNoOpGetter<int>;
                var meth = func.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(row.Schema.GetColumnType(col).RawType);
                return (Action)meth.Invoke(null, new object[] { row, col });
            }

            private static Action GetNoOpGetter<T>(IRow row, int col)
            {
                T value = default(T);
                var getter = row.GetGetter<T>(col);
                return () => getter(ref value);
            }

            private sealed class NGram : IEquatable<NGram>
            {
                // Items [0,Lim) hold the ngram hashed inputs.
                public readonly uint[] Grams;
                // The number of items in Grams that are actually valid.
                public readonly int Lim;
                // The per output column index into the source column array.
                public readonly int ISrcCol;

                public NGram(uint[] ngram, int lim, int srcCol)
                {
                    Contracts.AssertValue(ngram);
                    Contracts.Assert(1 <= lim & lim <= ngram.Length);
                    Contracts.Assert(0 <= srcCol);

                    Grams = ngram;
                    Lim = lim;
                    ISrcCol = srcCol;
                }

                public NGram Clone()
                {
                    var ngram = new uint[Lim];
                    Array.Copy(Grams, ngram, Lim);
                    return new NGram(ngram, Lim, ISrcCol);
                }

                public bool Equals(NGram other)
                {
                    // Note that ISrcCol is a little funky. It is not necessarily part of the
                    // hash that the transform produces, and it is an index per output column,
                    // so across two output columns two ISrcCols are not comparable. This means
                    // However, we are collating the descriptors per output column, so we
                    // accept this slight funkiness.
                    if (other != null && other.Lim == Lim && other.ISrcCol == ISrcCol)
                    {
                        for (int i = 0; i < Lim; ++i)
                        {
                            if (other.Grams[i] != Grams[i])
                                return false;
                        }
                        return true;
                    }
                    return false;
                }

                public override int GetHashCode()
                {
                    int hash = Lim;
                    hash = Hashing.CombineHash(hash, ISrcCol);
                    for (int i = 0; i < Lim; ++i)
                        hash = Hashing.CombineHash(hash, (int)Grams[i]);
                    return hash;
                }
            }

            private static void ClearDst(ref StringBuilder dst)
            {
                if (dst == null)
                    dst = new StringBuilder();
                else
                    dst.Clear();
            }

            public NgramIdFinder Decorate(int iinfo, NgramIdFinder finder)
            {
                Contracts.Assert(0 <= iinfo && iinfo < _parent._bindings.InfoCount);
                Contracts.Assert(_iinfoToCollector[iinfo] == null);
                Contracts.AssertValue(finder);

                var srcIndices = _parent._bindings.Infos[iinfo].SrcIndices;

                // Define the mapper from the ngram, to text.
                ValueMapper<NGram, StringBuilder> stringMapper;
                StringBuilder temp = null;
                char[] buffer = null;

                if (srcIndices.Length == 1)
                {
                    // No need to include the column name. This will just be "A" or "(A,B,C)" depending
                    // on the n-arity of the ngram.
                    var srcMap = _srcTextGetters[srcIndices[0]];
                    Contracts.AssertValue(srcMap);

                    stringMapper =
                        (ref NGram src, ref StringBuilder dst) =>
                        {
                            Contracts.Assert(src.ISrcCol == 0);
                            if (src.Lim == 1)
                            {
                                srcMap(ref src.Grams[0], ref dst);
                                return;
                            }
                            ClearDst(ref dst);
                            for (int i = 0; i < src.Lim; ++i)
                            {
                                if (i > 0)
                                    dst.Append('|');
                                srcMap(ref src.Grams[i], ref temp);
                                InvertHashUtils.AppendToEnd(temp, dst, ref buffer);
                            }
                        };
                }
                else
                {
                    Contracts.Assert(srcIndices.Length > 1);
                    string[] srcNames = _friendlyNames[iinfo];
                    if (srcNames == null)
                    {
                        srcNames = new string[srcIndices.Length];
                        for (int i = 0; i < srcIndices.Length; ++i)
                            srcNames[i] = _parent.Source.Schema.GetColumnName(srcIndices[i]);
                    }
                    Contracts.Assert(Utils.Size(srcNames) == srcIndices.Length);
                    string[] friendlyNames = _friendlyNames?[iinfo];
                    // We need to disambiguate the column name. This will be the same as the above format,
                    // just instead of "<Stuff>" it would be with "ColumnName:<Stuff>".
                    stringMapper =
                        (ref NGram src, ref StringBuilder dst) =>
                        {
                            var srcMap = _srcTextGetters[srcIndices[src.ISrcCol]];
                            Contracts.AssertValue(srcMap);
                            ClearDst(ref dst);
                            dst.Append(srcNames[src.ISrcCol]);
                            dst.Append(':');
                            for (int i = 0; i < src.Lim; ++i)
                            {
                                if (i > 0)
                                    dst.Append('|');
                                srcMap(ref src.Grams[i], ref temp);
                                InvertHashUtils.AppendToEnd(temp, dst, ref buffer);
                            }
                        };
                }

                var collector = _iinfoToCollector[iinfo] = new InvertHashCollector<NGram>(
                    _parent._bindings.Types[iinfo].VectorSize, _invertHashMaxCounts[iinfo],
                    stringMapper, EqualityComparer<NGram>.Default, (ref NGram src, ref NGram dst) => dst = src.Clone());

                return
                    (uint[] ngram, int lim, int icol, ref bool more) =>
                    {
                        Contracts.Assert(0 <= icol && icol < srcIndices.Length);
                        Contracts.AssertValue(_srcTextGetters[srcIndices[icol]]);
                        var result = finder(ngram, lim, icol, ref more);
                        // For the hashing NgramIdFinder, a result of -1 indicates that
                        // a slot does not exist for the given ngram. We do not pass ngrams
                        // that do not have a slot to the InvertHash collector.
                        if (result != -1)
                        {
                            // The following ngram is "unsafe", in that the ngram array is actually
                            // re-used. The collector will utilize its copier to make it safe, in
                            // the event that this is a key it needs to keep.
                            var ngramObj = new NGram(ngram, lim, icol);
                            collector.Add(result, ngramObj);
                        }
                        return result;
                    };
            }

            public VBuffer<ReadOnlyMemory<char>>[] SlotNamesMetadata(out ColumnType[] types)
            {
                var values = new VBuffer<ReadOnlyMemory<char>>[_iinfoToCollector.Length];
                types = new ColumnType[_iinfoToCollector.Length];
                for (int iinfo = 0; iinfo < _iinfoToCollector.Length; ++iinfo)
                {
                    if (_iinfoToCollector[iinfo] != null)
                    {
                        var vec = values[iinfo] = _iinfoToCollector[iinfo].GetMetadata();
                        Contracts.Assert(vec.Length == _parent._bindings.Types[iinfo].VectorSize);
                        types[iinfo] = new VectorType(TextType.Instance, vec.Length);
                    }
                }
                return values;
            }
        }
    }
}
