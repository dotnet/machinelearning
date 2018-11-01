// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Transforms.Text;

[assembly: LoadableClass(NgramTransform.Summary, typeof(NgramTransform), typeof(NgramTransform.Arguments), typeof(SignatureDataTransform),
    "Ngram Transform", "NgramTransform", "Ngram")]

[assembly: LoadableClass(NgramTransform.Summary, typeof(NgramTransform), null, typeof(SignatureLoadDataTransform),
    "Ngram Transform", NgramTransform.LoaderSignature)]

namespace Microsoft.ML.Transforms.Text
{
    using Conditional = System.Diagnostics.ConditionalAttribute;

    public sealed class NgramTransform : OneToOneTransformBase
    {
        /// <summary>
        /// Weighting criteria: a statistical measure used to evaluate how important a word is to a document in a corpus.
        /// This enumeration is serialized.
        /// </summary>
        public enum WeightingCriteria
        {
            [EnumValueDisplay("TF (Term Frequency)")]
            Tf = 0,

            [EnumValueDisplay("IDF (Inverse Document Frequency)")]
            Idf = 1,

            [EnumValueDisplay("TF-IDF")]
            TfIdf = 2
        }

        public sealed class Column : OneToOneColumn
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

            [Argument(ArgumentType.Multiple, HelpText = "Maximum number of ngrams to store in the dictionary", ShortName = "max")]
            public int[] MaxNumTerms = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Statistical measure used to evaluate how important a word is to a document in a corpus")]
            public WeightingCriteria? Weighting;

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
                if (NgramLength != null || AllLengths != null || SkipLength != null || Utils.Size(MaxNumTerms) != 0)
                    return false;
                return TryUnparseCore(sb);
            }
        }

        public sealed class Arguments : TransformInputBase
        {
            internal const int DefaultMaxTerms = 10000000;

            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum ngram length", ShortName = "ngram")]
            public int NgramLength = 2;

            [Argument(ArgumentType.AtMostOnce, HelpText =
                "Whether to store all ngram lengths up to ngramLength, or only ngramLength", ShortName = "all")]
            public bool AllLengths = true;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Maximum number of tokens to skip when constructing an ngram",
                ShortName = "skips")]
            public int SkipLength = 0;

            [Argument(ArgumentType.Multiple, HelpText = "Maximum number of ngrams to store in the dictionary", ShortName = "max")]
            public int[] MaxNumTerms = new int[] { DefaultMaxTerms };

            [Argument(ArgumentType.AtMostOnce, HelpText = "The weighting criteria")]
            public WeightingCriteria Weighting = WeightingCriteria.Tf;
        }

        private sealed class ColInfoEx
        {
            // Position i, indicates whether the pool contains any (i+1)-grams
            public readonly bool[] NonEmptyLevels;

            public readonly int NgramLength;
            public readonly int SkipLength;

            public readonly WeightingCriteria Weighting;

            public bool RequireIdf()
            {
                return Weighting == WeightingCriteria.Idf || Weighting == WeightingCriteria.TfIdf;
            }

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
                Contracts.CheckUserArg(Enum.IsDefined(typeof(WeightingCriteria), args.Weighting), nameof(args.Weighting));
                Weighting = item.Weighting ?? args.Weighting;

                NonEmptyLevels = new bool[NgramLength];
            }

            public ColInfoEx(ModelLoadContext ctx, bool readWeighting)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int: NgramLength
                // int: SkipLength
                // int: Weighting Criteria (if readWeighting == true)
                // bool[NgramLength]: NonEmptyLevels

                NgramLength = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(0 < NgramLength && NgramLength <= NgramBufferBuilder.MaxSkipNgramLength);
                SkipLength = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(0 <= SkipLength && SkipLength <= NgramBufferBuilder.MaxSkipNgramLength);
                Contracts.CheckDecode(NgramLength <= NgramBufferBuilder.MaxSkipNgramLength - SkipLength);

                if (readWeighting)
                    Weighting = (WeightingCriteria)ctx.Reader.ReadInt32();
                Contracts.CheckDecode(Enum.IsDefined(typeof(WeightingCriteria), Weighting));
                NonEmptyLevels = ctx.Reader.ReadBoolArray(NgramLength);
            }

            public void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int: NgramLength
                // int: SkipLength
                // int: Weighting Criteria
                // bool[NgramLength]: NonEmptyLevels

                Contracts.Assert(0 < NgramLength && NgramLength <= NgramBufferBuilder.MaxSkipNgramLength);
                ctx.Writer.Write(NgramLength);
                Contracts.Assert(0 <= SkipLength && SkipLength <= NgramBufferBuilder.MaxSkipNgramLength);
                Contracts.Assert(NgramLength + SkipLength <= NgramBufferBuilder.MaxSkipNgramLength);
                ctx.Writer.Write(SkipLength);
                Contracts.Assert(Enum.IsDefined(typeof(WeightingCriteria), Weighting));
                ctx.Writer.Write((int)Weighting);
                Contracts.Assert(Utils.Size(NonEmptyLevels) == NgramLength);
                ctx.Writer.WriteBoolBytesNoCount(NonEmptyLevels, NgramLength);
            }
        }

        private const uint VerTfIdfSupported = 0x00010002;

        public const string LoaderSignature = "NgramTransform";
        internal const string Summary = "Produces a bag of counts of ngrams (sequences of consecutive values of length 1-n) in a given vector of keys. "
            + "It does so by building a dictionary of ngrams and using the id in the dictionary as the index in the bag.";

        internal const string UserName = "NGram Transform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "NGRAMTRN",
                // verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Add support for TF-IDF
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(NgramTransform).Assembly.FullName);
        }

        private readonly VectorType[] _types;

        // REVIEW: The slot names types are not really needed. They are only used to "remember" which new
        // columns have slot names.
        private readonly VectorType[] _slotNamesTypes;

        private readonly ColInfoEx[] _exes;
        // These contain the ngram maps
        private readonly SequencePool[] _ngramMaps;

        // Ngram inverse document frequencies
        private readonly double[][] _invDocFreqs;

        private const string RegistrationName = "Ngram";

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public NgramTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, Contracts.CheckRef(args, nameof(args)).Column, input, TestType)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Utils.Size(Infos) == Utils.Size(args.Column));

            _exes = new ColInfoEx[Infos.Length];
            for (int iinfo = 0; iinfo < _exes.Length; iinfo++)
                _exes[iinfo] = new ColInfoEx(args.Column[iinfo], args);

            _ngramMaps = Train(args, input, out _invDocFreqs);

            InitColumnTypeAndMetadata(out _types, out _slotNamesTypes);
        }

        private NgramTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestType)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            // for each column
            //   ColInfoEx
            //   the ngram SequencePool
            //   the ngram inverse document frequencies

            _exes = new ColInfoEx[Infos.Length];
            _ngramMaps = new SequencePool[Infos.Length];
            _invDocFreqs = new double[Infos.Length][];
            for (int i = 0; i < Infos.Length; i++)
            {
                _exes[i] = new ColInfoEx(ctx, ctx.Header.ModelVerWritten >= VerTfIdfSupported);
                _ngramMaps[i] = new SequencePool(ctx.Reader);

                if (ctx.Header.ModelVerWritten >= VerTfIdfSupported)
                {
                    _invDocFreqs[i] = ctx.Reader.ReadDoubleArray();
                    for (int j = 0; j < Utils.Size(_invDocFreqs[i]); j++)
                        Host.CheckDecode(_invDocFreqs[i][j] >= 0);
                }
            }

            InitColumnTypeAndMetadata(out _types, out _slotNamesTypes);
        }

        public static NgramTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model",
                ch =>
                {
                    // *** Binary format ***
                    // int: sizeof(Float)
                    // <remainder handled in ctors>
                    int cbFloat = ctx.Reader.ReadInt32();
                    ch.CheckDecode(cbFloat == sizeof(Float));
                    return new NgramTransform(h, ctx, input);
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
            //   ColInfoEx
            //   the ngram SequencePool
            //   the ngram inverse document frequencies

            ctx.Writer.Write(sizeof(Float));
            SaveBase(ctx);
            var ngramsNames = default(VBuffer<ReadOnlyMemory<char>>);
            for (int i = 0; i < _exes.Length; i++)
            {
                _exes[i].Save(ctx);
                _ngramMaps[i].Save(ctx.Writer);
                ctx.Writer.WriteDoubleArray(_invDocFreqs[i]);

                if (_slotNamesTypes[i] != null)
                {
                    GetSlotNames(i, ref ngramsNames);
                    Host.Assert(_ngramMaps[i].Count == ngramsNames.Count);
                    Host.Assert(ngramsNames.IsDense);
                    ctx.SaveTextStream(string.Format("{0}-ngrams.txt", Infos[i].Name),
                        writer =>
                        {
                            writer.WriteLine("# Number of Ngrams terms = {0}", ngramsNames.Count);
                            for (int j = 0; j < ngramsNames.Count; j++)
                                writer.WriteLine("{0}\t{1}", j, ngramsNames.Values[j]);
                        });
                }
            }
        }

        private static string TestType(ColumnType type)
        {
            const string reason = "Expected vector of Key type, and Key is convertable to U4";
            Contracts.AssertValue(type);
            if (!type.IsVector)
                return reason;
            if (!type.ItemType.IsKey)
                return reason;
            // Can only accept key types that can be converted to U4.
            if (type.ItemType.KeyCount == 0 && type.ItemType.RawKind > DataKind.U4)
                return reason;
            return null;
        }

        private void InitColumnTypeAndMetadata(out VectorType[] types, out VectorType[] slotNamesTypes)
        {
            types = new VectorType[Infos.Length];
            slotNamesTypes = new VectorType[Infos.Length];

            var md = Metadata;
            for (int iinfo = 0; iinfo < _exes.Length; iinfo++)
            {
                types[iinfo] = new VectorType(NumberType.Float, _ngramMaps[iinfo].Count);
                var info = Infos[iinfo];
                if (!Source.Schema.HasKeyNames(info.Source, info.TypeSrc.ItemType.KeyCount))
                    continue;

                using (var bldr = md.BuildMetadata(iinfo))
                {
                    if (_ngramMaps[iinfo].Count > 0)
                    {
                        slotNamesTypes[iinfo] = new VectorType(TextType.Instance, _ngramMaps[iinfo].Count);
                        bldr.AddGetter<VBuffer<ReadOnlyMemory<char>>>(MetadataUtils.Kinds.SlotNames,
                            slotNamesTypes[iinfo], GetSlotNames);
                    }
                }
            }
            md.Seal();
        }

        private void GetSlotNames(int iinfo, ref VBuffer<ReadOnlyMemory<char>> dst)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            Host.Assert(_slotNamesTypes[iinfo] != null);

            var keyCount = Infos[iinfo].TypeSrc.ItemType.KeyCount;
            Host.Assert(Source.Schema.HasKeyNames(Infos[iinfo].Source, keyCount));

            var unigramNames = new VBuffer<ReadOnlyMemory<char>>();

            // Get the key values of the unigrams.
            Source.Schema.GetMetadata(MetadataUtils.Kinds.KeyValues, Infos[iinfo].Source, ref unigramNames);
            Host.Check(unigramNames.Length == keyCount);

            var pool = _ngramMaps[iinfo];
            var values = dst.Values;

            var ngramCount = pool.Count;
            if (Utils.Size(values) < ngramCount)
                Array.Resize(ref values, ngramCount);

            StringBuilder sb = new StringBuilder();
            uint[] ngram = new uint[_exes[iinfo].NgramLength];
            for (int slot = 0; slot < pool.Count; slot++)
            {
                var n = pool.GetById(slot, ref ngram);
                Host.Assert(n >= 0);

                // Get the unigrams composing the current ngram.
                ComposeNgramString(ngram, n, sb, keyCount,
                    unigramNames.GetItemOrDefault);
                values[slot] = sb.ToString().AsMemory();
            }

            dst = new VBuffer<ReadOnlyMemory<char>>(ngramCount, values, dst.Indices);
        }

        private delegate void TermGetter(int index, ref ReadOnlyMemory<char> term);

        private void ComposeNgramString(uint[] ngram, int count, StringBuilder sb, int keyCount, TermGetter termGetter)
        {
            Host.AssertValue(sb);
            Host.AssertValue(ngram);
            Host.Assert(keyCount > 0);

            sb.Clear();
            ReadOnlyMemory<char> term = default;
            string sep = "";
            for (int iterm = 0; iterm < count; iterm++)
            {
                sb.Append(sep);
                sep = "|";
                var unigram = ngram[iterm];
                if (unigram <= 0 || unigram > keyCount)
                    sb.Append("*");
                else
                {
                    termGetter((int)unigram - 1, ref term);
                    sb.AppendMemory(term);
                }
            }
        }

        private SequencePool[] Train(Arguments args, IDataView trainingData, out double[][] invDocFreqs)
        {
            // Contains the maximum number of grams to store in the dictionary, for each level of ngrams,
            // from 1 (in position 0) up to ngramLength (in position ngramLength-1)
            var lims = new int[Infos.Length][];
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
            {
                var all = args.Column[iinfo].AllLengths ?? args.AllLengths;
                var ngramLength = _exes[iinfo].NgramLength;
                var maxNumTerms = Utils.Size(args.Column[iinfo].MaxNumTerms) > 0 ? args.Column[iinfo].MaxNumTerms : args.MaxNumTerms;
                if (!all)
                {
                    Host.CheckUserArg(Utils.Size(maxNumTerms) == 0 ||
                        Utils.Size(maxNumTerms) == 1 && maxNumTerms[0] > 0, nameof(args.MaxNumTerms));
                    lims[iinfo] = new int[ngramLength];
                    lims[iinfo][ngramLength - 1] = Utils.Size(maxNumTerms) == 0 ? Arguments.DefaultMaxTerms : maxNumTerms[0];
                }
                else
                {
                    Host.CheckUserArg(Utils.Size(maxNumTerms) <= ngramLength, nameof(args.MaxNumTerms));
                    Host.CheckUserArg(Utils.Size(maxNumTerms) == 0 || maxNumTerms.All(i => i >= 0) && maxNumTerms[maxNumTerms.Length - 1] > 0, nameof(args.MaxNumTerms));
                    var extend = Utils.Size(maxNumTerms) == 0 ? Arguments.DefaultMaxTerms : maxNumTerms[maxNumTerms.Length - 1];
                    lims[iinfo] = Utils.BuildArray(ngramLength,
                        i => i < Utils.Size(maxNumTerms) ? maxNumTerms[i] : extend);
                }
            }

            var helpers = new NgramBufferBuilder[Infos.Length];
            var getters = new ValueGetter<VBuffer<uint>>[Infos.Length];
            var src = new VBuffer<uint>[Infos.Length];

            // Keep track of how many grams are in the pool for each value of n. Position
            // i in _counts counts how many (i+1)-grams are in the pool for column iinfo.
            var counts = new int[Infos.Length][];
            var ngramMaps = new SequencePool[Infos.Length];
            bool[] activeInput = new bool[trainingData.Schema.ColumnCount];
            foreach (var info in Infos)
                activeInput[info.Source] = true;
            using (var cursor = trainingData.GetRowCursor(col => activeInput[col]))
            using (var pch = Host.StartProgressChannel("Building n-gram dictionary"))
            {
                for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
                {
                    Host.Assert(Infos[iinfo].TypeSrc.IsVector && Infos[iinfo].TypeSrc.ItemType.IsKey);
                    var ngramLength = _exes[iinfo].NgramLength;
                    var skipLength = _exes[iinfo].SkipLength;

                    getters[iinfo] = RowCursorUtils.GetVecGetterAs<uint>(NumberType.U4, cursor, Infos[iinfo].Source);
                    src[iinfo] = default(VBuffer<uint>);
                    counts[iinfo] = new int[ngramLength];
                    ngramMaps[iinfo] = new SequencePool();

                    // Note: GetNgramIdFinderAdd will control how many ngrams of a specific length will
                    // be added (using lims[iinfo]), therefore we set slotLim to the maximum
                    helpers[iinfo] = new NgramBufferBuilder(ngramLength, skipLength, Utils.ArrayMaxSize,
                        GetNgramIdFinderAdd(counts[iinfo], lims[iinfo], ngramMaps[iinfo], _exes[iinfo].RequireIdf(), Host));
                }

                int cInfoFull = 0;
                bool[] infoFull = new bool[Infos.Length];

                invDocFreqs = new double[Infos.Length][];

                long totalDocs = 0;
                Double rowCount = trainingData.GetRowCount(true) ?? Double.NaN;
                var buffers = new VBuffer<float>[Infos.Length];
                pch.SetHeader(new ProgressHeader(new[] { "Total n-grams" }, new[] { "documents" }),
                    e => e.SetProgress(0, totalDocs, rowCount));
                while (cInfoFull < Infos.Length && cursor.MoveNext())
                {
                    totalDocs++;
                    for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
                    {
                        getters[iinfo](ref src[iinfo]);
                        var keyCount = (uint)Infos[iinfo].TypeSrc.ItemType.KeyCount;
                        if (keyCount == 0)
                            keyCount = uint.MaxValue;
                        if (!infoFull[iinfo])
                        {
                            if (_exes[iinfo].RequireIdf())
                                helpers[iinfo].Reset();

                            helpers[iinfo].AddNgrams(in src[iinfo], 0, keyCount);
                            if (_exes[iinfo].RequireIdf())
                            {
                                int totalNgrams = counts[iinfo].Sum();
                                Utils.EnsureSize(ref invDocFreqs[iinfo], totalNgrams);
                                helpers[iinfo].GetResult(ref buffers[iinfo]);
                                foreach (var pair in buffers[iinfo].Items())
                                {
                                    if (pair.Value >= 1)
                                        invDocFreqs[iinfo][pair.Key] += 1;
                                }
                            }
                        }
                        AssertValid(counts[iinfo], lims[iinfo], ngramMaps[iinfo]);
                    }
                }

                pch.Checkpoint(counts.Sum(c => c.Sum()), totalDocs);
                for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
                {
                    for (int i = 0; i < Utils.Size(invDocFreqs[iinfo]); i++)
                        if (invDocFreqs[iinfo][i] != 0)
                            invDocFreqs[iinfo][i] = Math.Log(totalDocs / invDocFreqs[iinfo][i]);
                }

                for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
                {
                    AssertValid(counts[iinfo], lims[iinfo], ngramMaps[iinfo]);

                    int ngramLength = _exes[iinfo].NgramLength;
                    for (int i = 0; i < ngramLength; i++)
                        _exes[iinfo].NonEmptyLevels[i] = counts[iinfo][i] > 0;
                }

                return ngramMaps;
            }
        }

        [Conditional("DEBUG")]
        private void AssertValid(int[] counts, int[] lims, SequencePool pool)
        {
            int count = 0;
            int countFull = 0;
            for (int i = 0; i < lims.Length; i++)
            {
                Host.Assert(counts[i] >= 0);
                Host.Assert(counts[i] <= lims[i]);
                if (counts[i] == lims[i])
                    countFull++;
                count += counts[i];
            }
            Host.Assert(count == pool.Count);
        }

        private static NgramIdFinder GetNgramIdFinderAdd(int[] counts, int[] lims, SequencePool pool, bool requireIdf, IHost host)
        {
            Contracts.AssertValue(host);
            host.Assert(Utils.Size(lims) > 0);
            host.Assert(Utils.Size(lims) == Utils.Size(counts));

            int numFull = lims.Count(l => l <= 0);
            int ngramLength = lims.Length;
            return
                (uint[] ngram, int lim, int icol, ref bool more) =>
                {
                    host.Assert(0 < lim && lim <= Utils.Size(ngram));
                    host.Assert(lim <= Utils.Size(counts));
                    host.Assert(lim <= Utils.Size(lims));
                    host.Assert(icol == 0);

                    var max = lim - 1;
                    int slot = -1;
                    if (counts[max] < lims[max] && pool.TryAdd(ngram, 0, lim, out slot) && ++counts[max] >= lims[max])
                        numFull++;

                    // Note: 'slot' is either the id of the added ngram or -1. In case it is -1, find its id.
                    // Note: 'more' controls whether more ngrams/skip-grams should be processed in the current
                    //       row. For IDF, as we are interested in counting the occurrence of ngrams/skip-
                    //       grams, more should not be updated.
                    if (requireIdf)
                        return slot != -1 ? slot : pool.Get(ngram, 0, lim);

                    more = numFull < ngramLength;
                    return -1;
                };
        }

        private NgramIdFinder GetNgramIdFinder(int iinfo)
        {
            return
                (uint[] ngram, int lim, int icol, ref bool more) =>
                {
                    Host.Assert(0 < lim && lim <= Utils.Size(ngram));
                    Host.Assert(lim <= Utils.Size(_exes[iinfo].NonEmptyLevels));

                    if (!_exes[iinfo].NonEmptyLevels[lim - 1])
                        return -1;
                    return _ngramMaps[iinfo].Get(ngram, 0, lim);
                };
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Check(0 <= iinfo & iinfo < Infos.Length);
            return _types[iinfo];
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            Host.Assert(Infos[iinfo].TypeSrc.IsVector);
            Host.Assert(Infos[iinfo].TypeSrc.ItemType.IsKey);

            disposer = null;

            var getSrc = RowCursorUtils.GetVecGetterAs<uint>(NumberType.U4, input, Infos[iinfo].Source);
            var src = default(VBuffer<uint>);
            var bldr = new NgramBufferBuilder(_exes[iinfo].NgramLength, _exes[iinfo].SkipLength,
                _ngramMaps[iinfo].Count, GetNgramIdFinder(iinfo));
            var keyCount = (uint)Infos[iinfo].TypeSrc.ItemType.KeyCount;
            if (keyCount == 0)
                keyCount = uint.MaxValue;

            ValueGetter<VBuffer<Float>> del;
            switch (_exes[iinfo].Weighting)
            {
                case WeightingCriteria.TfIdf:
                    Host.AssertValue(_invDocFreqs[iinfo]);
                    del =
                        (ref VBuffer<Float> dst) =>
                        {
                            getSrc(ref src);
                            if (!bldr.IsEmpty)
                            {
                                bldr.Reset();
                                bldr.AddNgrams(in src, 0, keyCount);
                                bldr.GetResult(ref dst);
                                VBufferUtils.Apply(ref dst, (int i, ref Float v) => v = (Float)(v * _invDocFreqs[iinfo][i]));
                            }
                            else
                                dst = new VBuffer<Float>(0, dst.Values, dst.Indices);
                        };
                    break;
                case WeightingCriteria.Idf:
                    Host.AssertValue(_invDocFreqs[iinfo]);
                    del =
                        (ref VBuffer<Float> dst) =>
                        {
                            getSrc(ref src);
                            if (!bldr.IsEmpty)
                            {
                                bldr.Reset();
                                bldr.AddNgrams(in src, 0, keyCount);
                                bldr.GetResult(ref dst);
                                VBufferUtils.Apply(ref dst, (int i, ref Float v) => v = v >= 1 ? (Float)_invDocFreqs[iinfo][i] : 0);
                            }
                            else
                                dst = new VBuffer<Float>(0, dst.Values, dst.Indices);
                        };
                    break;
                case WeightingCriteria.Tf:
                    del =
                        (ref VBuffer<Float> dst) =>
                        {
                            getSrc(ref src);
                            if (!bldr.IsEmpty)
                            {
                                bldr.Reset();
                                bldr.AddNgrams(in src, 0, keyCount);
                                bldr.GetResult(ref dst);
                            }
                            else
                                dst = new VBuffer<Float>(0, dst.Values, dst.Indices);
                        };
                    break;
                default:
                    throw Host.Except("Unsupported weighting criteria");
            }

            return del;
        }
    }
}
