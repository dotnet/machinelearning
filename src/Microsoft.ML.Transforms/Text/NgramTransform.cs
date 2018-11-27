// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;

[assembly: LoadableClass(NgramCountingTransformer.Summary, typeof(IDataTransform), typeof(NgramCountingTransformer), typeof(NgramCountingTransformer.Arguments), typeof(SignatureDataTransform),
    "Ngram Transform", "NgramTransform", "Ngram")]

[assembly: LoadableClass(NgramCountingTransformer.Summary, typeof(IDataTransform), typeof(NgramCountingTransformer), null, typeof(SignatureLoadDataTransform),
    "Ngram Transform", NgramCountingTransformer.LoaderSignature)]

[assembly: LoadableClass(NgramCountingTransformer.Summary, typeof(NgramCountingTransformer), null, typeof(SignatureLoadModel),
    "Ngram Transform", NgramCountingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(NgramCountingTransformer), null, typeof(SignatureLoadRowMapper),
    "Ngram Transform", NgramCountingTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms.Text
{
    using Conditional = System.Diagnostics.ConditionalAttribute;

    public sealed class NgramCountingTransformer : OneToOneTransformerBase
    {
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
            public NgramCountingEstimator.WeightingCriteria? Weighting;

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
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum ngram length", ShortName = "ngram")]
            public int NgramLength = NgramCountingEstimator.Defaults.NgramLength;

            [Argument(ArgumentType.AtMostOnce, HelpText =
                "Whether to store all ngram lengths up to ngramLength, or only ngramLength", ShortName = "all")]
            public bool AllLengths = NgramCountingEstimator.Defaults.AllLength;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Maximum number of tokens to skip when constructing an ngram",
                ShortName = "skips")]
            public int SkipLength = NgramCountingEstimator.Defaults.SkipLength;

            [Argument(ArgumentType.Multiple, HelpText = "Maximum number of ngrams to store in the dictionary", ShortName = "max")]
            public int[] MaxNumTerms = new int[] { NgramCountingEstimator.Defaults.MaxNumTerms };

            [Argument(ArgumentType.AtMostOnce, HelpText = "The weighting criteria")]
            public NgramCountingEstimator.WeightingCriteria Weighting = NgramCountingEstimator.Defaults.Weighting;
        }

        private const uint VerTfIdfSupported = 0x00010002;

        internal const string LoaderSignature = "NgramTransform";
        internal const string Summary = "Produces a bag of counts of ngrams (sequences of consecutive values of length 1-n) in a given vector of keys. "
            + "It does so by building a dictionary of ngrams and using the id in the dictionary as the index in the bag.";

        internal const string UserName = "NGram Transform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "NGRAMTRN",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010002, // Add support for TF-IDF
                verWrittenCur: 0x00010003, // Get rid of writing float size in model context
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(NgramCountingTransformer).Assembly.FullName);
        }

        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        public sealed class ColumnInfo
        {
            public readonly string Input;
            public readonly string Output;
            public readonly int NgramLength;
            public readonly int SkipLength;
            public readonly bool AllLengths;
            public readonly NgramCountingEstimator.WeightingCriteria Weighting;
            /// <summary>
            /// Contains the maximum number of grams to store in the dictionary, for each level of ngrams,
            /// from 1 (in position 0) up to ngramLength (in position ngramLength-1)
            /// </summary>
            public readonly ImmutableArray<int> Limits;

            /// <summary>
            /// Describes how the transformer handles one Gcn column pair.
            /// </summary>
            /// <param name="input">Name of input column.</param>
            /// <param name="output">Name of output column.</param>
            /// <param name="ngramLength">Maximum ngram length.</param>
            /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
            /// <param name="allLengths">"Whether to store all ngram lengths up to ngramLength, or only ngramLength.</param>
            /// <param name="weighting">The weighting criteria.</param>
            /// <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
            public ColumnInfo(string input, string output,
                int ngramLength = NgramCountingEstimator.Defaults.NgramLength,
                int skipLength = NgramCountingEstimator.Defaults.SkipLength,
                bool allLengths = NgramCountingEstimator.Defaults.AllLength,
                NgramCountingEstimator.WeightingCriteria weighting = NgramCountingEstimator.Defaults.Weighting,
                int maxNumTerms = NgramCountingEstimator.Defaults.MaxNumTerms) : this(input, output, ngramLength, skipLength, allLengths, weighting, new int[] { maxNumTerms })
            {
            }

            internal ColumnInfo(string input, string output,
                int ngramLength,
                int skipLength,
                bool allLengths,
                NgramCountingEstimator.WeightingCriteria weighting,
                int[] maxNumTerms)
            {
                Input = input;
                Output = output;
                NgramLength = ngramLength;
                Contracts.CheckUserArg(0 < NgramLength && NgramLength <= NgramBufferBuilder.MaxSkipNgramLength, nameof(ngramLength));
                SkipLength = skipLength;
                if (NgramLength + SkipLength > NgramBufferBuilder.MaxSkipNgramLength)
                {
                    throw Contracts.ExceptUserArg(nameof(skipLength),
                        $"The sum of skipLength and ngramLength must be less than or equal to {NgramBufferBuilder.MaxSkipNgramLength}");
                }
                AllLengths = allLengths;
                Weighting = weighting;
                var limits = new int[ngramLength];
                if (!AllLengths)
                {
                    Contracts.CheckUserArg(Utils.Size(maxNumTerms) == 0 ||
                        Utils.Size(maxNumTerms) == 1 && maxNumTerms[0] > 0, nameof(maxNumTerms));
                    limits[ngramLength - 1] = Utils.Size(maxNumTerms) == 0 ? NgramCountingEstimator.Defaults.MaxNumTerms : maxNumTerms[0];
                }
                else
                {
                    Contracts.CheckUserArg(Utils.Size(maxNumTerms) <= ngramLength, nameof(maxNumTerms));
                    Contracts.CheckUserArg(Utils.Size(maxNumTerms) == 0 || maxNumTerms.All(i => i >= 0) && maxNumTerms[maxNumTerms.Length - 1] > 0, nameof(maxNumTerms));
                    var extend = Utils.Size(maxNumTerms) == 0 ? NgramCountingEstimator.Defaults.MaxNumTerms : maxNumTerms[maxNumTerms.Length - 1];
                    limits = Utils.BuildArray(ngramLength, i => i < Utils.Size(maxNumTerms) ? maxNumTerms[i] : extend);
                }
                Limits = ImmutableArray.Create(limits);
            }
        }

        private sealed class TransformInfo
        {
            // Position i, indicates whether the pool contains any (i+1)-grams
            public readonly bool[] NonEmptyLevels;
            public readonly int NgramLength;
            public readonly int SkipLength;
            public readonly NgramCountingEstimator.WeightingCriteria Weighting;

            public bool RequireIdf => Weighting == NgramCountingEstimator.WeightingCriteria.Idf || Weighting == NgramCountingEstimator.WeightingCriteria.TfIdf;

            public TransformInfo(ColumnInfo info)
            {
                NgramLength = info.NgramLength;
                SkipLength = info.SkipLength;
                Weighting = info.Weighting;
                NonEmptyLevels = new bool[NgramLength];
            }

            public TransformInfo(ModelLoadContext ctx, bool readWeighting)
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
                    Weighting = (NgramCountingEstimator.WeightingCriteria)ctx.Reader.ReadInt32();
                Contracts.CheckDecode(Enum.IsDefined(typeof(NgramCountingEstimator.WeightingCriteria), Weighting));
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
                Contracts.Assert(Enum.IsDefined(typeof(NgramCountingEstimator.WeightingCriteria), Weighting));
                ctx.Writer.Write((int)Weighting);
                Contracts.Assert(Utils.Size(NonEmptyLevels) == NgramLength);
                ctx.Writer.WriteBoolBytesNoCount(NonEmptyLevels);
            }
        }

        private readonly ImmutableArray<TransformInfo> _transformInfos;

        // These contain the ngram maps
        private readonly SequencePool[] _ngramMaps;

        // Ngram inverse document frequencies
        private readonly double[][] _invDocFreqs;

        private static (string input, string output)[] GetColumnPairs(ColumnInfo[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Input, x.Output)).ToArray();
        }

        protected override void CheckInputColumn(ISchema inputSchema, int col, int srcCol)
        {
            var type = inputSchema.GetColumnType(srcCol);
            if (!NgramCountingEstimator.IsColumnTypeValid(type))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].input, NgramCountingEstimator.ExpectedColumnType, type.ToString());
        }

        internal NgramCountingTransformer(IHostEnvironment env, IDataView input, ColumnInfo[] columns)
           : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(NgramCountingTransformer)), GetColumnPairs(columns))
        {
            var transformInfos = new TransformInfo[columns.Length];
            for (int i = 0; i < columns.Length; i++)
            {
                input.Schema.TryGetColumnIndex(columns[i].Input, out int srcCol);
                var typeSrc = input.Schema.GetColumnType(srcCol);
                transformInfos[i] = new TransformInfo(columns[i]);
            }
            _transformInfos = transformInfos.ToImmutableArray();
            _ngramMaps = Train(Host, columns, _transformInfos, input, out _invDocFreqs);
        }

        private static SequencePool[] Train(IHostEnvironment env, ColumnInfo[] columns, ImmutableArray<TransformInfo> transformInfos, IDataView trainingData, out double[][] invDocFreqs)
        {
            var helpers = new NgramBufferBuilder[columns.Length];
            var getters = new ValueGetter<VBuffer<uint>>[columns.Length];
            var src = new VBuffer<uint>[columns.Length];

            // Keep track of how many grams are in the pool for each value of n. Position
            // i in _counts counts how many (i+1)-grams are in the pool for column iinfo.
            var counts = new int[columns.Length][];
            var ngramMaps = new SequencePool[columns.Length];
            var activeInput = new bool[trainingData.Schema.ColumnCount];
            var srcTypes = new ColumnType[columns.Length];
            var srcCols = new int[columns.Length];
            for (int iinfo = 0; iinfo < columns.Length; iinfo++)
            {
                trainingData.Schema.TryGetColumnIndex(columns[iinfo].Input, out srcCols[iinfo]);
                srcTypes[iinfo] = trainingData.Schema[srcCols[iinfo]].Type;
                activeInput[srcCols[iinfo]] = true;
            }
            using (var cursor = trainingData.GetRowCursor(col => activeInput[col]))
            using (var pch = env.StartProgressChannel("Building n-gram dictionary"))
            {
                for (int iinfo = 0; iinfo < columns.Length; iinfo++)
                {
                    env.Assert(srcTypes[iinfo].IsVector && srcTypes[iinfo].ItemType.IsKey);
                    var ngramLength = columns[iinfo].NgramLength;
                    var skipLength = columns[iinfo].SkipLength;

                    getters[iinfo] = RowCursorUtils.GetVecGetterAs<uint>(NumberType.U4, cursor, srcCols[iinfo]);
                    src[iinfo] = default;
                    counts[iinfo] = new int[ngramLength];
                    ngramMaps[iinfo] = new SequencePool();

                    // Note: GetNgramIdFinderAdd will control how many ngrams of a specific length will
                    // be added (using lims[iinfo]), therefore we set slotLim to the maximum
                    helpers[iinfo] = new NgramBufferBuilder(ngramLength, skipLength, Utils.ArrayMaxSize,
                        GetNgramIdFinderAdd(env, counts[iinfo], columns[iinfo].Limits, ngramMaps[iinfo], transformInfos[iinfo].RequireIdf));
                }

                int cInfoFull = 0;
                bool[] infoFull = new bool[columns.Length];

                invDocFreqs = new double[columns.Length][];

                long totalDocs = 0;
                var rowCount = trainingData.GetRowCount() ?? double.NaN;
                var buffers = new VBuffer<float>[columns.Length];
                pch.SetHeader(new ProgressHeader(new[] { "Total n-grams" }, new[] { "documents" }),
                    e => e.SetProgress(0, totalDocs, rowCount));
                while (cInfoFull < columns.Length && cursor.MoveNext())
                {
                    totalDocs++;
                    for (int iinfo = 0; iinfo < columns.Length; iinfo++)
                    {
                        getters[iinfo](ref src[iinfo]);
                        var keyCount = (uint)srcTypes[iinfo].ItemType.KeyCount;
                        if (keyCount == 0)
                            keyCount = uint.MaxValue;
                        if (!infoFull[iinfo])
                        {
                            if (transformInfos[iinfo].RequireIdf)
                                helpers[iinfo].Reset();

                            helpers[iinfo].AddNgrams(in src[iinfo], 0, keyCount);
                            if (transformInfos[iinfo].RequireIdf)
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
                        AssertValid(env, counts[iinfo], columns[iinfo].Limits, ngramMaps[iinfo]);
                    }
                }

                pch.Checkpoint(counts.Sum(c => c.Sum()), totalDocs);
                for (int iinfo = 0; iinfo < columns.Length; iinfo++)
                {
                    for (int i = 0; i < Utils.Size(invDocFreqs[iinfo]); i++)
                        if (invDocFreqs[iinfo][i] != 0)
                            invDocFreqs[iinfo][i] = Math.Log(totalDocs / invDocFreqs[iinfo][i]);
                }

                for (int iinfo = 0; iinfo < columns.Length; iinfo++)
                {
                    AssertValid(env, counts[iinfo], columns[iinfo].Limits, ngramMaps[iinfo]);

                    int ngramLength = transformInfos[iinfo].NgramLength;
                    for (int i = 0; i < ngramLength; i++)
                        transformInfos[iinfo].NonEmptyLevels[i] = counts[iinfo][i] > 0;
                }

                return ngramMaps;
            }
        }

        [Conditional("DEBUG")]
        private static void AssertValid(IHostEnvironment env, int[] counts, ImmutableArray<int> lims, SequencePool pool)
        {
            int count = 0;
            int countFull = 0;
            for (int i = 0; i < lims.Length; i++)
            {
                env.Assert(counts[i] >= 0);
                env.Assert(counts[i] <= lims[i]);
                if (counts[i] == lims[i])
                    countFull++;
                count += counts[i];
            }
            env.Assert(count == pool.Count);
        }

        private static NgramIdFinder GetNgramIdFinderAdd(IHostEnvironment env, int[] counts, ImmutableArray<int> lims, SequencePool pool, bool requireIdf)
        {
            Contracts.AssertValue(env);
            env.Assert(lims.Length > 0);
            env.Assert(lims.Length == Utils.Size(counts));

            int numFull = lims.Count(l => l <= 0);
            int ngramLength = lims.Length;
            return
                (uint[] ngram, int lim, int icol, ref bool more) =>
                {
                    env.Assert(0 < lim && lim <= Utils.Size(ngram));
                    env.Assert(lim <= Utils.Size(counts));
                    env.Assert(lim <= lims.Length);
                    env.Assert(icol == 0);

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

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(Schema.Create(inputSchema));

        private NgramCountingTransformer(IHost host, ModelLoadContext ctx) :
            base(host, ctx)
        {
            var columnsLength = ColumnPairs.Length;
            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            // for each column
            //   _transformInfo
            //   the ngram SequencePool
            //   the ngram inverse document frequencies
            var transformInfos = new TransformInfo[columnsLength];
            _ngramMaps = new SequencePool[columnsLength];
            _invDocFreqs = new double[columnsLength][];
            for (int i = 0; i < columnsLength; i++)
            {
                transformInfos[i] = new TransformInfo(ctx, ctx.Header.ModelVerWritten >= VerTfIdfSupported);
                _ngramMaps[i] = new SequencePool(ctx.Reader);

                if (ctx.Header.ModelVerWritten >= VerTfIdfSupported)
                {
                    _invDocFreqs[i] = ctx.Reader.ReadDoubleArray();
                    for (int j = 0; j < Utils.Size(_invDocFreqs[i]); j++)
                        Host.CheckDecode(_invDocFreqs[i][j] >= 0);
                }
            }
            _transformInfos = transformInfos.ToImmutableArray();
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            env.CheckValue(args.Column, nameof(args.Column));
            var cols = new ColumnInfo[args.Column.Length];
            using (var ch = env.Start("ValidateArgs"))
            {

                for (int i = 0; i < cols.Length; i++)
                {
                    var item = args.Column[i];
                    var maxNumTerms = Utils.Size(item.MaxNumTerms) > 0 ? item.MaxNumTerms : args.MaxNumTerms;
                    cols[i] = new ColumnInfo(item.Source ?? item.Name,
                        item.Name,
                        item.NgramLength ?? args.NgramLength,
                        item.SkipLength ?? args.SkipLength,
                        item.AllLengths ?? args.AllLengths,
                        item.Weighting ?? args.Weighting,
                        maxNumTerms);
                };
            }
            return new NgramCountingTransformer(env, input, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static NgramCountingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(nameof(NgramCountingTransformer));

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            if (ctx.Header.ModelVerWritten < 0x00010003)
            {
                int cbFloat = ctx.Reader.ReadInt32();
                env.CheckDecode(cbFloat == sizeof(float));
            }
            return new NgramCountingTransformer(host, ctx);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            // *** Binary format ***
            // <base>
            // for each added column
            //   _transformInfo
            //   the ngram SequencePool
            //   the ngram inverse document frequencies
            SaveColumns(ctx);
            for (int i = 0; i < _transformInfos.Length; i++)
            {
                _transformInfos[i].Save(ctx);
                _ngramMaps[i].Save(ctx.Writer);
                ctx.Writer.WriteDoubleArray(_invDocFreqs[i]);
            }
        }

        protected override IRowMapper MakeRowMapper(Schema schema) => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly ColumnType[] _srcTypes;
            private readonly int[] _srcCols;
            private readonly ColumnType[] _types;
            private readonly NgramCountingTransformer _parent;

            public Mapper(NgramCountingTransformer parent, Schema inputSchema)
               : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _types = new ColumnType[_parent.ColumnPairs.Length];
                _srcTypes = new ColumnType[_parent.ColumnPairs.Length];
                _srcCols = new int[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    _types[i] = new VectorType(NumberType.Float, _parent._ngramMaps[i].Count);
                    inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out _srcCols[i]);
                    _srcTypes[i] = inputSchema.GetColumnType(_srcCols[i]);
                }
            }

            protected override Schema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new Schema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    var builder = new MetadataBuilder();
                    AddMetadata(i, builder);

                    result[i] = new Schema.DetachedColumn(_parent.ColumnPairs[i].output, _types[i], builder.GetMetadata());
                }
                return result;
            }

            private void AddMetadata(int iinfo, MetadataBuilder builder)
            {
                if (InputSchema.HasKeyValues(_srcCols[iinfo], _srcTypes[iinfo].ItemType.KeyCount))
                {
                    ValueGetter<VBuffer<ReadOnlyMemory<char>>> getter = (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                    {
                        GetSlotNames(iinfo, _parent._ngramMaps[iinfo].Count, ref dst);
                    };

                    var slotNamesType = new VectorType(TextType.Instance, _parent._ngramMaps[iinfo].Count);
                    builder.AddSlotNames(_parent._ngramMaps[iinfo].Count, getter);
                }
            }

            private void GetSlotNames(int iinfo, int size, ref VBuffer<ReadOnlyMemory<char>> dst)
            {
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);

                var keyCount = _srcTypes[iinfo].ItemType.KeyCount;
                Host.Assert(InputSchema.HasKeyValues(_srcCols[iinfo], keyCount));

                var unigramNames = new VBuffer<ReadOnlyMemory<char>>();

                // Get the key values of the unigrams.
                InputSchema.GetMetadata(MetadataUtils.Kinds.KeyValues, _srcCols[iinfo], ref unigramNames);
                Host.Check(unigramNames.Length == keyCount);

                var pool = _parent._ngramMaps[iinfo];

                var ngramCount = pool.Count;
                var dstEditor = VBufferEditor.Create(ref dst, ngramCount);

                StringBuilder sb = new StringBuilder();
                uint[] ngram = new uint[_parent._transformInfos[iinfo].NgramLength];
                for (int slot = 0; slot < pool.Count; slot++)
                {
                    var n = pool.GetById(slot, ref ngram);
                    Host.Assert(n >= 0);

                    // Get the unigrams composing the current ngram.
                    ComposeNgramString(ngram, n, sb, keyCount,
                        unigramNames.GetItemOrDefault);
                    dstEditor.Values[slot] = sb.ToString().AsMemory();
                }

                dst = dstEditor.Commit();
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

            private NgramIdFinder GetNgramIdFinder(int iinfo)
            {
                return
                    (uint[] ngram, int lim, int icol, ref bool more) =>
                    {
                        Host.Assert(0 < lim && lim <= Utils.Size(ngram));
                        Host.Assert(lim <= Utils.Size(_parent._transformInfos[iinfo].NonEmptyLevels));

                        if (!_parent._transformInfos[iinfo].NonEmptyLevels[lim - 1])
                            return -1;
                        return _parent._ngramMaps[iinfo].Get(ngram, 0, lim);
                    };
            }

            protected override Delegate MakeGetter(IRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;

                var getSrc = RowCursorUtils.GetVecGetterAs<uint>(NumberType.U4, input, _srcCols[iinfo]);
                var src = default(VBuffer<uint>);
                var bldr = new NgramBufferBuilder(_parent._transformInfos[iinfo].NgramLength, _parent._transformInfos[iinfo].SkipLength,
                    _parent._ngramMaps[iinfo].Count, GetNgramIdFinder(iinfo));
                var keyCount = (uint)_srcTypes[iinfo].ItemType.KeyCount;
                if (keyCount == 0)
                    keyCount = uint.MaxValue;

                ValueGetter<VBuffer<float>> del;
                switch (_parent._transformInfos[iinfo].Weighting)
                {
                    case NgramCountingEstimator.WeightingCriteria.TfIdf:
                        Host.AssertValue(_parent._invDocFreqs[iinfo]);
                        del =
                            (ref VBuffer<float> dst) =>
                            {
                                getSrc(ref src);
                                if (!bldr.IsEmpty)
                                {
                                    bldr.Reset();
                                    bldr.AddNgrams(in src, 0, keyCount);
                                    bldr.GetResult(ref dst);
                                    VBufferUtils.Apply(ref dst, (int i, ref float v) => v = (float)(v * _parent._invDocFreqs[iinfo][i]));
                                }
                                else
                                    VBufferUtils.Resize(ref dst, 0);
                            };
                        break;
                    case NgramCountingEstimator.WeightingCriteria.Idf:
                        Host.AssertValue(_parent._invDocFreqs[iinfo]);
                        del =
                            (ref VBuffer<float> dst) =>
                            {
                                getSrc(ref src);
                                if (!bldr.IsEmpty)
                                {
                                    bldr.Reset();
                                    bldr.AddNgrams(in src, 0, keyCount);
                                    bldr.GetResult(ref dst);
                                    VBufferUtils.Apply(ref dst, (int i, ref float v) => v = v >= 1 ? (float)_parent._invDocFreqs[iinfo][i] : 0);
                                }
                                else
                                    VBufferUtils.Resize(ref dst, 0);
                            };
                        break;
                    case NgramCountingEstimator.WeightingCriteria.Tf:
                        del =
                            (ref VBuffer<float> dst) =>
                            {
                                getSrc(ref src);
                                if (!bldr.IsEmpty)
                                {
                                    bldr.Reset();
                                    bldr.AddNgrams(in src, 0, keyCount);
                                    bldr.GetResult(ref dst);
                                }
                                else
                                    VBufferUtils.Resize(ref dst, 0);
                            };
                        break;
                    default:
                        throw Host.Except("Unsupported weighting criteria");
                }
                return del;
            }
        }
    }

    /// <summary>
    /// Produces a bag of counts of ngrams(sequences of consecutive values of length 1-n) in a given vector of keys.
    /// It does so by building a dictionary of ngrams and using the id in the dictionary as the index in the bag.
    /// </summary>
    public sealed class NgramCountingEstimator : IEstimator<NgramCountingTransformer>
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

        internal static class Defaults
        {
            public const int NgramLength = 2;
            public const bool AllLength = true;
            public const int SkipLength = 0;
            public const int MaxNumTerms = 10000000;
            public const WeightingCriteria Weighting = WeightingCriteria.Tf;
        }

        private readonly IHost _host;
        private readonly NgramCountingTransformer.ColumnInfo[] _columns;

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words) in <paramref name="inputColumn"/>
        /// and outputs bag of word vector as <paramref name="outputColumn"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The column containing text to compute bag of word vector.</param>
        /// <param name="outputColumn">The column containing bag of word vector. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        public NgramCountingEstimator(IHostEnvironment env,
            string inputColumn,
            string outputColumn = null,
            int ngramLength = Defaults.NgramLength,
            int skipLength = Defaults.SkipLength,
            bool allLengths = Defaults.AllLength,
            int maxNumTerms = Defaults.MaxNumTerms,
            WeightingCriteria weighting = Defaults.Weighting)
            : this(env, new[] { (inputColumn, outputColumn ?? inputColumn) }, ngramLength, skipLength, allLengths, maxNumTerms, weighting)
        {
        }

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words) in <paramref name="columns.inputs"/>
        /// and outputs bag of word vector for each output in <paramref name="columns.output"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to compute bag of word vector.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        public NgramCountingEstimator(IHostEnvironment env,
            (string input, string output)[] columns,
            int ngramLength = Defaults.NgramLength,
            int skipLength = Defaults.SkipLength,
            bool allLengths = Defaults.AllLength,
            int maxNumTerms = Defaults.MaxNumTerms,
            WeightingCriteria weighting = Defaults.Weighting)
            : this(env, columns.Select(x => new NgramCountingTransformer.ColumnInfo(x.input, x.output, ngramLength, skipLength, allLengths, weighting, maxNumTerms)).ToArray())
        {
        }

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words) in <paramref name="columns.inputs"/>
        /// and outputs bag of word vector for each output in <paramref name="columns.output"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Array of columns with information how to transform data.</param>
        public NgramCountingEstimator(IHostEnvironment env, params NgramCountingTransformer.ColumnInfo[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(NgramCountingEstimator));
            _columns = columns;
        }

        public NgramCountingTransformer Fit(IDataView input) => new NgramCountingTransformer(_host, input, _columns);

        internal static bool IsColumnTypeValid(ColumnType type)
        {
            if (!type.IsVector)
                return false;
            if (!type.ItemType.IsKey)
                return false;
            // Can only accept key types that can be converted to U4.
            if (type.ItemType.KeyCount == 0 && type.ItemType.RawKind > DataKind.U4)
                return false;
            return true;
        }

        internal static bool IsSchemaColumnValid(SchemaShape.Column col)
        {
            if (col.Kind == SchemaShape.Column.VectorKind.Scalar)
                return false;
            if (!col.IsKey)
                return false;
            // Can only accept key types that can be converted to U4.
            if (col.ItemType.RawKind > DataKind.U4)
                return false;
            return true;
        }

        internal const string ExpectedColumnType = "Expected vector of Key type, and Key is convertable to U4";

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var colInfo in _columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.Input, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);
                if (!IsSchemaColumnValid(col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input, ExpectedColumnType, col.GetTypeString());
                var metadata = new List<SchemaShape.Column>();
                if (col.HasKeyValues())
                    metadata.Add(new SchemaShape.Column(MetadataUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, TextType.Instance, false));
                result[colInfo.Output] = new SchemaShape.Column(colInfo.Output, SchemaShape.Column.VectorKind.Vector, NumberType.R4, false, new SchemaShape(metadata));
            }
            return new SchemaShape(result.Values);
        }
    }
}
