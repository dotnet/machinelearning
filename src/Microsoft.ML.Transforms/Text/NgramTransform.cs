// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Text;

[assembly: LoadableClass(NgramExtractingTransformer.Summary, typeof(IDataTransform), typeof(NgramExtractingTransformer), typeof(NgramExtractingTransformer.Options), typeof(SignatureDataTransform),
    "Ngram Transform", "NgramTransform", "Ngram")]

[assembly: LoadableClass(NgramExtractingTransformer.Summary, typeof(IDataTransform), typeof(NgramExtractingTransformer), null, typeof(SignatureLoadDataTransform),
    "Ngram Transform", NgramExtractingTransformer.LoaderSignature)]

[assembly: LoadableClass(NgramExtractingTransformer.Summary, typeof(NgramExtractingTransformer), null, typeof(SignatureLoadModel),
    "Ngram Transform", NgramExtractingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(NgramExtractingTransformer), null, typeof(SignatureLoadRowMapper),
    "Ngram Transform", NgramExtractingTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms.Text
{
    /// <summary>
    /// <see cref="ITransformer"/> resulting from fitting an <see cref="NgramExtractingEstimator"/>.
    /// </summary>
    public sealed class NgramExtractingTransformer : OneToOneTransformerBase
    {
        internal sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum n-gram length", ShortName = "ngram")]
            public int? NgramLength;

            [Argument(ArgumentType.AtMostOnce, HelpText =
                "Whether to include all n-gram lengths up to " + nameof(NgramLength) + " or only " + nameof(NgramLength), Name = "AllLengths", ShortName = "all")]
            public bool? UseAllLengths;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Maximum number of tokens to skip when constructing an n-gram",
                ShortName = "skips")]
            public int? SkipLength;

            [Argument(ArgumentType.Multiple, HelpText = "Maximum number of n-grams to store in the dictionary", ShortName = "max")]
            public int[] MaxNumTerms = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Statistical measure used to evaluate how important a word is to a document in a corpus")]
            public NgramExtractingEstimator.WeightingCriteria? Weighting;

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
                if (NgramLength != null || UseAllLengths != null || SkipLength != null || Utils.Size(MaxNumTerms) != 0)
                    return false;
                return TryUnparseCore(sb);
            }
        }

        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum n-gram length", ShortName = "ngram")]
            public int NgramLength = NgramExtractingEstimator.Defaults.NgramLength;

            [Argument(ArgumentType.AtMostOnce, HelpText =
                "Whether to store all n-gram lengths up to ngramLength, or only ngramLength", Name = "AllLengths", ShortName = "all")]
            public bool UseAllLengths = NgramExtractingEstimator.Defaults.UseAllLengths;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Maximum number of tokens to skip when constructing an n-gram",
                ShortName = "skips")]
            public int SkipLength = NgramExtractingEstimator.Defaults.SkipLength;

            [Argument(ArgumentType.Multiple, HelpText = "Maximum number of n-grams to store in the dictionary", ShortName = "max")]
            public int[] MaxNumTerms = new int[] { NgramExtractingEstimator.Defaults.MaximumNgramsCount };

            [Argument(ArgumentType.AtMostOnce, HelpText = "The weighting criteria")]
            public NgramExtractingEstimator.WeightingCriteria Weighting = NgramExtractingEstimator.Defaults.Weighting;
        }

        private const uint VerTfIdfSupported = 0x00010002;

        internal const string LoaderSignature = "NgramTransform";
        internal const string Summary = "Produces a bag of counts of n-grams (sequences of consecutive values of length 1-n) in a given vector of keys. "
            + "It does so by building a dictionary of n-grams and using the id in the dictionary as the index in the bag.";

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
                loaderAssemblyName: typeof(NgramExtractingTransformer).Assembly.FullName);
        }

        private sealed class TransformInfo
        {
            // Position i, indicates whether the pool contains any (i+1)-grams
            public readonly bool[] NonEmptyLevels;
            public readonly int NgramLength;
            public readonly int SkipLength;
            public readonly bool UseAllLengths;
            public readonly NgramExtractingEstimator.WeightingCriteria Weighting;

            public bool RequireIdf => Weighting == NgramExtractingEstimator.WeightingCriteria.Idf || Weighting == NgramExtractingEstimator.WeightingCriteria.TfIdf;

            public TransformInfo(NgramExtractingEstimator.ColumnOptions info)
            {
                NgramLength = info.NgramLength;
                SkipLength = info.SkipLength;
                Weighting = info.Weighting;
                UseAllLengths = info.UseAllLengths;
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
                    Weighting = (NgramExtractingEstimator.WeightingCriteria)ctx.Reader.ReadInt32();
                Contracts.CheckDecode(Enum.IsDefined(typeof(NgramExtractingEstimator.WeightingCriteria), Weighting));
                NonEmptyLevels = ctx.Reader.ReadBoolArray(NgramLength);
            }

            internal void Save(ModelSaveContext ctx)
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
                Contracts.Assert(Enum.IsDefined(typeof(NgramExtractingEstimator.WeightingCriteria), Weighting));
                ctx.Writer.Write((int)Weighting);
                Contracts.Assert(Utils.Size(NonEmptyLevels) == NgramLength);
                ctx.Writer.WriteBoolBytesNoCount(NonEmptyLevels);
            }
        }

        private readonly ImmutableArray<TransformInfo> _transformInfos;

        // These contain the n-gram maps
        private readonly SequencePool[] _ngramMaps;

        // Ngram inverse document frequencies
        private readonly double[][] _invDocFreqs;

        private static (string outputColumnName, string inputColumnName)[] GetColumnPairs(NgramExtractingEstimator.ColumnOptions[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Name, x.InputColumnName)).ToArray();
        }

        private protected override void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            var type = inputSchema[srcCol].Type;
            if (!NgramExtractingEstimator.IsColumnTypeValid(type))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].inputColumnName, NgramExtractingEstimator.ExpectedColumnType, type.ToString());
        }

        internal NgramExtractingTransformer(IHostEnvironment env, IDataView input, NgramExtractingEstimator.ColumnOptions[] columns)
           : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(NgramExtractingTransformer)), GetColumnPairs(columns))
        {
            var transformInfos = new TransformInfo[columns.Length];
            for (int i = 0; i < columns.Length; i++)
            {
                input.Schema.TryGetColumnIndex(columns[i].InputColumnName, out int srcCol);
                var typeSrc = input.Schema[srcCol].Type;
                transformInfos[i] = new TransformInfo(columns[i]);
            }
            _transformInfos = transformInfos.ToImmutableArray();
            _ngramMaps = Train(Host, columns, _transformInfos, input, out _invDocFreqs);
        }

        private static SequencePool[] Train(IHostEnvironment env, NgramExtractingEstimator.ColumnOptions[] columns, ImmutableArray<TransformInfo> transformInfos, IDataView trainingData, out double[][] invDocFreqs)
        {
            var helpers = new NgramBufferBuilder[columns.Length];
            var getters = new ValueGetter<VBuffer<uint>>[columns.Length];
            var src = new VBuffer<uint>[columns.Length];

            // Keep track of how many grams are in the pool for each value of n. Position
            // i in _counts counts how many (i+1)-grams are in the pool for column iinfo.
            var counts = new int[columns.Length][];
            var ngramMaps = new SequencePool[columns.Length];
            var activeCols = new List<DataViewSchema.Column>();
            var srcTypes = new DataViewType[columns.Length];
            var srcCols = new int[columns.Length];
            for (int iinfo = 0; iinfo < columns.Length; iinfo++)
            {
                trainingData.Schema.TryGetColumnIndex(columns[iinfo].InputColumnName, out srcCols[iinfo]);
                srcTypes[iinfo] = trainingData.Schema[srcCols[iinfo]].Type;
                activeCols.Add(trainingData.Schema[srcCols[iinfo]]);
            }
            using (var cursor = trainingData.GetRowCursor(activeCols))
            using (var pch = env.StartProgressChannel("Building n-gram dictionary"))
            {
                for (int iinfo = 0; iinfo < columns.Length; iinfo++)
                {
                    env.Assert(srcTypes[iinfo] is VectorDataViewType vectorType && vectorType.ItemType is KeyDataViewType);
                    var ngramLength = columns[iinfo].NgramLength;
                    var skipLength = columns[iinfo].SkipLength;

                    getters[iinfo] = RowCursorUtils.GetVecGetterAs<uint>(NumberDataViewType.UInt32, cursor, srcCols[iinfo]);
                    src[iinfo] = default;
                    counts[iinfo] = new int[ngramLength];
                    ngramMaps[iinfo] = new SequencePool();

                    // Note: GetNgramIdFinderAdd will control how many n-grams of a specific length will
                    // be added (using lims[iinfo]), therefore we set slotLim to the maximum
                    helpers[iinfo] = new NgramBufferBuilder(ngramLength, skipLength, Utils.ArrayMaxSize,
                        GetNgramIdFinderAdd(env, counts[iinfo], columns[iinfo].MaximumNgramsCounts, ngramMaps[iinfo], transformInfos[iinfo].RequireIdf));
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
                        var keyCount = (uint)srcTypes[iinfo].GetItemType().GetKeyCount();
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
                        AssertValid(env, counts[iinfo], columns[iinfo].MaximumNgramsCounts, ngramMaps[iinfo]);
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
                    AssertValid(env, counts[iinfo], columns[iinfo].MaximumNgramsCounts, ngramMaps[iinfo]);

                    int ngramLength = transformInfos[iinfo].NgramLength;
                    for (int i = 0; i < ngramLength; i++)
                        transformInfos[iinfo].NonEmptyLevels[i] = counts[iinfo][i] > 0;
                }

                return ngramMaps;
            }
        }

        [Conditional("DEBUG")]
        private static void AssertValid(IHostEnvironment env, int[] counts, IReadOnlyList<int> lims, SequencePool pool)
        {
            int count = 0;
            int countFull = 0;
            for (int i = 0; i < lims.Count; i++)
            {
                env.Assert(counts[i] >= 0);
                env.Assert(counts[i] <= lims[i]);
                if (counts[i] == lims[i])
                    countFull++;
                count += counts[i];
            }
            env.Assert(count == pool.Count);
        }

        private static NgramIdFinder GetNgramIdFinderAdd(IHostEnvironment env, int[] counts, IReadOnlyList<int> lims, SequencePool pool, bool requireIdf)
        {
            Contracts.AssertValue(env);
            env.Assert(lims.Count > 0);
            env.Assert(lims.Count == Utils.Size(counts));

            int numFull = lims.Count(l => l <= 0);
            int ngramLength = lims.Count;
            return
                (uint[] ngram, int lim, int icol, ref bool more) =>
                {
                    env.Assert(0 < lim && lim <= Utils.Size(ngram));
                    env.Assert(lim <= Utils.Size(counts));
                    env.Assert(lim <= lims.Count);
                    env.Assert(icol == 0);

                    var max = lim - 1;
                    int slot = -1;
                    if (counts[max] < lims[max] && pool.TryAdd(ngram, 0, lim, out slot) && ++counts[max] >= lims[max])
                        numFull++;

                    // Note: 'slot' is either the id of the added n-gram or -1. In case it is -1, find its id.
                    // Note: 'more' controls whether more n-grams/skip-grams should be processed in the current
                    //       row. For IDF, as we are interested in counting the occurrence of n-grams/skip-
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
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private NgramExtractingTransformer(IHost host, ModelLoadContext ctx) :
            base(host, ctx)
        {
            var columnsLength = ColumnPairs.Length;
            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            // for each column
            //   _transformInfo
            //   the n-gram SequencePool
            //   the n-gram inverse document frequencies
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
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            env.CheckValue(options.Columns, nameof(options.Columns));
            var cols = new NgramExtractingEstimator.ColumnOptions[options.Columns.Length];
            using (var ch = env.Start("ValidateArgs"))
            {

                for (int i = 0; i < cols.Length; i++)
                {
                    var item = options.Columns[i];
                    var maxNumTerms = Utils.Size(item.MaxNumTerms) > 0 ? item.MaxNumTerms : options.MaxNumTerms;
                    cols[i] = new NgramExtractingEstimator.ColumnOptions(
                        item.Name,
                        item.NgramLength ?? options.NgramLength,
                        item.SkipLength ?? options.SkipLength,
                        item.UseAllLengths ?? options.UseAllLengths,
                        item.Weighting ?? options.Weighting,
                        maxNumTerms,
                        item.Source ?? item.Name);
                }
            }
            return new NgramExtractingTransformer(env, input, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static NgramExtractingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(nameof(NgramExtractingTransformer));

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            if (ctx.Header.ModelVerWritten < 0x00010003)
            {
                int cbFloat = ctx.Reader.ReadInt32();
                env.CheckDecode(cbFloat == sizeof(float));
            }
            return new NgramExtractingTransformer(host, ctx);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            // *** Binary format ***
            // <base>
            // for each added column
            //   _transformInfo
            //   the n-gram SequencePool
            //   the n-gram inverse document frequencies
            SaveColumns(ctx);
            for (int i = 0; i < _transformInfos.Length; i++)
            {
                _transformInfos[i].Save(ctx);
                _ngramMaps[i].Save(ctx.Writer);
                ctx.Writer.WriteDoubleArray(_invDocFreqs[i]);
            }
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase, ISaveAsOnnx
        {
            private readonly DataViewType[] _srcTypes;
            private readonly int[] _srcCols;
            private readonly DataViewType[] _types;
            private readonly NgramExtractingTransformer _parent;

            public Mapper(NgramExtractingTransformer parent, DataViewSchema inputSchema)
               : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _types = new DataViewType[_parent.ColumnPairs.Length];
                _srcTypes = new DataViewType[_parent.ColumnPairs.Length];
                _srcCols = new int[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    _types[i] = new VectorDataViewType(NumberDataViewType.Single, _parent._ngramMaps[i].Count);
                    inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out _srcCols[i]);
                    _srcTypes[i] = inputSchema[_srcCols[i]].Type;
                }
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    var builder = new DataViewSchema.Annotations.Builder();
                    AddMetadata(i, builder);

                    result[i] = new DataViewSchema.DetachedColumn(_parent.ColumnPairs[i].outputColumnName, _types[i], builder.ToAnnotations());
                }
                return result;
            }

            private void AddMetadata(int iinfo, DataViewSchema.Annotations.Builder builder)
            {
                if (InputSchema[_srcCols[iinfo]].HasKeyValues())
                {
                    ValueGetter<VBuffer<ReadOnlyMemory<char>>> getter = (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                    {
                        GetSlotNames(iinfo, _parent._ngramMaps[iinfo].Count, ref dst);
                    };

                    var slotNamesType = new VectorDataViewType(TextDataViewType.Instance, _parent._ngramMaps[iinfo].Count);
                    builder.AddSlotNames(_parent._ngramMaps[iinfo].Count, getter);
                }
            }

            private void GetSlotNames(int iinfo, int size, ref VBuffer<ReadOnlyMemory<char>> dst)
            {
                var itemType = _srcTypes[iinfo].GetItemType();
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                Host.Assert(InputSchema[_srcCols[iinfo]].HasKeyValues());

                var unigramNames = new VBuffer<ReadOnlyMemory<char>>();

                // Get the key values of the unigrams.
                var keyCount = itemType.GetKeyCountAsInt32(Host);
                InputSchema[_srcCols[iinfo]].GetKeyValues(ref unigramNames);
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

                    // Get the unigrams composing the current n-gram.
                    ComposeNgramString(ngram, n, sb, keyCount, in unigramNames);
                    dstEditor.Values[slot] = sb.ToString().AsMemory();
                }

                dst = dstEditor.Commit();
            }

            private IEnumerable<long> GetNgramData(int iinfo, out long[] ngramCounts, out double[] weights, out List<long> indexes)
            {
                var transformInfo = _parent._transformInfos[iinfo];
                var itemType = _srcTypes[iinfo].GetItemType();

                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                Host.Assert(InputSchema[_srcCols[iinfo]].HasKeyValues());

                // Get the key values of the unigrams.
                var keyCount = itemType.GetKeyCountAsInt32(Host);

                var maxNGramLength = transformInfo.NgramLength;

                var pool = _parent._ngramMaps[iinfo];

                // the ngrams in ML.NET are sequentially organized. e.g. {a, a|b, b, b|c...}
                // in onnx, they need to be separated by type. e.g. {a, b, c, a|b, b|c...}
                // since the resulting vectors need to match, we need to create a mapping
                // between the two and store it in the node attributes

                // create seprate lists to track the ids of 1-grams, 2-grams etc
                // because they need to be in adjacent regions in the same list
                // when supplied to onnx
                // We later concatenate all these separate n-gram lists
                var ngramIds = new List<long>[maxNGramLength];
                var ngramIndexes = new List<long>[maxNGramLength];
                for (int i = 0; i < ngramIds.Length; i++)
                {
                    ngramIds[i] = new List<long>();
                    ngramIndexes[i] = new List<long>();
                    //ngramWeights[i] = new List<float>();
                }

                weights = new double[pool.Count];

                uint[] ngram = new uint[maxNGramLength];
                for (int i = 0; i < pool.Count; i++)
                {
                    var n = pool.GetById(i, ref ngram);
                    Host.Assert(n >= 0);

                    // add the id of each gram to the corresponding ids list
                    for (int j = 0; j < n; j++)
                        ngramIds[n - 1].Add(ngram[j]);

                    // add the indexes to the corresponding list
                    ngramIndexes[n - 1].Add(i);

                    if (transformInfo.RequireIdf)
                        weights[i] = _parent._invDocFreqs[iinfo][i];
                    else
                        weights[i] = 1.0f;
                }

                // initialize the ngramCounts array with start-index of each n-gram
                int start = 0;
                ngramCounts = new long[maxNGramLength];
                for (int i = 0; i < maxNGramLength; i++)
                {
                    ngramCounts[i] = start;
                    start += ngramIds[i].Count;
                }

                // concatenate all the lists and return
                IEnumerable<long> allNGramIds = ngramIds[0];
                indexes = ngramIndexes[0];
                for (int i = 1; i < maxNGramLength; i++)
                {
                    allNGramIds = Enumerable.Concat(allNGramIds, ngramIds[i]);
                    indexes = indexes.Concat(ngramIndexes[i]).ToList();
                }

                return allNGramIds;
            }

            private void ComposeNgramString(uint[] ngram, int count, StringBuilder sb, int keyCount, in VBuffer<ReadOnlyMemory<char>> terms)
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
                        term = terms.GetItemOrDefault((int)unigram - 1);
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

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;

                var getSrc = RowCursorUtils.GetVecGetterAs<uint>(NumberDataViewType.UInt32, input, _srcCols[iinfo]);
                var src = default(VBuffer<uint>);
                var bldr = new NgramBufferBuilder(_parent._transformInfos[iinfo].NgramLength, _parent._transformInfos[iinfo].SkipLength,
                    _parent._ngramMaps[iinfo].Count, GetNgramIdFinder(iinfo));
                var keyCount = (uint)_srcTypes[iinfo].GetItemType().GetKeyCount();
                if (keyCount == 0)
                    keyCount = uint.MaxValue;

                ValueGetter<VBuffer<float>> del;
                switch (_parent._transformInfos[iinfo].Weighting)
                {
                    case NgramExtractingEstimator.WeightingCriteria.TfIdf:
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
                    case NgramExtractingEstimator.WeightingCriteria.Idf:
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
                    case NgramExtractingEstimator.WeightingCriteria.Tf:
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

            public bool CanSaveOnnx(OnnxContext ctx) => true;

            public void SaveAsOnnx(OnnxContext ctx)
            {
                Host.CheckValue(ctx, nameof(ctx));

                int numColumns = _parent.ColumnPairs.Length;
                for (int iinfo = 0; iinfo < numColumns; ++iinfo)
                {
                    string inputColumnName = _parent.ColumnPairs[iinfo].inputColumnName;
                    if (!ctx.ContainsColumn(inputColumnName))
                        continue;

                    string outputColumnName = _parent.ColumnPairs[iinfo].outputColumnName;

                    // NOTE:
                    // There is a subtle side effect in how OnnxContextImpl works
                    // AddIntermediateVariable picks a unique name for the variable being added
                    // and overwrites any existing old name with the new name in its column map
                    // If the pipeline consists of the same column name being transformed by more
                    // than one transformer, then the srcVariableName used must be the old name
                    // and the dstVariableName must be the newly chosen unique name
                    // Therefore, GetVariableName must always be called before AddIntermediateVariable here
                    string srcVariableName = ctx.GetVariableName(inputColumnName);
                    string dstVariableName = ctx.AddIntermediateVariable(_srcTypes[iinfo], outputColumnName, true);
                    SaveAsOnnxCore(ctx, iinfo, srcVariableName, dstVariableName);
                }
            }

            private void SaveAsOnnxCore(OnnxContext ctx, int iinfo, string srcVariableName, string dstVariableName)
            {
                const int minimumOpSetVersion = 9;
                ctx.CheckOpSetVersion(minimumOpSetVersion, LoaderSignature);

                var transformInfo = _parent._transformInfos[iinfo];

                // TfIdfVectorizer accepts strings, int32 and int64 tensors.
                // But in the ML.NET implementation of the NGramTransform, it only accepts keys as inputs
                // That are the result of ValueToKeyMapping transformer, which outputs UInt32 values,
                // Or TokenizingByCharacters, which outputs UInt16 values
                // So, if it is UInt32, UInt64, or UInt16, cast the output here to Int32/Int64
                string opType;
                var vectorType = _srcTypes[iinfo] as VectorDataViewType;

                if ((vectorType != null) &&
                    ((vectorType.RawType == typeof(VBuffer<UInt32>)) || (vectorType.RawType == typeof(VBuffer<UInt64>)) ||
                    (vectorType.RawType == typeof(VBuffer<UInt16>))))
                {
                    var dataKind = _srcTypes[iinfo] == NumberDataViewType.UInt32 ? DataKind.Int32 : DataKind.Int64;

                    opType = "Cast";
                    string castOutput = ctx.AddIntermediateVariable(_srcTypes[iinfo], "CastOutput", true);

                    var castNode = ctx.CreateNode(opType, srcVariableName, castOutput, ctx.GetNodeName(opType), "");
                    var t = InternalDataKindExtensions.ToInternalDataKind(dataKind).ToType();
                    castNode.AddAttribute("to", t);

                    srcVariableName = castOutput;
                }

                opType = "TfIdfVectorizer";
                var node = ctx.CreateNode(opType, srcVariableName, dstVariableName, ctx.GetNodeName(opType), "");
                node.AddAttribute("max_gram_length", transformInfo.NgramLength);
                node.AddAttribute("max_skip_count", transformInfo.SkipLength);
                node.AddAttribute("min_gram_length", transformInfo.UseAllLengths ? 1 : transformInfo.NgramLength);

                string mode;
                if (transformInfo.RequireIdf)
                {
                    mode = transformInfo.Weighting == NgramExtractingEstimator.WeightingCriteria.Idf ? "IDF" : "TFIDF";
                }
                else
                {
                    mode = "TF";
                }
                node.AddAttribute("mode", mode);

                long[] ngramCounts;
                double[] ngramWeights;
                List<long> ngramIndexes;

                var ngramIds = GetNgramData(iinfo, out ngramCounts, out ngramWeights, out ngramIndexes);

                node.AddAttribute("ngram_counts", ngramCounts);
                node.AddAttribute("pool_int64s", ngramIds);
                node.AddAttribute("ngram_indexes", ngramIndexes);
                node.AddAttribute("weights", ngramWeights);
            }

        }
    }

    /// <summary>
    /// Produces a vector of counts of n-grams (sequences of consecutive words) encountered in the input text.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | Yes |
    /// | Input column data type | Vector of [key](xref:Microsoft.ML.Data.KeyDataViewType) type. |
    /// | Output column data type | Known-sized vector of <xref:System.Single> |
    /// | Exportable to ONNX | Yes |
    ///
    /// The resulting <xref:Microsoft.ML.Transforms.Text.NgramExtractingTransformer>
    /// creates a new column, named as specified in the output column name parameters, where each
    /// input vector is mapped to a vector of counts of n-grams (sequences of consecutive words) encountered in the input text.
    ///
    /// The estimator builds a dictionary of n-grams and the <xref:Microsoft.ML.Transforms.Text.NgramExtractingTransformer>
    /// uses the id in the dictionary as the index in the count vector that it produces.
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]></format>
    /// </remarks>
    /// <seealso cref="TextCatalog.ProduceNgrams(TransformsCatalog.TextTransforms, string, string, int, int, bool, int, WeightingCriteria)"/>
    public sealed class NgramExtractingEstimator : IEstimator<NgramExtractingTransformer>
    {
        /// <summary>
        /// A statistical measure used to evaluate how important a word is to a document in a corpus.
        /// This enumeration is serialized.
        /// </summary>
        public enum WeightingCriteria
        {
            /// <summary>Term Frequency. Calculated based on the number of occurrences in the document.</summary>
            [EnumValueDisplay("TF (Term Frequency)")]
            Tf = 0,

            /// <summary>
            /// Inverse Document Frequency. A ratio (the logarithm of inverse relative frequency)
            /// that measures the information a slot provides by determining how common or rare it is across the entire corpus.
            /// </summary>
            [EnumValueDisplay("IDF (Inverse Document Frequency)")]
            Idf = 1,

            /// <summary>The product of the term frequency and the inverse document frequency.</summary>
            [EnumValueDisplay("TF-IDF")]
            TfIdf = 2
        }

        internal static class Defaults
        {
            public const int NgramLength = 2;
            public const bool UseAllLengths = true;
            public const int SkipLength = 0;
            public const int MaximumNgramsCount = 10000000;
            public const WeightingCriteria Weighting = WeightingCriteria.Tf;
        }

        private readonly IHost _host;
        private readonly ColumnOptions[] _columns;

        /// <summary>
        /// Produces a bag of counts of n-grams (sequences of consecutive words) in <paramref name="inputColumnName"/>
        /// and outputs bag of word vector as <paramref name="outputColumnName"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an n-gram.</param>
        /// <param name="useAllLengths">Whether to include all n-gram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maximumNgramsCount">Maximum number of n-grams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        internal NgramExtractingEstimator(IHostEnvironment env,
            string outputColumnName, string inputColumnName = null,
            int ngramLength = Defaults.NgramLength,
            int skipLength = Defaults.SkipLength,
            bool useAllLengths = Defaults.UseAllLengths,
            int maximumNgramsCount = Defaults.MaximumNgramsCount,
            WeightingCriteria weighting = Defaults.Weighting)
            : this(env, new[] { (outputColumnName, inputColumnName ?? outputColumnName) }, ngramLength, skipLength, useAllLengths, maximumNgramsCount, weighting)
        {
        }

        /// <summary>
        /// Produces a bag of counts of n-grams (sequences of consecutive words) in <paramref name="columns.inputs"/>
        /// and outputs bag of word vector for each output in <paramref name="columns.output"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to compute bag of word vector.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an n-gram.</param>
        /// <param name="useAllLengths">Whether to include all n-gram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maximumNgramsCount">Maximum number of n-grams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        internal NgramExtractingEstimator(IHostEnvironment env,
            (string outputColumnName, string inputColumnName)[] columns,
            int ngramLength = Defaults.NgramLength,
            int skipLength = Defaults.SkipLength,
            bool useAllLengths = Defaults.UseAllLengths,
            int maximumNgramsCount = Defaults.MaximumNgramsCount,
            WeightingCriteria weighting = Defaults.Weighting)
            : this(env, columns.Select(x => new ColumnOptions(x.outputColumnName, x.inputColumnName, ngramLength, skipLength, useAllLengths, weighting, maximumNgramsCount)).ToArray())
        {
        }

        /// <summary>
        /// Produces a bag of counts of n-grams (sequences of consecutive words) in <paramref name="columns.inputs"/>
        /// and outputs bag of word vector for each output in <paramref name="columns.output"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Array of columns with information how to transform data.</param>
        internal NgramExtractingEstimator(IHostEnvironment env, params ColumnOptions[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(NgramExtractingEstimator));
            _columns = columns;
        }

        /// <summary>
        /// Trains and returns a <see cref="NgramExtractingTransformer"/>.
        /// </summary>
        public NgramExtractingTransformer Fit(IDataView input) => new NgramExtractingTransformer(_host, input, _columns);

        internal static bool IsColumnTypeValid(DataViewType type)
        {
            if (!(type is VectorDataViewType vectorType))
                return false;
            if (!(vectorType.ItemType is KeyDataViewType itemKeyType))
                return false;
            // Can only accept key types that can be converted to U4.
            if (itemKeyType.Count == 0 && !NgramUtils.IsValidNgramRawType(itemKeyType.RawType))
                return false;
            return true;
        }

        internal static bool IsSchemaColumnValid(SchemaShape.Column col)
        {
            if (col.Kind == SchemaShape.Column.VectorKind.Scalar)
                return false;
            if (!col.IsKey)
                return false;
            // Can only accept key types that can be converted to U8.
            if (!NgramUtils.IsValidNgramRawType(col.ItemType.RawType))
                return false;
            return true;
        }

        internal const string ExpectedColumnType = "Expected vector of Key type, and Key is convertible to UInt32";

        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        [BestFriend]
        internal sealed class ColumnOptions
        {
            /// <summary>Name of the column resulting from the transformation of <see cref="InputColumnName"/>.</summary>
            public readonly string Name;
            /// <summary>Name of column to transform.</summary>
            public readonly string InputColumnName;
            /// <summary>Maximum n-gram length.</summary>
            public readonly int NgramLength;
            /// <summary>Maximum number of tokens to skip when constructing an n-gram.</summary>
            public readonly int SkipLength;
            /// <summary>Whether to store all n-gram lengths up to ngramLength, or only ngramLength.</summary>
            public readonly bool UseAllLengths;
            /// <summary>The weighting criteria.</summary>
            public readonly WeightingCriteria Weighting;
            /// <summary>
            /// Underlying state of <see cref="MaximumNgramsCounts"/>.
            /// </summary>
            private readonly ImmutableArray<int> _maximumNgramsCounts;
            /// <summary>
            /// Contains the maximum number of terms (that is, n-grams) to store in the dictionary, for each level of n-grams,
            /// from n=1 (in position 0) up to n=<see cref="NgramLength"/> (in position <see cref="NgramLength"/>-1)
            /// </summary>
            public IReadOnlyList<int> MaximumNgramsCounts => _maximumNgramsCounts;

            /// <summary>
            /// Describes how the transformer handles one Gcn column pair.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="name"/> will be used as source.</param>
            /// <param name="ngramLength">Maximum n-gram length.</param>
            /// <param name="skipLength">Maximum number of tokens to skip when constructing an n-gram.</param>
            /// <param name="useAllLengths">Whether to store all n-gram lengths up to ngramLength, or only ngramLength.</param>
            /// <param name="weighting">The weighting criteria.</param>
            /// <param name="maximumNgramsCount">Maximum number of n-grams to store in the dictionary.</param>
            public ColumnOptions(string name, string inputColumnName = null,
                int ngramLength = Defaults.NgramLength,
                int skipLength = Defaults.SkipLength,
                bool useAllLengths = Defaults.UseAllLengths,
                WeightingCriteria weighting = Defaults.Weighting,
                int maximumNgramsCount = Defaults.MaximumNgramsCount)
                : this(name, ngramLength, skipLength, useAllLengths, weighting, new int[] { maximumNgramsCount }, inputColumnName ?? name)
            {
            }

            internal ColumnOptions(string name,
                int ngramLength,
                int skipLength,
                bool useAllLengths,
                WeightingCriteria weighting,
                int[] maximumNgramsCounts,
                string inputColumnName = null)
            {
                if (ngramLength == 1 && skipLength != 0)
                    throw Contracts.ExceptUserArg(nameof(skipLength), string.Format(
                        "{0} (actual value: {1}) can only be zero when {2} set to one.", nameof(skipLength), skipLength, nameof(ngramLength)));
                if (ngramLength + skipLength > NgramBufferBuilder.MaxSkipNgramLength)
                    throw Contracts.ExceptUserArg(nameof(skipLength),
                        $"The sum of skipLength and ngramLength must be less than or equal to {NgramBufferBuilder.MaxSkipNgramLength}");
                Contracts.CheckUserArg(0 < ngramLength && ngramLength <= NgramBufferBuilder.MaxSkipNgramLength, nameof(ngramLength));

                var limits = new int[ngramLength];
                if (!useAllLengths)
                {
                    Contracts.CheckUserArg(Utils.Size(maximumNgramsCounts) == 0 ||
                        Utils.Size(maximumNgramsCounts) == 1 && maximumNgramsCounts[0] > 0, nameof(maximumNgramsCounts));
                    limits[ngramLength - 1] = Utils.Size(maximumNgramsCounts) == 0 ? Defaults.MaximumNgramsCount : maximumNgramsCounts[0];
                }
                else
                {
                    Contracts.CheckUserArg(Utils.Size(maximumNgramsCounts) <= ngramLength, nameof(maximumNgramsCounts));
                    Contracts.CheckUserArg(Utils.Size(maximumNgramsCounts) == 0 || maximumNgramsCounts.All(i => i >= 0) && maximumNgramsCounts[maximumNgramsCounts.Length - 1] > 0, nameof(maximumNgramsCounts));
                    var extend = Utils.Size(maximumNgramsCounts) == 0 ? Defaults.MaximumNgramsCount : maximumNgramsCounts[maximumNgramsCounts.Length - 1];
                    limits = Utils.BuildArray(ngramLength, i => i < Utils.Size(maximumNgramsCounts) ? maximumNgramsCounts[i] : extend);
                }
                _maximumNgramsCounts = ImmutableArray.Create(limits);

                Name = name;
                InputColumnName = inputColumnName ?? name;
                NgramLength = ngramLength;
                SkipLength = skipLength;
                UseAllLengths = useAllLengths;
                Weighting = weighting;
            }
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colInfo in _columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.InputColumnName, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName);
                if (!IsSchemaColumnValid(col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName, ExpectedColumnType, col.GetTypeString());
                var metadata = new List<SchemaShape.Column>();
                if (col.NeedsSlotNames())
                    metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, TextDataViewType.Instance, false));
                result[colInfo.Name] = new SchemaShape.Column(colInfo.Name, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false, new SchemaShape(metadata));
            }
            return new SchemaShape(result.Values);
        }
    }
}
