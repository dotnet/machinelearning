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
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Text;

[assembly: LoadableClass(NgramHashingTransformer.Summary, typeof(IDataTransform), typeof(NgramHashingTransformer), typeof(NgramHashingTransformer.Options), typeof(SignatureDataTransform),
    "Ngram Hash Transform", "NgramHashTransform", "NgramHash")]

[assembly: LoadableClass(NgramHashingTransformer.Summary, typeof(IDataTransform), typeof(NgramHashingTransformer), null, typeof(SignatureLoadDataTransform),
    "Ngram Hash Transform", NgramHashingTransformer.LoaderSignature)]

[assembly: LoadableClass(NgramHashingTransformer.Summary, typeof(NgramHashingTransformer), null, typeof(SignatureLoadModel),
    "Ngram Hash Transform", NgramHashingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(NgramHashingTransformer), null, typeof(SignatureLoadRowMapper),
    "Ngram Hash Transform", NgramHashingTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms.Text
{
    /// <summary>
    /// Produces a bag of counts of ngrams (sequences of consecutive words of length 1-n) in a given text.
    /// It does so by hashing each ngram and using the hash value as the index in the bag.
    /// </summary>
    public sealed class NgramHashingTransformer : RowToRowTransformerBase
    {
        internal sealed class Column : ManyToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum ngram length", ShortName = "ngram")]
            public int? NgramLength;

            [Argument(ArgumentType.AtMostOnce, HelpText =
                "Whether to include all ngram lengths up to " + nameof(NgramLength) + " or only " + nameof(NgramLength), Name = "AllLengths", ShortName = "all")]
            public bool? UseAllLengths;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Maximum number of tokens to skip when constructing an ngram",
                ShortName = "skips")]
            public int? SkipLength;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Number of bits to hash into. Must be between 1 and 30, inclusive.",
                Name = "HashBits", ShortName = "bits")]
            public int? NumberOfBits;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
            public uint? Seed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to rehash unigrams", ShortName = "rehash")]
            public bool? RehashUnigrams;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the position of each source column should be included in the hash (when there are multiple source columns).", ShortName = "ord")]
            public bool? Ordered;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.",
                ShortName = "ih")]
            public int? MaximumNumberOfInverts;

            internal static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            private protected override bool TryParse(string str)
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
                NumberOfBits = bits;
                return true;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (NgramLength != null || UseAllLengths != null || SkipLength != null || Seed != null ||
                    RehashUnigrams != null || Ordered != null || MaximumNumberOfInverts != null)
                {
                    return false;
                }
                if (NumberOfBits == null)
                    return TryUnparseCore(sb);

                string extra = NumberOfBits.Value.ToString();
                return TryUnparseCore(sb, extra);
            }
        }

        internal sealed class Options
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:numberOfBits:src)",
                ShortName = "col",
                SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum ngram length", ShortName = "ngram", SortOrder = 3)]
            public int NgramLength = NgramHashingEstimator.Defaults.NgramLength;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Whether to include all ngram lengths up to " + nameof(NgramLength) + " or only " + nameof(NgramLength),
                Name = "AllLengths", ShortName = "all", SortOrder = 4)]
            public bool UseAllLengths = NgramHashingEstimator.Defaults.UseAllLengths;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Maximum number of tokens to skip when constructing an ngram",
                ShortName = "skips", SortOrder = 3)]
            public int SkipLength = NgramHashingEstimator.Defaults.SkipLength;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Number of bits to hash into. Must be between 1 and 30, inclusive.",
                Name = "HashBits", ShortName = "bits", SortOrder = 2)]
            public int NumberOfBits = NgramHashingEstimator.Defaults.NumberOfBits;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
            public uint Seed = NgramHashingEstimator.Defaults.Seed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to rehash unigrams", ShortName = "rehash")]
            public bool RehashUnigrams = NgramHashingEstimator.Defaults.RehashUnigrams;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Whether the position of each source column should be included in the hash (when there are multiple source columns).",
                ShortName = "ord", SortOrder = 6)]
            public bool Ordered = NgramHashingEstimator.Defaults.UseOrderedHashing;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.",
                ShortName = "ih")]
            public int MaximumNumberOfInverts = NgramHashingEstimator.Defaults.MaximumNumberOfInverts;
        }

        internal const string Summary = "Produces a bag of counts of ngrams (sequences of consecutive values of length 1-n) in a given vector of keys. "
          + "It does so by hashing each ngram and using the hash value as the index in the bag.";

        internal const string LoaderSignature = "NgramHashTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "HASHGRAM",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010002, // Invert hash key values, hash fix
                verWrittenCur: 0x00010003, // Get rid of writing float size in model context and change saving format
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(NgramHashingTransformer).Assembly.FullName);
        }

        private const int VersionTransformer = 0x00010003;

        private readonly ImmutableArray<NgramHashingEstimator.ColumnOptions> _columns;
        private readonly VBuffer<ReadOnlyMemory<char>>[] _slotNames;
        private readonly VectorType[] _slotNamesTypes;

        /// <summary>
        /// Constructor for case where you don't need to 'train' transform on data, for example, MaximumNumberOfInverts for all columns set to zero.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="columns">Description of dataset columns and how to process them.</param>
        internal NgramHashingTransformer(IHostEnvironment env, params NgramHashingEstimator.ColumnOptions[] columns) :
            base(Contracts.CheckRef(env, nameof(env)).Register(nameof(NgramHashingTransformer)))
        {
            _columns = columns.ToImmutableArray();
            foreach (var column in _columns)
            {
                if (column.MaximumNumberOfInverts != 0)
                    throw Host.ExceptParam(nameof(columns), $"Found colunm with {nameof(column.MaximumNumberOfInverts)} set to non zero value, please use { nameof(NgramHashingEstimator)} instead");
            }
        }

        internal NgramHashingTransformer(IHostEnvironment env, IDataView input, params NgramHashingEstimator.ColumnOptions[] columns) :
            base(Contracts.CheckRef(env, nameof(env)).Register(nameof(NgramHashingTransformer)))
        {
            Contracts.CheckValue(columns, nameof(columns));
            _columns = columns.ToImmutableArray();

            // Let's validate input schema and check which columns requried maximumNumberOfInverts.
            int[] invertHashMaxCounts = new int[_columns.Length];
            HashSet<int> columnWithInvertHash = new HashSet<int>();
            var sourceColumnsForInvertHash = new List<DataViewSchema.Column>();
            for (int i = 0; i < _columns.Length; i++)
            {
                int invertHashMaxCount;
                if (_columns[i].MaximumNumberOfInverts == -1)
                    invertHashMaxCount = int.MaxValue;
                else
                    invertHashMaxCount = _columns[i].MaximumNumberOfInverts;
                if (invertHashMaxCount > 0)
                {
                    columnWithInvertHash.Add(i);
                    invertHashMaxCounts[i] = invertHashMaxCount;
                    for (int j = 0; j < _columns[i].InputColumnNamesArray.Length; j++)
                    {
                        if (!input.Schema.TryGetColumnIndex(_columns[i].InputColumnNamesArray[j], out int srcCol))
                            throw Host.ExceptSchemaMismatch(nameof(input), "input", _columns[i].InputColumnNamesArray[j]);
                        var columnType = input.Schema[srcCol].Type;
                        if (!NgramHashingEstimator.IsColumnTypeValid(input.Schema[srcCol].Type))
                            throw Host.ExceptSchemaMismatch(nameof(input), "input", _columns[i].InputColumnNamesArray[j], NgramHashingEstimator.ExpectedColumnType, columnType.ToString());
                        sourceColumnsForInvertHash.Add(input.Schema[srcCol]);
                    }
                }
            }
            // In case of maximumNumberOfInverts set to non zero value for at least one column.
            if (Utils.Size(columnWithInvertHash) > 0)
            {
                var active = new bool[1];
                string[][] friendlyNames = _columns.Select(c => c.FriendlyNames).ToArray();
                // We will create invert hash helper class, which would store in itself all original ngrams and their mapping into hash values.
                var helper = new InvertHashHelper(this, input.Schema, friendlyNames, sourceColumnsForInvertHash, invertHashMaxCounts);
                // in order to get all original ngrams we have to go data in same way as we would process it, so let's create mapper with decorate function.
                var mapper = new Mapper(this, input.Schema, helper.Decorate);
                // Let's create cursor to iterate over input data.
                using (var rowCursor = input.GetRowCursor(sourceColumnsForInvertHash))
                {
                    Action disp;
                    // We create mapper getters on top of input cursor
                    var del = (mapper as IRowMapper).CreateGetters(rowCursor, columnWithInvertHash.Contains, out disp);
                    var valueGetters = new ValueGetter<VBuffer<float>>[columnWithInvertHash.Count];
                    for (int i = 0; i < columnWithInvertHash.Count; i++)
                        valueGetters[i] = del[i] as ValueGetter<VBuffer<float>>;
                    VBuffer<float> value = default;
                    // and invoke each getter for each row.
                    while (rowCursor.MoveNext())
                    {
                        for (int i = 0; i < columnWithInvertHash.Count; i++)
                            valueGetters[i](ref value);
                    }
                    // decorate function of helper object captured all encountered ngrams so, we ask it to give us metadata information for slot names.
                    _slotNames = helper.SlotNamesMetadata(out _slotNamesTypes);
                }
            }
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int number of columns
            // columns
            ctx.Writer.Write(_columns.Length);
            foreach (var column in _columns)
                column.Save(ctx);
            TextModelHelper.SaveAll(Host, ctx, _columns.Length, _slotNames);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private NgramHashingTransformer(IHostEnvironment env, ModelLoadContext ctx, bool loadLegacy = false) :
            base(Contracts.CheckRef(env, nameof(env)).Register(nameof(NgramHashingTransformer)))
        {
            Host.CheckValue(ctx, nameof(ctx));
            if (loadLegacy)
            {
                int cbFloat = ctx.Reader.ReadInt32();
                Host.CheckDecode(cbFloat == sizeof(float));
            }
            var columnsLength = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(columnsLength > 0);
            var columns = new NgramHashingEstimator.ColumnOptions[columnsLength];
            if (!loadLegacy)
            {
                // *** Binary format ***
                // int number of columns
                // columns
                for (int i = 0; i < columnsLength; i++)
                    columns[i] = new NgramHashingEstimator.ColumnOptions(ctx);
            }
            else
            {
                // *** Binary format ***
                // int: number of added columns
                // for each added column
                //   int: id of output column name
                //   int: number of input column names
                //   int[]: ids of input column names
                var outputs = new string[columnsLength];
                var inputs = new string[columnsLength][];
                for (int i = 0; i < columnsLength; i++)
                {
                    outputs[i] = ctx.LoadNonEmptyString();

                    int csrc = ctx.Reader.ReadInt32();
                    Contracts.CheckDecode(csrc > 0);
                    inputs[i] = new string[csrc];
                    for (int j = 0; j < csrc; j++)
                    {
                        string src = ctx.LoadNonEmptyString();
                        inputs[i][j] = src;
                    }
                }

                // *** Binary format ***
                // int number of columns
                // columns
                for (int i = 0; i < columnsLength; i++)
                    columns[i] = new NgramHashingEstimator.ColumnOptions(ctx, outputs[i], inputs[i]);
            }
            _columns = columns.ToImmutableArray();
            TextModelHelper.LoadAll(Host, ctx, columnsLength, out _slotNames, out _slotNamesTypes);
        }

        // Factory method for SignatureDataTransform.
        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            env.CheckValue(options.Column, nameof(options.Column));
            var cols = new NgramHashingEstimator.ColumnOptions[options.Column.Length];
            using (var ch = env.Start("ValidateArgs"))
            {

                for (int i = 0; i < cols.Length; i++)
                {
                    var item = options.Column[i];
                    cols[i] = new NgramHashingEstimator.ColumnOptions(
                        item.Name,
                        item.Source ?? new string[] { item.Name },
                        item.NgramLength ?? options.NgramLength,
                        item.SkipLength ?? options.SkipLength,
                        item.UseAllLengths ?? options.UseAllLengths,
                        item.NumberOfBits ?? options.NumberOfBits,
                        item.Seed ?? options.Seed,
                        item.Ordered ?? options.Ordered,
                        item.MaximumNumberOfInverts ?? options.MaximumNumberOfInverts,
                        item.RehashUnigrams ?? options.RehashUnigrams
                        );
                };
            }
            return new NgramHashingTransformer(env, input, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static NgramHashingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(nameof(NgramHashingTransformer));
            ctx.CheckAtModel(GetVersionInfo());
            return new NgramHashingTransformer(host, ctx, ctx.Header.ModelVerWritten < VersionTransformer);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private sealed class Mapper : MapperBase
        {
            private readonly NgramHashingTransformer _parent;
            private readonly VectorType[] _types;
            private readonly int[][] _srcIndices;
            private readonly DataViewType[][] _srcTypes;
            private readonly FinderDecorator _decorator;

            public Mapper(NgramHashingTransformer parent, DataViewSchema inputSchema, FinderDecorator decorator = null) :
                base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                _parent = parent;
                _decorator = decorator;
                _types = new VectorType[_parent._columns.Length];
                _srcIndices = new int[_parent._columns.Length][];
                _srcTypes = new DataViewType[_parent._columns.Length][];
                for (int i = 0; i < _parent._columns.Length; i++)
                {
                    _srcIndices[i] = new int[_parent._columns[i].InputColumnNamesArray.Length];
                    _srcTypes[i] = new DataViewType[_parent._columns[i].InputColumnNamesArray.Length];
                    for (int j = 0; j < _parent._columns[i].InputColumnNamesArray.Length; j++)
                    {
                        var srcName = _parent._columns[i].InputColumnNamesArray[j];
                        if (!inputSchema.TryGetColumnIndex(srcName, out int srcCol))
                            throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", srcName);
                        var columnType = inputSchema[srcCol].Type;
                        if (!NgramHashingEstimator.IsColumnTypeValid(columnType))
                            throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", srcName, NgramHashingEstimator.ExpectedColumnType, columnType.ToString());
                        var srcType = inputSchema[srcCol].Type;
                        _srcIndices[i][j] = srcCol;
                        _srcTypes[i][j] = srcType;
                    }

                    _types[i] = new VectorType(NumberDataViewType.Single, 1 << _parent._columns[i].NumberOfBits);
                }
            }

            private NgramIdFinder GetNgramIdFinder(int iinfo)
            {
                uint mask = (1U << _parent._columns[iinfo].NumberOfBits) - 1;
                int ngramLength = _parent._columns[iinfo].NgramLength;
                bool rehash = _parent._columns[iinfo].RehashUnigrams;
                bool ordered = _parent._columns[iinfo].UseOrderedHashing;
                bool all = _parent._columns[iinfo].UseAllLengths;
                uint seed = _parent._columns[iinfo].Seed;

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

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                int srcCount = _srcIndices[iinfo].Length;
                ValueGetter<VBuffer<uint>>[] getSrc = new ValueGetter<VBuffer<uint>>[srcCount];
                for (int isrc = 0; isrc < srcCount; isrc++)
                    getSrc[isrc] = RowCursorUtils.GetVecGetterAs<uint>(NumberDataViewType.UInt32, input, _srcIndices[iinfo][isrc]);
                var src = default(VBuffer<uint>);
                var ngramIdFinder = GetNgramIdFinder(iinfo);
                if (_decorator != null)
                    ngramIdFinder = _decorator(iinfo, ngramIdFinder);
                var bldr = new NgramBufferBuilder(_parent._columns[iinfo].NgramLength, _parent._columns[iinfo].SkipLength,
                    _types[iinfo].Size, ngramIdFinder);
                var keyCounts = _srcTypes[iinfo].Select(
                    t => (t.GetItemType() is KeyType keyType && keyType.Count > 0) ? (uint)keyType.Count : uint.MaxValue).ToArray();

                // REVIEW: Special casing the srcCount==1 case could potentially improve perf.
                ValueGetter<VBuffer<float>> del =
                    (ref VBuffer<float> dst) =>
                    {
                        bldr.Reset();
                        for (int i = 0; i < srcCount; i++)
                        {
                            getSrc[i](ref src);
                            bldr.AddNgrams(in src, i, keyCounts[i]);
                        }
                        bldr.GetResult(ref dst);
                    };
                return del;
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                var active = new bool[InputSchema.Count];
                for (int i = 0; i < _srcIndices.Length; i++)
                {
                    if (activeOutput(i))
                    {
                        foreach (var src in _srcIndices[i])
                            active[src] = true;
                    }
                }
                return col => active[col];
            }

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_parent._columns.Length];
                for (int i = 0; i < _parent._columns.Length; i++)
                {
                    var builder = new DataViewSchema.Annotations.Builder();
                    AddMetadata(i, builder);
                    result[i] = new DataViewSchema.DetachedColumn(_parent._columns[i].Name, _types[i], builder.ToAnnotations());
                }
                return result;
            }

            private void AddMetadata(int i, DataViewSchema.Annotations.Builder builder)
            {
                if (_parent._slotNamesTypes != null && _parent._slotNamesTypes[i] != null)
                {
                    ValueGetter<VBuffer<ReadOnlyMemory<char>>> getter = (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                    {
                        _parent._slotNames[i].CopyTo(ref dst);
                    };
                    builder.Add(AnnotationUtils.Kinds.SlotNames, _parent._slotNamesTypes[i], getter);
                }
            }
        }

        private delegate NgramIdFinder FinderDecorator(int iinfo, NgramIdFinder finder);

        private sealed class InvertHashHelper
        {
            private readonly NgramHashingTransformer _parent;
            // One per output column (will be null if invert hashing is not specified for
            // this column).
            private readonly InvertHashCollector<NGram>[] _iinfoToCollector;
            // One per source column that we want to convert (will be null if we don't hash
            // them in a column where we've said we want invert hashing).
            private readonly ValueMapper<uint, StringBuilder>[] _srcTextGetters;
            // If null, or specific element is null, then just use the input column name.
            private readonly string[][] _friendlyNames;
            private readonly int[] _invertHashMaxCounts;
            private readonly int[][] _srcIndices;

            public InvertHashHelper(NgramHashingTransformer parent, DataViewSchema inputSchema, string[][] friendlyNames, IEnumerable<DataViewSchema.Column> columnsNeeded, int[] invertHashMaxCounts)
            {
                Contracts.AssertValue(parent);
                Contracts.AssertValue(friendlyNames);
                Contracts.Assert(friendlyNames.Length == parent._columns.Length);
                Contracts.AssertValue(columnsNeeded);
                Contracts.AssertValue(invertHashMaxCounts);
                Contracts.Assert(invertHashMaxCounts.Length == parent._columns.Length);
                _parent = parent;
                // One per iinfo (some may be null).
                _iinfoToCollector = new InvertHashCollector<NGram>[_parent._columns.Length];
                // One per source column (some may be null).
                _srcTextGetters = new ValueMapper<uint, StringBuilder>[inputSchema.Count];
                _invertHashMaxCounts = invertHashMaxCounts;

                foreach (var col in columnsNeeded)
                {
                    Contracts.Assert(col.Index < _srcTextGetters.Length);
                    _srcTextGetters[col.Index] = InvertHashUtils.GetSimpleMapper<uint>(inputSchema, col.Index);
                }

                _srcIndices = new int[_parent._columns.Length][];
                for (int i = 0; i < _parent._columns.Length; i++)
                {
                    _srcIndices[i] = new int[_parent._columns[i].InputColumnNamesArray.Length];
                    for (int j = 0; j < _parent._columns[i].InputColumnNamesArray.Length; j++)
                    {
                        var srcName = _parent._columns[i].InputColumnNamesArray[j];
                        if (!inputSchema.TryGetColumnIndex(srcName, out int srcCol))
                            throw _parent.Host.ExceptSchemaMismatch(nameof(inputSchema), "input", srcName);
                        _srcIndices[i][j] = srcCol;
                    }
                }
                _friendlyNames = friendlyNames;
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
                Contracts.Assert(0 <= iinfo && iinfo < _parent._columns.Length);
                Contracts.Assert(_iinfoToCollector[iinfo] == null);
                Contracts.AssertValue(finder);

                var srcIndices = _srcIndices[iinfo];

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
                        (in NGram src, ref StringBuilder dst) =>
                        {
                            Contracts.Assert(src.ISrcCol == 0);
                            if (src.Lim == 1)
                            {
                                srcMap(in src.Grams[0], ref dst);
                                return;
                            }
                            ClearDst(ref dst);
                            for (int i = 0; i < src.Lim; ++i)
                            {
                                if (i > 0)
                                    dst.Append('|');
                                srcMap(in src.Grams[i], ref temp);
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
                            srcNames[i] = _parent._columns[iinfo].InputColumnNamesArray[i];
                    }
                    Contracts.Assert(Utils.Size(srcNames) == srcIndices.Length);
                    string[] friendlyNames = _friendlyNames?[iinfo];
                    // We need to disambiguate the column name. This will be the same as the above format,
                    // just instead of "<Stuff>" it would be with "ColumnName:<Stuff>".
                    stringMapper =
                        (in NGram src, ref StringBuilder dst) =>
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
                                srcMap(in src.Grams[i], ref temp);
                                InvertHashUtils.AppendToEnd(temp, dst, ref buffer);
                            }
                        };
                }

                var collector = _iinfoToCollector[iinfo] = new InvertHashCollector<NGram>(
                    1 << _parent._columns[iinfo].NumberOfBits, _invertHashMaxCounts[iinfo],
                    stringMapper, EqualityComparer<NGram>.Default, (in NGram src, ref NGram dst) => dst = src.Clone());

                return
                    (uint[] ngram, int lim, int icol, ref bool more) =>
                    {
                        Contracts.Assert(0 <= icol && icol < srcIndices.Length);
                        Contracts.AssertValue(_srcTextGetters[srcIndices[icol]]);
                        var result = finder(ngram, lim, icol, ref more);
                        // For the hashing NgramIdFinder, a result of -1 indicates that
                        // a slot does not exist for the given ngram. We do not pass ngrams
                        // that do not have a slot to the MaximumNumberOfInverts collector.
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

            public VBuffer<ReadOnlyMemory<char>>[] SlotNamesMetadata(out VectorType[] types)
            {
                var values = new VBuffer<ReadOnlyMemory<char>>[_iinfoToCollector.Length];
                types = new VectorType[_iinfoToCollector.Length];
                for (int iinfo = 0; iinfo < _iinfoToCollector.Length; ++iinfo)
                {
                    if (_iinfoToCollector[iinfo] != null)
                    {
                        var vec = values[iinfo] = _iinfoToCollector[iinfo].GetMetadata();
                        Contracts.Assert(vec.Length == 1 << _parent._columns[iinfo].NumberOfBits);
                        types[iinfo] = new VectorType(TextDataViewType.Instance, vec.Length);
                    }
                }
                return values;
            }
        }
    }

    /// <summary>
    /// Produces a bag of counts of ngrams (sequences of consecutive words of length 1-n) in a given text.
    /// It does so by hashing each ngram and using the hash value as the index in the bag.
    ///
    /// <see cref="NgramHashingEstimator"/> is different from <see cref="WordHashBagEstimator"/> in a way that <see cref="NgramHashingEstimator"/>
    /// takes tokenized text as input while <see cref="WordHashBagEstimator"/> tokenizes text internally.
    /// </summary>
    public sealed class NgramHashingEstimator : IEstimator<NgramHashingTransformer>
    {
        /// <summary>
        /// Describes how the transformer handles one pair of mulitple inputs - singular output columns.
        /// </summary>
        [BestFriend]
        internal sealed class ColumnOptions
        {
            /// <summary>Name of the column resulting from the transformation of <see cref="InputColumnNames"/>.</summary>
            public readonly string Name;
            /// <summary>Names of the columns to transform.</summary>
            public IReadOnlyList<string> InputColumnNames => InputColumnNamesArray;
            internal readonly string[] InputColumnNamesArray;
            /// <summary>Maximum ngram length.</summary>
            public readonly int NgramLength;
            /// <summary>Maximum number of tokens to skip when constructing an ngram.</summary>
            public readonly int SkipLength;
            /// <summary>Whether to store all ngram lengths up to <see cref="NgramLength"/>, or only <see cref="NgramLength"/>.</summary>
            public readonly bool UseAllLengths;
            /// <summary>Number of bits to hash into. Must be between 1 and 31, inclusive.</summary>
            public readonly int NumberOfBits;
            /// <summary>Hashing seed.</summary>
            public readonly uint Seed;
            /// <summary>Whether the position of each term should be included in the hash.</summary>
            public readonly bool UseOrderedHashing;
            /// <summary>
            /// During hashing we constuct mappings between original values and the produced hash values.
            /// Text representation of original values are stored in the slot names of the  metadata for the new column.
            /// Hashing, as such, can map many initial values to one.
            /// <see cref="MaximumNumberOfInverts"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
            /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.
            /// </summary>
            public readonly int MaximumNumberOfInverts;
            /// <summary>Whether to rehash unigrams.</summary>
            public readonly bool RehashUnigrams;
            // For all source columns, use these friendly names for the source
            // column names instead of the real column names.
            internal string[] FriendlyNames;

            /// <summary>
            /// Describes how the transformer maps several input columns, <paramref name="inputColumnNames"/>, to a output column, <paramref name="name"/>.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnNames"/>.</param>
            /// <param name="inputColumnNames">Names of the columns to transform. </param>
            /// <param name="ngramLength">Maximum ngram length.</param>
            /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
            /// <param name="useAllLengths">Whether to store all ngram lengths up to <paramref name="ngramLength"/>, or only <paramref name="ngramLength"/>.</param>
            /// <param name="numberOfBits">Number of bits to hash into. Must be between 1 and 31, inclusive.</param>
            /// <param name="seed">Hashing seed.</param>
            /// <param name="useOrderedHashing">Whether the position of each term should be included in the hash.</param>
            /// <param name="maximumNumberOfInverts">During hashing we constuct mappings between original values and the produced hash values.
            /// Text representation of original values are stored in the slot names of the metadata for the new column.
            /// Hashing, as such, can map many initial values to one.
            /// <paramref name="maximumNumberOfInverts"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
            /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
            /// <param name="rehashUnigrams">Whether to rehash unigrams.</param>
            public ColumnOptions(string name,
                string[] inputColumnNames,
                int ngramLength = NgramHashingEstimator.Defaults.NgramLength,
                int skipLength = NgramHashingEstimator.Defaults.SkipLength,
                bool useAllLengths = NgramHashingEstimator.Defaults.UseAllLengths,
                int numberOfBits = NgramHashingEstimator.Defaults.NumberOfBits,
                uint seed = NgramHashingEstimator.Defaults.Seed,
                bool useOrderedHashing = NgramHashingEstimator.Defaults.UseOrderedHashing,
                int maximumNumberOfInverts = NgramHashingEstimator.Defaults.MaximumNumberOfInverts,
                bool rehashUnigrams = NgramHashingEstimator.Defaults.RehashUnigrams)
            {
                Contracts.CheckValue(name, nameof(name));
                Contracts.CheckValue(inputColumnNames, nameof(inputColumnNames));
                Contracts.CheckParam(!inputColumnNames.Any(r => string.IsNullOrWhiteSpace(r)), nameof(inputColumnNames),
                    "Contained some null or empty items");
                if (maximumNumberOfInverts < -1)
                    throw Contracts.ExceptParam(nameof(maximumNumberOfInverts), "Value too small, must be -1 or larger");
                // If the bits is 31 or higher, we can't declare a KeyValues of the appropriate length,
                // this requiring a VBuffer of length 1u << 31 which exceeds int.MaxValue.
                if (maximumNumberOfInverts != 0 && numberOfBits >= 31)
                    throw Contracts.ExceptParam(nameof(numberOfBits), $"Cannot support maximumNumberOfInverts for a {0} bit hash. 30 is the maximum possible.", numberOfBits);

                if (ngramLength == 1 && skipLength != 0)
                    throw Contracts.ExceptUserArg(nameof(skipLength), string.Format(
                        "{0} (actual value: {1}) can only be zero when {2} set to one.", nameof(skipLength), skipLength, nameof(ngramLength)));
                if (ngramLength + skipLength > NgramBufferBuilder.MaxSkipNgramLength)
                {
                    throw Contracts.ExceptUserArg(nameof(skipLength),
                        $"The sum of skipLength and ngramLength must be less than or equal to {NgramBufferBuilder.MaxSkipNgramLength}");
                }

                FriendlyNames = null;
                Name = name;
                InputColumnNamesArray = inputColumnNames;
                NgramLength = ngramLength;
                SkipLength = skipLength;
                UseAllLengths = useAllLengths;
                NumberOfBits = numberOfBits;
                Seed = seed;
                UseOrderedHashing = useOrderedHashing;
                MaximumNumberOfInverts = maximumNumberOfInverts;
                RehashUnigrams = rehashUnigrams;
            }

            internal ColumnOptions(ModelLoadContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // size of Inputs
                // string[] Inputs;
                // string Output;
                // int: NgramLength
                // int: SkipLength
                // int: NumberOfBits
                // uint: Seed
                // byte: Rehash
                // byte: Ordered
                // byte: AllLengths
                var inputsLength = ctx.Reader.ReadInt32();
                InputColumnNamesArray = new string[inputsLength];
                for (int i = 0; i < InputColumnNamesArray.Length; i++)
                    InputColumnNamesArray[i] = ctx.LoadNonEmptyString();
                Name = ctx.LoadNonEmptyString();
                NgramLength = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(0 < NgramLength && NgramLength <= NgramBufferBuilder.MaxSkipNgramLength);
                SkipLength = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(0 <= SkipLength && SkipLength <= NgramBufferBuilder.MaxSkipNgramLength);
                Contracts.CheckDecode(SkipLength <= NgramBufferBuilder.MaxSkipNgramLength - NgramLength);
                NumberOfBits = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(1 <= NumberOfBits && NumberOfBits <= 30);
                Seed = ctx.Reader.ReadUInt32();
                RehashUnigrams = ctx.Reader.ReadBoolByte();
                UseOrderedHashing = ctx.Reader.ReadBoolByte();
                UseAllLengths = ctx.Reader.ReadBoolByte();
            }

            internal ColumnOptions(ModelLoadContext ctx, string name, string[] inputColumnNames)
            {
                Contracts.AssertValue(ctx);
                Contracts.CheckValue(inputColumnNames, nameof(inputColumnNames));
                Contracts.CheckParam(!inputColumnNames.Any(r => string.IsNullOrWhiteSpace(r)), nameof(inputColumnNames),
                   "Contained some null or empty items");
                InputColumnNamesArray = inputColumnNames;
                Name = name;
                // *** Binary format ***
                // string Output;
                // int: NgramLength
                // int: SkipLength
                // int: NumberOfBits
                // uint: Seed
                // byte: Rehash
                // byte: Ordered
                // byte: AllLengths
                NgramLength = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(0 < NgramLength && NgramLength <= NgramBufferBuilder.MaxSkipNgramLength);
                SkipLength = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(0 <= SkipLength && SkipLength <= NgramBufferBuilder.MaxSkipNgramLength);
                Contracts.CheckDecode(SkipLength <= NgramBufferBuilder.MaxSkipNgramLength - NgramLength);
                NumberOfBits = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(1 <= NumberOfBits && NumberOfBits <= 30);
                Seed = ctx.Reader.ReadUInt32();
                RehashUnigrams = ctx.Reader.ReadBoolByte();
                UseOrderedHashing = ctx.Reader.ReadBoolByte();
                UseAllLengths = ctx.Reader.ReadBoolByte();
            }

            internal void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // size of Inputs
                // string[] Inputs;
                // string Output;
                // int: NgramLength
                // int: SkipLength
                // int: NumberOfBits
                // uint: Seed
                // byte: Rehash
                // byte: Ordered
                // byte: AllLengths
                Contracts.Assert(InputColumnNamesArray.Length > 0);
                ctx.Writer.Write(InputColumnNamesArray.Length);
                for (int i = 0; i < InputColumnNamesArray.Length; i++)
                    ctx.SaveNonEmptyString(InputColumnNamesArray[i]);
                ctx.SaveNonEmptyString(Name);

                Contracts.Assert(0 < NgramLength && NgramLength <= NgramBufferBuilder.MaxSkipNgramLength);
                ctx.Writer.Write(NgramLength);
                Contracts.Assert(0 <= SkipLength && SkipLength <= NgramBufferBuilder.MaxSkipNgramLength);
                Contracts.Assert(NgramLength + SkipLength <= NgramBufferBuilder.MaxSkipNgramLength);
                ctx.Writer.Write(SkipLength);
                Contracts.Assert(1 <= NumberOfBits && NumberOfBits <= 30);
                ctx.Writer.Write(NumberOfBits);
                ctx.Writer.Write(Seed);
                ctx.Writer.WriteBoolByte(RehashUnigrams);
                ctx.Writer.WriteBoolByte(UseOrderedHashing);
                ctx.Writer.WriteBoolByte(UseAllLengths);
            }
        }

        internal static class Defaults
        {
            internal const int NgramLength = 2;
            internal const bool UseAllLengths = true;
            internal const int SkipLength = 0;
            internal const int NumberOfBits = 16;
            internal const uint Seed = 314489979;
            internal const bool RehashUnigrams = false;
            internal const bool UseOrderedHashing = true;
            internal const int MaximumNumberOfInverts = 0;
        }

        private readonly IHost _host;
        private readonly ColumnOptions[] _columns;

        /// <summary>
        /// Produces a bag of counts of hashed ngrams in <paramref name="inputColumnName"/>
        /// and outputs ngram vector as <paramref name="outputColumnName"/>
        ///
        /// <see cref="NgramHashingEstimator"/> is different from <see cref="WordHashBagEstimator"/> in a way that <see cref="NgramHashingEstimator"/>
        /// takes tokenized text as input while <see cref="WordHashBagEstimator"/> tokenizes text internally.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="outputColumnName">Name of output column, will contain the ngram vector. Null means <paramref name="inputColumnName"/> is replaced.</param>
        /// <param name="inputColumnName">Name of input column containing tokenized text.</param>
        /// <param name="numberOfBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="useAllLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="useOrderedHashing">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="maximumNumberOfInverts">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="maximumNumberOfInverts"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        internal NgramHashingEstimator(IHostEnvironment env,
            string outputColumnName,
            string inputColumnName = null,
            int numberOfBits = 16,
            int ngramLength = 2,
            int skipLength = 0,
            bool useAllLengths = true,
            uint seed = 314489979,
            bool useOrderedHashing = true,
            int maximumNumberOfInverts = 0)
            : this(env, outputColumnName, new[] { inputColumnName ?? outputColumnName }, numberOfBits, ngramLength, skipLength, useAllLengths, seed, useOrderedHashing, maximumNumberOfInverts)
        {
        }

        /// <summary>
        /// Produces a bag of counts of hashed ngrams in <paramref name="inputColumnNames"/>
        /// and outputs ngram vector as <paramref name="outputColumnName"/>
        ///
        /// <see cref="NgramHashingEstimator"/> is different from <see cref="WordHashBagEstimator"/> in a way that <see cref="NgramHashingEstimator"/>
        /// takes tokenized text as input while <see cref="WordHashBagEstimator"/> tokenizes text internally.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="outputColumnName">Name of output column, will contain the ngram vector.</param>
        /// <param name="inputColumnNames">Name of input columns containing tokenized text.</param>
        /// <param name="numberOfBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="useAllLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="useOrderedHashing">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="maximumNumberOfInverts">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="maximumNumberOfInverts"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        internal NgramHashingEstimator(IHostEnvironment env,
            string outputColumnName,
            string[] inputColumnNames,
            int numberOfBits = 16,
            int ngramLength = 2,
            int skipLength = 0,
            bool useAllLengths = true,
            uint seed = 314489979,
            bool useOrderedHashing = true,
            int maximumNumberOfInverts = 0)
            : this(env, new ColumnOptions(outputColumnName, inputColumnNames, ngramLength, skipLength, useAllLengths, numberOfBits, seed, useOrderedHashing, maximumNumberOfInverts))
        {
        }

        /// <summary>
        /// Produces a bag of counts of hashed ngrams in <paramref name="columns.inputs"/>
        /// and outputs ngram vector for each output in <paramref name="columns.output"/>
        ///
        /// <see cref="NgramHashingEstimator"/> is different from <see cref="WordHashBagEstimator"/> in a way that <see cref="NgramHashingEstimator"/>
        /// takes tokenized text as input while <see cref="WordHashBagEstimator"/> tokenizes text internally.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Array of columns which specifies the behavior of the transformation.</param>
        internal NgramHashingEstimator(IHostEnvironment env, params ColumnOptions[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(NgramHashingEstimator));
            _columns = columns;
        }

        internal static bool IsColumnTypeValid(DataViewType type)
        {
            if (!(type is VectorType vectorType))
                return false;
            if (!(vectorType.ItemType is KeyType itemKeyType))
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
            // Can only accept key types that can be converted to U4.
            if (!NgramUtils.IsValidNgramRawType(col.ItemType.RawType))
                return false;
            return true;
        }

        internal const string ExpectedColumnType = "Expected vector of Key type, and Key is convertible to U4";

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
                foreach (var input in colInfo.InputColumnNamesArray)
                {
                    if (!inputSchema.TryFindColumn(input, out var col))
                        throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", input);
                    if (!IsSchemaColumnValid(col))
                        throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, ExpectedColumnType, col.GetTypeString());
                }
                var metadata = new List<SchemaShape.Column>();
                metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, TextDataViewType.Instance, false));
                result[colInfo.Name] = new SchemaShape.Column(colInfo.Name, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false, new SchemaShape(metadata));
            }
            return new SchemaShape(result.Values);
        }

        /// <summary>
        /// Trains and returns a <see cref="NgramHashingTransformer"/>.
        /// </summary>
        public NgramHashingTransformer Fit(IDataView input) => new NgramHashingTransformer(_host, input, _columns);
    }
}
