// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;

[assembly: LoadableClass(HashTransformer.Summary, typeof(IDataTransform), typeof(HashTransformer), typeof(HashTransformer.Arguments), typeof(SignatureDataTransform),
    "Hash Transform", "HashTransform", "Hash", DocName = "transform/HashTransform.md")]

[assembly: LoadableClass(HashTransformer.Summary, typeof(IDataTransform), typeof(HashTransformer), null, typeof(SignatureLoadDataTransform),
    "Hash Transform", HashTransformer.LoaderSignature)]

[assembly: LoadableClass(HashTransformer.Summary, typeof(HashTransformer), null, typeof(SignatureLoadModel),
     "Hash Transform", HashTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(HashTransformer), null, typeof(SignatureLoadRowMapper),
   "Hash Transform", HashTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    using Conditional = System.Diagnostics.ConditionalAttribute;

    /// <summary>
    /// This transform can hash either single valued columns or vector columns. For vector columns,
    /// it hashes each slot separately.
    /// It can hash either text values or key values.
    /// </summary>
    public sealed class HashTransformer : OneToOneTransformerBase
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col",
                SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of bits to hash into. Must be between 1 and 31, inclusive",
                ShortName = "bits", SortOrder = 2)]
            public int HashBits = HashEstimator.Defaults.HashBits;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
            public uint Seed = HashEstimator.Defaults.Seed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the position of each term should be included in the hash",
                ShortName = "ord")]
            public bool Ordered = HashEstimator.Defaults.Ordered;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.",
                ShortName = "ih")]
            public int InvertHash = HashEstimator.Defaults.InvertHash;
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
                if (!base.TryParse(str, out string extra))
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

        public sealed class ColumnInfo
        {
            public readonly string Input;
            public readonly string Output;
            public readonly int HashBits;
            public readonly uint Seed;
            public readonly bool Ordered;
            public readonly int InvertHash;

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="input">Name of input column.</param>
            /// <param name="output">Name of output column.</param>
            /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 31, inclusive.</param>
            /// <param name="seed">Hashing seed.</param>
            /// <param name="ordered">Whether the position of each term should be included in the hash.</param>
            /// <param name="invertHash">Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.</param>
            public ColumnInfo(string input, string output,
                int hashBits = HashEstimator.Defaults.HashBits,
                uint seed = HashEstimator.Defaults.Seed,
                bool ordered = HashEstimator.Defaults.Ordered,
                int invertHash = HashEstimator.Defaults.InvertHash)
            {
                if (invertHash < -1)
                    throw Contracts.ExceptParam(nameof(invertHash), "Value too small, must be -1 or larger");
                if (invertHash != 0 && hashBits >= 31)
                    throw Contracts.ExceptParam(nameof(hashBits), $"Cannot support invertHash for a {0} bit hash. 30 is the maximum possible.", hashBits);

                Input = input;
                Output = output;
                HashBits = hashBits;
                Seed = seed;
                Ordered = ordered;
                InvertHash = invertHash;
            }

            internal ColumnInfo(string input, string output, ModelLoadContext ctx)
            {
                Input = input;
                Output = output;
                // *** Binary format ***
                // int: HashBits
                // uint: HashSeed
                // byte: Ordered
                HashBits = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(HashEstimator.NumBitsMin <= HashBits && HashBits < HashEstimator.NumBitsLim);
                Seed = ctx.Reader.ReadUInt32();
                Ordered = ctx.Reader.ReadBoolByte();
            }

            internal void Save(ModelSaveContext ctx)
            {
                // *** Binary format ***
                // int: HashBits
                // uint: HashSeed
                // byte: Ordered

                Contracts.Assert(HashEstimator.NumBitsMin <= HashBits && HashBits < HashEstimator.NumBitsLim);
                ctx.Writer.Write(HashBits);

                ctx.Writer.Write(Seed);
                ctx.Writer.WriteBoolByte(Ordered);
            }
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

        private readonly ColumnInfo[] _columns;
        private readonly VBuffer<ReadOnlyMemory<char>>[] _keyValues;
        private readonly ColumnType[] _kvTypes;

        protected override void CheckInputColumn(ISchema inputSchema, int col, int srcCol)
        {
            var type = inputSchema.GetColumnType(srcCol);
            if (!HashEstimator.IsColumnTypeValid(type))
                throw Host.ExceptParam(nameof(inputSchema), HashEstimator.ExpectedColumnType);
        }

        private static (string input, string output)[] GetColumnPairs(ColumnInfo[] columns)
        {
            Contracts.CheckNonEmpty(columns, nameof(columns));
            return columns.Select(x => (x.Input, x.Output)).ToArray();
        }

        private ColumnType GetOutputType(ISchema inputSchema, ColumnInfo column)
        {
            var keyCount = column.HashBits < 31 ? 1 << column.HashBits : 0;
            inputSchema.TryGetColumnIndex(column.Input, out int srcCol);
            var itemType = new KeyType(DataKind.U4, 0, keyCount, keyCount > 0);
            var srcType = inputSchema.GetColumnType(srcCol);
            if (!srcType.IsVector)
                return itemType;
            else
                return new VectorType(itemType, srcType.VectorSize);
        }

        /// <summary>
        /// Constructor for case where you don't need to 'train' transform on data, e.g. InvertHash for all columns set to zero.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="columns">Description of dataset columns and how to process them.</param>
        public HashTransformer(IHostEnvironment env, ColumnInfo[] columns) :
              base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            _columns = columns.ToArray();
            foreach (var column in _columns)
            {
                if (column.InvertHash != 0)
                    throw Host.ExceptParam(nameof(columns), $"Found colunm with {nameof(column.InvertHash)} set to non zero value, please use { nameof(HashEstimator)} instead");
            }
        }

        internal HashTransformer(IHostEnvironment env, IDataView input, ColumnInfo[] columns) :
            base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            _columns = columns.ToArray();
            var types = new ColumnType[_columns.Length];
            List<int> invertIinfos = null;
            List<int> invertHashMaxCounts = null;
            HashSet<int> sourceColumnsForInvertHash = new HashSet<int>();
            for (int i = 0; i < _columns.Length; i++)
            {
                if (!input.Schema.TryGetColumnIndex(ColumnPairs[i].input, out int srcCol))
                    throw Host.ExceptSchemaMismatch(nameof(input), "input", ColumnPairs[i].input);
                CheckInputColumn(input.Schema, i, srcCol);

                types[i] = GetOutputType(input.Schema, _columns[i]);
                int invertHashMaxCount;
                if (_columns[i].InvertHash == -1)
                    invertHashMaxCount = int.MaxValue;
                else
                    invertHashMaxCount = _columns[i].InvertHash;
                if (invertHashMaxCount > 0)
                {
                    Utils.Add(ref invertIinfos, i);
                    Utils.Add(ref invertHashMaxCounts, invertHashMaxCount);
                    sourceColumnsForInvertHash.Add(srcCol);
                }
            }
            if (Utils.Size(sourceColumnsForInvertHash) > 0)
            {
                using (IRowCursor srcCursor = input.GetRowCursor(sourceColumnsForInvertHash.Contains))
                {
                    using (var ch = Host.Start("Invert hash building"))
                    {
                        InvertHashHelper[] helpers = new InvertHashHelper[invertIinfos.Count];
                        Action disposer = null;
                        for (int i = 0; i < helpers.Length; ++i)
                        {
                            int iinfo = invertIinfos[i];
                            Host.Assert(types[iinfo].ItemType.KeyCount > 0);
                            var dstGetter = GetGetterCore(srcCursor, iinfo, out disposer);
                            Host.Assert(disposer == null);
                            var ex = _columns[iinfo];
                            var maxCount = invertHashMaxCounts[i];
                            helpers[i] = InvertHashHelper.Create(srcCursor, ex, maxCount, dstGetter);
                        }
                        while (srcCursor.MoveNext())
                        {
                            for (int i = 0; i < helpers.Length; ++i)
                                helpers[i].Process();
                        }
                        _keyValues = new VBuffer<ReadOnlyMemory<char>>[_columns.Length];
                        _kvTypes = new ColumnType[_columns.Length];
                        for (int i = 0; i < helpers.Length; ++i)
                        {
                            _keyValues[invertIinfos[i]] = helpers[i].GetKeyValuesMetadata();
                            Host.Assert(_keyValues[invertIinfos[i]].Length == types[invertIinfos[i]].ItemType.KeyCount);
                            _kvTypes[invertIinfos[i]] = new VectorType(TextType.Instance, _keyValues[invertIinfos[i]].Length);
                        }
                        ch.Done();
                    }
                }
            }
        }

        private Delegate GetGetterCore(IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < _columns.Length);
            disposer = null;
            input.Schema.TryGetColumnIndex(_columns[iinfo].Input, out int srcCol);
            var srcType = input.Schema.GetColumnType(srcCol);
            if (!srcType.IsVector)
                return ComposeGetterOne(input, iinfo, srcCol, srcType);
            return ComposeGetterVec(input, iinfo, srcCol, srcType);
        }

        protected override IRowMapper MakeRowMapper(ISchema schema) => new Mapper(this, schema);

        // Factory method for SignatureLoadModel.
        private static HashTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new HashTransformer(host, ctx);
        }

        private HashTransformer(IHost host, ModelLoadContext ctx)
          : base(host, ctx)
        {
            var columnsLength = ColumnPairs.Length;
            _columns = new ColumnInfo[columnsLength];
            for (int i = 0; i < columnsLength; i++)
                _columns[i] = new ColumnInfo(ColumnPairs[i].input, ColumnPairs[i].output, ctx);
            TextModelHelper.LoadAll(Host, ctx, columnsLength, out _keyValues, out _kvTypes);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));

            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            SaveColumns(ctx);

            // <prefix handled in static Create method>
            // <base>
            // <columns>
            Host.Assert(_columns.Length == ColumnPairs.Length);
            foreach (var col in _columns)
                col.Save(ctx);

            TextModelHelper.SaveAll(Host, ctx, _columns.Length, _keyValues);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        // Factory method for SignatureDataTransform.
        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            env.CheckValue(args.Column, nameof(args.Column));
            var cols = new ColumnInfo[args.Column.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                var item = args.Column[i];
                var kind = item.InvertHash ?? args.InvertHash;
                cols[i] = new ColumnInfo(item.Source ?? item.Name,
                    item.Name,
                    item.HashBits ?? args.HashBits,
                    item.Seed ?? args.Seed,
                    item.Ordered ?? args.Ordered,
                    item.InvertHash ?? args.InvertHash);
            };
            return new HashTransformer(env, input, cols).MakeDataTransform(input);
        }

        #region Getters
        private ValueGetter<uint> ComposeGetterOne(IRow input, int iinfo, int srcCol, ColumnType srcType)
        {
            Host.Assert(srcType.IsText || srcType.IsKey || srcType == NumberType.R4 || srcType == NumberType.R8);

            var mask = (1U << _columns[iinfo].HashBits) - 1;
            uint seed = _columns[iinfo].Seed;
            // In case of single valued input column, hash in 0 for the slot index.
            if (_columns[iinfo].Ordered)
                seed = Hashing.MurmurRound(seed, 0);

            switch (srcType.RawKind)
            {
                case DataKind.Text:
                    return ComposeGetterOneCore(input.GetGetter<ReadOnlyMemory<char>>(srcCol), seed, mask);
                case DataKind.U1:
                    return ComposeGetterOneCore(input.GetGetter<byte>(srcCol), seed, mask);
                case DataKind.U2:
                    return ComposeGetterOneCore(input.GetGetter<ushort>(srcCol), seed, mask);
                case DataKind.U4:
                    return ComposeGetterOneCore(input.GetGetter<uint>(srcCol), seed, mask);
                case DataKind.R4:
                    return ComposeGetterOneCore(input.GetGetter<float>(srcCol), seed, mask);
                case DataKind.R8:
                    return ComposeGetterOneCore(input.GetGetter<double>(srcCol), seed, mask);
                default:
                    Host.Assert(srcType.RawKind == DataKind.U8);
                    return ComposeGetterOneCore(input.GetGetter<ulong>(srcCol), seed, mask);
            }
        }

        private ValueGetter<uint> ComposeGetterOneCore(ValueGetter<ReadOnlyMemory<char>> getSrc, uint seed, uint mask)
        {
            ReadOnlyMemory<char> src = default;
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

        // This is a delegate for a function that loops over the first count elements of src, and hashes
        // them (either with their index or without) into dst. Additionally it fills in zero hashes in the rest of dst elements.
        private delegate void HashLoopWithZeroHash<TSrc>(int count, int[] indices, TSrc[] src, uint[] dst, int dstCount, uint seed, uint mask);

        private ValueGetter<VBuffer<uint>> ComposeGetterVec(IRow input, int iinfo, int srcCol, ColumnType srcType)
        {
            Host.Assert(srcType.IsVector);
            Host.Assert(srcType.ItemType.IsText || srcType.ItemType.IsKey || srcType.ItemType == NumberType.R4 || srcType.ItemType == NumberType.R8);

            switch (srcType.ItemType.RawKind)
            {
                case DataKind.Text:
                    return ComposeGetterVecCore<ReadOnlyMemory<char>>(input, iinfo, srcCol, srcType, HashUnord, HashDense, HashSparse);
                case DataKind.U1:
                    return ComposeGetterVecCore<byte>(input, iinfo, srcCol, srcType, HashUnord, HashDense, HashSparse);
                case DataKind.U2:
                    return ComposeGetterVecCore<ushort>(input, iinfo, srcCol, srcType, HashUnord, HashDense, HashSparse);
                case DataKind.U4:
                    return ComposeGetterVecCore<uint>(input, iinfo, srcCol, srcType, HashUnord, HashDense, HashSparse);
                case DataKind.R4:
                    return ComposeGetterVecCoreFloat<float>(input, iinfo, srcCol, srcType, HashSparseUnord, HashUnord, HashDense);
                case DataKind.R8:
                    return ComposeGetterVecCoreFloat<double>(input, iinfo, srcCol, srcType, HashSparseUnord, HashUnord, HashDense);
                default:
                    Host.Assert(srcType.ItemType.RawKind == DataKind.U8);
                    return ComposeGetterVecCore<ulong>(input, iinfo, srcCol, srcType, HashUnord, HashDense, HashSparse);
            }
        }

        private ValueGetter<VBuffer<uint>> ComposeGetterVecCore<T>(IRow input, int iinfo, int srcCol, ColumnType srcType,
             HashLoop<T> hasherUnord, HashLoop<T> hasherDense, HashLoop<T> hasherSparse)
        {
            Host.Assert(srcType.IsVector);
            Host.Assert(srcType.ItemType.RawType == typeof(T));

            var getSrc = input.GetGetter<VBuffer<T>>(srcCol);
            var ex = _columns[iinfo];
            var mask = (1U << ex.HashBits) - 1;
            var seed = ex.Seed;
            var len = srcType.VectorSize;
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

        private ValueGetter<VBuffer<uint>> ComposeGetterVecCoreFloat<T>(IRow input, int iinfo, int srcCol, ColumnType srcType,
            HashLoopWithZeroHash<T> hasherSparseUnord, HashLoop<T> hasherDenseUnord, HashLoop<T> hasherDenseOrdered)
        {
            Host.Assert(srcType.IsVector);
            Host.Assert(srcType.ItemType.RawType == typeof(T));

            var getSrc = input.GetGetter<VBuffer<T>>(srcCol);
            var ex = _columns[iinfo];
            var mask = (1U << ex.HashBits) - 1;
            var seed = ex.Seed;
            var len = srcType.VectorSize;
            var src = default(VBuffer<T>);
            T[] denseValues = null;
            int expectedSrcLength = srcType.VectorSize;
            HashLoop<T> hasherDense = ex.Ordered ? hasherDenseOrdered : hasherDenseUnord;

            return
                (ref VBuffer<uint> dst) =>
                {
                    getSrc(ref src);
                    if (len > 0 && src.Length != len)
                        throw Host.Except("Hash transform expected {0} slots, but got {1}", len, src.Length);

                    T[] values = src.Values;
                    var valuesCount = src.Count;
                    var srcIsDense = src.IsDense;
                    var hashes = dst.Values;

                    // force-densify the input in case of ordered hash.
                    if (!srcIsDense && ex.Ordered)
                    {
                        if (denseValues == null)
                            denseValues = new T[expectedSrcLength];
                        values = denseValues;
                        src.CopyTo(values);
                        valuesCount = expectedSrcLength;
                        srcIsDense = true;
                    }

                    if (srcIsDense)
                    {
                        if (Utils.Size(hashes) < valuesCount)
                            hashes = new uint[valuesCount];
                        hasherDense(valuesCount, null, values, hashes, seed, mask);
                        dst = new VBuffer<uint>(values.Length, hashes, dst.Indices);
                        return;
                    }

                    // source is sparse at this point and hash is unordered
                    if (Utils.Size(hashes) < expectedSrcLength)
                        hashes = new uint[expectedSrcLength];
                    hasherSparseUnord(src.Count, src.Indices, values, hashes, expectedSrcLength, seed, mask);
                    dst = new VBuffer<uint>(expectedSrcLength, hashes, dst.Indices);
                };
        }

        #endregion

        #region Core Hash functions, with and without index
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint HashCore(uint seed, ref ReadOnlyMemory<char> value, uint mask)
        {
            Contracts.Assert(Utils.IsPowerOfTwo(mask + 1));
            if (value.IsEmpty)
                return 0;
            return (Hashing.MurmurHash(seed, value.Span.Trim(' ')) & mask) + 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint HashCore(uint seed, ref ReadOnlyMemory<char> value, int i, uint mask)
        {
            Contracts.Assert(Utils.IsPowerOfTwo(mask + 1));
            if (value.IsEmpty)
                return 0;
            return (Hashing.MurmurHash(Hashing.MurmurRound(seed, (uint)i), value.Span.Trim(' ')) & mask) + 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint HashCore(uint seed, ref float value, uint mask)
        {
            Contracts.Assert(Utils.IsPowerOfTwo(mask + 1));
            if (float.IsNaN(value))
                return 0;
            // (value == 0 ? 0 : value) takes care of negative 0, its equal to positive 0 according to the IEEE 754 standard
            return (Hashing.MixHash(Hashing.MurmurRound(seed, FloatUtils.GetBits(value == 0 ? 0 : value))) & mask) + 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint HashCore(uint seed, ref float value, int i, uint mask)
        {
            Contracts.Assert(Utils.IsPowerOfTwo(mask + 1));
            if (float.IsNaN(value))
                return 0;
            return (Hashing.MixHash(Hashing.MurmurRound(Hashing.MurmurRound(seed, (uint)i),
                FloatUtils.GetBits(value == 0 ? 0 : value))) & mask) + 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint HashCore(uint seed, ref double value, uint mask)
        {
            Contracts.Assert(Utils.IsPowerOfTwo(mask + 1));
            if (double.IsNaN(value))
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
            if (double.IsNaN(value))
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
        private static void HashUnord(int count, int[] indices, ReadOnlyMemory<char>[] src, uint[] dst, uint seed, uint mask)
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
        private static void HashDense(int count, int[] indices, ReadOnlyMemory<char>[] src, uint[] dst, uint seed, uint mask)
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
        private static void HashSparse(int count, int[] indices, ReadOnlyMemory<char>[] src, uint[] dst, uint seed, uint mask)
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

        private static void HashSparseUnord(int count, int[] indices, float[] src, uint[] dst, int dstCount, uint seed, uint mask)
        {
            AssertValid(count, src, dst);
            Contracts.Assert(count <= dstCount);
            Contracts.Assert(dstCount <= Utils.Size(dst));

            float zero = 0.0f;
            uint zeroHash = HashCore(seed, ref zero, mask);

            int j = 0;
            for (int i = 0; i < dstCount; i++)
            {
                if (count <= j || indices[j] > i)
                    dst[i] = zeroHash;
                else if (indices[j] == i)
                    dst[i] = HashCore(seed, ref src[j++], mask);
                else
                    Contracts.Assert(false, "this should have never happened.");
            }
        }

        private static void HashSparseUnord(int count, int[] indices, double[] src, uint[] dst, int dstCount, uint seed, uint mask)
        {
            AssertValid(count, src, dst);
            Contracts.Assert(count <= dstCount);
            Contracts.Assert(dstCount <= Utils.Size(dst));

            double zero = 0.0;
            uint zeroHash = HashCore(seed, ref zero, mask);

            int j = 0;
            for (int i = 0; i < dstCount; i++)
            {
                if (count <= j || indices[j] > i)
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

        private sealed class Mapper : MapperBase
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

            private readonly ColumnType[] _types;
            private readonly HashTransformer _parent;

            public Mapper(HashTransformer parent, ISchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _types = new ColumnType[_parent._columns.Length];
                for (int i = 0; i < _types.Length; i++)
                    _types[i] = _parent.GetOutputType(inputSchema, _parent._columns[i]);
            }

            public override RowMapperColumnInfo[] GetOutputColumns()
            {
                var result = new RowMapperColumnInfo[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out int colIndex);
                    var colMetaInfo = new ColumnMetadataInfo(_parent.ColumnPairs[i].output);

                    foreach (var type in InputSchema.GetMetadataTypes(colIndex).Where(x => x.Key == MetadataUtils.Kinds.SlotNames))
                        Utils.MarshalInvoke(AddMetaGetter<int>, type.Value.RawType, colMetaInfo, InputSchema, type.Key, type.Value, colIndex);
                    if (_parent._kvTypes != null && _parent._kvTypes[i] != null)
                        AddMetaKeyValues(i, colMetaInfo);
                    result[i] = new RowMapperColumnInfo(_parent.ColumnPairs[i].output, _types[i], colMetaInfo);
                }
                return result;
            }
            private void AddMetaKeyValues(int i, ColumnMetadataInfo colMetaInfo)
            {
                MetadataUtils.MetadataGetter<VBuffer<ReadOnlyMemory<char>>> getter = (int col, ref VBuffer<ReadOnlyMemory<char>> dst) =>
                {
                    _parent._keyValues[i].CopyTo(ref dst);
                };
                var info = new MetadataInfo<VBuffer<ReadOnlyMemory<char>>>(_parent._kvTypes[i], getter);
                colMetaInfo.Add(MetadataUtils.Kinds.KeyValues, info);
            }

            private int AddMetaGetter<T>(ColumnMetadataInfo colMetaInfo, ISchema schema, string kind, ColumnType ct, int originalCol)
            {
                MetadataUtils.MetadataGetter<T> getter = (int col, ref T dst) =>
                {
                    // We don't care about 'col': this getter is specialized for a column 'originalCol',
                    // and 'col' in this case is the 'metadata kind index', not the column index.
                    schema.GetMetadata<T>(kind, originalCol, ref dst);
                };
                var info = new MetadataInfo<T>(ct, getter);
                colMetaInfo.Add(kind, info);
                return 0;
            }

            protected override Delegate MakeGetter(IRow input, int iinfo, out Action disposer) => _parent.GetGetterCore(input, iinfo, out disposer);
        }

        private abstract class InvertHashHelper
        {
            protected readonly IRow Row;
            private readonly bool _includeSlot;
            private readonly ColumnInfo _ex;
            private readonly ColumnType _srcType;
            private readonly int _srcCol;

            private InvertHashHelper(IRow row, ColumnInfo ex)
            {
                Contracts.AssertValue(row);
                Row = row;
                row.Schema.TryGetColumnIndex(ex.Input, out int srcCol);
                _srcCol = srcCol;
                _srcType = row.Schema.GetColumnType(srcCol);
                _ex = ex;
                // If this is a vector and ordered, then we must include the slot as part of the representation.
                _includeSlot = _srcType.IsVector && _ex.Ordered;
            }

            /// <summary>
            /// Constructs an <see cref="InvertHashHelper"/> instance to accumulate hash/value pairs
            /// from a single column as parameterized by this transform, with values fetched from
            /// the row.
            /// </summary>
            /// <param name="row">The input source row, from which the hashed values can be fetched</param>
            /// <param name="ex">The extra column info</param>
            /// <param name="invertHashMaxCount">The number of input hashed valuPres to accumulate per output hash value</param>
            /// <param name="dstGetter">A hash getter, built on top of <paramref name="row"/>.</param>
            public static InvertHashHelper Create(IRow row, ColumnInfo ex, int invertHashMaxCount, Delegate dstGetter)
            {
                row.Schema.TryGetColumnIndex(ex.Input, out int srcCol);
                ColumnType typeSrc = row.Schema.GetColumnType(srcCol);
                Type t = typeSrc.IsVector ? (ex.Ordered ? typeof(ImplVecOrdered<>) : typeof(ImplVec<>)) : typeof(ImplOne<>);
                t = t.MakeGenericType(typeSrc.ItemType.RawType);
                var consTypes = new Type[] { typeof(IRow), typeof(ColumnInfo), typeof(int), typeof(Delegate) };
                var constructorInfo = t.GetConstructor(consTypes);
                return (InvertHashHelper)constructorInfo.Invoke(new object[] { row, ex, invertHashMaxCount, dstGetter });
            }

            /// <summary>
            /// This calculates the hash/value pair from the current value of the column, and does
            /// appropriate processing of them to build the invert hash map.
            /// </summary>
            public abstract void Process();

            public abstract VBuffer<ReadOnlyMemory<char>> GetKeyValuesMetadata();

            private sealed class TextEqualityComparer : IEqualityComparer<ReadOnlyMemory<char>>
            {
                // REVIEW: Is this sufficiently useful? Should we be using term map, instead?
                private readonly uint _seed;

                public TextEqualityComparer(uint seed)
                {
                    _seed = seed;
                }

                public bool Equals(ReadOnlyMemory<char> x, ReadOnlyMemory<char> y) => x.Span.SequenceEqual(y.Span);

                public int GetHashCode(ReadOnlyMemory<char> obj)
                {
                    if (obj.IsEmpty)
                        return 0;
                    return (int)Hashing.MurmurHash(_seed, obj.Span.Trim(' ')) + 1;
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
                Contracts.Assert(_srcType.ItemType.RawType == typeof(T));
                if (typeof(T) == typeof(ReadOnlyMemory<char>))
                {
                    // We are hashing twice, once to assign to the slot, and then again,
                    // to build a set of encountered elements. Obviously we cannot use the
                    // same seed used to assign to a slot, or otherwise this per-slot hash
                    // would have a lot of collisions. We ensure that we have different
                    // hash function by inverting the seed's bits.
                    var c = new TextEqualityComparer(~_ex.Seed);
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

                protected Impl(IRow row, ColumnInfo ex, int invertHashMaxCount)
                    : base(row, ex)
                {
                    Contracts.AssertValue(row);
                    Contracts.AssertValue(ex);

                    Collector = new InvertHashCollector<T>(1 << ex.HashBits, invertHashMaxCount, GetTextMap(), GetComparer());
                }

                protected virtual ValueMapper<T, StringBuilder> GetTextMap()
                {
                    return InvertHashUtils.GetSimpleMapper<T>(Row.Schema, _srcCol);
                }

                protected virtual IEqualityComparer<T> GetComparer()
                {
                    return GetSimpleComparer<T>();
                }

                public override VBuffer<ReadOnlyMemory<char>> GetKeyValuesMetadata()
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

                public ImplOne(IRow row, ColumnInfo ex, int invertHashMaxCount, Delegate dstGetter)
                    : base(row, ex, invertHashMaxCount)
                {
                    _srcGetter = Row.GetGetter<T>(_srcCol);
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

                public ImplVec(IRow row, ColumnInfo ex, int invertHashMaxCount, Delegate dstGetter)
                    : base(row, ex, invertHashMaxCount)
                {
                    _srcGetter = Row.GetGetter<VBuffer<T>>(_srcCol);
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

                public ImplVecOrdered(IRow row, ColumnInfo ex, int invertHashMaxCount, Delegate dstGetter)
                    : base(row, ex, invertHashMaxCount)
                {
                    _srcGetter = Row.GetGetter<VBuffer<T>>(_srcCol);
                    _dstGetter = dstGetter as ValueGetter<VBuffer<uint>>;
                    Contracts.AssertValue(_dstGetter);
                }

                protected override ValueMapper<KeyValuePair<int, T>, StringBuilder> GetTextMap()
                {
                    var simple = InvertHashUtils.GetSimpleMapper<T>(Row.Schema, _srcCol);
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

    /// <summary>
    /// Estimator for <see cref="HashTransformer"/>
    /// </summary>
    public sealed class HashEstimator : IEstimator<HashTransformer>
    {
        public const int NumBitsMin = 1;
        public const int NumBitsLim = 32;

        public static class Defaults
        {
            public const int HashBits = NumBitsLim - 1;
            public const uint Seed = 314489979;
            public const bool Ordered = false;
            public const int InvertHash = 0;
        }

        private readonly IHost _host;
        private readonly HashTransformer.ColumnInfo[] _columns;

        public static bool IsColumnTypeValid(ColumnType type) => (type.ItemType.IsText || type.ItemType.IsKey || type.ItemType == NumberType.R4 || type.ItemType == NumberType.R8);

        internal const string ExpectedColumnType = "Expected Text, Key, Single or Double item type";

        /// <summary>
        /// Convinence constructor for simple one column case
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="inputColumn">Name of the output column.</param>
        /// <param name="outputColumn">Name of the column to be transformed. If this is null '<paramref name="inputColumn"/>' will be used.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 31, inclusive.</param>
        /// <param name="invertHash">Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.</param>
        public HashEstimator(IHostEnvironment env, string inputColumn, string outputColumn = null,
            int hashBits = Defaults.HashBits, int invertHash = Defaults.InvertHash)
            : this(env, new HashTransformer.ColumnInfo(inputColumn, outputColumn ?? inputColumn, hashBits: hashBits, invertHash: invertHash))
        {
        }

        /// <param name="env">Host Environment.</param>
        /// <param name="columns">Description of dataset columns and how to process them.</param>
        public HashEstimator(IHostEnvironment env, params HashTransformer.ColumnInfo[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(HashEstimator));
            _columns = columns.ToArray();
        }

        public HashTransformer Fit(IDataView input) => new HashTransformer(_host, input, _columns);

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var colInfo in _columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.Input, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);
                if (!IsColumnTypeValid(col.ItemType))
                    throw _host.ExceptParam(nameof(inputSchema), ExpectedColumnType);
                var metadata = new List<SchemaShape.Column>();
                if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.SlotNames, out var slotMeta))
                    metadata.Add(slotMeta);
                if (colInfo.InvertHash != 0)
                    metadata.Add(new SchemaShape.Column(MetadataUtils.Kinds.KeyValues, SchemaShape.Column.VectorKind.Vector, TextType.Instance, false));
                result[colInfo.Output] = new SchemaShape.Column(colInfo.Output, col.ItemType.IsVector ? SchemaShape.Column.VectorKind.Vector : SchemaShape.Column.VectorKind.Scalar, NumberType.U4, true, new SchemaShape(metadata));
            }
            return new SchemaShape(result.Values);
        }
    }
}

