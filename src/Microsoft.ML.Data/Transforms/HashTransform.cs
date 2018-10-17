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
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(HashTransformer).Assembly.FullName);
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
        /// Constructor for case where you don't need to 'train' transform on data, for example, InvertHash for all columns set to zero.
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

        protected override IRowMapper MakeRowMapper(ISchema schema) => new Mapper(this, Schema.Create(schema));

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

            if (srcType.IsKey)
            {
                switch (srcType.RawKind)
                {
                    case DataKind.U1:
                        return MakeScalarHashGetter<byte, HashKey1>(seed, mask, input.GetGetter<byte>(srcCol));
                    case DataKind.U2:
                        return MakeScalarHashGetter<ushort, HashKey2>(seed, mask, input.GetGetter<ushort>(srcCol));
                    case DataKind.U4:
                        return MakeScalarHashGetter<uint, HashKey4>(seed, mask, input.GetGetter<uint>(srcCol));
                    default:
                        Host.Assert(srcType.RawKind == DataKind.U8);
                        return MakeScalarHashGetter<ulong, HashKey8>(seed, mask, input.GetGetter<ulong>(srcCol));
                }
            }

            switch (srcType.RawKind)
            {
                case DataKind.R4:
                    return MakeScalarHashGetter<float, HashFloat>(seed, mask, input.GetGetter<float>(srcCol));
                case DataKind.R8:
                    return MakeScalarHashGetter<double, HashDouble>(seed, mask, input.GetGetter<double>(srcCol));
                default:
                    Host.Assert(srcType.RawKind == DataKind.Text);
                    return MakeScalarHashGetter<ReadOnlyMemory<char>, HashText>(seed, mask, input.GetGetter<ReadOnlyMemory<char>>(srcCol));
            }
        }

        private ValueGetter<VBuffer<uint>> ComposeGetterVec(IRow input, int iinfo, int srcCol, ColumnType srcType)
        {
            Host.Assert(srcType.IsVector);
            Host.Assert(srcType.ItemType.IsText || srcType.ItemType.IsKey || srcType.ItemType == NumberType.R4 || srcType.ItemType == NumberType.R8);

            if (srcType.ItemType.IsKey)
            {
                switch (srcType.ItemType.RawKind)
                {
                    case DataKind.U1:
                        return ComposeGetterVecCore<byte, HashKey1>(input, iinfo, srcCol, srcType);
                    case DataKind.U2:
                        return ComposeGetterVecCore<ushort, HashKey2>(input, iinfo, srcCol, srcType);
                    case DataKind.U4:
                        return ComposeGetterVecCore<uint, HashKey4>(input, iinfo, srcCol, srcType);
                    default:
                        Host.Assert(srcType.ItemType.RawKind == DataKind.U8);
                        return ComposeGetterVecCore<ulong, HashKey8>(input, iinfo, srcCol, srcType);
                }
            }

            switch (srcType.ItemType.RawKind)
            {
                case DataKind.R4:
                    return ComposeGetterVecCore<float, HashFloat>(input, iinfo, srcCol, srcType);
                case DataKind.R8:
                    return ComposeGetterVecCore<double, HashDouble>(input, iinfo, srcCol, srcType);
                default:
                    Host.Assert(srcType.ItemType.RawKind == DataKind.TX);
                    return ComposeGetterVecCore<ReadOnlyMemory<char>, HashText>(input, iinfo, srcCol, srcType);
            }
        }

        private ValueGetter<VBuffer<uint>> ComposeGetterVecCore<T, THash>(IRow input, int iinfo, int srcCol, ColumnType srcType)
            where THash : struct, IHasher<T>
        {
            Host.Assert(srcType.IsVector);
            Host.Assert(srcType.ItemType.RawType == typeof(T));

            var getSrc = input.GetGetter<VBuffer<T>>(srcCol);
            var ex = _columns[iinfo];
            var mask = (1U << ex.HashBits) - 1;
            var seed = ex.Seed;

            if (!ex.Ordered)
                return MakeVectorHashGetter<T, THash>(seed, mask, getSrc);
            return MakeVectorOrderedHashGetter<T, THash>(seed, mask, getSrc);
        }

        #endregion

        /// <summary>
        /// The usage of this interface may seem a bit strange, but it is deliberately structured in this way.
        /// One will note all implementors of this interface are structs, and that where used, you never use
        /// the interface itself but instead a type. This is due to how .NET and the JIT handles generic.
        /// For value types, it will actually generate new assembly code, which will allow effectively code
        /// generation in a way that would not happen if the hasher implementor was a class, or if the hasher
        /// implementation was just passed in with a delegate, or the hashing logic was encapsulated as the
        /// abstract method of some class.
        ///
        /// In a prior time, there were methods for all possible combinations of types, scalarness, vector
        /// sparsity/density, whether the hash was sparsity preserving or not, whether it was ordered or not.
        /// This resulted in an explosion of methods that made the hash transform code somewhat hard to maintain.
        /// On the other hand, the methods were fast, since they were effectively (by brute enumeration) completely
        /// inlined, so introducing any levels of abstraction would slow things down. By doing things in this
        /// fashion using generics over struct types, we are effectively (via the JIT) doing code generation so
        /// things are inlined and just as fast as the explicit implementation, while making the code rather
        /// easier to maintain.
        /// </summary>
        private interface IHasher<T>
        {
            uint HashCore(uint seed, uint mask, in T value);
        }

        private readonly struct HashFloat : IHasher<float>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in float value)
                => float.IsNaN(value) ? 0 : (Hashing.MixHash(Hashing.MurmurRound(seed, FloatUtils.GetBits(value == 0 ? 0 : value))) & mask) + 1;
        }

        private readonly struct HashDouble : IHasher<double>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]

            public uint HashCore(uint seed, uint mask, in double value)
            {
                if (double.IsNaN(value))
                    return 0;

                ulong v = FloatUtils.GetBits(value == 0 ? 0 : value);
                var hash = Hashing.MurmurRound(seed, Utils.GetLo(v));
                var hi = Utils.GetHi(v);
                if (hi != 0)
                    hash = Hashing.MurmurRound(hash, hi);
                return (Hashing.MixHash(hash) & mask) + 1;
            }
        }

        private readonly struct HashText : IHasher<ReadOnlyMemory<char>>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in ReadOnlyMemory<char> value)
                => value.IsEmpty ? 0 : (Hashing.MurmurHash(seed, value.Span.Trim(' ')) & mask) + 1;
        }

        private readonly struct HashKey1 : IHasher<byte>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in byte value)
                => value == 0 ? 0 : (Hashing.MixHash(Hashing.MurmurRound(seed, value)) & mask) + 1;
        }

        private readonly struct HashKey2 : IHasher<ushort>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in ushort value)
                => value == 0 ? 0 : (Hashing.MixHash(Hashing.MurmurRound(seed, value)) & mask) + 1;
        }

        private readonly struct HashKey4 : IHasher<uint>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in uint value)
                => value == 0 ? 0 : (Hashing.MixHash(Hashing.MurmurRound(seed, value)) & mask) + 1;
        }

        private readonly struct HashKey8 : IHasher<ulong>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in ulong value)
            {
                if (value == 0)
                    return 0;
                var hash = Hashing.MurmurRound(seed, Utils.GetLo(value));
                var hi = Utils.GetHi(value);
                if (hi != 0)
                    hash = Hashing.MurmurRound(hash, hi);
                return (Hashing.MixHash(hash) & mask) + 1;
            }
        }

        private static ValueGetter<uint> MakeScalarHashGetter<T, THash>(uint seed, uint mask, ValueGetter<T> srcGetter)
            where THash : struct, IHasher<T>
        {
            Contracts.Assert(Utils.IsPowerOfTwo(mask + 1));
            Contracts.AssertValue(srcGetter);
            T src = default;
            THash hasher = default;
            return (ref uint dst) =>
                {
                    srcGetter(ref src);
                    dst = hasher.HashCore(seed, mask, src);
                };
        }

        private static ValueGetter<VBuffer<uint>> MakeVectorHashGetter<T, THash>(uint seed, uint mask, ValueGetter<VBuffer<T>> srcGetter)
            where THash : struct, IHasher<T>
        {
            Contracts.Assert(Utils.IsPowerOfTwo(mask + 1));
            Contracts.AssertValue(srcGetter);
            VBuffer<T> src = default;
            THash hasher = default;

            // Determine whether this transformation is sparsity preserving, or not. It is sparsity preserving
            // if the default of T maps to the "missing" key value, that is, 0.
            uint defaultHash = hasher.HashCore(seed, mask, default);
            if (defaultHash == 0)
            {
                // It is sparsity preserving.
                return (ref VBuffer<uint> dst) =>
                {
                    srcGetter(ref src);
                    int[] indices = dst.Indices;
                    if (src.Count == 0)
                    {
                        dst = new VBuffer<uint>(src.Length, 0, dst.Values, dst.Indices);
                        return;
                    }
                    if (!src.IsDense)
                    {
                        Utils.EnsureSize(ref indices, src.Count, keepOld: false);
                        Array.Copy(src.Indices, 0, indices, 0, src.Count);
                    }
                    var values = dst.Values;
                    Utils.EnsureSize(ref values, src.Count, keepOld: false);
                    var srcValuesSpan = src.Values.AsSpan(0, src.Count);
                    for (int i = 0; i < srcValuesSpan.Length; ++i)
                        values[i] = hasher.HashCore(seed, mask, srcValuesSpan[i]);
                    dst = new VBuffer<uint>(src.Length, src.Count, values, indices);
                };
            }
            // It is not sparsity preserving.
            return (ref VBuffer<uint> dst) =>
            {
                srcGetter(ref src);
                uint[] values = dst.Values;
                Utils.EnsureSize(ref values, src.Length, keepOld: false);
                var srcValuesSpan = src.Values.AsSpan(0, src.Count);
                if (src.IsDense)
                {
                    for (int i = 0; i < srcValuesSpan.Length; ++i)
                        values[i] = hasher.HashCore(seed, mask, srcValuesSpan[i]);
                }
                else
                {
                    // First fill in the values of the destination. This strategy assumes, of course,
                    // that it is more performant to intiialize then fill in the exceptional (non-sparse)
                    // values, rather than having complicated logic to do a simultaneous traversal of the
                    // sparse vs. dense array.
                    for (int i = 0; i < src.Length; ++i)
                        values[i] = defaultHash;
                    // Next overwrite the values in the explicit entries.
                    for (int i = 0; i < srcValuesSpan.Length; ++i)
                        values[src.Indices[i]] = hasher.HashCore(seed, mask, srcValuesSpan[i]);
                }
                dst = new VBuffer<uint>(src.Length, values, dst.Indices);
            };
        }

        private static ValueGetter<VBuffer<uint>> MakeVectorOrderedHashGetter<T, THash>(uint seed, uint mask, ValueGetter<VBuffer<T>> srcGetter)
            where THash : struct, IHasher<T>
        {
            Contracts.Assert(Utils.IsPowerOfTwo(mask + 1));
            Contracts.AssertValue(srcGetter);
            VBuffer<T> src = default;
            THash hasher = default;

            // Determine whether this transformation is sparsity preserving, or not. It is sparsity preserving
            // if the default of T maps to the "missing" key value, that is, 0.
            uint defaultHash = hasher.HashCore(seed, mask, default);
            if (defaultHash == 0)
            {
                // It is sparsity preserving.
                return (ref VBuffer<uint> dst) =>
                {
                    srcGetter(ref src);
                    int[] indices = dst.Indices;
                    if (src.Count == 0)
                    {
                        dst = new VBuffer<uint>(src.Length, 0, dst.Values, dst.Indices);
                        return;
                    }
                    if (!src.IsDense)
                    {
                        Utils.EnsureSize(ref indices, src.Count, keepOld: false);
                        Array.Copy(src.Indices, 0, indices, 0, src.Count);
                    }
                    var values = dst.Values;
                    Utils.EnsureSize(ref values, src.Count, keepOld: false);
                    var srcValuesSpan = src.Values.AsSpan(0, src.Count);
                    if (src.IsDense)
                    {
                        for (int i = 0; i < srcValuesSpan.Length; ++i)
                            values[i] = hasher.HashCore(Hashing.MurmurRound(seed, (uint)i), mask, srcValuesSpan[i]);
                    }
                    else
                    {
                        for (int i = 0; i < srcValuesSpan.Length; ++i)
                            values[i] = hasher.HashCore(Hashing.MurmurRound(seed, (uint)src.Indices[i]), mask, srcValuesSpan[i]);
                    }
                    dst = new VBuffer<uint>(src.Length, src.Count, values, indices);
                };
            }
            // It is not sparsity preserving.
            return (ref VBuffer<uint> dst) =>
            {
                srcGetter(ref src);
                uint[] values = dst.Values;
                Utils.EnsureSize(ref values, src.Length, keepOld: false);
                var srcValuesSpan = src.Values.AsSpan(0, src.Count);
                if (src.IsDense)
                {
                    for (int i = 0; i < srcValuesSpan.Length; ++i)
                        values[i] = hasher.HashCore(Hashing.MurmurRound(seed, (uint)i), mask, srcValuesSpan[i]);
                }
                else
                {
                    int j = 0;
                    for (int i = 0; i < src.Length; i++)
                    {
                        if (src.Count <= j || src.Indices[j] > i)
                            values[i] = hasher.HashCore(Hashing.MurmurRound(seed, (uint)i), mask, default);
                        else if (src.Indices[j] == i)
                            values[i] = hasher.HashCore(seed, mask, srcValuesSpan[j++]);
                        else
                            Contracts.Assert(false, "this should have never happened.");
                    }
                }
                dst = new VBuffer<uint>(src.Length, values, dst.Indices);
            };
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

            public Mapper(HashTransformer parent, Schema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _types = new ColumnType[_parent._columns.Length];
                for (int i = 0; i < _types.Length; i++)
                    _types[i] = _parent.GetOutputType(inputSchema, _parent._columns[i]);
            }

            public override Schema.Column[] GetOutputColumns()
            {
                var result = new Schema.Column[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out int colIndex);
                    var meta = new Schema.Metadata.Builder();

                    meta.Add(InputSchema[colIndex].Metadata, name => name == MetadataUtils.Kinds.SlotNames);

                    if (_parent._kvTypes != null && _parent._kvTypes[i] != null)
                        AddMetaKeyValues(i, meta);
                    result[i] = new Schema.Column(_parent.ColumnPairs[i].output, _types[i], meta.GetMetadata());
                }
                return result;
            }
            private void AddMetaKeyValues(int i, Schema.Metadata.Builder builder)
            {
                ValueGetter<VBuffer<ReadOnlyMemory<char>>> getter = (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                {
                    _parent._keyValues[i].CopyTo(ref dst);
                };
                builder.AddKeyValues(_parent._kvTypes[i].VectorSize, _parent._kvTypes[i].ItemType.AsPrimitive, getter);
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
        internal const int NumBitsMin = 1;
        internal const int NumBitsLim = 32;

        internal static class Defaults
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

