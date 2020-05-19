// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(HashingTransformer.Summary, typeof(IDataTransform), typeof(HashingTransformer), typeof(HashingTransformer.Options), typeof(SignatureDataTransform),
    "Hash Transform", "HashTransform", "Hash", DocName = "transform/HashTransform.md")]

[assembly: LoadableClass(HashingTransformer.Summary, typeof(IDataTransform), typeof(HashingTransformer), null, typeof(SignatureLoadDataTransform),
    "Hash Transform", HashingTransformer.LoaderSignature)]

[assembly: LoadableClass(HashingTransformer.Summary, typeof(HashingTransformer), null, typeof(SignatureLoadModel),
     "Hash Transform", HashingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(HashingTransformer), null, typeof(SignatureLoadRowMapper),
   "Hash Transform", HashingTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// <see cref="ITransformer"/> resulting from fitting a <see cref="HashingEstimator"/>.
    /// </summary>
    public sealed class HashingTransformer : OneToOneTransformerBase
    {
        internal sealed class Options
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:src)",
                Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of bits to hash into. Must be between 1 and 31, inclusive",
                ShortName = "bits", SortOrder = 2)]
            public int NumberOfBits = HashingEstimator.Defaults.NumberOfBits;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
            public uint Seed = HashingEstimator.Defaults.Seed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the position of each term should be included in the hash",
                ShortName = "ord")]
            public bool Ordered = HashingEstimator.Defaults.UseOrderedHashing;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.",
                ShortName = "ih")]
            public int MaximumNumberOfInverts = HashingEstimator.Defaults.MaximumNumberOfInverts;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the slots of a vector column should be hashed into a single value.")]
            public bool Combine = HashingEstimator.Defaults.Combine;
        }

        internal sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of bits to hash into. Must be between 1 and 31, inclusive", ShortName = "bits")]
            public int? NumberOfBits;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
            public uint? Seed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the position of each term should be included in the hash",
                ShortName = "ord")]
            public bool? Ordered;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.",
                ShortName = "ih")]
            public int? MaximumNumberOfInverts;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the slots of a vector column should be hashed into a single value.")]
            public bool? Combine;

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

                int bits;
                if (!int.TryParse(extra, out bits))
                    return false;
                NumberOfBits = bits;
                return true;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (Seed != null || Ordered != null || MaximumNumberOfInverts != null || Combine != null)
                    return false;
                if (NumberOfBits == null)
                    return TryUnparseCore(sb);
                string extra = NumberOfBits.Value.ToString();
                return TryUnparseCore(sb, extra);
            }
        }

        private const string RegistrationName = "Hash";

        internal const string Summary = "Converts column values into hashes. This transform accepts text and keys as inputs. It works on single- and vector-valued columns, "
            + "and hashes each slot in a vector separately.";

        internal const string LoaderSignature = "HashTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "HASHTRNS",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010002, // Invert hash key values, hash fix
                //verWrittenCur: 0x00010003, // Uses MurmurHash3_x86_32
                verWrittenCur: 0x00010004, // Add Combine option
                verReadableCur: 0x00010004,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(HashingTransformer).Assembly.FullName);
        }

        private readonly HashingEstimator.ColumnOptions[] _columns;
        private readonly VBuffer<ReadOnlyMemory<char>>[] _keyValues;
        private readonly VectorDataViewType[] _kvTypes;
        private readonly bool _nonOnnxExportableVersion;

        private protected override void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            var type = inputSchema[srcCol].Type;
            if (!HashingEstimator.IsColumnTypeValid(type))
                throw Host.ExceptParam(nameof(inputSchema), HashingEstimator.ExpectedColumnType);
        }

        private static (string outputColumnName, string inputColumnName)[] GetColumnPairs(HashingEstimator.ColumnOptions[] columns)
        {
            Contracts.CheckNonEmpty(columns, nameof(columns));
            return columns.Select(x => (x.Name, x.InputColumnName)).ToArray();
        }

        private DataViewType GetOutputType(DataViewSchema inputSchema, HashingEstimator.ColumnOptions column)
        {
            var keyCount = (ulong)1 << column.NumberOfBits;
            inputSchema.TryGetColumnIndex(column.InputColumnName, out int srcCol);
            var itemType = new KeyDataViewType(typeof(uint), keyCount);
            var srcType = inputSchema[srcCol].Type;
            if (srcType is VectorDataViewType vectorType && !column.Combine)
                return new VectorDataViewType(itemType, vectorType.Size);
            else
                return itemType;
        }

        /// <summary>
        /// Constructor for case where you don't need to 'train' transform on data, for example, InvertHash for all columns set to zero.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="columns">Description of dataset columns and how to process them.</param>
        internal HashingTransformer(IHostEnvironment env, params HashingEstimator.ColumnOptions[] columns) :
              base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            _columns = columns;
            foreach (var column in _columns)
            {
                if (column.MaximumNumberOfInverts != 0)
                    throw Host.ExceptParam(nameof(columns), $"Found column with {nameof(column.MaximumNumberOfInverts)} set to non zero value, please use { nameof(HashingEstimator)} instead");

                if (column.Combine && column.UseOrderedHashing)
                    throw Host.ExceptParam(nameof(HashingEstimator.ColumnOptions.Combine), "When the 'Combine' option is specified, ordered hashing is not supported.");
            }
        }

        internal HashingTransformer(IHostEnvironment env, IDataView input, params HashingEstimator.ColumnOptions[] columns) :
            base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            _columns = columns;
            var types = new DataViewType[_columns.Length];
            List<int> invertIinfos = null;
            List<int> invertHashMaxCounts = null;
            var sourceColumnsForInvertHash = new List<DataViewSchema.Column>();
            for (int i = 0; i < _columns.Length; i++)
            {
                DataViewSchema.Column? srcCol = input.Schema.GetColumnOrNull(ColumnPairs[i].inputColumnName);
                if (srcCol == null)
                    throw Host.ExceptSchemaMismatch(nameof(input), "input", ColumnPairs[i].inputColumnName);
                CheckInputColumn(input.Schema, i, srcCol.Value.Index);

                types[i] = GetOutputType(input.Schema, _columns[i]);

                // If Combine is specified, there are no invert hashes.
                if (_columns[i].Combine)
                    continue;

                int invertHashMaxCount;
                if (_columns[i].MaximumNumberOfInverts == -1)
                    invertHashMaxCount = int.MaxValue;
                else
                    invertHashMaxCount = _columns[i].MaximumNumberOfInverts;
                if (invertHashMaxCount > 0)
                {
                    Utils.Add(ref invertIinfos, i);
                    Utils.Add(ref invertHashMaxCounts, invertHashMaxCount);
                    sourceColumnsForInvertHash.Add(srcCol.Value);
                }
            }
            if (Utils.Size(sourceColumnsForInvertHash) > 0)
            {
                using (DataViewRowCursor srcCursor = input.GetRowCursor(sourceColumnsForInvertHash))
                {
                    using (var ch = Host.Start("Invert hash building"))
                    {
                        InvertHashHelper[] helpers = new InvertHashHelper[invertIinfos.Count];
                        Action disposer = null;
                        for (int i = 0; i < helpers.Length; ++i)
                        {
                            int iinfo = invertIinfos[i];
                            Host.Assert(types[iinfo].GetItemType().GetKeyCount() > 0);
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
                        _kvTypes = new VectorDataViewType[_columns.Length];
                        for (int i = 0; i < helpers.Length; ++i)
                        {
                            _keyValues[invertIinfos[i]] = helpers[i].GetKeyValuesMetadata();
                            Host.Assert(_keyValues[invertIinfos[i]].Length == types[invertIinfos[i]].GetItemType().GetKeyCountAsInt32(Host));
                            _kvTypes[invertIinfos[i]] = new VectorDataViewType(TextDataViewType.Instance, _keyValues[invertIinfos[i]].Length);
                        }
                    }
                }
            }
        }

        private Delegate GetGetterCore(DataViewRow input, int iinfo, out Action disposer)
        {
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < _columns.Length);
            disposer = null;
            input.Schema.TryGetColumnIndex(_columns[iinfo].InputColumnName, out int srcCol);
            var srcType = input.Schema[srcCol].Type;
            if (!(srcType is VectorDataViewType vectorType))
                return ComposeGetterOne(input, iinfo, srcCol, srcType);
            if (_columns[iinfo].Combine)
                return ComposeGetterCombined(input, iinfo, srcCol, vectorType);
            return ComposeGetterVec(input, iinfo, srcCol, vectorType);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        // Factory method for SignatureLoadModel.
        private static HashingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new HashingTransformer(host, ctx);
        }

        private HashingTransformer(IHost host, ModelLoadContext ctx)
          : base(host, ctx)
        {
            var columnsLength = ColumnPairs.Length;
            _columns = new HashingEstimator.ColumnOptions[columnsLength];
            _nonOnnxExportableVersion = (ctx.Header.ModelVerWritten < 0x00010003); // Version 0x00010003 will use MurmurHash3_x86_32
            for (int i = 0; i < columnsLength; i++)
                _columns[i] = new HashingEstimator.ColumnOptions(ColumnPairs[i].outputColumnName, ColumnPairs[i].inputColumnName, ctx);
            TextModelHelper.LoadAll(Host, ctx, columnsLength, out _keyValues, out _kvTypes);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
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
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        // Factory method for SignatureDataTransform.
        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            env.CheckValue(options.Columns, nameof(options.Columns));
            var cols = new HashingEstimator.ColumnOptions[options.Columns.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                var item = options.Columns[i];
                cols[i] = new HashingEstimator.ColumnOptions(
                    item.Name,
                    item.Source ?? item.Name,
                    item.NumberOfBits ?? options.NumberOfBits,
                    item.Seed ?? options.Seed,
                    item.Ordered ?? options.Ordered,
                    item.MaximumNumberOfInverts ?? options.MaximumNumberOfInverts);
            };
            return new HashingTransformer(env, input, cols).MakeDataTransform(input);
        }

        #region Getters
        private ValueGetter<uint> ComposeGetterOne(DataViewRow input, int iinfo, int srcCol, DataViewType srcType)
        {
            Host.Assert(HashingEstimator.IsColumnTypeValid(srcType));

            var mask = (1U << _columns[iinfo].NumberOfBits) - 1;
            uint seed = _columns[iinfo].Seed;
            // In case of single valued input column, hash in 0 for the slot index.
            if (_columns[iinfo].UseOrderedHashing)
                seed = Hashing.MurmurRound(seed, 0);

            if (srcType is KeyDataViewType)
            {
                if (srcType.RawType == typeof(uint))
                    return MakeScalarHashGetter<uint, HashKey4>(input, srcCol, seed, mask, _nonOnnxExportableVersion);
                else if (srcType.RawType == typeof(ulong))
                    return MakeScalarHashGetter<ulong, HashKey8>(input, srcCol, seed, mask, _nonOnnxExportableVersion);
                else if (srcType.RawType == typeof(ushort))
                    return MakeScalarHashGetter<ushort, HashKey2>(input, srcCol, seed, mask, _nonOnnxExportableVersion);

                Host.Assert(srcType.RawType == typeof(byte));
                return MakeScalarHashGetter<byte, HashKey1>(input, srcCol, seed, mask, _nonOnnxExportableVersion);
            }

            if (srcType.RawType == typeof(ReadOnlyMemory<char>))
                return MakeScalarHashGetter<ReadOnlyMemory<char>, HashText>(input, srcCol, seed, mask, _nonOnnxExportableVersion);
            else if (srcType.RawType == typeof(float))
                return MakeScalarHashGetter<float, HashFloat>(input, srcCol, seed, mask, _nonOnnxExportableVersion);
            else if (srcType.RawType == typeof(double))
                return MakeScalarHashGetter<double, HashDouble>(input, srcCol, seed, mask, _nonOnnxExportableVersion);
            else if (srcType.RawType == typeof(sbyte))
                return MakeScalarHashGetter<sbyte, HashI1>(input, srcCol, seed, mask, _nonOnnxExportableVersion);
            else if (srcType.RawType == typeof(short))
                return MakeScalarHashGetter<short, HashI2>(input, srcCol, seed, mask, _nonOnnxExportableVersion);
            else if (srcType.RawType == typeof(int))
                return MakeScalarHashGetter<int, HashI4>(input, srcCol, seed, mask, _nonOnnxExportableVersion);
            else if (srcType.RawType == typeof(long))
                return MakeScalarHashGetter<long, HashI8>(input, srcCol, seed, mask, _nonOnnxExportableVersion);
            else if (srcType.RawType == typeof(byte))
                return MakeScalarHashGetter<byte, HashU1>(input, srcCol, seed, mask, _nonOnnxExportableVersion);
            else if (srcType.RawType == typeof(ushort))
                return MakeScalarHashGetter<ushort, HashU2>(input, srcCol, seed, mask, _nonOnnxExportableVersion);
            else if (srcType.RawType == typeof(uint))
                return MakeScalarHashGetter<uint, HashU4>(input, srcCol, seed, mask, _nonOnnxExportableVersion);
            else if (srcType.RawType == typeof(ulong))
                return MakeScalarHashGetter<ulong, HashU8>(input, srcCol, seed, mask, _nonOnnxExportableVersion);
            else if (srcType.RawType == typeof(DataViewRowId))
                return MakeScalarHashGetter<DataViewRowId, HashU16>(input, srcCol, seed, mask, _nonOnnxExportableVersion);

            Host.Assert(srcType.RawType == typeof(bool));
            return MakeScalarHashGetter<bool, HashBool>(input, srcCol, seed, mask, _nonOnnxExportableVersion);
        }

        private ValueGetter<VBuffer<uint>> ComposeGetterVec(DataViewRow input, int iinfo, int srcCol, VectorDataViewType srcType)
        {
            Host.Assert(HashingEstimator.IsColumnTypeValid(srcType.ItemType));

            Type rawType = srcType.ItemType.RawType;
            if (srcType.ItemType is KeyDataViewType)
            {
                if (rawType == typeof(byte))
                    return ComposeGetterVecCore<byte, HashKey1>(input, iinfo, srcCol, srcType);
                else if (rawType == typeof(ushort))
                    return ComposeGetterVecCore<ushort, HashKey2>(input, iinfo, srcCol, srcType);
                else if (rawType == typeof(uint))
                    return ComposeGetterVecCore<uint, HashKey4>(input, iinfo, srcCol, srcType);

                Host.Assert(rawType == typeof(ulong));
                return ComposeGetterVecCore<ulong, HashKey8>(input, iinfo, srcCol, srcType);
            }

            if (rawType == typeof(byte))
                return ComposeGetterVecCore<byte, HashU1>(input, iinfo, srcCol, srcType);
            else if (rawType == typeof(ushort))
                return ComposeGetterVecCore<ushort, HashU2>(input, iinfo, srcCol, srcType);
            else if (rawType == typeof(uint))
                return ComposeGetterVecCore<uint, HashU4>(input, iinfo, srcCol, srcType);
            else if (rawType == typeof(ulong))
                return ComposeGetterVecCore<ulong, HashU8>(input, iinfo, srcCol, srcType);
            else if (rawType == typeof(DataViewRowId))
                return ComposeGetterVecCore<DataViewRowId, HashU16>(input, iinfo, srcCol, srcType);
            else if (rawType == typeof(sbyte))
                return ComposeGetterVecCore<sbyte, HashI1>(input, iinfo, srcCol, srcType);
            else if (rawType == typeof(short))
                return ComposeGetterVecCore<short, HashI2>(input, iinfo, srcCol, srcType);
            else if (rawType == typeof(int))
                return ComposeGetterVecCore<int, HashI4>(input, iinfo, srcCol, srcType);
            else if (rawType == typeof(long))
                return ComposeGetterVecCore<long, HashI8>(input, iinfo, srcCol, srcType);
            else if (rawType == typeof(float))
                return ComposeGetterVecCore<float, HashFloat>(input, iinfo, srcCol, srcType);
            else if (rawType == typeof(double))
                return ComposeGetterVecCore<double, HashDouble>(input, iinfo, srcCol, srcType);
            else if (rawType == typeof(bool))
                return ComposeGetterVecCore<bool, HashBool>(input, iinfo, srcCol, srcType);

            Host.Assert(srcType.ItemType == TextDataViewType.Instance);
            return ComposeGetterVecCore<ReadOnlyMemory<char>, HashText>(input, iinfo, srcCol, srcType);
        }

        private ValueGetter<VBuffer<uint>> ComposeGetterVecCore<T, THash>(DataViewRow input, int iinfo, int srcCol, VectorDataViewType srcType)
            where THash : struct, IHasher<T>
        {
            Host.Assert(srcType.ItemType.RawType == typeof(T));

            var getSrc = input.GetGetter<VBuffer<T>>(input.Schema[srcCol]);
            var ex = _columns[iinfo];
            var mask = (1U << ex.NumberOfBits) - 1;
            var seed = ex.Seed;

            if (!ex.UseOrderedHashing)
                return MakeVectorHashGetter<T, THash>(seed, mask, getSrc, _nonOnnxExportableVersion);
            return MakeVectorOrderedHashGetter<T, THash>(seed, mask, getSrc, _nonOnnxExportableVersion);
        }

        private ValueGetter<uint> ComposeGetterCombined(DataViewRow input, int iinfo, int srcCol, VectorDataViewType srcType)
        {
            Host.Assert(HashingEstimator.IsColumnTypeValid(srcType));

            var mask = (1U << _columns[iinfo].NumberOfBits) - 1;
            uint seed = _columns[iinfo].Seed;

            Type rawType = srcType.ItemType.RawType;
            if (srcType.ItemType is KeyDataViewType)
            {
                if (rawType == typeof(uint))
                    return MakeCombinedVectorHashGetter<uint, HashKey4>(input, srcCol, seed, mask);
                else if (rawType == typeof(ulong))
                    return MakeCombinedVectorHashGetter<ulong, HashKey8>(input, srcCol, seed, mask);
                else if (rawType == typeof(ushort))
                    return MakeCombinedVectorHashGetter<ushort, HashKey2>(input, srcCol, seed, mask);

                Host.Assert(rawType == typeof(byte));
                return MakeCombinedVectorHashGetter<byte, HashKey1>(input, srcCol, seed, mask);
            }

            if (rawType == typeof(ReadOnlyMemory<char>))
                return MakeCombinedVectorHashGetter<ReadOnlyMemory<char>, HashText>(input, srcCol, seed, mask);
            else if (rawType == typeof(float))
                return MakeCombinedVectorHashGetter<float, HashFloat>(input, srcCol, seed, mask);
            else if (rawType == typeof(double))
                return MakeCombinedVectorHashGetter<double, HashDouble>(input, srcCol, seed, mask);
            else if (rawType == typeof(sbyte))
                return MakeCombinedVectorHashGetter<sbyte, HashI1>(input, srcCol, seed, mask);
            else if (rawType == typeof(short))
                return MakeCombinedVectorHashGetter<short, HashI2>(input, srcCol, seed, mask);
            else if (rawType == typeof(int))
                return MakeCombinedVectorHashGetter<int, HashI4>(input, srcCol, seed, mask);
            else if (rawType == typeof(long))
                return MakeCombinedVectorHashGetter<long, HashI8>(input, srcCol, seed, mask);
            else if (rawType == typeof(byte))
                return MakeCombinedVectorHashGetter<byte, HashU1>(input, srcCol, seed, mask);
            else if (rawType == typeof(ushort))
                return MakeCombinedVectorHashGetter<ushort, HashU2>(input, srcCol, seed, mask);
            else if (rawType == typeof(uint))
                return MakeCombinedVectorHashGetter<uint, HashU4>(input, srcCol, seed, mask);
            else if (rawType == typeof(ulong))
                return MakeCombinedVectorHashGetter<ulong, HashU8>(input, srcCol, seed, mask);
            else if (rawType == typeof(DataViewRowId))
                return MakeCombinedVectorHashGetter<DataViewRowId, HashU16>(input, srcCol, seed, mask);

            Host.Assert(rawType == typeof(bool));
            return MakeCombinedVectorHashGetter<bool, HashBool>(input, srcCol, seed, mask);
        }

        #endregion

        /// <summary>
        /// The usage of this interface may seem a bit strange, but it is deliberately structured in this way.
        /// One will note all implementors of this interface are structs, and that where used, you never use
        /// the interface itself, but instead an implementing type. This is due to how .NET and the JIT handles
        /// generic types that are also value types. For value types, it will actually generate new assembly
        /// code, which will allow effectively code generation in a way that would not happen if the hasher
        /// implementor was a class, or if the hasher implementation was just passed in with a delegate, or
        /// the hashing logic was encapsulated as the abstract method of some class.
        ///
        /// In a prior time, there were methods for all possible combinations of types, scalar-ness, vector
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
            uint HashCoreOld(uint seed, uint mask, in T value);
            uint HashCore(uint seed, uint mask, in T value);
            uint HashCore(uint seed, uint mask, in VBuffer<T> values);
        }

        private readonly struct HashFloat : IHasher<float>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCoreOld(uint seed, uint mask, in float value)
                => float.IsNaN(value) ? 0 : (Hashing.MixHash(Hashing.MurmurRound(seed, FloatUtils.GetBits(value == 0 ? 0 : value))) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in float value)
                => float.IsNaN(value) ? 0 : (Hashing.MixHash(Hashing.MurmurRound(seed, FloatUtils.GetBits(value == 0 ? 0 : value)), sizeof(float)) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in VBuffer<float> values)
            {
                var hash = seed;
                foreach (var value in values.DenseValues())
                {
                    if (float.IsNaN(value))
                        return 0;
                    hash = Hashing.MurmurRound(hash, FloatUtils.GetBits(value == 0 ? 0 : value));
                }
                return (Hashing.MixHash(hash, sizeof(uint)) & mask) + 1;
            }
        }

        private readonly struct HashDouble : IHasher<double>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]

            public uint HashCoreOld(uint seed, uint mask, in double value)
            {
                if (double.IsNaN(value))
                    return 0;

                return (Hashing.MixHash(HashRound(seed, value)) & mask) + 1;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in double value)
            {
                if (double.IsNaN(value))
                    return 0;

                return (Hashing.MixHash(HashRound(seed, value), sizeof(double)) & mask) + 1;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in VBuffer<double> values)
            {
                var hash = seed;
                foreach (var value in values.DenseValues())
                {
                    if (double.IsNaN(value))
                        return 0;
                    hash = HashRound(hash, value);
                }
                return (Hashing.MixHash(hash, sizeof(uint)) & mask) + 1;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private uint HashRound(uint seed, double value)
            {
                ulong v = FloatUtils.GetBits(value == 0 ? 0 : value);
                var hash = Hashing.MurmurRound(seed, Utils.GetLo(v));
                var hi = Utils.GetHi(v);
                return Hashing.MurmurRound(hash, hi);
            }
        }

        private readonly struct HashText : IHasher<ReadOnlyMemory<char>>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCoreOld(uint seed, uint mask, in ReadOnlyMemory<char> value)
                => value.IsEmpty ? 0 : (Hashing.MurmurHash(seed, value.Span.Trim(' ')) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in ReadOnlyMemory<char> value)
                => value.IsEmpty ? 0 : (Hashing.MurmurHashV2(seed, value.Span.Trim(' ')) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in VBuffer<ReadOnlyMemory<char>> values)
            {
                var hash = seed;
                foreach (var value in values.DenseValues())
                {
                    if (value.IsEmpty)
                        return 0;
                    hash = Hashing.MurmurHashV2(hash, value.Span.Trim(' '));
                }
                return (hash & mask) + 1;
            }
        }

        private readonly struct HashKey1 : IHasher<byte>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCoreOld(uint seed, uint mask, in byte value)
                => value == 0 ? 0 : (Hashing.MixHash(Hashing.MurmurRound(seed, value)) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in byte value)
                => value == 0 ? 0 : (Hashing.MixHash(Hashing.MurmurRound(seed, value), sizeof(uint)) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in VBuffer<byte> values)
            {
                var hash = seed;
                foreach (var value in values.DenseValues())
                {
                    if (value == 0)
                        return 0;
                    hash = Hashing.MurmurRound(hash, value);
                }
                return (Hashing.MixHash(hash, sizeof(uint)) & mask) + 1;
            }
        }

        private readonly struct HashKey2 : IHasher<ushort>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCoreOld(uint seed, uint mask, in ushort value)
                => value == 0 ? 0 : (Hashing.MixHash(Hashing.MurmurRound(seed, value)) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in ushort value)
                => value == 0 ? 0 : (Hashing.MixHash(Hashing.MurmurRound(seed, value), sizeof(uint)) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in VBuffer<ushort> values)
            {
                var hash = seed;
                foreach (var value in values.DenseValues())
                {
                    if (value == 0)
                        return 0;
                    hash = Hashing.MurmurRound(hash, value);
                }
                return (Hashing.MixHash(hash, sizeof(uint)) & mask) + 1;
            }
        }

        private readonly struct HashKey4 : IHasher<uint>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCoreOld(uint seed, uint mask, in uint value)
                => value == 0 ? 0 : (Hashing.MixHash(Hashing.MurmurRound(seed, value)) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in uint value)
                => value == 0 ? 0 : (Hashing.MixHash(Hashing.MurmurRound(seed, value), sizeof(uint)) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in VBuffer<uint> values)
            {
                var hash = seed;
                foreach (var value in values.DenseValues())
                {
                    if (value == 0)
                        return 0;
                    hash = Hashing.MurmurRound(hash, value);
                }
                return (Hashing.MixHash(hash, sizeof(uint)) & mask) + 1;
            }
        }

        private readonly struct HashKey8 : IHasher<ulong>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCoreOld(uint seed, uint mask, in ulong value)
            {
                if (value == 0)
                    return 0;
                return (Hashing.MixHash(HashRound(seed, value)) & mask) + 1;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in ulong value)
            {
                if (value == 0)
                    return 0;
                return (Hashing.MixHash(HashRound(seed, value), sizeof(uint)) & mask) + 1;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in VBuffer<ulong> values)
            {
                var hash = seed;
                foreach (var value in values.DenseValues())
                {
                    if (value == 0)
                        return 0;
                    hash = HashRound(hash, value);
                }
                return (Hashing.MixHash(hash, sizeof(uint)) & mask) + 1;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private uint HashRound(uint seed, ulong value)
            {
                var hash = Hashing.MurmurRound(seed, Utils.GetLo(value));
                var hi = Utils.GetHi(value);
                if (hi == 0)
                    return hash;
                return Hashing.MurmurRound(hash, hi);
            }
        }

        private readonly struct HashU1 : IHasher<byte>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCoreOld(uint seed, uint mask, in byte value)
                => (Hashing.MixHash(Hashing.MurmurRound(seed, value)) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in byte value)
                => (Hashing.MixHash(Hashing.MurmurRound(seed, value), sizeof(uint)) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in VBuffer<byte> values)
            {
                var hash = seed;
                foreach (var value in values.DenseValues())
                    hash = Hashing.MurmurRound(hash, value);
                return (Hashing.MixHash(hash, sizeof(uint)) & mask) + 1;
            }
        }

        private readonly struct HashU2 : IHasher<ushort>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCoreOld(uint seed, uint mask, in ushort value)
                => (Hashing.MixHash(Hashing.MurmurRound(seed, value)) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in ushort value)
                => (Hashing.MixHash(Hashing.MurmurRound(seed, value), sizeof(uint)) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in VBuffer<ushort> values)
            {
                var hash = seed;
                foreach (var value in values.DenseValues())
                    hash = Hashing.MurmurRound(hash, value);
                return (Hashing.MixHash(hash, sizeof(uint)) & mask) + 1;
            }
        }

        private readonly struct HashU4 : IHasher<uint>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCoreOld(uint seed, uint mask, in uint value)
                => (Hashing.MixHash(Hashing.MurmurRound(seed, value)) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in uint value)
                => (Hashing.MixHash(Hashing.MurmurRound(seed, value), sizeof(uint)) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in VBuffer<uint> values)
            {
                var hash = seed;
                foreach (var value in values.DenseValues())
                    hash = Hashing.MurmurRound(hash, value);
                return (Hashing.MixHash(hash, sizeof(uint)) & mask) + 1;
            }
        }

        private readonly struct HashU8 : IHasher<ulong>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCoreOld(uint seed, uint mask, in ulong value)
            {
                return (Hashing.MixHash(HashRound(seed, value)) & mask) + 1;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in ulong value)
            {
                return (Hashing.MixHash(HashRound(seed, value), sizeof(ulong)) & mask) + 1;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in VBuffer<ulong> values)
            {
                var hash = seed;
                foreach (var value in values.DenseValues())
                    hash = HashRound(hash, value);
                return (Hashing.MixHash(hash, sizeof(uint)) & mask) + 1;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private uint HashRound(uint seed, ulong value)
            {
                var hash = Hashing.MurmurRound(seed, Utils.GetLo(value));
                var hi = Utils.GetHi(value);
                return Hashing.MurmurRound(hash, hi);
            }
        }

        private readonly struct HashU16 : IHasher<DataViewRowId>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCoreOld(uint seed, uint mask, in DataViewRowId value)
            {
                return (Hashing.MixHash(HashRound(seed, value)) & mask) + 1;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in DataViewRowId value)
            {
                return (Hashing.MixHash(HashRound(seed, value), sizeof(uint)) & mask) + 1;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in VBuffer<DataViewRowId> values)
            {
                var hash = seed;
                foreach (var value in values.DenseValues())
                    hash = HashRound(hash, value);
                return (Hashing.MixHash(hash, sizeof(uint)) & mask) + 1;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private uint HashRound(uint seed, DataViewRowId value)
            {
                var hash = Hashing.MurmurRound(seed, Utils.GetLo(value.Low));
                var hi = Utils.GetHi(value.Low);
                if (hi != 0)
                    hash = Hashing.MurmurRound(hash, hi);
                if (value.High != 0)
                {
                    hash = Hashing.MurmurRound(hash, Utils.GetLo(value.High));
                    hi = Utils.GetHi(value.High);
                    if (hi != 0)
                        hash = Hashing.MurmurRound(hash, hi);
                }
                return hash;
            }
        }

        private readonly struct HashBool : IHasher<bool>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCoreOld(uint seed, uint mask, in bool value)
                => (Hashing.MixHash(Hashing.MurmurRound(seed, value ? 1u : 0u)) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in bool value)
                => (Hashing.MixHash(Hashing.MurmurRound(seed, value ? 1u : 0u), sizeof(uint)) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in VBuffer<bool> values)
            {
                var hash = seed;
                foreach (var value in values.DenseValues())
                    hash = Hashing.MurmurRound(hash, value ? 1u : 0u);
                return (Hashing.MixHash(hash, sizeof(uint)) & mask) + 1;
            }
        }

        private readonly struct HashI1 : IHasher<sbyte>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCoreOld(uint seed, uint mask, in sbyte value)
                => (Hashing.MixHash(Hashing.MurmurRound(seed, (uint)value)) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in sbyte value)
                => (Hashing.MixHash(Hashing.MurmurRound(seed, (uint)value), sizeof(uint)) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in VBuffer<sbyte> values)
            {
                var hash = seed;
                foreach (var value in values.DenseValues())
                    hash = Hashing.MurmurRound(hash, (uint)value);
                return (Hashing.MixHash(hash, sizeof(uint)) & mask) + 1;
            }
        }

        private readonly struct HashI2 : IHasher<short>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCoreOld(uint seed, uint mask, in short value)
                => (Hashing.MixHash(Hashing.MurmurRound(seed, (uint)value)) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in short value)
                => (Hashing.MixHash(Hashing.MurmurRound(seed, (uint)value), sizeof(uint)) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in VBuffer<short> values)
            {
                var hash = seed;
                foreach (var value in values.DenseValues())
                    hash = Hashing.MurmurRound(hash, (uint)value);
                return (Hashing.MixHash(hash, sizeof(uint)) & mask) + 1;
            }
        }

        private readonly struct HashI4 : IHasher<int>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCoreOld(uint seed, uint mask, in int value)
                => (Hashing.MixHash(Hashing.MurmurRound(seed, (uint)value)) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in int value)
                => (Hashing.MixHash(Hashing.MurmurRound(seed, (uint)value), sizeof(uint)) & mask) + 1;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in VBuffer<int> values)
            {
                var hash = seed;
                foreach (var value in values.DenseValues())
                    hash = Hashing.MurmurRound(hash, (uint)value);
                return (Hashing.MixHash(hash, sizeof(uint)) & mask) + 1;
            }
        }

        private readonly struct HashI8 : IHasher<long>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCoreOld(uint seed, uint mask, in long value)
            {
                return (Hashing.MixHash(HashRound(seed, value)) & mask) + 1;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in long value)
            {
                return (Hashing.MixHash(HashRound(seed, value), sizeof(long)) & mask) + 1;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public uint HashCore(uint seed, uint mask, in VBuffer<long> values)
            {
                var hash = seed;
                foreach (var value in values.DenseValues())
                    hash = HashRound(hash, value);
                return (Hashing.MixHash(hash, sizeof(uint)) & mask) + 1;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private uint HashRound(uint seed, long value)
            {
                var hash = Hashing.MurmurRound(seed, Utils.GetLo((ulong)value));
                var hi = Utils.GetHi((ulong)value);
                return Hashing.MurmurRound(hash, hi);
            }
        }

        private static ValueGetter<uint> MakeScalarHashGetter<T, THash>(DataViewRow input, int srcCol, uint seed, uint mask, bool useOldHashing)
            where THash : struct, IHasher<T>
        {
            Contracts.Assert(Utils.IsPowerOfTwo(mask + 1));
            Contracts.AssertValue(input);
            Contracts.Assert(0 <= srcCol && srcCol < input.Schema.Count);
            Contracts.Assert(input.Schema[srcCol].Type.RawType == typeof(T));

            var srcGetter = input.GetGetter<T>(input.Schema[srcCol]);
            T src = default;
            THash hasher = default;
            if (useOldHashing)
            {
                return (ref uint dst) =>
                {
                    srcGetter(ref src);
                    dst = hasher.HashCoreOld(seed, mask, src);
                };
            }
            return (ref uint dst) =>
            {
                srcGetter(ref src);
                dst = hasher.HashCore(seed, mask, src);
            };
        }

        private static ValueGetter<VBuffer<uint>> MakeVectorHashGetter<T, THash>(uint seed, uint mask, ValueGetter<VBuffer<T>> srcGetter, bool useOldHashing)
            where THash : struct, IHasher<T>
        {
            Contracts.Assert(Utils.IsPowerOfTwo(mask + 1));
            Contracts.AssertValue(srcGetter);
            VBuffer<T> src = default;
            THash hasher = default;

            if (useOldHashing)
            {
                // Determine whether this transformation is sparsity preserving, or not. It is sparsity preserving
                // if the default of T maps to the "missing" key value, that is, 0.
                uint defaultHashOld = hasher.HashCoreOld(seed, mask, default(T));
                if (defaultHashOld == 0)
                {
                    // It is sparsity preserving.
                    return (ref VBuffer<uint> dst) =>
                    {
                        srcGetter(ref src);
                        var srcValues = src.GetValues();
                        if (srcValues.Length == 0)
                        {
                            VBufferUtils.Resize(ref dst, src.Length, 0);
                            return;
                        }
                        var editor = VBufferEditor.Create(ref dst, src.Length, srcValues.Length);

                        for (int i = 0; i < srcValues.Length; ++i)
                            editor.Values[i] = hasher.HashCoreOld(seed, mask, srcValues[i]);
                        if (!src.IsDense)
                            src.GetIndices().CopyTo(editor.Indices);

                        dst = editor.Commit();
                    };
                }
                // It is not sparsity preserving.
                return (ref VBuffer<uint> dst) =>
                {
                    srcGetter(ref src);
                    var editor = VBufferEditor.Create(ref dst, src.Length);

                    var srcValues = src.GetValues();
                    if (src.IsDense)
                    {
                        for (int i = 0; i < srcValues.Length; ++i)
                            editor.Values[i] = hasher.HashCoreOld(seed, mask, srcValues[i]);
                    }
                    else
                    {
                        // First fill in the values of the destination. This strategy assumes, of course,
                        // that it is more performant to initialize then fill in the exceptional (non-sparse)
                        // values, rather than having complicated logic to do a simultaneous traversal of the
                        // sparse vs. dense array.
                        for (int i = 0; i < src.Length; ++i)
                            editor.Values[i] = defaultHashOld;
                        // Next overwrite the values in the explicit entries.
                        var srcIndices = src.GetIndices();
                        for (int i = 0; i < srcValues.Length; ++i)
                            editor.Values[srcIndices[i]] = hasher.HashCoreOld(seed, mask, srcValues[i]);
                    }
                    dst = editor.Commit();
                };
            }

            // Determine whether this transformation is sparsity preserving, or not. It is sparsity preserving
            // if the default of T maps to the "missing" key value, that is, 0.
            uint defaultHash = hasher.HashCore(seed, mask, default(T));
            if (defaultHash == 0)
            {
                // It is sparsity preserving.
                return (ref VBuffer<uint> dst) =>
                {
                    srcGetter(ref src);
                    var srcValues = src.GetValues();
                    if (srcValues.Length == 0)
                    {
                        VBufferUtils.Resize(ref dst, src.Length, 0);
                        return;
                    }
                    var editor = VBufferEditor.Create(ref dst, src.Length, srcValues.Length);

                    for (int i = 0; i < srcValues.Length; ++i)
                        editor.Values[i] = hasher.HashCore(seed, mask, srcValues[i]);
                    if (!src.IsDense)
                        src.GetIndices().CopyTo(editor.Indices);

                    dst = editor.Commit();
                };
            }
            // It is not sparsity preserving.
            return (ref VBuffer<uint> dst) =>
            {
                srcGetter(ref src);
                var editor = VBufferEditor.Create(ref dst, src.Length);

                var srcValues = src.GetValues();
                if (src.IsDense)
                {
                    for (int i = 0; i < srcValues.Length; ++i)
                        editor.Values[i] = hasher.HashCore(seed, mask, srcValues[i]);
                }
                else
                {
                    // First fill in the values of the destination. This strategy assumes, of course,
                    // that it is more performant to initialize then fill in the exceptional (non-sparse)
                    // values, rather than having complicated logic to do a simultaneous traversal of the
                    // sparse vs. dense array.
                    for (int i = 0; i < src.Length; ++i)
                        editor.Values[i] = defaultHash;
                    // Next overwrite the values in the explicit entries.
                    var srcIndices = src.GetIndices();
                    for (int i = 0; i < srcValues.Length; ++i)
                        editor.Values[srcIndices[i]] = hasher.HashCore(seed, mask, srcValues[i]);
                }
                dst = editor.Commit();
            };
        }

        private static ValueGetter<VBuffer<uint>> MakeVectorOrderedHashGetter<T, THash>(uint seed, uint mask, ValueGetter<VBuffer<T>> srcGetter, bool useOldHashing)
            where THash : struct, IHasher<T>
        {
            Contracts.Assert(Utils.IsPowerOfTwo(mask + 1));
            Contracts.AssertValue(srcGetter);
            VBuffer<T> src = default;
            THash hasher = default;

            if (useOldHashing)
            {
                // Determine whether this transformation is sparsity preserving, or not. It is sparsity preserving
                // if the default of T maps to the "missing" key value, that is, 0.
                uint defaultHashOld = hasher.HashCoreOld(seed, mask, default(T));
                if (defaultHashOld == 0)
                {
                    // It is sparsity preserving.
                    return (ref VBuffer<uint> dst) =>
                    {
                        srcGetter(ref src);
                        var srcValues = src.GetValues();
                        if (srcValues.Length == 0)
                        {
                            VBufferUtils.Resize(ref dst, src.Length, 0);
                            return;
                        }
                        var editor = VBufferEditor.Create(ref dst, src.Length, srcValues.Length);

                        if (src.IsDense)
                        {
                            for (int i = 0; i < srcValues.Length; ++i)
                                editor.Values[i] = hasher.HashCoreOld(Hashing.MurmurRound(seed, (uint)i), mask, srcValues[i]);
                        }
                        else
                        {
                            var srcIndices = src.GetIndices();
                            for (int i = 0; i < srcValues.Length; ++i)
                                editor.Values[i] = hasher.HashCoreOld(Hashing.MurmurRound(seed, (uint)srcIndices[i]), mask, srcValues[i]);
                            srcIndices.CopyTo(editor.Indices);

                        }
                        dst = editor.Commit();
                    };
                }
                // It is not sparsity preserving.
                return (ref VBuffer<uint> dst) =>
                {
                    srcGetter(ref src);
                    var editor = VBufferEditor.Create(ref dst, src.Length);

                    var srcValues = src.GetValues();
                    if (src.IsDense)
                    {
                        for (int i = 0; i < srcValues.Length; ++i)
                            editor.Values[i] = hasher.HashCoreOld(Hashing.MurmurRound(seed, (uint)i), mask, srcValues[i]);
                    }
                    else
                    {
                        var srcIndices = src.GetIndices();
                        int j = 0;
                        for (int i = 0; i < src.Length; i++)
                        {
                            uint indexSeed = Hashing.MurmurRound(seed, (uint)i);
                            if (srcIndices.Length <= j || srcIndices[j] > i)
                                editor.Values[i] = hasher.HashCoreOld(indexSeed, mask, default(T));
                            else if (srcIndices[j] == i)
                                editor.Values[i] = hasher.HashCoreOld(indexSeed, mask, srcValues[j++]);
                            else
                                Contracts.Assert(false, "this should have never happened.");
                        }
                    }
                    dst = editor.Commit();
                };
            }

            // Determine whether this transformation is sparsity preserving, or not. It is sparsity preserving
            // if the default of T maps to the "missing" key value, that is, 0.
            uint defaultHash = hasher.HashCore(seed, mask, default(T));
            if (defaultHash == 0)
            {
                // It is sparsity preserving.
                return (ref VBuffer<uint> dst) =>
                {
                    srcGetter(ref src);
                    var srcValues = src.GetValues();
                    if (srcValues.Length == 0)
                    {
                        VBufferUtils.Resize(ref dst, src.Length, 0);
                        return;
                    }
                    var editor = VBufferEditor.Create(ref dst, src.Length, srcValues.Length);

                    if (src.IsDense)
                    {
                        for (int i = 0; i < srcValues.Length; ++i)
                            editor.Values[i] = hasher.HashCore(Hashing.MurmurRound(seed, (uint)i), mask, srcValues[i]);
                    }
                    else
                    {
                        var srcIndices = src.GetIndices();
                        for (int i = 0; i < srcValues.Length; ++i)
                            editor.Values[i] = hasher.HashCore(Hashing.MurmurRound(seed, (uint)srcIndices[i]), mask, srcValues[i]);
                        srcIndices.CopyTo(editor.Indices);

                    }
                    dst = editor.Commit();
                };
            }
            // It is not sparsity preserving.
            return (ref VBuffer<uint> dst) =>
            {
                srcGetter(ref src);
                var editor = VBufferEditor.Create(ref dst, src.Length);

                var srcValues = src.GetValues();
                if (src.IsDense)
                {
                    for (int i = 0; i < srcValues.Length; ++i)
                        editor.Values[i] = hasher.HashCore(Hashing.MurmurRound(seed, (uint)i), mask, srcValues[i]);
                }
                else
                {
                    var srcIndices = src.GetIndices();
                    int j = 0;
                    for (int i = 0; i < src.Length; i++)
                    {
                        uint indexSeed = Hashing.MurmurRound(seed, (uint)i);
                        if (srcIndices.Length <= j || srcIndices[j] > i)
                            editor.Values[i] = hasher.HashCore(indexSeed, mask, default(T));
                        else if (srcIndices[j] == i)
                            editor.Values[i] = hasher.HashCore(indexSeed, mask, srcValues[j++]);
                        else
                            Contracts.Assert(false, "this should have never happened.");
                    }
                }
                dst = editor.Commit();
            };
        }

        private static ValueGetter<uint> MakeCombinedVectorHashGetter<T, THash>(DataViewRow input, int srcCol, uint seed, uint mask)
            where THash : struct, IHasher<T>
        {
            Contracts.Assert(Utils.IsPowerOfTwo(mask + 1));
            Contracts.AssertValue(input);

            var getSrc = input.GetGetter<VBuffer<T>>(input.Schema[srcCol]);
            VBuffer<T> src = default;
            THash hasher = default;

            return (ref uint dst) =>
            {
                getSrc(ref src);
                dst = hasher.HashCore(seed, mask, src);
            };
        }

        private sealed class Mapper : OneToOneMapperBase, ISaveAsOnnx
        {
            private sealed class ColInfo
            {
                public readonly string Name;
                public readonly string InputColumnName;
                public readonly DataViewType TypeSrc;

                public ColInfo(string outputColumnName, string inputColumnName, DataViewType type)
                {
                    Name = outputColumnName;
                    InputColumnName = inputColumnName;
                    TypeSrc = type;
                }
            }

            private readonly DataViewType[] _srcTypes;
            private readonly DataViewType[] _dstTypes;
            private readonly HashingTransformer _parent;

            public Mapper(HashingTransformer parent, DataViewSchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _srcTypes = new DataViewType[_parent._columns.Length];
                _dstTypes = new DataViewType[_parent._columns.Length];
                for (int i = 0; i < _dstTypes.Length; i++)
                {
                    _dstTypes[i] = _parent.GetOutputType(inputSchema, _parent._columns[i]);
                    _srcTypes[i] = inputSchema[_parent.ColumnPairs[i].inputColumnName].Type;
                }
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out int colIndex);
                    var meta = new DataViewSchema.Annotations.Builder();

                    meta.Add(InputSchema[colIndex].Annotations, name => name == AnnotationUtils.Kinds.SlotNames);

                    if (_parent._kvTypes != null && _parent._kvTypes[i] != null)
                        AddMetaKeyValues(i, meta);
                    result[i] = new DataViewSchema.DetachedColumn(_parent.ColumnPairs[i].outputColumnName, _dstTypes[i], meta.ToAnnotations());
                }
                return result;
            }
            private void AddMetaKeyValues(int i, DataViewSchema.Annotations.Builder builder)
            {
                ValueGetter<VBuffer<ReadOnlyMemory<char>>> getter = (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                {
                    _parent._keyValues[i].CopyTo(ref dst);
                };
                builder.AddKeyValues(_parent._kvTypes[i].Size, (PrimitiveDataViewType)_parent._kvTypes[i].ItemType, getter);
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer) => _parent.GetGetterCore(input, iinfo, out disposer);

            private bool SaveAsOnnxCore(OnnxContext ctx, int iinfo, string srcVariable, string dstVariable)
            {
                string castOutput;
                OnnxNode castNode;
                OnnxNode murmurNode;

                var srcType = _srcTypes[iinfo].GetItemType();
                if (srcType is KeyDataViewType)
                    return false;
                if (_parent._columns[iinfo].Combine)
                    return false;

                var opType = "MurmurHash3";
                string murmurOutput = ctx.AddIntermediateVariable(_dstTypes[iinfo], "MurmurOutput");

                // Numeric input types are limited to those supported by the Onnxruntime MurmurHash operator, which currently only supports
                // uints and ints. Thus, ulongs, longs, doubles and floats are not supported.
                if (srcType == NumberDataViewType.UInt16 || srcType == NumberDataViewType.Int16 ||
                    srcType == NumberDataViewType.SByte || srcType == NumberDataViewType.Byte ||
                    srcType == BooleanDataViewType.Instance)
                {
                    castOutput = ctx.AddIntermediateVariable(NumberDataViewType.UInt32, "CastOutput", true);
                    castNode = ctx.CreateNode("Cast", srcVariable, castOutput, ctx.GetNodeName(opType), "");
                    castNode.AddAttribute("to", NumberDataViewType.UInt32.RawType);
                    murmurNode = ctx.CreateNode(opType, castOutput, murmurOutput, ctx.GetNodeName(opType), "com.microsoft");
                }
                else if (srcType == NumberDataViewType.UInt32 || srcType == NumberDataViewType.Int32 || srcType == NumberDataViewType.UInt64 ||
                         srcType == NumberDataViewType.Int64 || srcType == NumberDataViewType.Single || srcType == NumberDataViewType.Double || srcType == TextDataViewType.Instance)

                {
                    murmurNode = ctx.CreateNode(opType, srcVariable, murmurOutput, ctx.GetNodeName(opType), "com.microsoft");
                }
                else
                {
                    return false;
                }

                murmurNode.AddAttribute("positive", 1);
                var seed = _parent._columns[iinfo].Seed;
                if (_parent._columns[iinfo].UseOrderedHashing)
                {
                    if (_srcTypes[iinfo] is VectorDataViewType)
                        return false;
                    seed = Hashing.MurmurRound(seed, 0);
                }
                murmurNode.AddAttribute("seed", seed);

                // masking is done via bitshifts, until bitwise AND is supported by Onnxruntime
                opType = "BitShift";
                string bitShiftOutput = ctx.AddIntermediateVariable(_dstTypes[iinfo], "BitShiftOutput");
                var shiftValue = ctx.AddInitializer((ulong)(32 - _parent._columns[iinfo].NumberOfBits), false);
                var shiftNode = ctx.CreateNode(opType, new[] { murmurOutput, shiftValue }, new[] { bitShiftOutput }, ctx.GetNodeName(opType), "");
                shiftNode.AddAttribute("direction", "LEFT");

                opType = "BitShift";
                string bitShiftOutput2 = ctx.AddIntermediateVariable(_dstTypes[iinfo], "BitShiftOutput2");
                shiftNode = ctx.CreateNode(opType, new[] { bitShiftOutput, shiftValue }, new[] { bitShiftOutput2 }, ctx.GetNodeName(opType), "");
                shiftNode.AddAttribute("direction", "RIGHT");

                var vectorShape = new VectorDataViewType(NumberDataViewType.Int64, _srcTypes[iinfo].GetValueCount());
                opType = "Cast";
                castOutput = ctx.AddIntermediateVariable(vectorShape, "CastOutput2");
                castNode = ctx.CreateNode(opType, bitShiftOutput2, castOutput, ctx.GetNodeName(opType), "");
                var t = NumberDataViewType.Int64.RawType;
                castNode.AddAttribute("to", t);

                opType = "Add";
                string addOutput = ctx.AddIntermediateVariable(vectorShape, "AddOutput");
                string one = ctx.AddInitializer(1);
                ctx.CreateNode(opType, new[] { castOutput, one }, new[] { addOutput }, ctx.GetNodeName(opType), "");

                opType = "Cast";
                var castNodeFinal = ctx.CreateNode(opType, addOutput, dstVariable, ctx.GetNodeName(opType), "");
                castNodeFinal.AddAttribute("to", _dstTypes[iinfo].GetItemType().RawType);

                return true;
            }

            void ISaveAsOnnx.SaveAsOnnx(OnnxContext ctx)
            {
                Host.CheckValue(ctx, nameof(ctx));
                for (int iinfo = 0; iinfo < _parent._columns.Length; ++iinfo)
                {
                    var info = _parent.ColumnPairs[iinfo];
                    string inputColumnName = info.inputColumnName;
                    if (!ctx.ContainsColumn(inputColumnName))
                    {
                        ctx.RemoveColumn(inputColumnName, false);
                        continue;
                    }

                    if (!SaveAsOnnxCore(ctx, iinfo, ctx.GetVariableName(inputColumnName), ctx.AddIntermediateVariable(_dstTypes[iinfo], info.outputColumnName)))
                    {
                        ctx.RemoveColumn(inputColumnName, true);
                    }
                }
            }

            bool ICanSaveOnnx.CanSaveOnnx(OnnxContext ctx) => true;
        }

        private abstract class InvertHashHelper
        {
            protected readonly DataViewRow Row;
            private readonly bool _includeSlot;
            private readonly HashingEstimator.ColumnOptions _ex;
            private readonly DataViewType _srcType;
            private readonly int _srcCol;

            private InvertHashHelper(DataViewRow row, HashingEstimator.ColumnOptions ex)
            {
                Contracts.AssertValue(row);
                Row = row;
                row.Schema.TryGetColumnIndex(ex.InputColumnName, out int srcCol);
                _srcCol = srcCol;
                _srcType = row.Schema[srcCol].Type;
                _ex = ex;
                // If this is a vector and ordered, then we must include the slot as part of the representation.
                _includeSlot = _srcType is VectorDataViewType && _ex.UseOrderedHashing;
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
            public static InvertHashHelper Create(DataViewRow row, HashingEstimator.ColumnOptions ex, int invertHashMaxCount, Delegate dstGetter)
            {
                row.Schema.TryGetColumnIndex(ex.InputColumnName, out int srcCol);
                DataViewType typeSrc = row.Schema[srcCol].Type;
                VectorDataViewType vectorTypeSrc = typeSrc as VectorDataViewType;

                Type t = vectorTypeSrc != null ? (ex.UseOrderedHashing ? typeof(ImplVecOrdered<>) : typeof(ImplVec<>)) : typeof(ImplOne<>);
                DataViewType itemType = vectorTypeSrc?.ItemType ?? typeSrc;

                t = t.MakeGenericType(itemType.RawType);

                var consTypes = new Type[] { typeof(DataViewRow), typeof(HashingEstimator.ColumnOptions), typeof(int), typeof(Delegate) };
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
                Contracts.Assert(_srcType.GetItemType().RawType == typeof(T));
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

                protected Impl(DataViewRow row, HashingEstimator.ColumnOptions ex, int invertHashMaxCount)
                    : base(row, ex)
                {
                    Contracts.AssertValue(row);
                    Contracts.AssertValue(ex);

                    Collector = new InvertHashCollector<T>(1 << ex.NumberOfBits, invertHashMaxCount, GetTextMap(), GetComparer());
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

                public ImplOne(DataViewRow row, HashingEstimator.ColumnOptions ex, int invertHashMaxCount, Delegate dstGetter)
                    : base(row, ex, invertHashMaxCount)
                {
                    _srcGetter = Row.GetGetter<T>(Row.Schema[_srcCol]);
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

                public ImplVec(DataViewRow row, HashingEstimator.ColumnOptions ex, int invertHashMaxCount, Delegate dstGetter)
                    : base(row, ex, invertHashMaxCount)
                {
                    _srcGetter = Row.GetGetter<VBuffer<T>>(Row.Schema[_srcCol]);
                    _dstGetter = dstGetter as ValueGetter<VBuffer<uint>>;
                    Contracts.AssertValue(_dstGetter);
                }

                public override void Process()
                {
                    _srcGetter(ref _value);
                    _dstGetter(ref _hash);

                    var valueValues = _value.GetValues();
                    var hashValues = _hash.GetValues();

                    // The two arrays should be consistent in their density, length, count, etc.
                    Contracts.Assert(_value.IsDense == _hash.IsDense);
                    Contracts.Assert(_value.Length == _hash.Length);
                    Contracts.Assert(valueValues.Length == hashValues.Length);

                    for (int i = 0; i < valueValues.Length; ++i)
                        Collector.Add(hashValues[i], valueValues[i]);
                }
            }

            private sealed class ImplVecOrdered<T> : Impl<KeyValuePair<int, T>>
            {
                private readonly ValueGetter<VBuffer<T>> _srcGetter;
                private readonly ValueGetter<VBuffer<uint>> _dstGetter;

                private VBuffer<T> _value;
                private VBuffer<uint> _hash;

                public ImplVecOrdered(DataViewRow row, HashingEstimator.ColumnOptions ex, int invertHashMaxCount, Delegate dstGetter)
                    : base(row, ex, invertHashMaxCount)
                {
                    _srcGetter = Row.GetGetter<VBuffer<T>>(Row.Schema[_srcCol]);
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

                    var valueValues = _value.GetValues();
                    var hashValues = _hash.GetValues();

                    // The two arrays should be consistent in their density, length, count, etc.
                    Contracts.Assert(_value.IsDense == _hash.IsDense);
                    Contracts.Assert(_value.Length == _hash.Length);
                    Contracts.Assert(valueValues.Length == hashValues.Length);
                    if (_hash.IsDense)
                    {
                        for (int i = 0; i < valueValues.Length; ++i)
                            Collector.Add(hashValues[i], new KeyValuePair<int, T>(i, valueValues[i]));
                    }
                    else
                    {
                        var hashIndices = _hash.GetIndices();
                        for (int i = 0; i < valueValues.Length; ++i)
                            Collector.Add(hashValues[i], new KeyValuePair<int, T>(hashIndices[i], valueValues[i]));
                    }
                }
            }
        }
    }

    /// <summary>
    /// Estimator for <see cref="HashingTransformer"/>, which hashes either single valued columns or vector columns. For vector columns,
    /// it hashes each slot separately.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | Yes, if the mapping of the hashes to the values is required. |
    /// | Input column data type | Vector or scalars of numeric, boolean, [text](xref:Microsoft.ML.Data.TextDataViewType), [DateTime](xref:System.DateTime) and [key](xref:Microsoft.ML.Data.KeyDataViewType) type. |
    /// | Output column data type | Vector or scalar [key](xref:Microsoft.ML.Data.KeyDataViewType) type. |
    /// | Exportable to ONNX | Yes - on estimators trained on v1.5 and up. <xref:System.Int64>, <xref:System.UInt64>, <xref:System.Single>, <xref:System.Double> and OrderedHashing are not supported. |
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]></format>
    /// </remarks>
    /// <seealso cref="ConversionsExtensionsCatalog.Hash(TransformsCatalog.ConversionTransforms, string, string, int, int)"/>
    /// <seealso cref="ConversionsExtensionsCatalog.Hash(TransformsCatalog.ConversionTransforms, HashingEstimator.ColumnOptions[])"/>
    public sealed class HashingEstimator : IEstimator<HashingTransformer>
    {
        internal const int NumBitsMin = 1;
        internal const int NumBitsLim = 32;

        internal static class Defaults
        {
            public const int NumberOfBits = NumBitsLim - 1;
            public const uint Seed = 314489979;
            public const bool UseOrderedHashing = false;
            public const int MaximumNumberOfInverts = 0;
            public const bool Combine = false;
        }

        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        public sealed class ColumnOptions
        {
            /// <summary>
            /// Name of the column resulting from the transformation of <see cref="InputColumnName"/>.
            /// </summary>
            public string Name { get; set; }

            /// <summary> Name of column to transform.</summary>
            public string InputColumnName { get; set; }

            /// <summary> Number of bits to hash into. Must be between 1 and 31, inclusive.</summary>
            public int NumberOfBits { get; set; }

            /// <summary> Hashing seed.</summary>
            public uint Seed { get; set; }

            /// <summary> Whether the position of each term should be included in the hash, only applies to inputs of vector type.</summary>
            public bool UseOrderedHashing { get; set; }

            /// <summary>
            /// During hashing we constuct mappings between original values and the produced hash values.
            /// Text representation of original values are stored in the key names of the annotations for the new column. Hashing, as such, can map many initial values to one.
            /// <see cref="MaximumNumberOfInverts"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
            /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.
            /// </summary>
            public int MaximumNumberOfInverts { get; set; }

            /// <summary>
            /// Whether the slots of a vector column should be hashed into a single value.
            /// </summary>
            public bool Combine { get; set; }

            private const uint VersionNoCombineOption = 0x00010003;

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="name"/> will be used as source.</param>
            /// <param name="numberOfBits">Number of bits to hash into. Must be between 1 and 31, inclusive.</param>
            /// <param name="seed">Hashing seed.</param>
            /// <param name="useOrderedHashing">Whether the position of each term should be included in the hash, only applies to inputs of vector type.</param>
            /// <param name="maximumNumberOfInverts">During hashing we construct mappings between original values and the produced hash values.
            /// Text representation of original values are stored in the slot names of the annotations for the new column.Hashing, as such, can map many initial values to one.
            /// <paramref name="maximumNumberOfInverts"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
            /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
            /// <param name="combine">Whether the slots of a vector column should be hashed into a single value.</param>
            public ColumnOptions(string name,
                string inputColumnName = null,
                int numberOfBits = Defaults.NumberOfBits,
                uint seed = Defaults.Seed,
                bool useOrderedHashing = Defaults.UseOrderedHashing,
                int maximumNumberOfInverts = Defaults.MaximumNumberOfInverts,
                bool combine = Defaults.Combine)
            {
                if (maximumNumberOfInverts < -1)
                    throw Contracts.ExceptParam(nameof(maximumNumberOfInverts), "Value too small, must be -1 or larger");
                if (maximumNumberOfInverts != 0 && numberOfBits >= 31)
                    throw Contracts.ExceptParam(nameof(numberOfBits), $"Cannot support maximumNumberOfInverts for a {0} bit hash. 30 is the maximum possible.", numberOfBits);
                Contracts.CheckNonWhiteSpace(name, nameof(name));
                Name = name;
                InputColumnName = inputColumnName ?? name;
                NumberOfBits = numberOfBits;
                Seed = seed;
                UseOrderedHashing = useOrderedHashing;
                MaximumNumberOfInverts = maximumNumberOfInverts;
                Combine = combine;
            }

            internal ColumnOptions(string name, string inputColumnName, ModelLoadContext ctx)
            {
                Name = name;
                InputColumnName = inputColumnName;

                // *** Binary format ***
                // int: NumberOfBits
                // uint: HashSeed
                // byte: Ordered
                // byte: Combine

                NumberOfBits = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(NumBitsMin <= NumberOfBits && NumberOfBits < NumBitsLim);
                Seed = ctx.Reader.ReadUInt32();
                UseOrderedHashing = ctx.Reader.ReadBoolByte();
                if (ctx.Header.ModelVerWritten > VersionNoCombineOption)
                    Combine = ctx.Reader.ReadBoolByte();
            }

            internal void Save(ModelSaveContext ctx)
            {
                // *** Binary format ***
                // int: NumberOfBits
                // uint: HashSeed
                // byte: Ordered
                // byte: Combine

                Contracts.Assert(NumBitsMin <= NumberOfBits && NumberOfBits < NumBitsLim);
                ctx.Writer.Write(NumberOfBits);

                ctx.Writer.Write(Seed);
                ctx.Writer.WriteBoolByte(UseOrderedHashing);
                ctx.Writer.WriteBoolByte(Combine);
            }
        }

        private readonly IHost _host;
        private readonly ColumnOptions[] _columns;

        internal static bool IsColumnTypeValid(DataViewType type)
        {
            var itemType = type.GetItemType();
            return itemType is TextDataViewType || itemType is KeyDataViewType || itemType is NumberDataViewType ||
                itemType is BooleanDataViewType || itemType is RowIdDataViewType;
        }

        internal const string ExpectedColumnType = "Expected Text, Key, numeric, Boolean or DataViewRowId item type";

        /// <summary>
        /// Initializes a new instance of <see cref="HashingEstimator"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform.
        /// If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="numberOfBits">Number of bits to hash into. Must be between 1 and 31, inclusive.</param>
        /// <param name="maximumNumberOfInverts">During hashing we construct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="maximumNumberOfInverts"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        /// <param name="useOrderedHashing">Whether the position of each term should be included in the hash, only applies to inputs of vector type.</param>
        /// <param name="combine">Whether the slots of a vector column should be hashed into a single value.</param>
        internal HashingEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null,
            int numberOfBits = Defaults.NumberOfBits, int maximumNumberOfInverts = Defaults.MaximumNumberOfInverts,
            bool useOrderedHashing = Defaults.UseOrderedHashing, bool combine = Defaults.Combine)
            : this(env, new ColumnOptions(outputColumnName, inputColumnName ?? outputColumnName,
                numberOfBits: numberOfBits, useOrderedHashing: useOrderedHashing, maximumNumberOfInverts: maximumNumberOfInverts, combine: combine))
        {
        }

        /// <summary>
        /// Initializes a new instance of <see cref="HashingEstimator"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="columns">Description of dataset columns and how to process them.</param>
        [BestFriend]
        internal HashingEstimator(IHostEnvironment env, params ColumnOptions[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(HashingEstimator));
            _columns = columns;

            // Validate the options.
            foreach (var columnOptions in _columns)
            {
                if (columnOptions.Combine && columnOptions.MaximumNumberOfInverts != 0)
                    throw _host.ExceptParam(nameof(ColumnOptions.Combine), "When the 'Combine' option is specified, invert hashes are not supported.");
                if (columnOptions.Combine && columnOptions.UseOrderedHashing)
                    throw _host.ExceptParam(nameof(ColumnOptions.Combine), "When the 'Combine' option is specified, ordered hashing is not supported.");
            }
        }

        /// <summary>
        /// Trains and returns a <see cref="HashingTransformer"/>.
        /// </summary>
        public HashingTransformer Fit(IDataView input) => new HashingTransformer(_host, input, _columns);

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
                if (!IsColumnTypeValid(col.ItemType))
                    throw _host.ExceptParam(nameof(inputSchema), ExpectedColumnType);
                var metadata = new List<SchemaShape.Column>();
                if (!colInfo.Combine && col.Annotations.TryFindColumn(AnnotationUtils.Kinds.SlotNames, out var slotMeta))
                    metadata.Add(slotMeta);
                if (colInfo.MaximumNumberOfInverts != 0)
                    metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.KeyValues, SchemaShape.Column.VectorKind.Vector, TextDataViewType.Instance, false));
                result[colInfo.Name] = new SchemaShape.Column(colInfo.Name, colInfo.Combine ? SchemaShape.Column.VectorKind.Scalar : col.Kind,
                    NumberDataViewType.UInt32, true, new SchemaShape(metadata));
            }
            return new SchemaShape(result.Values);
        }
    }
}

