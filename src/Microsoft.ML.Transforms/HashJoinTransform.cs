// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Transforms.Conversions;
using System;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;
using Float = System.Single;

[assembly: LoadableClass(HashJoinTransform.Summary, typeof(HashJoinTransform), typeof(HashJoinTransform.Arguments), typeof(SignatureDataTransform),
    HashJoinTransform.UserName, "HashJoinTransform", HashJoinTransform.RegistrationName)]

[assembly: LoadableClass(HashJoinTransform.Summary, typeof(HashJoinTransform), null, typeof(SignatureLoadDataTransform),
    HashJoinTransform.UserName, HashJoinTransform.LoaderSignature, "HashJoinFunction")]

[assembly: EntryPointModule(typeof(HashJoin))]

namespace Microsoft.ML.Transforms.Conversions
{
    /// <summary>
    /// This transform hashes its input columns. Each column is hashed separately, and within each
    /// column there is an option to specify which slots should be hashed together into one output slot.
    /// This transform can be applied either to single valued columns or to known length vector columns.
    /// </summary>
    public sealed class HashJoinTransform : OneToOneTransformBase
    {
        public const int NumBitsMin = 1;
        public const int NumBitsLim = 32;

        private static class Defaults
        {
            public const bool Join = true;
            public const int HashBits = NumBitsLim - 1;
            public const uint Seed = 314489979;
            public const bool Ordered = true;
        }

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)",
                ShortName = "col",
                SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the values need to be combined for a single hash")]
            public bool Join = Defaults.Join;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of bits to hash into. Must be between 1 and 31, inclusive.",
                ShortName = "bits", SortOrder = 2)]
            public int HashBits = Defaults.HashBits;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
            public uint Seed = Defaults.Seed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the position of each term should be included in the hash", ShortName = "ord")]
            public bool Ordered = Defaults.Ordered;
        }

        public sealed class Column : OneToOneColumn
        {
            // REVIEW: rename to 'combine' (with 'join' as a secondary name) once possible
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the values need to be combined for a single hash")]
            public bool? Join;

            // REVIEW: maybe the language could support ranges
            [Argument(ArgumentType.AtMostOnce, HelpText = "Which slots should be combined together. Example: 0,3,5;0,1;3;2,1,0. Overrides 'join'.")]
            public string CustomSlotMap;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of bits to hash into. Must be between 1 and 31, inclusive.", ShortName = "bits")]
            public int? HashBits;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
            public uint? Seed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the position of each term should be included in the hash", ShortName = "ord")]
            public bool? Ordered;

            public static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (Join != null || !string.IsNullOrEmpty(CustomSlotMap) || HashBits != null ||
                    Seed != null || Ordered != null)
                {
                    return false;
                }
                return TryUnparseCore(sb);
            }
        }

        public sealed class ColumnInfoEx
        {
            // Either VBuffer<Key<U4>> or a single Key<U4>.
            // Note that if CustomSlotMap contains only one array, the output type of the transform will a single Key<U4>.
            // This corresponds to the join=+ case, although now it's possible to omit certain slots entirely.
            // If # of hash bits is less than 31, the key type will have a positive count.
            public readonly ColumnType OutputColumnType;

            public readonly int HashBits;
            public readonly uint HashSeed;
            public readonly bool Ordered;
            public readonly int[][] SlotMap; // null if the input is a single column

            public int OutputValueCount
            {
                get { return OutputColumnType.ValueCount; }
            }

            public ColumnInfoEx(int[][] slotMap, int hashBits, uint hashSeed, bool ordered)
            {
                Contracts.CheckValueOrNull(slotMap);
                Contracts.Check(NumBitsMin <= hashBits && hashBits < NumBitsLim);

                SlotMap = slotMap;
                HashBits = hashBits;
                HashSeed = hashSeed;
                Ordered = ordered;
                var itemType = GetItemType(hashBits);
                if (Utils.Size(SlotMap) <= 1)
                    OutputColumnType = itemType;
                else
                    OutputColumnType = new VectorType(itemType, SlotMap.Length);
            }

            /// <summary>
            /// Constructs the correct KeyType for the given hash bits.
            /// Because of array size limitation, if hashBits = 31, the key type is not contiguous (not transformable into indicator array)
            /// </summary>
            private static KeyType GetItemType(int hashBits)
            {
                var keyCount = hashBits < 31 ? 1 << hashBits : 0;
                return new KeyType(DataKind.U4, 0, keyCount, keyCount > 0);
            }
        }

        internal const string RegistrationName = "HashJoin";

        internal const string Summary = "Converts column values into hashes. This transform accepts both numeric and text inputs, both single and vector-valued columns. ";

        internal const string UserName = "Hash Join Transform";

        public const string LoaderSignature = "HashJoinTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "HSHJOINF",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010002, // Added slot maps
                //verWrittenCur: 0x00010003, // Added per-column hash bits
                //verWrittenCur: 0x00010004, // Added per-column hash seed and ordered flag
                verWrittenCur: 0x00010005, // Hash fix
                verReadableCur: 0x00010005,
                verWeCanReadBack: 0x00010005,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(HashJoinTransform).Assembly.FullName);
        }

        private readonly ColumnInfoEx[] _exes;

        /// <summary>
        /// Initializes a new instance of <see cref="HashJoinTransform"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="source">Name of the column to be transformed. If this is null '<paramref name="name"/>' will be used.</param>
        /// <param name="join">Whether the values need to be combined for a single hash.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 31, inclusive.</param>
        public HashJoinTransform(IHostEnvironment env,
            IDataView input,
            string name,
            string source = null,
             bool join = Defaults.Join,
            int hashBits = Defaults.HashBits)
            : this(env, new Arguments() { Column = new[] { new Column() { Source = source ?? name, Name = name } }, Join = join, HashBits = hashBits }, input)
        {
        }

        /// <include file='doc.xml' path='doc/members/member[@name="HashJoin"]/*' />
        public HashJoinTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, Contracts.CheckRef(args, nameof(args)).Column, input, TestColumnType)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));

            if (args.HashBits < NumBitsMin || args.HashBits >= NumBitsLim)
                throw Host.ExceptUserArg(nameof(args.HashBits), "hashBits should be between {0} and {1} inclusive", NumBitsMin, NumBitsLim - 1);

            _exes = new ColumnInfoEx[Infos.Length];
            for (int i = 0; i < Infos.Length; i++)
            {
                var hashBits = args.Column[i].HashBits ?? args.HashBits;
                Host.CheckUserArg(NumBitsMin <= hashBits && hashBits < NumBitsLim, nameof(args.HashBits));
                _exes[i] = CreateColumnInfoEx(
                    args.Column[i].Join ?? args.Join,
                    args.Column[i].CustomSlotMap,
                    args.Column[i].HashBits ?? args.HashBits,
                    args.Column[i].Seed ?? args.Seed,
                    args.Column[i].Ordered ?? args.Ordered,
                    Infos[i]);
            }

            SetMetadata();
        }

        private HashJoinTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestColumnType)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <base>
            // For each of the infos:
            //     int: hash bits
            //     uint: hash seed
            //     bool: ordered
            //     int N: number of slot maps (must be zero for single-value inputs)
            //     Then, for each of the N maps:
            //          Slot map itself (array of ints, including length)

            Host.AssertNonEmpty(Infos);

            _exes = new ColumnInfoEx[Infos.Length];
            for (int i = 0; i < Infos.Length; i++)
            {
                int hashBits = ctx.Reader.ReadInt32();
                Host.CheckDecode(NumBitsMin <= hashBits && hashBits < NumBitsLim);

                uint hashSeed = ctx.Reader.ReadUInt32();
                bool ordered = ctx.Reader.ReadBoolByte();

                int slotMapCount = ctx.Reader.ReadInt32();
                Host.CheckDecode(slotMapCount >= 0);

                int[][] slotMap = null;
                if (slotMapCount > 0)
                {
                    Host.CheckDecode(Infos[i].TypeSrc.IsVector);

                    slotMap = new int[slotMapCount][];
                    for (int j = 0; j < slotMapCount; j++)
                    {
                        slotMap[j] = ctx.Reader.ReadIntArray();
                        Host.CheckDecode(Utils.Size(slotMap[j]) > 0); // null array could be returned by the call above

                        // the slots should be distinct and between 0 and vector size
                        Host.CheckDecode(slotMap[j].Distinct().Count() == slotMap[j].Length);
                        Host.CheckDecode(
                            slotMap[j].All(slot => 0 <= slot && slot < Infos[i].TypeSrc.ValueCount));
                    }
                }

                _exes[i] = new ColumnInfoEx(slotMap, hashBits, hashSeed, ordered);
            }

            SetMetadata();
        }

        public static HashJoinTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);

            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            h.CheckValue(input, nameof(input));

            return h.Apply("Loading Model", ch => new HashJoinTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            // For each of the infos:
            //     int: hash bits
            //     uint: hash seed
            //     bool: ordered
            //     int N: number of slot maps (must be zero for single-value inputs)
            //     Then, for each of the N maps:
            //          Slot map itself (array of ints, including length)

            SaveBase(ctx);

            for (int iColumn = 0; iColumn < Infos.Length; iColumn++)
            {
                var ex = _exes[iColumn];

                Host.Assert(NumBitsMin <= ex.HashBits && ex.HashBits < NumBitsLim);
                ctx.Writer.Write(ex.HashBits);

                ctx.Writer.Write(ex.HashSeed);
                ctx.Writer.WriteBoolByte(ex.Ordered);

                ctx.Writer.Write(Utils.Size(ex.SlotMap));
                if (ex.SlotMap == null)
                    continue;

                for (int i = 0; i < ex.SlotMap.Length; i++)
                {
                    Host.Assert(ex.SlotMap[i].Distinct().Count() == ex.SlotMap[i].Length);
                    Host.Assert(ex.SlotMap[i].All(slot => 0 <= slot && slot < Infos[iColumn].TypeSrc.ValueCount));
                    ctx.Writer.WriteIntArray(ex.SlotMap[i]);
                }
            }
        }

        private ColumnInfoEx CreateColumnInfoEx(bool join, string customSlotMap, int hashBits, uint hashSeed, bool ordered, ColInfo colInfo)
        {
            int[][] slotMap = null;
            if (colInfo.TypeSrc.IsVector)
            {
                // fill in the slot map
                if (!string.IsNullOrWhiteSpace(customSlotMap))
                    slotMap = CompileSlotMap(customSlotMap, colInfo.TypeSrc.ValueCount);
                else
                    slotMap = CreateDefaultSlotMap(join, colInfo.TypeSrc.ValueCount);
                Host.Assert(Utils.Size(slotMap) >= 1);
            }

            return new ColumnInfoEx(slotMap, hashBits, hashSeed, ordered);
        }

        private int[][] CompileSlotMap(string slotMapString, int srcSlotCount)
        {
            var parts = ReadOnlyMemoryUtils.Split(slotMapString.AsMemory(), new[] { ';' }).ToArray();
            var slotMap = new int[parts.Length][];
            for (int i = 0; i < slotMap.Length; i++)
            {
                var slotIndices = ReadOnlyMemoryUtils.Split(parts[i], new[] { ',' }).ToArray();
                var slots = new int[slotIndices.Length];
                slotMap[i] = slots;
                for (int j = 0; j < slots.Length; j++)
                {
                    int index;
                    if (!int.TryParse(slotIndices[j].ToString(), out index) || index < 0 || index >= srcSlotCount)
                        throw Host.Except("Unexpected slot index '{1}' in group {0}. Expected 0 to {2}", i, slotIndices[j], srcSlotCount - 1);
                    slots[j] = index;
                }

                if (slots.Distinct().Count() < slots.Length)
                    throw Host.Except("Group '{0}' has duplicate slot indices", parts[i]);
            }

            return slotMap;
        }

        private static int[][] CreateDefaultSlotMap(bool join, int srcSlotCount)
        {
            if (join)
            {
                // map all input slots into one output slot
                return new[] { Utils.GetIdentityPermutation(srcSlotCount) };
            }
            else
            {
                // map every input slot into a separate output slot
                return Enumerable.Range(0, srcSlotCount).Select(v => new[] { v }).ToArray();
            }
        }

        private static string TestColumnType(ColumnType type)
        {
            // REVIEW: list all types that can be hashed.
            if (type.ValueCount > 0)
                return null;
            return "Unknown vector size";
        }

        private void SetMetadata()
        {
            var md = Metadata;
            for (int i = 0; i < _exes.Length; i++)
            {
                var ex = _exes[i];
                if (Utils.Size(ex.SlotMap) <= 1)
                    continue;
                using (var bldr = md.BuildMetadata(i))
                {
                    bldr.AddGetter<VBuffer<ReadOnlyMemory<char>>>(MetadataUtils.Kinds.SlotNames,
                        new VectorType(TextType.Instance, ex.SlotMap.Length), GetSlotNames);
                }
            }
            md.Seal();
        }

        private void GetSlotNames(int iinfo, ref VBuffer<ReadOnlyMemory<char>> dst)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);

            Host.AssertValue(_exes[iinfo].SlotMap);

            int n = _exes[iinfo].OutputValueCount;
            var output = dst.Values;
            if (Utils.Size(output) < n)
                output = new ReadOnlyMemory<char>[n];

            var srcColumnName = Source.Schema.GetColumnName(Infos[iinfo].Source);
            bool useDefaultSlotNames = !Source.Schema.HasSlotNames(Infos[iinfo].Source, Infos[iinfo].TypeSrc.VectorSize);
            VBuffer<ReadOnlyMemory<char>> srcSlotNames = default;
            if (!useDefaultSlotNames)
            {
                Source.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, Infos[iinfo].Source, ref srcSlotNames);
                useDefaultSlotNames =
                    !srcSlotNames.IsDense
                    || srcSlotNames.Length != Infos[iinfo].TypeSrc.ValueCount;
            }

            var outputSlotName = new StringBuilder();
            for (int slot = 0; slot < n; slot++)
            {
                var slotList = _exes[iinfo].SlotMap[slot];

                outputSlotName.Clear();

                foreach (var inputSlotIndex in slotList)
                {
                    if (outputSlotName.Length > 0)
                        outputSlotName.Append("+");

                    if (useDefaultSlotNames)
                        outputSlotName.AppendFormat("{0}[{1}]", srcColumnName, inputSlotIndex);
                    else
                        outputSlotName.Append(srcSlotNames.Values[inputSlotIndex]);
                }

                output[slot] = outputSlotName.ToString().AsMemory();
            }

            dst = new VBuffer<ReadOnlyMemory<char>>(n, output, dst.Indices);
        }

        private delegate uint HashDelegate<TSrc>(in TSrc value, uint seed);

        // generic method generators
        private static MethodInfo _methGetterOneToOne;
        private static MethodInfo _methGetterVecToVec;
        private static MethodInfo _methGetterVecToOne;

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            // Construct MethodInfos templates that we need for the generic methods.
            if (_methGetterOneToOne == null)
            {
                Func<IRow, int, ValueGetter<uint>> del = ComposeGetterOneToOne<int>;
                Interlocked.CompareExchange(ref _methGetterOneToOne, del.GetMethodInfo().GetGenericMethodDefinition(), null);
            }
            if (_methGetterVecToVec == null)
            {
                Func<IRow, int, ValueGetter<VBuffer<uint>>> del = ComposeGetterVecToVec<int>;
                Interlocked.CompareExchange(ref _methGetterVecToVec, del.GetMethodInfo().GetGenericMethodDefinition(), null);
            }
            if (_methGetterVecToOne == null)
            {
                Func<IRow, int, ValueGetter<uint>> del = ComposeGetterVecToOne<int>;
                Interlocked.CompareExchange(ref _methGetterVecToOne, del.GetMethodInfo().GetGenericMethodDefinition(), null);
            }

            // Magic code to generate a correct getter.
            // First, we take a method info for GetGetter<int>
            // Then, we replace <int> with correct type of the input
            // And then we generate a delegate using the generic delegate generator
            MethodInfo mi;
            if (!Infos[iinfo].TypeSrc.IsVector)
                mi = _methGetterOneToOne;
            else if (_exes[iinfo].OutputValueCount == 1)
                mi = _methGetterVecToOne;
            else
                mi = _methGetterVecToVec;

            mi = mi.MakeGenericMethod(Infos[iinfo].TypeSrc.ItemType.RawType);
            return (Delegate)mi.Invoke(this, new object[] { input, iinfo });
        }

        /// <summary>
        /// Getter generator for inputs of type <typeparamref name="TSrc"/>
        /// </summary>
        /// <typeparam name="TSrc">Input type. Must be a non-vector</typeparam>
        /// <param name="input">Row inout</param>
        /// <param name="iinfo">Index of the getter</param>
        private ValueGetter<uint> ComposeGetterOneToOne<TSrc>(IRow input, int iinfo)
        {
            Host.AssertValue(input);
            Host.Assert(!Infos[iinfo].TypeSrc.IsVector);

            var getSrc = GetSrcGetter<TSrc>(input, iinfo);
            var hashFunction = ComposeHashDelegate<TSrc>();
            var src = default(TSrc);
            var mask = (1U << _exes[iinfo].HashBits) - 1;
            var hashSeed = _exes[iinfo].HashSeed;
            return
                (ref uint dst) =>
                {
                    getSrc(ref src);
                    dst = (hashFunction(in src, hashSeed) & mask) + 1; // +1 to offset from zero, which has special meaning for KeyType
                };
        }

        /// <summary>
        /// Getter generator for inputs of type <typeparamref name="TSrc"/>, where output type is a vector of hashes
        /// </summary>
        /// <typeparam name="TSrc">Input type. Must be a vector</typeparam>
        /// <param name="input">Row input</param>
        /// <param name="iinfo">Index of the getter</param>
        private ValueGetter<VBuffer<uint>> ComposeGetterVecToVec<TSrc>(IRow input, int iinfo)
        {
            Host.AssertValue(input);
            Host.Assert(Infos[iinfo].TypeSrc.IsVector);

            var getSrc = GetSrcGetter<VBuffer<TSrc>>(input, iinfo);
            var hashFunction = ComposeHashDelegate<TSrc>();
            var src = default(VBuffer<TSrc>);
            int n = _exes[iinfo].OutputValueCount;
            int expectedSrcLength = Infos[iinfo].TypeSrc.VectorSize;
            int[][] slotMap = _exes[iinfo].SlotMap;
            // REVIEW: consider adding a fix-zero functionality (subtract emptyTextHash from all hashes)
            var mask = (1U << _exes[iinfo].HashBits) - 1;
            var hashSeed = _exes[iinfo].HashSeed;
            bool ordered = _exes[iinfo].Ordered;
            TSrc[] denseValues = null;
            return
                (ref VBuffer<uint> dst) =>
                {
                    getSrc(ref src);
                    Host.Check(src.Length == expectedSrcLength);
                    TSrc[] values;

                    // force-densify the input
                    // REVIEW: this performs poorly if only a fraction of sparse vector is used for hashing.
                    // This scenario was unlikely at the time of writing. Regardless of performance, the hash value
                    // needs to be consistent across equivalent representations - sparse vs dense.
                    if (src.IsDense)
                        values = src.Values;
                    else
                    {
                        if (denseValues == null)
                            denseValues = new TSrc[expectedSrcLength];
                        values = denseValues;
                        src.CopyTo(values);
                    }

                    var hashes = dst.Values;
                    if (Utils.Size(hashes) < n)
                        hashes = new uint[n];

                    for (int i = 0; i < n; i++)
                    {
                        uint hash = hashSeed;

                        foreach (var srcSlot in slotMap[i])
                        {
                            // REVIEW: some legacy code hashes 0 for srcSlot in ord- case, do we need to preserve this behavior?
                            if (ordered)
                                hash = Hashing.MurmurRound(hash, (uint)srcSlot);
                            hash = hashFunction(in values[srcSlot], hash);
                        }

                        hashes[i] = (Hashing.MixHash(hash) & mask) + 1; // +1 to offset from zero, which has special meaning for KeyType
                    }

                    dst = new VBuffer<uint>(n, hashes, dst.Indices);
                };
        }

        /// <summary>
        /// Getter generator for inputs of type <typeparamref name="TSrc"/>, where output type is a single hash
        /// </summary>
        /// <typeparam name="TSrc">Input type. Must be a vector</typeparam>
        /// <param name="input">Row input</param>
        /// <param name="iinfo">Index of the getter</param>
        private ValueGetter<uint> ComposeGetterVecToOne<TSrc>(IRow input, int iinfo)
        {
            Host.AssertValue(input);
            Host.Assert(Infos[iinfo].TypeSrc.IsVector);
            Host.Assert(Utils.Size(_exes[iinfo].SlotMap) == 1);

            var slots = _exes[iinfo].SlotMap[0];

            var getSrc = GetSrcGetter<VBuffer<TSrc>>(input, iinfo);
            var hashFunction = ComposeHashDelegate<TSrc>();
            var src = default(VBuffer<TSrc>);
            int expectedSrcLength = Infos[iinfo].TypeSrc.VectorSize;
            var mask = (1U << _exes[iinfo].HashBits) - 1;
            var hashSeed = _exes[iinfo].HashSeed;
            bool ordered = _exes[iinfo].Ordered;
            TSrc[] denseValues = null;
            return
                (ref uint dst) =>
                {
                    getSrc(ref src);
                    Host.Check(src.Length == expectedSrcLength);

                    TSrc[] values;
                    // force-densify the input
                    // REVIEW: this performs poorly if only a fraction of sparse vector is used for hashing.
                    // This scenario was unlikely at the time of writing. Regardless of performance, the hash value
                    // needs to be consistent across equivalent representations - sparse vs dense.
                    if (src.IsDense)
                        values = src.Values;
                    else
                    {
                        if (denseValues == null)
                            denseValues = new TSrc[expectedSrcLength];
                        values = denseValues;
                        src.CopyTo(values);
                    }

                    uint hash = hashSeed;
                    foreach (var srcSlot in slots)
                    {
                        if (ordered)
                            hash = Hashing.MurmurRound(hash, (uint)srcSlot);
                        hash = hashFunction(in values[srcSlot], hash);
                    }
                    dst = (Hashing.MixHash(hash) & mask) + 1; // +1 to offset from zero, which has special meaning for KeyType
                };
        }

        /// <summary>
        /// Generic hash function
        /// </summary>
        private HashDelegate<TSrc> ComposeHashDelegate<TSrc>()
        {
            // REVIEW: Add a specialized hashing for ints, once numeric bin mapper is done http://sqlbuvsts01:8080/Main/Advanced%20Analytics/_workitems/edit/5823788
            if (typeof(TSrc) == typeof(Float))
                return (HashDelegate<TSrc>)(Delegate)ComposeFloatHashDelegate();

            if (typeof(TSrc) == typeof(Double))
                return (HashDelegate<TSrc>)(Delegate)ComposeDoubleHashDelegate();

            // Default case: convert to text and hash as a string.
            var sb = default(StringBuilder);
            var conv = Runtime.Data.Conversion.Conversions.Instance.GetStringConversion<TSrc>();
            return
                (in TSrc value, uint seed) =>
                {
                    conv(in value, ref sb);
                    return Hashing.MurmurHash(seed, sb, 0, sb.Length);
                };
        }

        /// <summary>
        /// Generate a specialized hash function for floats
        /// </summary>
        private HashDelegate<Float> ComposeFloatHashDelegate()
        {
            return Hash;
        }

        /// <summary>
        /// Generate a specialized hash function for doubles
        /// </summary>
        private HashDelegate<Double> ComposeDoubleHashDelegate()
        {
            return Hash;
        }

        private uint Hash(in float value, uint seed)
        {
            return Hashing.MurmurRound(seed, FloatUtils.GetBits(value));
        }

        private uint Hash(in double value, uint seed)
        {
            ulong v = FloatUtils.GetBits(value);
            uint hash = Hashing.MurmurRound(seed, Utils.GetLo(v));
            return Hashing.MurmurRound(hash, Utils.GetHi(v));
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(iinfo >= 0 && iinfo < _exes.Length);
            return _exes[iinfo].OutputColumnType;
        }
    }

    public static class HashJoin
    {
        [TlcModule.EntryPoint(Name = "Transforms.HashConverter",
            Desc = HashJoinTransform.Summary,
            UserName = HashJoinTransform.UserName,
            ShortName = HashJoinTransform.RegistrationName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name=""HashJoin""]/*' />",
                                 @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/example[@name=""HashJoin""]/*' />"})]
        public static CommonOutputs.TransformOutput Apply(IHostEnvironment env, HashJoinTransform.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "HashJoin", input);
            var view = new HashJoinTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, view, input.Data),
                OutputData = view
            };
        }
    }
}
