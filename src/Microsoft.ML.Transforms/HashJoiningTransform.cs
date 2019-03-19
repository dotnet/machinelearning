// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(HashJoiningTransform.Summary, typeof(HashJoiningTransform), typeof(HashJoiningTransform.Arguments), typeof(SignatureDataTransform),
    HashJoiningTransform.UserName, "HashJoinTransform", HashJoiningTransform.RegistrationName)]

[assembly: LoadableClass(HashJoiningTransform.Summary, typeof(HashJoiningTransform), null, typeof(SignatureLoadDataTransform),
    HashJoiningTransform.UserName, HashJoiningTransform.LoaderSignature, "HashJoinFunction")]

[assembly: EntryPointModule(typeof(HashJoin))]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// This transform hashes its input columns. Each column is hashed separately, and within each
    /// column there is an option to specify which slots should be hashed together into one output slot.
    /// This transform can be applied either to single valued columns or to known length vector columns.
    /// </summary>
    [BestFriend]
    internal sealed class HashJoiningTransform : OneToOneTransformBase
    {
        public const int NumBitsMin = 1;
        public const int NumBitsLim = 32;

        private static class Defaults
        {
            public const bool Join = true;
            public const int NumberOfBits = NumBitsLim - 1;
            public const uint Seed = 314489979;
            public const bool Ordered = true;
        }

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)",
                Name = "Column",
                ShortName = "col",
                SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the values need to be combined for a single hash")]
            public bool Join = Defaults.Join;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of bits to hash into. Must be between 1 and 31, inclusive.",
                ShortName = "bits", SortOrder = 2)]
            public int NumberOfBits = Defaults.NumberOfBits;

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
            public int? NumberOfBits;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
            public uint? Seed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the position of each term should be included in the hash", ShortName = "ord")]
            public bool? Ordered;

            internal static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (Join != null || !string.IsNullOrEmpty(CustomSlotMap) || NumberOfBits != null ||
                    Seed != null || Ordered != null)
                {
                    return false;
                }
                return TryUnparseCore(sb);
            }
        }

        [BestFriend]
        internal sealed class ColumnOptions
        {
            // Either VBuffer<Key<U4>> or a single Key<U4>.
            // Note that if CustomSlotMap contains only one array, the output type of the transform will a single Key<U4>.
            // This corresponds to the join=+ case, although now it's possible to omit certain slots entirely.
            // If # of hash bits is less than 31, the key type will have a positive count.
            public readonly DataViewType OutputColumnType;

            public readonly int NumberOfBits;
            public readonly uint HashSeed;
            public readonly bool Ordered;
            public readonly int[][] SlotMap; // null if the input is a single column

            public int OutputValueCount
            {
                get { return OutputColumnType.GetValueCount(); }
            }

            public ColumnOptions(int[][] slotMap, int numberOfBits, uint hashSeed, bool ordered)
            {
                Contracts.CheckValueOrNull(slotMap);
                Contracts.Check(NumBitsMin <= numberOfBits && numberOfBits < NumBitsLim);

                SlotMap = slotMap;
                NumberOfBits = numberOfBits;
                HashSeed = hashSeed;
                Ordered = ordered;
                var itemType = GetItemType(numberOfBits);
                if (Utils.Size(SlotMap) <= 1)
                    OutputColumnType = itemType;
                else
                    OutputColumnType = new VectorType(itemType, SlotMap.Length);
            }

            /// <summary>
            /// Constructs the correct KeyType for the given hash bits.
            /// Because of array size limitation, if numberOfBits = 31, the key type is not contiguous (not transformable into indicator array)
            /// </summary>
            private static KeyType GetItemType(int numberOfBits)
            {
                var keyCount = (ulong)1 << numberOfBits;
                return new KeyType(typeof(uint), keyCount);
            }
        }

        internal const string RegistrationName = "HashJoin";

        internal const string Summary = "Converts column values into hashes. This transform accepts both numeric and text inputs, both single and vector-valued columns. ";

        internal const string UserName = "Hash Join Transform";

        internal const string LoaderSignature = "HashJoinTransform";
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
                loaderAssemblyName: typeof(HashJoiningTransform).Assembly.FullName);
        }

        private readonly ColumnOptions[] _exes;

        /// <summary>
        /// Initializes a new instance of <see cref="HashJoiningTransform"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="source">Name of the column to be transformed. If this is null '<paramref name="name"/>' will be used.</param>
        /// <param name="join">Whether the values need to be combined for a single hash.</param>
        /// <param name="numberOfBits">Number of bits to hash into. Must be between 1 and 31, inclusive.</param>
        public HashJoiningTransform(IHostEnvironment env,
            IDataView input,
            string name,
            string source = null,
             bool join = Defaults.Join,
            int numberOfBits = Defaults.NumberOfBits)
            : this(env, new Arguments() { Columns = new[] { new Column() { Source = source ?? name, Name = name } }, Join = join, NumberOfBits = numberOfBits }, input)
        {
        }

        /// <include file='doc.xml' path='doc/members/member[@name="HashJoin"]/*' />
        public HashJoiningTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, Contracts.CheckRef(args, nameof(args)).Columns, input, TestColumnType)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Columns));

            if (args.NumberOfBits < NumBitsMin || args.NumberOfBits >= NumBitsLim)
                throw Host.ExceptUserArg(nameof(args.NumberOfBits), "numberOfBits should be between {0} and {1} inclusive", NumBitsMin, NumBitsLim - 1);

            _exes = new ColumnOptions[Infos.Length];
            for (int i = 0; i < Infos.Length; i++)
            {
                var numberOfBits = args.Columns[i].NumberOfBits ?? args.NumberOfBits;
                Host.CheckUserArg(NumBitsMin <= numberOfBits && numberOfBits < NumBitsLim, nameof(args.NumberOfBits));
                _exes[i] = CreateColumnOptionsEx(
                    args.Columns[i].Join ?? args.Join,
                    args.Columns[i].CustomSlotMap,
                    args.Columns[i].NumberOfBits ?? args.NumberOfBits,
                    args.Columns[i].Seed ?? args.Seed,
                    args.Columns[i].Ordered ?? args.Ordered,
                    Infos[i]);
            }

            SetMetadata();
        }

        private HashJoiningTransform(IHost host, ModelLoadContext ctx, IDataView input)
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

            _exes = new ColumnOptions[Infos.Length];
            for (int i = 0; i < Infos.Length; i++)
            {
                int numberOfBits = ctx.Reader.ReadInt32();
                Host.CheckDecode(NumBitsMin <= numberOfBits && numberOfBits < NumBitsLim);

                uint hashSeed = ctx.Reader.ReadUInt32();
                bool ordered = ctx.Reader.ReadBoolByte();

                int slotMapCount = ctx.Reader.ReadInt32();
                Host.CheckDecode(slotMapCount >= 0);

                int[][] slotMap = null;
                if (slotMapCount > 0)
                {
                    Host.CheckDecode(Infos[i].TypeSrc is VectorType);

                    slotMap = new int[slotMapCount][];
                    for (int j = 0; j < slotMapCount; j++)
                    {
                        slotMap[j] = ctx.Reader.ReadIntArray();
                        Host.CheckDecode(Utils.Size(slotMap[j]) > 0); // null array could be returned by the call above

                        // the slots should be distinct and between 0 and vector size
                        Host.CheckDecode(slotMap[j].Distinct().Count() == slotMap[j].Length);
                        Host.CheckDecode(
                            slotMap[j].All(slot => 0 <= slot && slot < Infos[i].TypeSrc.GetValueCount()));
                    }
                }

                _exes[i] = new ColumnOptions(slotMap, numberOfBits, hashSeed, ordered);
            }

            SetMetadata();
        }

        public static HashJoiningTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);

            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            h.CheckValue(input, nameof(input));

            return h.Apply("Loading Model", ch => new HashJoiningTransform(h, ctx, input));
        }

        private protected override void SaveModel(ModelSaveContext ctx)
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

                Host.Assert(NumBitsMin <= ex.NumberOfBits && ex.NumberOfBits < NumBitsLim);
                ctx.Writer.Write(ex.NumberOfBits);

                ctx.Writer.Write(ex.HashSeed);
                ctx.Writer.WriteBoolByte(ex.Ordered);

                ctx.Writer.Write(Utils.Size(ex.SlotMap));
                if (ex.SlotMap == null)
                    continue;

                for (int i = 0; i < ex.SlotMap.Length; i++)
                {
                    Host.Assert(ex.SlotMap[i].Distinct().Count() == ex.SlotMap[i].Length);
                    Host.Assert(ex.SlotMap[i].All(slot => 0 <= slot && slot < Infos[iColumn].TypeSrc.GetValueCount()));
                    ctx.Writer.WriteIntArray(ex.SlotMap[i]);
                }
            }
        }

        private ColumnOptions CreateColumnOptionsEx(bool join, string customSlotMap, int numberOfBits, uint hashSeed, bool ordered, ColInfo colInfo)
        {
            int[][] slotMap = null;
            if (colInfo.TypeSrc is VectorType vectorType)
            {
                // fill in the slot map
                if (!string.IsNullOrWhiteSpace(customSlotMap))
                    slotMap = CompileSlotMap(customSlotMap, vectorType.Size);
                else
                    slotMap = CreateDefaultSlotMap(join, vectorType.Size);
                Host.Assert(Utils.Size(slotMap) >= 1);
            }

            return new ColumnOptions(slotMap, numberOfBits, hashSeed, ordered);
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

        private static string TestColumnType(DataViewType type)
        {
            // REVIEW: list all types that can be hashed.
            if (type.GetValueCount() > 0)
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
                    bldr.AddGetter<VBuffer<ReadOnlyMemory<char>>>(AnnotationUtils.Kinds.SlotNames,
                        new VectorType(TextDataViewType.Instance, ex.SlotMap.Length), GetSlotNames);
                }
            }
            md.Seal();
        }

        private void GetSlotNames(int iinfo, ref VBuffer<ReadOnlyMemory<char>> dst)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);

            Host.AssertValue(_exes[iinfo].SlotMap);

            int n = _exes[iinfo].OutputValueCount;
            var dstEditor = VBufferEditor.Create(ref dst, n);

            var srcColumnName = Source.Schema[Infos[iinfo].Source].Name;
            bool useDefaultSlotNames = !Source.Schema[Infos[iinfo].Source].HasSlotNames(Infos[iinfo].TypeSrc.GetVectorSize());
            VBuffer<ReadOnlyMemory<char>> srcSlotNames = default;
            if (!useDefaultSlotNames)
            {
                Source.Schema[Infos[iinfo].Source].Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref srcSlotNames);
                useDefaultSlotNames =
                    !srcSlotNames.IsDense
                    || srcSlotNames.Length != Infos[iinfo].TypeSrc.GetValueCount();
            }

            var outputSlotName = new StringBuilder();
            var srcSlotNameValues = srcSlotNames.GetValues();
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
                        outputSlotName.Append(srcSlotNameValues[inputSlotIndex]);
                }

                dstEditor.Values[slot] = outputSlotName.ToString().AsMemory();
            }

            dst = dstEditor.Commit();
        }

        private delegate uint HashDelegate<TSrc>(in TSrc value, uint seed);

        // generic method generators
        private static MethodInfo _methGetterOneToOne;
        private static MethodInfo _methGetterVecToVec;
        private static MethodInfo _methGetterVecToOne;

        protected override Delegate GetGetterCore(IChannel ch, DataViewRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            // Construct MethodInfos templates that we need for the generic methods.
            if (_methGetterOneToOne == null)
            {
                Func<DataViewRow, int, ValueGetter<uint>> del = ComposeGetterOneToOne<int>;
                Interlocked.CompareExchange(ref _methGetterOneToOne, del.GetMethodInfo().GetGenericMethodDefinition(), null);
            }
            if (_methGetterVecToVec == null)
            {
                Func<DataViewRow, int, ValueGetter<VBuffer<uint>>> del = ComposeGetterVecToVec<int>;
                Interlocked.CompareExchange(ref _methGetterVecToVec, del.GetMethodInfo().GetGenericMethodDefinition(), null);
            }
            if (_methGetterVecToOne == null)
            {
                Func<DataViewRow, int, ValueGetter<uint>> del = ComposeGetterVecToOne<int>;
                Interlocked.CompareExchange(ref _methGetterVecToOne, del.GetMethodInfo().GetGenericMethodDefinition(), null);
            }

            // Magic code to generate a correct getter.
            // First, we take a method info for GetGetter<int>
            // Then, we replace <int> with correct type of the input
            // And then we generate a delegate using the generic delegate generator
            DataViewType itemType;
            MethodInfo mi;
            if (!(Infos[iinfo].TypeSrc is VectorType vectorType))
            {
                itemType = Infos[iinfo].TypeSrc;
                mi = _methGetterOneToOne;
            }
            else
            {
                itemType = vectorType.ItemType;
                if (_exes[iinfo].OutputValueCount == 1)
                    mi = _methGetterVecToOne;
                else
                    mi = _methGetterVecToVec;
            }

            mi = mi.MakeGenericMethod(itemType.RawType);
            return (Delegate)mi.Invoke(this, new object[] { input, iinfo });
        }

        /// <summary>
        /// Getter generator for inputs of type <typeparamref name="TSrc"/>
        /// </summary>
        /// <typeparam name="TSrc">Input type. Must be a non-vector</typeparam>
        /// <param name="input">Row inout</param>
        /// <param name="iinfo">Index of the getter</param>
        private ValueGetter<uint> ComposeGetterOneToOne<TSrc>(DataViewRow input, int iinfo)
        {
            Host.AssertValue(input);
            Host.Assert(!(Infos[iinfo].TypeSrc is VectorType));

            var getSrc = GetSrcGetter<TSrc>(input, iinfo);
            var hashFunction = ComposeHashDelegate<TSrc>();
            var src = default(TSrc);
            var mask = (1U << _exes[iinfo].NumberOfBits) - 1;
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
        private ValueGetter<VBuffer<uint>> ComposeGetterVecToVec<TSrc>(DataViewRow input, int iinfo)
        {
            Host.AssertValue(input);
            VectorType srcType = Infos[iinfo].TypeSrc as VectorType;
            Host.Assert(srcType != null);

            var getSrc = GetSrcGetter<VBuffer<TSrc>>(input, iinfo);
            var hashFunction = ComposeHashDelegate<TSrc>();
            var src = default(VBuffer<TSrc>);
            int n = _exes[iinfo].OutputValueCount;
            int expectedSrcLength = srcType.Size;
            int[][] slotMap = _exes[iinfo].SlotMap;
            // REVIEW: consider adding a fix-zero functionality (subtract emptyTextHash from all hashes)
            var mask = (1U << _exes[iinfo].NumberOfBits) - 1;
            var hashSeed = _exes[iinfo].HashSeed;
            bool ordered = _exes[iinfo].Ordered;
            var denseSource = default(VBuffer<TSrc>);
            return
                (ref VBuffer<uint> dst) =>
                {
                    getSrc(ref src);
                    Host.Check(src.Length == expectedSrcLength);
                    var hashes = VBufferEditor.Create(ref dst, n);
                    src.CopyToDense(ref denseSource);
                    for (int i = 0; i < n; i++)
                    {
                        uint hash = hashSeed;
                        foreach (var srcSlot in slotMap[i])
                        {
                            // REVIEW: some legacy code hashes 0 for srcSlot in ord- case, do we need to preserve this behavior?
                            if (ordered)
                                hash = Hashing.MurmurRound(hash, (uint)srcSlot);
                            hash = hashFunction(denseSource.GetItemOrDefault(srcSlot), hash);
                        }

                        hashes.Values[i] = (Hashing.MixHash(hash) & mask) + 1; // +1 to offset from zero, which has special meaning for KeyType
                    }

                    dst = hashes.Commit();
                };
        }

        /// <summary>
        /// Getter generator for inputs of type <typeparamref name="TSrc"/>, where output type is a single hash
        /// </summary>
        /// <typeparam name="TSrc">Input type. Must be a vector</typeparam>
        /// <param name="input">Row input</param>
        /// <param name="iinfo">Index of the getter</param>
        private ValueGetter<uint> ComposeGetterVecToOne<TSrc>(DataViewRow input, int iinfo)
        {
            Host.AssertValue(input);
            VectorType srcType = Infos[iinfo].TypeSrc as VectorType;
            Host.Assert(srcType != null);
            Host.Assert(Utils.Size(_exes[iinfo].SlotMap) == 1);

            var slots = _exes[iinfo].SlotMap[0];

            var getSrc = GetSrcGetter<VBuffer<TSrc>>(input, iinfo);
            var hashFunction = ComposeHashDelegate<TSrc>();
            var src = default(VBuffer<TSrc>);
            int expectedSrcLength = srcType.Size;
            var mask = (1U << _exes[iinfo].NumberOfBits) - 1;
            var hashSeed = _exes[iinfo].HashSeed;
            bool ordered = _exes[iinfo].Ordered;
            var denseSource = default(VBuffer<TSrc>);
            return
                (ref uint dst) =>
                {
                    getSrc(ref src);
                    Host.Check(src.Length == expectedSrcLength);
                    src.CopyToDense(ref denseSource);
                    uint hash = hashSeed;
                    foreach (var srcSlot in slots)
                    {
                        if (ordered)
                            hash = Hashing.MurmurRound(hash, (uint)srcSlot);
                        hash = hashFunction(denseSource.GetItemOrDefault(srcSlot), hash);
                    }
                    dst = (Hashing.MixHash(hash) & mask) + 1; // +1 to offset from zero, which has special meaning for KeyType
                };
        }

        /// <summary>
        /// Generic hash function
        /// </summary>
        private HashDelegate<TSrc> ComposeHashDelegate<TSrc>()
        {
            // REVIEW: Add a specialized hashing for ints, once numeric bin mapper is done.
            if (typeof(TSrc) == typeof(float))
                return (HashDelegate<TSrc>)(Delegate)ComposeFloatHashDelegate();

            if (typeof(TSrc) == typeof(double))
                return (HashDelegate<TSrc>)(Delegate)ComposeDoubleHashDelegate();

            // Default case: convert to text and hash as a string.
            var sb = default(StringBuilder);
            var conv = Data.Conversion.Conversions.Instance.GetStringConversion<TSrc>();
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
        private HashDelegate<float> ComposeFloatHashDelegate()
        {
            return Hash;
        }

        /// <summary>
        /// Generate a specialized hash function for doubles
        /// </summary>
        private HashDelegate<double> ComposeDoubleHashDelegate() => Hash;

        private uint Hash(in float value, uint seed) => Hashing.MurmurRound(seed, FloatUtils.GetBits(value));

        private uint Hash(in double value, uint seed)
        {
            ulong v = FloatUtils.GetBits(value);
            uint hash = Hashing.MurmurRound(seed, Utils.GetLo(v));
            return Hashing.MurmurRound(hash, Utils.GetHi(v));
        }

        protected override DataViewType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(iinfo >= 0 && iinfo < _exes.Length);
            return _exes[iinfo].OutputColumnType;
        }
    }

    internal static class HashJoin
    {
        [TlcModule.EntryPoint(Name = "Transforms.HashConverter",
            Desc = HashJoiningTransform.Summary,
            UserName = HashJoiningTransform.UserName,
            ShortName = HashJoiningTransform.RegistrationName)]
        public static CommonOutputs.TransformOutput Apply(IHostEnvironment env, HashJoiningTransform.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "HashJoin", input);
            var view = new HashJoiningTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, input.Data),
                OutputData = view
            };
        }
    }
}
