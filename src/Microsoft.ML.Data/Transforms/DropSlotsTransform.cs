// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;

[assembly: LoadableClass(DropSlotsTransform.Summary, typeof(DropSlotsTransform), typeof(DropSlotsTransform.Arguments), typeof(SignatureDataTransform),
    DropSlotsTransform.FriendlyName, "DropSlots", "DropSlotsTransform")]

[assembly: LoadableClass(DropSlotsTransform.Summary, typeof(DropSlotsTransform), null, typeof(SignatureLoadDataTransform),
    DropSlotsTransform.FriendlyName, DropSlotsTransform.LoaderSignature)]

[assembly: LoadableClass(DropSlotsTransform.Summary, typeof(DropSlotsTransform), null, typeof(SignatureLoadModel),
    DropSlotsTransform.FriendlyName, DropSlotsTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(DropSlotsTransform), null, typeof(SignatureLoadRowMapper),
   DropSlotsTransform.FriendlyName, DropSlotsTransform.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Transform to drop slots from columns. If the column is scalar, the only slot that can be dropped is slot 0.
    /// If all the slots are to be dropped, a vector valued column will be changed to a vector of length 1 (a scalar column will retain its type) and
    /// the value will be the default value.
    /// </summary>
    public sealed class DropSlotsTransform : OneToOneTransformerBase
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Columns to drop the slots for", ShortName = "col", SortOrder = 1)]
            public Column[] Column;
        }

        public sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.Multiple, HelpText = "Source slot index range(s) of the column to drop")]
            public Range[] Slots;

            public static Column Parse(string str)
            {
                Contracts.CheckNonWhiteSpace(str, nameof(str));

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            protected override bool TryParse(string str)
            {
                Contracts.AssertNonEmpty(str);

                // Allow name:src:slots and src:slots
                int ich = str.LastIndexOf(':');
                if (ich <= 0 || ich >= str.Length - 1)
                    return false;
                if (!base.TryParse(str.Substring(0, ich)))
                    return false;
                return TryParseSlots(str.Substring(ich + 1));
            }

            private bool TryParseSlots(string str)
            {
                Contracts.AssertValue(str);
                var strs = str.Split(',');
                if (str.Length == 0)
                    return false;
                Slots = new Range[strs.Length];
                for (int i = 0; i < strs.Length; i++)
                {
                    if ((Slots[i] = Range.Parse(strs[i])) == null)
                        return false;
                }
                return true;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.CheckValue(sb, nameof(sb));

                int ich = sb.Length;

                if (!TryUnparseCore(sb))
                    return false;

                sb.Append(':');
                string pre = "";
                foreach (var src in Slots)
                {
                    sb.Append(pre);
                    if (!src.TryUnparse(sb))
                    {
                        sb.Length = ich;
                        return false;
                    }
                    pre = ",";
                }
                return true;
            }
        }

        public sealed class Range
        {
            [Argument(ArgumentType.Required, HelpText = "First index in the range")]
            public int Min;

            // If null, it means int.MaxValue - 1. There are two reasons for this:
            // 1. max is an index, so it has to be strictly less than int.MaxValue.
            // 2. to prevent overflows when adding 1 to it.
            [Argument(ArgumentType.AtMostOnce, HelpText = "Last index in the range")]
            public int? Max;

            /// <summary>
            /// Source slot index range of the column to drop.
            /// </summary>
            /// <param name="min">Start index.</param>
            /// <param name="max">End index. If null it means int.MaxValue - 1.</param>
            public Range(int min, int? max)
            {
                Min = min;
                Max = max;
            }

            internal Range()
            {
            }

            public static Range Parse(string str)
            {
                Contracts.CheckNonWhiteSpace(str, nameof(str));

                var res = new Range();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            private bool TryParse(string str)
            {
                Contracts.AssertNonEmpty(str);

                int ich = str.IndexOf('-');
                if (ich < 0)
                {
                    if (!int.TryParse(str, out Min))
                        return false;
                    Max = Min;
                    return true;
                }

                if (ich == 0 || ich >= str.Length - 1)
                {
                    return false;
                }

                if (!int.TryParse(str.Substring(0, ich), out Min))
                    return false;

                string rest = str.Substring(ich + 1);
                if (rest == "*")
                    return true;

                int tmp;
                if (!int.TryParse(rest, out tmp))
                    return false;
                Max = tmp;
                return true;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.CheckValue(sb, nameof(sb));
                sb.Append(Min);
                if (Max != null)
                {
                    if (Max != Min)
                        sb.Append("-").Append(Max);
                }
                else
                    sb.Append("-*");
                return true;
            }

            /// <summary>
            /// Returns true if the range is valid.
            /// </summary>
            public bool IsValid()
            {
                return Min >= 0 && (Max == null || Min <= Max);
            }
        }

        /// <summary>
        /// Extra information for each column (in addition to ColumnInfo).
        /// </summary>
        public sealed class ColumnInfo
        {
            public readonly string Input;
            public readonly string Output;
            public readonly Range[] Slots;

            public ColumnInfo(string input, string output, Range[] slots)
            {
                foreach (var range in slots)
                    Contracts.Assert(range.IsValid());
                Input = input;
                Output = output;
                Slots = slots;
                //GetSlotsMinMax(slots, out SlotsMin, out SlotsMax);

            }

            //internal ColumnInfo(string input, string output, int[] slotsMin, int[] slotsMax)
            //{
            //    Input = input;
            //    Output = output;
            //    SlotsMin = slotsMin;
            //    SlotsMax = slotsMax;
            //}

            internal ColumnInfo(Column column)
            {
                Input = column.Source ?? column.Name;
                Output = column.Name;
                Slots = column.Slots;
                //GetSlotsMinMax(column.Slots, out SlotsMin, out SlotsMax);
            }
        }

        private const string RegistrationName = "DropSlots";
        internal const string Summary = "Removes the selected slots from the column.";
        internal const string FriendlyName = "Drop Slots Transform";
        internal const string LoaderSignature = "DropSlotsTransform";

        internal readonly int[][] SlotsMin;
        internal readonly int[][] SlotsMax;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "DROPSLOT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(DropSlotsTransform).Assembly.FullName);
        }

        /// <summary>
        /// Initializes a new <see cref="DropSlotsTransform"/> object.
        /// </summary>
        internal DropSlotsTransform(IHostEnvironment env, params ColumnInfo[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            Host.AssertNonEmpty(ColumnPairs);
            GetSlotsMinMax(columns, out SlotsMin, out SlotsMax);
            Host.CheckUserArg(AreRangesValid(SlotsMin, SlotsMax), nameof(columns), "The range min and max must be non-negative and min must be less than or equal to max.");
        }

        private DropSlotsTransform(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), ctx)
        {
            Host.AssertValue(ctx);
            // *** Binary format ***
            // <base>
            // for each added column
            //   int[]: slotsMin
            //   int[]: slotsMax (no count)
            Host.AssertNonEmpty(ColumnPairs);
            var size = ColumnPairs.Length;
            for (int i = 0; i < size; i++)
            {
                SlotsMin[i] = ctx.Reader.ReadIntArray();
                Host.CheckDecode(Utils.Size(SlotsMin) > 0);
                SlotsMax[i] = ctx.Reader.ReadIntArray(SlotsMin.Length);
            }
            Host.Assert(AreRangesValid(SlotsMin, SlotsMax));
        }

        // Factory method for SignatureLoadModel
        internal static DropSlotsTransform Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            ctx.CheckAtModel(GetVersionInfo());
            return new DropSlotsTransform(env, ctx);
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            var columns = args.Column.Select(column => new ColumnInfo(column)).ToArray();
            return new DropSlotsTransform(env, columns).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        internal static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(Schema.Create(inputSchema));

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            // for each added column
            //   int[]: slotsMin
            //   int[]: slotsMax (no count)
            SaveColumns(ctx);
            Host.Assert(AreRangesValid(SlotsMin, SlotsMax));
            for (int i = 0; i < ColumnPairs.Length; i++)
            {
                Host.Assert(SlotsMin[i].Length == SlotsMax[i].Length);
                ctx.Writer.WriteIntArray(SlotsMin[i]);
                ctx.Writer.WriteIntsNoCount(SlotsMax[i], SlotsMax[i].Length);
            }
        }

        private static void GetSlotsMinMax(ColumnInfo[] columns, out int[][] slotsMin, out int[][] slotsMax)
        {
            slotsMin = new int[columns.Length][];
            slotsMax = new int[columns.Length][];
            for (int i = 0; i < columns.Length; i++)
            {
                var slots = columns[i].Slots;
                slotsMin[i] = new int[slots.Length];
                slotsMax[i] = new int[slots.Length];
                for (int j = 0; j < slots.Length; j++)
                {
                    var range = slots[j];
                    slotsMin[i][j] = range.Min;
                    // There are two reasons for setting the max to int.MaxValue - 1:
                    // 1. max is an index, so it has to be strictly less than int.MaxValue.
                    // 2. to prevent overflows when adding 1 to it.
                    slotsMax[i][j] = range.Max ?? int.MaxValue - 1;
                }
                Array.Sort(slotsMin, slotsMax);
                var iDst = 0;
                for (int j = 1; j < slots.Length; j++)
                {
                    if (slotsMin[i][j] <= slotsMax[i][iDst] + 1)
                        slotsMax[i][iDst] = Math.Max(slotsMax[i][iDst], slotsMax[i][j]);
                    else
                    {
                        iDst++;
                        slotsMin[i][iDst] = slotsMin[i][j];
                        slotsMax[i][iDst] = slotsMax[i][j];
                    }
                }
                iDst++;
                Array.Resize(ref slotsMin[i], iDst);
                Array.Resize(ref slotsMax[i], iDst);
            }
        }

        private static (string input, string output)[] GetColumnPairs(ColumnInfo[] columns)
            => columns.Select(c => (c.Input, c.Output ?? c.Input)).ToArray();

        private static bool AreRangesValid(int[][] slotsMin, int[][] slotsMax)
        {
            if (slotsMin.Length != slotsMax.Length)
                return false;
            for (int iinfo = 0; iinfo < slotsMin.Length; iinfo++)
            {
                var prevmax = -2;
                for (int i = 0; i < slotsMin.Length; i++)
                {
                    if (!(0 <= slotsMin[iinfo][i] && slotsMin[iinfo][i] < int.MaxValue))
                        return false;
                    if (!(0 <= slotsMax[iinfo][i] && slotsMax[iinfo][i] < int.MaxValue))
                        return false;
                    if (!(slotsMin[iinfo][i] <= slotsMax[iinfo][i]))
                        return false;
                    if (!(slotsMin[iinfo][i] - 1 > prevmax))
                        return false;
                    prevmax = slotsMax[iinfo][i];
                }
            }
            return true;
        }

        //protected override void ActivateSourceColumns(int iinfo, bool[] active)
        //{
        //    if (!_columns[iinfo].Suppressed)
        //        active[ColumnPairs[iinfo].input] = true;
        //}

        //protected override bool WantParallelCursors(Func<int, bool> predicate)
        //{
        //    Host.AssertValue(predicate);

        //    // Parallel helps when some, but not all, slots are dropped.
        //    for (int iinfo = 0; iinfo < ColumnPairs.Length; iinfo++)
        //    {
        //        int col = ColumnIndex(iinfo);
        //        if (!predicate(col))
        //            continue;

        //        var ex = _columns[iinfo];
        //        var info = ColumnPairs[iinfo];
        //        Contracts.Assert(ex.TypeDst.IsVector == info.TypeSrc.IsVector);
        //        Contracts.Assert(ex.TypeDst.IsKnownSizeVector == info.TypeSrc.IsKnownSizeVector);
        //        Contracts.Assert(!ex.TypeDst.IsKnownSizeVector || ex.TypeDst.ValueCount <= info.TypeSrc.ValueCount);
        //        if (!ex.Suppressed &&
        //            (!ex.TypeDst.IsKnownSizeVector || ex.TypeDst.ValueCount != info.TypeSrc.ValueCount))
        //        {
        //            return true;
        //        }
        //    }

        //    // No real work to do.
        //    return false;
        //}

        protected override IRowMapper MakeRowMapper(Schema schema)
            => new Mapper(this, schema);

        private sealed class Mapper : MapperBase
        {
            private readonly DropSlotsTransform _parent;
            private readonly int[] _cols;
            private readonly ColumnType[] _srcTypes;
            private readonly ColumnType[] _dstTypes;

            private readonly SlotDropper[] _slotDropper;
            //// Track if all the slots of the column are to be dropped.
            private readonly bool[] _suppressed;
            private readonly int[][] _categoricalRanges;

            public Mapper(DropSlotsTransform parent, Schema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _cols = new int[_parent.ColumnPairs.Length];
                _srcTypes = new ColumnType[_parent.ColumnPairs.Length];
                _dstTypes = new ColumnType[_parent.ColumnPairs.Length];

                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    if (!InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out _cols[i]))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.ColumnPairs[i].input);
                    _srcTypes[i] = inputSchema.GetColumnType(_cols[i]);
                    // TODO check that the types are ok if needed
                    SlotDropper slotDropper = new SlotDropper(_srcTypes[i].ValueCount, _parent.SlotsMin[i], _parent.SlotsMax[i]);
                    ComputeType(inputSchema, i, _slotDropper[i], out _suppressed[i], out _dstTypes[i], out _categoricalRanges[i]);
                }
            }

            /// <summary>
            /// Computes the types (column and slotnames), the length reduction, categorical feature indices
            /// and whether the column is suppressed.
            /// The slotsMin and slotsMax arrays should be sorted and the intervals should not overlap.
            /// </summary>
            /// <param name="input">The input schema</param>
            /// <param name="iinfo">The column index in Infos</param>
            /// <param name="slotDropper">The slots to be dropped.</param>
            /// <param name="suppressed">Whether the column is suppressed (all slots dropped)</param>
            /// <param name="type">The column type</param>
            /// <param name="categoricalRanges">Categorical feature indices.</param>
            private void ComputeType(Schema input, int iinfo, SlotDropper slotDropper,
                out bool suppressed, out ColumnType type, out int[] categoricalRanges)
            {
                var slotsMin = _parent.SlotsMin[iinfo];
                var slotsMax = _parent.SlotsMax[iinfo];
                Host.AssertValue(slotDropper);
                Host.AssertValue(input);
                Host.AssertNonEmpty(slotsMin);
                Host.AssertNonEmpty(slotsMax);
                Host.Assert(slotsMin.Length == slotsMax.Length);
                Host.Assert(0 <= iinfo && iinfo < slotsMax.Length);

                categoricalRanges = null;
                //// Register for metadata. Propagate the IsNormalized metadata.
                //using (var bldr = Metadata.BuildMetadata(iinfo, input, _parent.ColumnPairs[iinfo].input,
                //    MetadataUtils.Kinds.IsNormalized, MetadataUtils.Kinds.KeyValues))
                //{
                    var typeSrc = _srcTypes[iinfo];
                    if (!typeSrc.IsVector)
                    {
                        type = typeSrc;
                        suppressed = slotsMin.Length > 0 && slotsMin[0] == 0;
                    }
                    else if (!typeSrc.IsKnownSizeVector)
                    {
                        type = typeSrc;
                        suppressed = false;
                    }
                    else
                    {
                        Host.Assert(typeSrc.IsKnownSizeVector);
                        var dstLength = slotDropper.DstLength;
                        var hasSlotNames = input.HasSlotNames(_cols[iinfo], _srcTypes[iinfo].VectorSize);
                        type = new VectorType(typeSrc.ItemType.AsPrimitive, Math.Max(dstLength, 1));
                        suppressed = dstLength == 0;

                        //if (hasSlotNames && dstLength > 0)
                        //{
                        //    // Add slot name metadata.
                        //    bldr.AddGetter<VBuffer<ReadOnlyMemory<char>>>(MetadataUtils.Kinds.SlotNames,
                        //        new VectorType(TextType.Instance, dstLength), GetSlotNames);
                        //}
                    }

                    //if (!suppressed)
                    //{
                    //    if (MetadataUtils.TryGetCategoricalFeatureIndices(input, _cols[iinfo], out categoricalRanges))
                    //    {
                    //        VBuffer<int> dst = default(VBuffer<int>);
                    //        GetCategoricalSlotRangesCore(iinfo, slotDropper.SlotsMin, slotDropper.SlotsMax, categoricalRanges, ref dst);
                    //        // REVIEW: cache dst as opposed to caculating it again.
                    //        if (dst.Length > 0)
                    //        {
                    //            Contracts.Assert(dst.Length % 2 == 0);

                    //            bldr.AddGetter<VBuffer<int>>(MetadataUtils.Kinds.CategoricalSlotRanges,
                    //                MetadataUtils.GetCategoricalType(dst.Length / 2), GetCategoricalSlotRanges);
                    //        }
                    //    }
                    //}
                //}
            }

            //protected override ColumnType GetColumnTypeCore(int iinfo)
            //{
            //    Host.Assert(0 <= iinfo & iinfo < _parent.ColumnPairs.Length);
            //    return _columns[iinfo].TypeDst;
            //}

            private void GetSlotNames(int iinfo, ref VBuffer<ReadOnlyMemory<char>> dst)
            {
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);

                var names = default(VBuffer<ReadOnlyMemory<char>>);
                InputSchema.GetMetadata(MetadataUtils.Kinds.SlotNames, _cols[iinfo], ref names);
                _slotDropper[iinfo].DropSlots(ref names, ref dst);
            }

            private void GetCategoricalSlotRanges(int iinfo, ref VBuffer<int> dst)
            {
                if (_categoricalRanges[iinfo] != null)
                {
                    GetCategoricalSlotRangesCore(iinfo, _parent.SlotsMin[iinfo],
                        _parent.SlotsMax[iinfo], _categoricalRanges[iinfo], ref dst);
                }
            }

            private void GetCategoricalSlotRangesCore(int iinfo, int[] slotsMin, int[] slotsMax, int[] catRanges, ref VBuffer<int> dst)
            {
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                Host.Assert(slotsMax != null && slotsMin != null);
                Host.Assert(slotsMax.Length == slotsMin.Length);

                Contracts.Assert(catRanges.Length > 0 && catRanges.Length % 2 == 0);

                var ranges = new int[catRanges.Length];
                catRanges.CopyTo(ranges, 0);
                int rangesIndex = 0;
                int dropSlotsIndex = 0;
                int previousDropSlotsIndex = 0;
                int droppedSlotsCount = 0;
                bool combine = false;
                int min = -1;
                int max = -1;
                List<int> newCategoricalSlotRanges = new List<int>();

                // Six possible ways a drop slot range interacts with categorical slots range.
                //
                //                    +--------------Drop-------------+
                //                    |                               |
                //
                //                +---Drop---+   +---Drop---+   +---Drop---+
                //  +---Drop---+  |          |   |          |   |          |  +---Drop---+
                //  |          |       |____________Range____________|        |          |
                //
                // The below code is better understood as a state machine.

                while (dropSlotsIndex < slotsMin.Length && rangesIndex < ranges.Length)
                {
                    Contracts.Assert(rangesIndex % 2 == 0);
                    Contracts.Assert(ranges[rangesIndex] <= ranges[rangesIndex + 1]);

                    if (slotsMax[dropSlotsIndex] < ranges[rangesIndex])
                        dropSlotsIndex++;
                    else if (slotsMin[dropSlotsIndex] > ranges[rangesIndex + 1])
                    {
                        if (combine)
                        {
                            CombineRanges(min, max, ranges[rangesIndex] - droppedSlotsCount,
                                ranges[rangesIndex + 1] - droppedSlotsCount, out min, out max);
                        }
                        else
                        {
                            Contracts.Assert(min == -1 && max == -1);
                            min = ranges[rangesIndex] - droppedSlotsCount;
                            max = ranges[rangesIndex + 1] - droppedSlotsCount;
                        }

                        newCategoricalSlotRanges.Add(min);
                        newCategoricalSlotRanges.Add(max);
                        min = max = -1;
                        rangesIndex += 2;
                        combine = false;
                    }
                    else if (slotsMin[dropSlotsIndex] <= ranges[rangesIndex] &&
                             slotsMax[dropSlotsIndex] >= ranges[rangesIndex + 1])
                    {
                        rangesIndex += 2;
                        if (combine)
                        {
                            Contracts.Assert(min >= 0 && min <= max);
                            newCategoricalSlotRanges.Add(min);
                            newCategoricalSlotRanges.Add(max);
                            min = max = -1;
                            combine = false;
                        }

                        Contracts.Assert(min == -1 && max == -1);

                    }
                    else if (slotsMin[dropSlotsIndex] > ranges[rangesIndex] &&
                             slotsMax[dropSlotsIndex] < ranges[rangesIndex + 1])
                    {
                        if (combine)
                        {
                            CombineRanges(min, max, ranges[rangesIndex] - droppedSlotsCount,
                                slotsMin[dropSlotsIndex] - 1 - droppedSlotsCount, out min, out max);
                        }
                        else
                        {
                            Contracts.Assert(min == -1 && max == -1);

                            min = ranges[rangesIndex] - droppedSlotsCount;
                            max = slotsMin[dropSlotsIndex] - 1 - droppedSlotsCount;
                            combine = true;
                        }

                        ranges[rangesIndex] = slotsMax[dropSlotsIndex] + 1;
                        dropSlotsIndex++;
                    }
                    else if (slotsMax[dropSlotsIndex] < ranges[rangesIndex + 1])
                    {
                        ranges[rangesIndex] = slotsMax[dropSlotsIndex] + 1;
                        dropSlotsIndex++;
                    }
                    else
                        ranges[rangesIndex + 1] = slotsMin[dropSlotsIndex] - 1;

                    if (previousDropSlotsIndex < dropSlotsIndex)
                    {
                        Contracts.Assert(dropSlotsIndex - previousDropSlotsIndex == 1);
                        droppedSlotsCount += slotsMax[previousDropSlotsIndex] - slotsMin[previousDropSlotsIndex] + 1;
                        previousDropSlotsIndex = dropSlotsIndex;
                    }
                }

                Contracts.Assert(rangesIndex % 2 == 0);

                if (combine)
                {
                    Contracts.Assert(rangesIndex < ranges.Length - 1);
                    CombineRanges(min, max, ranges[rangesIndex] - droppedSlotsCount,
                        ranges[rangesIndex + 1] - droppedSlotsCount, out min, out max);

                    newCategoricalSlotRanges.Add(min);
                    newCategoricalSlotRanges.Add(max);
                    rangesIndex += 2;
                    combine = false;
                    min = max = -1;
                }

                Contracts.Assert(min == -1 && max == -1);

                for (int i = rangesIndex; i < ranges.Length; i++)
                    newCategoricalSlotRanges.Add(ranges[i] - droppedSlotsCount);

                Contracts.Assert(newCategoricalSlotRanges.Count % 2 == 0);
                Contracts.Assert(newCategoricalSlotRanges.TrueForAll(x => x >= 0));
                Contracts.Assert(0 <= droppedSlotsCount && droppedSlotsCount <= slotsMax[slotsMax.Length - 1] + 1);

                if (newCategoricalSlotRanges.Count > 0)
                    dst = new VBuffer<int>(newCategoricalSlotRanges.Count, newCategoricalSlotRanges.ToArray());
            }

            private void CombineRanges(
                int minRange1, int maxRange1, int minRange2, int maxRange2,
                out int newRangeMin, out int newRangeMax)
            {
                Contracts.Assert(minRange2 >= 0 && maxRange2 >= 0);
                Contracts.Assert(minRange2 <= maxRange2);
                Contracts.Assert(minRange1 >= 0 && maxRange1 >= 0);
                Contracts.Assert(minRange1 <= maxRange1);
                Contracts.Assert(maxRange1 + 1 == minRange2);

                newRangeMin = minRange1;
                newRangeMax = maxRange2;
            }

            protected override Delegate MakeGetter(IRow input, int iinfo, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;

                var typeSrc = _srcTypes[iinfo];

                if (!typeSrc.IsVector)
                {
                    if (_suppressed[iinfo])
                        return MakeOneTrivialGetter(input, iinfo);
                    return GetSrcGetter(typeSrc, input, _cols[iinfo]);
                }
                if (_suppressed[iinfo])
                    return MakeVecTrivialGetter(input, iinfo);
                return MakeVecGetter(input, iinfo);
            }

            private Delegate MakeOneTrivialGetter(IRow input, int iinfo)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                Host.Assert(!_srcTypes[iinfo].IsVector);
                Host.Assert(_suppressed[iinfo]);

                Func<ValueGetter<int>> del = MakeOneTrivialGetter<int>;
                var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(_srcTypes[iinfo].RawType);
                return (Delegate)methodInfo.Invoke(this, new object[0]);
            }

            private ValueGetter<TDst> MakeOneTrivialGetter<TDst>()
            {
                return OneTrivialGetter;
            }

            // Delegates onto instance methods are more efficient than delegates onto static methods.
            private void OneTrivialGetter<TDst>(ref TDst value)
            {
                value = default(TDst);
            }

            private Delegate MakeVecTrivialGetter(IRow input, int iinfo)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                Host.Assert(_srcTypes[iinfo].IsVector);
                Host.Assert(_suppressed[iinfo]);

                Func<ValueGetter<VBuffer<int>>> del = MakeVecTrivialGetter<int>;
                var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(_srcTypes[iinfo].ItemType.RawType);
                return (Delegate)methodInfo.Invoke(this, new object[0]);
            }

            private ValueGetter<VBuffer<TDst>> MakeVecTrivialGetter<TDst>()
            {
                return VecTrivialGetter;
            }

            // Delegates onto instance methods are more efficient than delegates onto static methods.
            private void VecTrivialGetter<TDst>(ref VBuffer<TDst> value)
            {
                value = new VBuffer<TDst>(1, 0, value.Values, value.Indices);
            }

            private Delegate MakeVecGetter(IRow input, int iinfo)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                Host.Assert(_srcTypes[iinfo].IsVector);
                Host.Assert(!_suppressed[iinfo]);

                Func<IRow, int, ValueGetter<VBuffer<int>>> del = MakeVecGetter<int>;
                var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(_srcTypes[iinfo].ItemType.RawType);
                return (Delegate)methodInfo.Invoke(this, new object[] { input, iinfo });
            }

            private ValueGetter<VBuffer<TDst>> MakeVecGetter<TDst>(IRow input, int iinfo)
            {
                var srcGetter = GetSrcGetter<VBuffer<TDst>>(input, iinfo);
                var typeDst = _dstTypes[iinfo];
                int srcValueCount = _srcTypes[iinfo].ValueCount;
                if (typeDst.IsKnownSizeVector && typeDst.ValueCount == srcValueCount)
                    return srcGetter;

                var buffer = default(VBuffer<TDst>);
                return
                    (ref VBuffer<TDst> value) =>
                    {
                        srcGetter(ref buffer);
                        _slotDropper[iinfo].DropSlots(ref buffer, ref value);
                    };
            }

            private ValueGetter<T> GetSrcGetter<T>(IRow input, int iinfo)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                int src = _cols[iinfo];
                Host.Assert(input.IsColumnActive(src));
                return input.GetGetter<T>(src);
            }

            private Delegate GetSrcGetter(ColumnType typeDst, IRow row, int iinfo)
            {
                Host.CheckValue(typeDst, nameof(typeDst));
                Host.CheckValue(row, nameof(row));

                Func<IRow, int, ValueGetter<int>> del = GetSrcGetter<int>;
                var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeDst.RawType);
                return (Delegate)methodInfo.Invoke(this, new object[] { row, iinfo });
            }

            public override Schema.Column[] GetOutputColumns()
            {
                var result = new Schema.Column[_parent.ColumnPairs.Length];
                for (int iinfo = 0; iinfo < _parent.ColumnPairs.Length; iinfo++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[iinfo].input, out int colIndex);
                    Host.Assert(colIndex >= 0);
                    var builder = new Schema.Metadata.Builder();
                    //builder.Add(InputSchema[colIndex].Metadata, x => x == MetadataUtils.Kinds.SlotNames);

                    // Add SlotNames metadata.
                    if (_srcTypes[iinfo].IsVector && _srcTypes[iinfo].IsKnownSizeVector)
                    {
                        var dstLength = _slotDropper[iinfo].DstLength;
                        var hasSlotNames = InputSchema.HasSlotNames(_cols[iinfo], _srcTypes[iinfo].VectorSize);
                        var type = new VectorType(_srcTypes[iinfo].ItemType.AsPrimitive, Math.Max(dstLength, 1));

                        if (hasSlotNames && dstLength > 0)
                        {
                            // Add slot name metadata.
                            builder.Add(new Schema.Column(MetadataUtils.Kinds.SlotNames, new VectorType(TextType.Instance, dstLength), null), getter)<VBuffer<ReadOnlyMemory<char>>>(MetadataUtils.Kinds.SlotNames,
                                new VectorType(TextType.Instance, dstLength), GetSlotNames);
                        }
                    }

                    // Add CategoricalSlotRanges metadata.
                    if (!_suppressed[iinfo])
                    {
                        if (MetadataUtils.TryGetCategoricalFeatureIndices(InputSchema, _cols[iinfo], out _categoricalRanges[iinfo]))
                        {
                            VBuffer<int> dst = default(VBuffer<int>);
                            GetCategoricalSlotRangesCore(iinfo, _slotDropper[iinfo].SlotsMin, _slotDropper[iinfo].SlotsMax, _categoricalRanges[iinfo], ref dst);
                            // REVIEW: cache dst as opposed to caculating it again.
                            if (dst.Length > 0)
                            {
                                Contracts.Assert(dst.Length % 2 == 0);

                                builder.AddGetter<VBuffer<int>>(MetadataUtils.Kinds.CategoricalSlotRanges,
                                    MetadataUtils.GetCategoricalType(dst.Length / 2), GetCategoricalSlotRanges);
                            }
                        }
                    }

                    // Add isNormalize metada.
                    ValueGetter<bool> getter = (ref bool dst) => { dst = true; };
                    builder.Add(new Schema.Column(MetadataUtils.Kinds.IsNormalized, BoolType.Instance, null), getter);

                    result[iinfo] = new Schema.Column(_parent.ColumnPairs[iinfo].output, _dstTypes[iinfo], builder.GetMetadata());
                }
                return result;
            }
        }
    }
}
