// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(SlotsDroppingTransformer.Summary, typeof(IDataTransform), typeof(SlotsDroppingTransformer), typeof(SlotsDroppingTransformer.Options), typeof(SignatureDataTransform),
    SlotsDroppingTransformer.FriendlyName, SlotsDroppingTransformer.LoaderSignature, "DropSlots")]

[assembly: LoadableClass(SlotsDroppingTransformer.Summary, typeof(IDataTransform), typeof(SlotsDroppingTransformer), null, typeof(SignatureLoadDataTransform),
    SlotsDroppingTransformer.FriendlyName, SlotsDroppingTransformer.LoaderSignature)]

[assembly: LoadableClass(SlotsDroppingTransformer.Summary, typeof(SlotsDroppingTransformer), null, typeof(SignatureLoadModel),
    SlotsDroppingTransformer.FriendlyName, SlotsDroppingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(SlotsDroppingTransformer), null, typeof(SignatureLoadRowMapper),
   SlotsDroppingTransformer.FriendlyName, SlotsDroppingTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Transform to drop slots from columns. If the column is scalar, the only slot that can be dropped is slot 0.
    /// If all the slots are to be dropped, a vector valued column will be changed to a vector of length 1 (a scalar column will retain its type) and
    /// the value will be the default value.
    /// </summary>
    [BestFriend]
    internal sealed class SlotsDroppingTransformer : OneToOneTransformerBase
    {
        internal sealed class Options
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Columns to drop the slots for",
                Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;
        }

        [BestFriend]
        internal sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.Multiple, HelpText = "Source slot index range(s) of the column to drop")]
            public Range[] Slots;

            internal static Column Parse(string str)
            {
                Contracts.CheckNonWhiteSpace(str, nameof(str));

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            private protected override bool TryParse(string str)
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

            internal bool TryUnparse(StringBuilder sb)
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

        internal sealed class Range
        {
            [Argument(ArgumentType.Required, HelpText = "First index in the range")]
            public int Min;

            // If null, it means int.MaxValue - 1. There are two reasons for this:
            // 1. max is an index, so it has to be strictly less than int.MaxValue.
            // 2. to prevent overflows when adding 1 to it.
            [Argument(ArgumentType.AtMostOnce, HelpText = "Last index in the range")]
            public int? Max;

            internal static Range Parse(string str)
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

            internal bool TryUnparse(StringBuilder sb)
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
        /// Describes how the transformer handles one input-output column pair.
        /// </summary>
        [BestFriend]
        internal sealed class ColumnOptions
        {
            public readonly string Name;
            public readonly string InputColumnName;
            public readonly (int min, int? max)[] Slots;

            /// <summary>
            /// Describes how the transformer handles one input-output column pair.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="inputColumnName">Name of the column to transform.
            /// If set to <see langword="null"/>, the value of the <paramref name="name"/> will be used as source.</param>
            /// <param name="slots">Ranges of indices in the input column to be dropped. Setting max in <paramref name="slots"/> to null sets max to int.MaxValue.</param>
            public ColumnOptions(string name, string inputColumnName = null, params (int min, int? max)[] slots)
            {
                Name = name;
                Contracts.CheckValue(Name, nameof(Name));
                InputColumnName = inputColumnName ?? name;
                Contracts.CheckValue(InputColumnName, nameof(InputColumnName));

                // By default drop everything.
                Slots = (slots.Length > 0) ? slots : new (int min, int? max)[1];
                foreach (var (min, max) in Slots)
                    Contracts.Assert(min >= 0 && (max == null || min <= max));
            }

            internal ColumnOptions(Column column)
            {
                Name = column.Name;
                Contracts.CheckValue(Name, nameof(Name));
                InputColumnName = column.Source ?? column.Name;
                Contracts.CheckValue(InputColumnName, nameof(InputColumnName));
                Slots = column.Slots.Select(range => (range.Min, range.Max)).ToArray();
                foreach (var (min, max) in Slots)
                    Contracts.Assert(min >= 0 && (max == null || min <= max));
            }
        }

        private const string RegistrationName = "DropSlots";
        internal const string Summary = "Removes the selected slots from the column.";
        internal const string FriendlyName = "Drop Slots Transform";
        internal const string LoaderSignature = "DropSlotsTransform";

        // Store the lower (SlotsMin) and upper (SlotsMax) bounds of ranges of slots to be dropped for each column pair.
        // SlotsMin[i] and SlotsMax[i] are the bounds of the ranges for the i-th column pair.
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
                loaderAssemblyName: typeof(SlotsDroppingTransformer).Assembly.FullName);
        }

        /// <summary>
        /// Initializes a new <see cref="SlotsDroppingTransformer"/> object.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="min">Specifies the lower bound of the range of slots to be dropped. The lower bound is inclusive. </param>
        /// <param name="max">Specifies the upper bound of the range of slots to be dropped. The upper bound is exclusive.</param>
        internal SlotsDroppingTransformer(IHostEnvironment env, string outputColumnName, string inputColumnName = null, int min = default, int? max = null)
            : this(env, new ColumnOptions(outputColumnName, inputColumnName, (min, max)))
        {
        }

        /// <summary>
        /// Initializes a new <see cref="SlotsDroppingTransformer"/> object.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="columns">Specifies the ranges of slots to drop for each column pair.</param>
        internal SlotsDroppingTransformer(IHostEnvironment env, params ColumnOptions[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            Host.AssertNonEmpty(ColumnPairs);
            GetSlotsMinMax(columns, out SlotsMin, out SlotsMax);
            Host.CheckUserArg(AreRangesValid(SlotsMin, SlotsMax), nameof(columns), "The range min and max must be non-negative and min must be less than or equal to max.");
        }

        private SlotsDroppingTransformer(IHostEnvironment env, ModelLoadContext ctx)
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
            SlotsMin = new int[size][];
            SlotsMax = new int[size][];
            for (int i = 0; i < size; i++)
            {
                SlotsMin[i] = ctx.Reader.ReadIntArray();
                Host.CheckDecode(Utils.Size(SlotsMin[i]) > 0);
                SlotsMax[i] = ctx.Reader.ReadIntArray(SlotsMin[i].Length);
            }
            Host.Assert(AreRangesValid(SlotsMin, SlotsMax));
        }

        // Factory method for SignatureLoadModel.
        private static SlotsDroppingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            ctx.CheckAtModel(GetVersionInfo());
            return new SlotsDroppingTransformer(env, ctx);
        }

        // Factory method for SignatureDataTransform.
        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            var columns = options.Columns.Select(column => new ColumnOptions(column)).ToArray();
            return new SlotsDroppingTransformer(env, columns).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override void SaveModel(ModelSaveContext ctx)
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
                ctx.Writer.WriteIntsNoCount(SlotsMax[i]);
            }
        }

        private void GetSlotsMinMax(Column col, out int[] slotsMin, out int[] slotsMax)
        {
            slotsMin = new int[col.Slots.Length];
            slotsMax = new int[col.Slots.Length];
            for (int j = 0; j < col.Slots.Length; j++)
            {
                var range = col.Slots[j];
                Host.CheckUserArg(range.IsValid(), nameof(col.Slots), "The range min and max must be non-negative and min must be less than or equal to max.");
                slotsMin[j] = range.Min;
                // There are two reasons for setting the max to int.MaxValue - 1:
                // 1. max is an index, so it has to be strictly less than int.MaxValue.
                // 2. to prevent overflows when adding 1 to it.
                slotsMax[j] = range.Max ?? int.MaxValue - 1;
            }
            Array.Sort(slotsMin, slotsMax);
            var iDst = 0;
            for (int j = 1; j < col.Slots.Length; j++)
            {
                if (slotsMin[j] <= slotsMax[iDst] + 1)
                    slotsMax[iDst] = Math.Max(slotsMax[iDst], slotsMax[j]);
                else
                {
                    iDst++;
                    slotsMin[iDst] = slotsMin[j];
                    slotsMax[iDst] = slotsMax[j];
                }
            }
        }

        private static void GetSlotsMinMax(ColumnOptions[] columns, out int[][] slotsMin, out int[][] slotsMax)
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
                    slotsMin[i][j] = range.min;
                    // There are two reasons for setting the max to int.MaxValue - 1:
                    // 1. max is an index, so it has to be strictly less than int.MaxValue.
                    // 2. to prevent overflows when adding 1 to it.
                    slotsMax[i][j] = range.max ?? int.MaxValue - 1;
                }
                Array.Sort(slotsMin[i], slotsMax[i]);
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

        private static (string outputColumnName, string inputColumnName)[] GetColumnPairs(ColumnOptions[] columns)
            => columns.Select(c => (c.Name, c.InputColumnName ?? c.Name)).ToArray();

        private static bool AreRangesValid(int[][] slotsMin, int[][] slotsMax)
        {
            if (slotsMin.Length != slotsMax.Length)
                return false;
            for (int iinfo = 0; iinfo < slotsMin.Length; iinfo++)
            {
                var prevmax = -2;
                for (int i = 0; i < slotsMin[iinfo].Length; i++)
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

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema)
            => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly SlotsDroppingTransformer _parent;
            private readonly int[] _cols;
            private readonly DataViewType[] _srcTypes;
            private readonly DataViewType[] _dstTypes;
            private readonly SlotDropper[] _slotDropper;
            // Track if all the slots of the column are to be dropped.
            private readonly bool[] _suppressed;
            private readonly int[][] _categoricalRanges;

            public Mapper(SlotsDroppingTransformer parent, DataViewSchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _cols = new int[_parent.ColumnPairs.Length];
                _srcTypes = new DataViewType[_parent.ColumnPairs.Length];
                _dstTypes = new DataViewType[_parent.ColumnPairs.Length];
                _slotDropper = new SlotDropper[_parent.ColumnPairs.Length];
                _suppressed = new bool[_parent.ColumnPairs.Length];
                _categoricalRanges = new int[_parent.ColumnPairs.Length][];

                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    if (!InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out _cols[i]))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.ColumnPairs[i].inputColumnName);
                    _srcTypes[i] = inputSchema[_cols[i]].Type;
                    VectorType srcVectorType = _srcTypes[i] as VectorType;

                    DataViewType itemType = srcVectorType?.ItemType ?? _srcTypes[i];
                    if (!IsValidColumnType(itemType))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.ColumnPairs[i].inputColumnName);

                    int valueCount = srcVectorType?.Size ?? 1;
                    _slotDropper[i] = new SlotDropper(valueCount, _parent.SlotsMin[i], _parent.SlotsMax[i]);
                    ComputeType(inputSchema, i, _slotDropper[i], out _suppressed[i], out _dstTypes[i], out _categoricalRanges[i]);
                }
            }

            /// <summary>
            /// Both scalars and vectors are acceptable types, but the item type must have a default value which means it must be
            /// a string, a key, a float or a double.
            /// </summary>
            private static bool IsValidColumnType(DataViewType type)
                => (type is KeyType keytype && 0 < keytype.Count && keytype.Count < Utils.ArrayMaxSize)
                || type == NumberDataViewType.Single || type == NumberDataViewType.Double || type is TextDataViewType;

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
            private void ComputeType(DataViewSchema input, int iinfo, SlotDropper slotDropper,
                out bool suppressed, out DataViewType type, out int[] categoricalRanges)
            {
                var slotsMin = _parent.SlotsMin[iinfo];
                var slotsMax = _parent.SlotsMax[iinfo];
                Host.AssertValue(slotDropper);
                Host.AssertValue(input);
                Host.AssertNonEmpty(slotsMin);
                Host.AssertNonEmpty(slotsMax);
                Host.Assert(slotsMin.Length == slotsMax.Length);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);

                categoricalRanges = null;
                var typeSrc = _srcTypes[iinfo];
                if (!(typeSrc is VectorType vectorType))
                {
                    type = typeSrc;
                    suppressed = slotsMin.Length > 0 && slotsMin[0] == 0;
                }
                else if (!vectorType.IsKnownSize)
                {
                    type = typeSrc;
                    suppressed = false;
                }
                else
                {
                    Host.Assert(vectorType.IsKnownSize);
                    var dstLength = slotDropper.DstLength;
                    var hasSlotNames = input[_cols[iinfo]].HasSlotNames(vectorType.Size);
                    type = new VectorType(vectorType.ItemType, Math.Max(dstLength, 1));
                    suppressed = dstLength == 0;
                }
            }

            private void GetSlotNames(int iinfo, ref VBuffer<ReadOnlyMemory<char>> dst)
            {
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);

                var names = default(VBuffer<ReadOnlyMemory<char>>);
                InputSchema[_cols[iinfo]].GetSlotNames(ref names);
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

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;

                var typeSrc = _srcTypes[iinfo];

                if (!(typeSrc is VectorType))
                {
                    if (_suppressed[iinfo])
                        return MakeOneTrivialGetter(input, iinfo);
                    return GetSrcGetter(typeSrc, input, _cols[iinfo]);
                }
                if (_suppressed[iinfo])
                    return MakeVecTrivialGetter(input, iinfo);
                return MakeVecGetter(input, iinfo);
            }

            private Delegate MakeOneTrivialGetter(DataViewRow input, int iinfo)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                Host.Assert(!(_srcTypes[iinfo] is VectorType));
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

            private Delegate MakeVecTrivialGetter(DataViewRow input, int iinfo)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                VectorType vectorType = (VectorType)_srcTypes[iinfo];
                Host.Assert(_suppressed[iinfo]);

                Func<ValueGetter<VBuffer<int>>> del = MakeVecTrivialGetter<int>;
                var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(vectorType.ItemType.RawType);
                return (Delegate)methodInfo.Invoke(this, new object[0]);
            }

            private ValueGetter<VBuffer<TDst>> MakeVecTrivialGetter<TDst>()
            {
                return VecTrivialGetter;
            }

            // Delegates onto instance methods are more efficient than delegates onto static methods.
            private void VecTrivialGetter<TDst>(ref VBuffer<TDst> value)
            {
                VBufferUtils.Resize(ref value, 1, 0);
            }

            private Delegate MakeVecGetter(DataViewRow input, int iinfo)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                VectorType vectorType = (VectorType)_srcTypes[iinfo];
                Host.Assert(!_suppressed[iinfo]);

                Func<DataViewRow, int, ValueGetter<VBuffer<int>>> del = MakeVecGetter<int>;
                var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(vectorType.ItemType.RawType);
                return (Delegate)methodInfo.Invoke(this, new object[] { input, iinfo });
            }

            private ValueGetter<VBuffer<TDst>> MakeVecGetter<TDst>(DataViewRow input, int iinfo)
            {
                var srcGetter = GetSrcGetter<VBuffer<TDst>>(input, iinfo);
                var typeDst = _dstTypes[iinfo];
                int srcValueCount = _srcTypes[iinfo].GetValueCount();
                if (typeDst is VectorType dstVector && dstVector.IsKnownSize && dstVector.Size == srcValueCount)
                    return srcGetter;

                var buffer = default(VBuffer<TDst>);
                return
                    (ref VBuffer<TDst> value) =>
                    {
                        srcGetter(ref buffer);
                        _slotDropper[iinfo].DropSlots(ref buffer, ref value);
                    };
            }

            private ValueGetter<T> GetSrcGetter<T>(DataViewRow input, int iinfo)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                var src = input.Schema[_cols[iinfo]];
                Host.Assert(input.IsColumnActive(src));
                return input.GetGetter<T>(src);
            }

            private Delegate GetSrcGetter(DataViewType typeDst, DataViewRow row, int iinfo)
            {
                Host.CheckValue(typeDst, nameof(typeDst));
                Host.CheckValue(row, nameof(row));

                Func<DataViewRow, int, ValueGetter<int>> del = GetSrcGetter<int>;
                var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeDst.RawType);
                return (Delegate)methodInfo.Invoke(this, new object[] { row, iinfo });
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    // Avoid closure when adding metadata.
                    int iinfo = i;

                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[iinfo].inputColumnName, out int colIndex);
                    Host.Assert(colIndex >= 0);
                    var builder = new DataViewSchema.Annotations.Builder();

                    // Add SlotNames metadata.
                    if (_srcTypes[iinfo] is VectorType vectorType && vectorType.IsKnownSize)
                    {
                        var dstLength = _slotDropper[iinfo].DstLength;
                        var hasSlotNames = InputSchema[_cols[iinfo]].HasSlotNames(vectorType.Size);
                        var type = new VectorType(vectorType.ItemType, Math.Max(dstLength, 1));

                        if (hasSlotNames && dstLength > 0)
                        {
                            // Add slot name metadata.
                            ValueGetter<VBuffer<ReadOnlyMemory<char>>> slotNamesGetter = (ref VBuffer<ReadOnlyMemory<char>> dst) => GetSlotNames(iinfo, ref dst);
                            builder.Add(AnnotationUtils.Kinds.SlotNames, new VectorType(TextDataViewType.Instance, dstLength), slotNamesGetter);
                        }
                    }

                    // Add CategoricalSlotRanges metadata.
                    if (!_suppressed[iinfo])
                    {
                        if (AnnotationUtils.TryGetCategoricalFeatureIndices(InputSchema, _cols[iinfo], out _categoricalRanges[iinfo]))
                        {
                            VBuffer<int> dst = default(VBuffer<int>);
                            GetCategoricalSlotRangesCore(iinfo, _slotDropper[iinfo].SlotsMin, _slotDropper[iinfo].SlotsMax, _categoricalRanges[iinfo], ref dst);
                            // REVIEW: cache dst as opposed to caculating it again.
                            if (dst.Length > 0)
                            {
                                Contracts.Assert(dst.Length % 2 == 0);
                                // Add slot name metadata.
                                ValueGetter<VBuffer<int>> categoricalSlotRangesGetter = (ref VBuffer<int> dest) => GetCategoricalSlotRanges(iinfo, ref dest);
                                builder.Add(AnnotationUtils.Kinds.CategoricalSlotRanges, AnnotationUtils.GetCategoricalType(dst.Length / 2), categoricalSlotRangesGetter);
                            }
                        }
                    }

                    // Add isNormalize and KeyValues metadata.
                    builder.Add(InputSchema[_cols[iinfo]].Annotations, x => x == AnnotationUtils.Kinds.KeyValues || x == AnnotationUtils.Kinds.IsNormalized);

                    result[iinfo] = new DataViewSchema.DetachedColumn(_parent.ColumnPairs[iinfo].outputColumnName, _dstTypes[iinfo], builder.ToAnnotations());
                }
                return result;
            }
        }
    }
}
