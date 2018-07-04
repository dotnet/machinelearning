// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Internal.Internallearn;

[assembly: LoadableClass(DropSlotsTransform.Summary, typeof(DropSlotsTransform), typeof(DropSlotsTransform.Arguments), typeof(SignatureDataTransform),
    "Drop Slots Transform", "DropSlots", "DropSlotsTransform")]

[assembly: LoadableClass(DropSlotsTransform.Summary, typeof(DropSlotsTransform), null, typeof(SignatureLoadDataTransform),
    "Drop Slots Transform", DropSlotsTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Transform to drop slots from columns. If the column is scalar, the only slot that can be dropped is slot 0.
    /// If all the slots are to be dropped, a vector valued column will be changed to a vector of length 1 (a scalar column will retain its type) and
    /// the value will be the default value.
    /// </summary>
    public sealed class DropSlotsTransform : OneToOneTransformBase
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

        internal const string Summary = "Removes the selected slots from the column.";

        public const string LoaderSignature = "DropSlotsTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "DROPSLOT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private const string RegistrationName = "DropSlots";

        /// <summary>
        /// Extra information for each column (in addition to ColumnInfo).
        /// </summary>
        private sealed class ColInfoEx
        {
            public readonly SlotDropper SlotDropper;
            // Track if all the slots of the column are to be dropped.
            public readonly bool Suppressed;
            public readonly ColumnType TypeDst;
            public readonly int[] CategoricalRanges;

            public ColInfoEx(SlotDropper slotDropper, bool suppressed, ColumnType typeDst, int[] categoricalRanges)
            {
                Contracts.AssertValue(slotDropper);
                SlotDropper = slotDropper;
                Suppressed = suppressed;
                TypeDst = typeDst;
                CategoricalRanges = categoricalRanges;
            }
        }

        private readonly ColInfoEx[] _exes;

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="source">Name of the input column.  If this is null '<paramref name="name"/>' will be used.</param>
        public DropSlotsTransform(IHostEnvironment env, IDataView input, string name, string source = null)
            : this(env, new Arguments() { Column = new[] { new Column() { Source = source ?? name, Name = name } } }, input)
        {
        }

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public DropSlotsTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(Contracts.CheckRef(env, nameof(env)), RegistrationName, env.CheckRef(args, nameof(args)).Column, input, null)
        {
            Host.CheckNonEmpty(args.Column, nameof(args.Column));

            var size = Infos.Length;
            _exes = new ColInfoEx[size];
            for (int i = 0; i < size; i++)
            {
                var col = args.Column[i];
                int[] slotsMin;
                int[] slotsMax;
                GetSlotsMinMax(col, out slotsMin, out slotsMax);
                SlotDropper slotDropper = new SlotDropper(Infos[i].TypeSrc.ValueCount, slotsMin, slotsMax);
                bool suppressed;
                ColumnType typeDst;
                int[] categoricalRanges;
                ComputeType(Source.Schema, slotsMin, slotsMax, i, slotDropper, out suppressed, out typeDst, out categoricalRanges);
                _exes[i] = new ColInfoEx(slotDropper, suppressed, typeDst, categoricalRanges);
            }
            Metadata.Seal();
        }

        public static DropSlotsTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new DropSlotsTransform(h, ctx, input));
        }

        private DropSlotsTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, null)
        {
            Host.AssertValue(ctx);
            // *** Binary format ***
            // <base>
            // for each added column
            //   int[]: slotsMin
            //   int[]: slotsMax (no count)
            Host.AssertNonEmpty(Infos);
            var size = Infos.Length;
            _exes = new ColInfoEx[size];
            for (int i = 0; i < size; i++)
            {
                int[] slotsMin = ctx.Reader.ReadIntArray();
                Host.CheckDecode(Utils.Size(slotsMin) > 0);
                int[] slotsMax = ctx.Reader.ReadIntArray(slotsMin.Length);
                bool suppressed;
                ColumnType typeDst;
                SlotDropper slotDropper = new SlotDropper(Infos[i].TypeSrc.ValueCount, slotsMin, slotsMax);
                int[] categoricalRanges;
                ComputeType(input.Schema, slotsMin, slotsMax, i, slotDropper, out suppressed, out typeDst, out categoricalRanges);
                _exes[i] = new ColInfoEx(slotDropper, suppressed, typeDst, categoricalRanges);
                Host.CheckDecode(AreRangesValid(i));
            }
            Metadata.Seal();
        }

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
            SaveBase(ctx);
            for (int i = 0; i < Infos.Length; i++)
            {
                var infoEx = _exes[i];
                Host.Assert(infoEx.SlotDropper.SlotsMin.Length == infoEx.SlotDropper.SlotsMax.Length);
                Host.Assert(AreRangesValid(i));
                ctx.Writer.WriteIntArray(infoEx.SlotDropper.SlotsMin);
                ctx.Writer.WriteIntsNoCount(infoEx.SlotDropper.SlotsMax, infoEx.SlotDropper.SlotsMax.Length);
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
            iDst++;
            Array.Resize(ref slotsMin, iDst);
            Array.Resize(ref slotsMax, iDst);
        }

        /// <summary>
        /// Computes the types (column and slotnames), the length reduction, categorical feature indices
        /// and whether the column is suppressed.
        /// The slotsMin and slotsMax arrays should be sorted and the intervals should not overlap.
        /// </summary>
        /// <param name="input">The input schema</param>
        /// <param name="slotsMin">The beginning indices of the ranges of slots to be dropped</param>
        /// <param name="slotsMax">The end indices of the ranges of slots to be dropped</param>
        /// <param name="iinfo">The column index in Infos</param>
        /// <param name="slotDropper">The slots to be dropped.</param>
        /// <param name="suppressed">Whether the column is suppressed (all slots dropped)</param>
        /// <param name="type">The column type</param>
        /// <param name="categoricalRanges">Categorical feature indices.</param>
        private void ComputeType(ISchema input, int[] slotsMin, int[] slotsMax, int iinfo,
            SlotDropper slotDropper, out bool suppressed, out ColumnType type, out int[] categoricalRanges)
        {
            Contracts.AssertValue(slotDropper);
            Contracts.AssertValue(input);
            Contracts.AssertNonEmpty(slotsMin);
            Contracts.AssertNonEmpty(slotsMax);
            Contracts.Assert(slotsMin.Length == slotsMax.Length);
            Contracts.Assert(0 <= iinfo && iinfo < Infos.Length);

            categoricalRanges = null;
            // Register for metadata. Propagate the IsNormalized metadata.
            using (var bldr = Metadata.BuildMetadata(iinfo, input, Infos[iinfo].Source,
                MetadataUtils.Kinds.IsNormalized, MetadataUtils.Kinds.KeyValues))
            {
                var typeSrc = Infos[iinfo].TypeSrc;
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
                    var hasSlotNames = input.HasSlotNames(Infos[iinfo].Source, Infos[iinfo].TypeSrc.VectorSize);
                    type = new VectorType(typeSrc.ItemType.AsPrimitive, Math.Max(dstLength, 1));
                    suppressed = dstLength == 0;

                    if (hasSlotNames && dstLength > 0)
                    {
                        // Add slot name metadata.
                        bldr.AddGetter<VBuffer<DvText>>(MetadataUtils.Kinds.SlotNames,
                            new VectorType(TextType.Instance, dstLength), GetSlotNames);
                    }
                }

                if (!suppressed)
                {
                    if (MetadataUtils.TryGetCategoricalFeatureIndices(Source.Schema, Infos[iinfo].Source, out categoricalRanges))
                    {
                        VBuffer<DvInt4> dst = default(VBuffer<DvInt4>);
                        GetCategoricalSlotRangesCore(iinfo, slotDropper.SlotsMin, slotDropper.SlotsMax, categoricalRanges, ref dst);
                        // REVIEW: cache dst as opposed to caculating it again.
                        if (dst.Length > 0)
                        {
                            Contracts.Assert(dst.Length % 2 == 0);

                            bldr.AddGetter<VBuffer<DvInt4>>(MetadataUtils.Kinds.CategoricalSlotRanges,
                                MetadataUtils.GetCategoricalType(dst.Length / 2), GetCategoricalSlotRanges);
                        }
                    }
                }
            }
        }

        private bool AreRangesValid(int iinfo)
        {
            var infoEx = _exes[iinfo];
            var prevmax = -2;
            for (int i = 0; i < infoEx.SlotDropper.SlotsMin.Length; i++)
            {
                if (!(0 <= infoEx.SlotDropper.SlotsMin[i] && infoEx.SlotDropper.SlotsMin[i] < int.MaxValue))
                    return false;
                if (!(0 <= infoEx.SlotDropper.SlotsMax[i] && infoEx.SlotDropper.SlotsMax[i] < int.MaxValue))
                    return false;
                if (!(infoEx.SlotDropper.SlotsMin[i] <= infoEx.SlotDropper.SlotsMax[i]))
                    return false;
                if (!(infoEx.SlotDropper.SlotsMin[i] - 1 > prevmax))
                    return false;
                prevmax = infoEx.SlotDropper.SlotsMax[i];
            }
            return true;
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo & iinfo < Infos.Length);
            return _exes[iinfo].TypeDst;
        }

        private void GetSlotNames(int iinfo, ref VBuffer<DvText> dst)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);

            var names = default(VBuffer<DvText>);
            Source.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, Infos[iinfo].Source, ref names);
            var infoEx = _exes[iinfo];
            infoEx.SlotDropper.DropSlots(ref names, ref dst);
        }

        private void GetCategoricalSlotRanges(int iinfo, ref VBuffer<DvInt4> dst)
        {
            if (_exes[iinfo].CategoricalRanges != null)
            {
                GetCategoricalSlotRangesCore(iinfo, _exes[iinfo].SlotDropper.SlotsMin,
                    _exes[iinfo].SlotDropper.SlotsMax, _exes[iinfo].CategoricalRanges, ref dst);
            }
        }

        private void GetCategoricalSlotRangesCore(int iinfo, int[] slotsMin, int[] slotsMax, int[] catRanges, ref VBuffer<DvInt4> dst)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
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
            DvInt4 min = -1;
            DvInt4 max = -1;
            List<DvInt4> newCategoricalSlotRanges = new List<DvInt4>();

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
                        Contracts.Assert(min.RawValue == -1 && max.RawValue == -1);
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
                        Contracts.Assert(min.RawValue >= 0 && min.RawValue <= max.RawValue);
                        newCategoricalSlotRanges.Add(min);
                        newCategoricalSlotRanges.Add(max);
                        min = max = -1;
                        combine = false;
                    }

                    Contracts.Assert(min.RawValue == -1 && max.RawValue == -1);

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
                        Contracts.Assert(min.RawValue == -1 && max.RawValue == -1);

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

            Contracts.Assert(min.RawValue == -1 && max.RawValue == -1);

            for (int i = rangesIndex; i < ranges.Length; i++)
                newCategoricalSlotRanges.Add(ranges[i] - droppedSlotsCount);

            Contracts.Assert(newCategoricalSlotRanges.Count % 2 == 0);
            Contracts.Assert(newCategoricalSlotRanges.TrueForAll(x => x.RawValue >= 0));
            Contracts.Assert(0 <= droppedSlotsCount && droppedSlotsCount <= slotsMax[slotsMax.Length - 1] + 1);

            if (newCategoricalSlotRanges.Count > 0)
                dst = new VBuffer<DvInt4>(newCategoricalSlotRanges.Count, newCategoricalSlotRanges.ToArray());
        }

        private void CombineRanges(
            DvInt4 minRange1, DvInt4 maxRange1, DvInt4 minRange2, DvInt4 maxRange2,
            out DvInt4 newRangeMin, out DvInt4 newRangeMax)
        {
            Contracts.Assert(minRange2.RawValue >= 0 && maxRange2.RawValue >= 0);
            Contracts.Assert(minRange2.RawValue <= maxRange2.RawValue);
            Contracts.Assert(minRange1.RawValue >= 0 && maxRange1.RawValue >= 0);
            Contracts.Assert(minRange1.RawValue <= maxRange1.RawValue);
            Contracts.Assert(maxRange1.RawValue + 1 == minRange2.RawValue);

            newRangeMin = minRange1;
            newRangeMax = maxRange2;
        }

        protected override void ActivateSourceColumns(int iinfo, bool[] active)
        {
            if (!_exes[iinfo].Suppressed)
                active[Infos[iinfo].Source] = true;
        }

        protected override bool WantParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate);

            // Parallel helps when some, but not all, slots are dropped.
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
            {
                int col = ColumnIndex(iinfo);
                if (!predicate(col))
                    continue;

                var ex = _exes[iinfo];
                var info = Infos[iinfo];
                Contracts.Assert(ex.TypeDst.IsVector == info.TypeSrc.IsVector);
                Contracts.Assert(ex.TypeDst.IsKnownSizeVector == info.TypeSrc.IsKnownSizeVector);
                Contracts.Assert(!ex.TypeDst.IsKnownSizeVector || ex.TypeDst.ValueCount <= info.TypeSrc.ValueCount);
                if (!ex.Suppressed &&
                    (!ex.TypeDst.IsKnownSizeVector || ex.TypeDst.ValueCount != info.TypeSrc.ValueCount))
                {
                    return true;
                }
            }

            // No real work to do.
            return false;
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            var typeSrc = Infos[iinfo].TypeSrc;

            if (!typeSrc.IsVector)
            {
                if (_exes[iinfo].Suppressed)
                    return MakeOneTrivialGetter(input, iinfo);
                return GetSrcGetter(typeSrc, input, Infos[iinfo].Source);
            }
            if (_exes[iinfo].Suppressed)
                return MakeVecTrivialGetter(input, iinfo);
            return MakeVecGetter(input, iinfo);
        }

        private Delegate MakeOneTrivialGetter(IRow input, int iinfo)
        {
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            Host.Assert(!Infos[iinfo].TypeSrc.IsVector);
            Host.Assert(_exes[iinfo].Suppressed);

            Func<ValueGetter<int>> del = MakeOneTrivialGetter<int>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(Infos[iinfo].TypeSrc.RawType);
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
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            Host.Assert(Infos[iinfo].TypeSrc.IsVector);
            Host.Assert(_exes[iinfo].Suppressed);

            Func<ValueGetter<VBuffer<int>>> del = MakeVecTrivialGetter<int>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(Infos[iinfo].TypeSrc.ItemType.RawType);
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
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            Host.Assert(Infos[iinfo].TypeSrc.IsVector);
            Host.Assert(!_exes[iinfo].Suppressed);

            Func<IRow, int, ValueGetter<VBuffer<int>>> del = MakeVecGetter<int>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(Infos[iinfo].TypeSrc.ItemType.RawType);
            return (Delegate)methodInfo.Invoke(this, new object[] { input, iinfo });
        }

        private ValueGetter<VBuffer<TDst>> MakeVecGetter<TDst>(IRow input, int iinfo)
        {
            var srcGetter = GetSrcGetter<VBuffer<TDst>>(input, iinfo);
            var typeDst = _exes[iinfo].TypeDst;
            int srcValueCount = Infos[iinfo].TypeSrc.ValueCount;
            if (typeDst.IsKnownSizeVector && typeDst.ValueCount == srcValueCount)
                return srcGetter;

            var buffer = default(VBuffer<TDst>);
            var infoEx = _exes[iinfo];
            return
                (ref VBuffer<TDst> value) =>
                {
                    srcGetter(ref buffer);
                    infoEx.SlotDropper.DropSlots(ref buffer, ref value);
                };
        }
    }
}
