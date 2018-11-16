// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.Internal.Utilities;
using System;

namespace Microsoft.ML.Transforms
{
    using Conditional = System.Diagnostics.ConditionalAttribute;

    public sealed partial class MissingValueReplacingTransformer
    {
        private static StatAggregator CreateStatAggregator(IChannel ch, ColumnType type, ReplacementKind? kind, bool bySlot, IRowCursor cursor, int col)
        {
            ch.Assert(type.ItemType.IsNumber);
            if (!type.IsVector)
            {
                // The type is a scalar.
                if (kind == ReplacementKind.Mean)
                {
                    switch (type.RawKind)
                    {
                    case DataKind.R4:
                        return new R4.MeanAggregatorOne(ch, cursor, col);
                    case DataKind.R8:
                        return new R8.MeanAggregatorOne(ch, cursor, col);
                    default:
                        break;
                    }
                }
                if (kind == ReplacementKind.Min || kind == ReplacementKind.Max)
                {
                    switch (type.RawKind)
                    {
                    case DataKind.R4:
                        return new R4.MinMaxAggregatorOne(ch, cursor, col, kind == ReplacementKind.Max);
                    case DataKind.R8:
                        return new R8.MinMaxAggregatorOne(ch, cursor, col, kind == ReplacementKind.Max);
                    default:
                        break;
                    }
                }
            }
            else if (bySlot)
            {
                // Imputation by slot.
                // REVIEW: It may be more appropriate to have a warning here instead.
                if (type.ValueCount == 0)
                    throw ch.Except("Imputation by slot is not allowed for vectors of unknown size.");

                if (kind == ReplacementKind.Mean)
                {
                    switch (type.ItemType.RawKind)
                    {
                    case DataKind.R4:
                        return new R4.MeanAggregatorBySlot(ch, type, cursor, col);
                    case DataKind.R8:
                        return new R8.MeanAggregatorBySlot(ch, type, cursor, col);
                    default:
                        break;
                    }
                }
                else if (kind == ReplacementKind.Min || kind == ReplacementKind.Max)
                {
                    switch (type.ItemType.RawKind)
                    {
                    case DataKind.R4:
                        return new R4.MinMaxAggregatorBySlot(ch, type, cursor, col, kind == ReplacementKind.Max);
                    case DataKind.R8:
                        return new R8.MinMaxAggregatorBySlot(ch, type, cursor, col, kind == ReplacementKind.Max);
                    default:
                        break;
                    }
                }
            }
            else
            {
                // Imputation across slots.
                if (kind == ReplacementKind.Mean)
                {
                    switch (type.ItemType.RawKind)
                    {
                    case DataKind.R4:
                        return new R4.MeanAggregatorAcrossSlots(ch, cursor, col);
                    case DataKind.R8:
                        return new R8.MeanAggregatorAcrossSlots(ch, cursor, col);
                    default:
                        break;
                    }
                }
                else if (kind == ReplacementKind.Min || kind == ReplacementKind.Max)
                {
                    switch (type.ItemType.RawKind)
                    {
                    case DataKind.R4:
                        return new R4.MinMaxAggregatorAcrossSlots(ch, cursor, col, kind == ReplacementKind.Max);
                    case DataKind.R8:
                        return new R8.MinMaxAggregatorAcrossSlots(ch, cursor, col, kind == ReplacementKind.Max);
                    default:
                        break;
                    }
                }
            }
            ch.Assert(false);
            throw ch.Except("Internal error, unrecognized imputation method ReplacementKind '{0}' or unrecognized type '{1}' " +
                "assigned in NAReplaceTransform.", kind, type);
        }

        /// <summary>
        /// The base class for stat aggregators for imputing mean, min, and max for the NAReplaceTransform.
        /// </summary>
        private abstract class StatAggregator
        {
            protected readonly IChannel Ch;

            protected StatAggregator(IChannel ch)
            {
                Contracts.AssertValue(ch);
                Ch = ch;
            }

            // Is called every time that the stat aggregator processes another value.
            public abstract void ProcessRow();
            // Returns the final computed stat after all values have been processed.
            public abstract object GetStat();
        }

        /// <summary>
        /// The base class for stat aggregators with knowledge of types.
        /// </summary>
        /// <typeparam name="TValue">The type for the column being aggregated.</typeparam>
        /// <typeparam name="TStat">The type of the stat being computed by the stat aggregator.</typeparam>
        private abstract class StatAggregator<TValue, TStat> : StatAggregator
        {
            private readonly ValueGetter<TValue> _getter;
            private TValue _val;

            // This is the number of times that ProcessValue has been called.
            private long _rowCount;

            // The currently computed statistics.
            protected TStat Stat;

            /// <summary>
            /// Returns the number of times that ProcessRow has been called.
            /// </summary>
            public long RowCount { get { return _rowCount; } }

            protected StatAggregator(IChannel ch, IRowCursor cursor, int col)
                : base(ch)
            {
                Ch.AssertValue(cursor);
                Ch.Assert(0 <= col);
                _getter = cursor.GetGetter<TValue>(col);
            }

            public sealed override void ProcessRow()
            {
                _rowCount++;
                _getter(ref _val);
                ProcessRow(in _val);
            }

            protected abstract void ProcessRow(in TValue val);
        }

        private abstract class StatAggregatorAcrossSlots<TItem, TStat> : StatAggregator<VBuffer<TItem>, TStat>
        {
            // The number of values that have been processed.
            private UInt128 _valueCount;

            /// <summary>
            /// Returns the number of values that have been processed so far.
            /// </summary>
            public UInt128 ValueCount { get { return _valueCount; } }

            protected StatAggregatorAcrossSlots(IChannel ch, IRowCursor cursor, int col)
                : base(ch, cursor, col)
            {
            }

            protected sealed override void ProcessRow(in VBuffer<TItem> src)
            {
                var srcCount = src.Count;
                var srcValues = src.Values;
                Ch.Assert(Utils.Size(srcValues) >= srcCount);

                for (int slot = 0; slot < srcCount; slot++)
                    ProcessValue(in srcValues[slot]);

                _valueCount = _valueCount + (ulong)src.Length;
            }

            protected abstract void ProcessValue(in TItem val);
        }

        private abstract class StatAggregatorBySlot<TItem, TStatItem> : StatAggregator<VBuffer<TItem>, TStatItem[]>
        {
            protected StatAggregatorBySlot(IChannel ch, ColumnType type, IRowCursor cursor, int col)
                : base(ch, cursor, col)
            {
                Ch.AssertValue(type);
                Ch.Assert(type.IsKnownSizeVector);
                Stat = new TStatItem[type.VectorSize];
            }

            protected sealed override void ProcessRow(in VBuffer<TItem> src)
            {
                var srcCount = src.Count;
                var srcValues = src.Values;
                Ch.Assert(Utils.Size(srcValues) >= srcCount);
                if (src.IsDense)
                {
                    // The src vector is dense.
                    for (int slot = 0; slot < srcCount; slot++)
                        ProcessValue(in srcValues[slot], slot);
                }
                else
                {
                    // The src vector is sparse.
                    var srcIndices = src.Indices;
                    Ch.Assert(Utils.Size(srcIndices) >= srcCount);
                    for (int islot = 0; islot < srcCount; islot++)
                        ProcessValue(in srcValues[islot], srcIndices[islot]);
                }
            }

            protected abstract void ProcessValue(in TItem val, int slot);
        }

        private abstract class MinMaxAggregatorOne<TValue, TStat> : StatAggregator<TValue, TStat>
        {
            protected readonly bool ReturnMax;
            private delegate void ProcessValueDelegate(in TValue val);
            private readonly ProcessValueDelegate _processValueDelegate;

            protected MinMaxAggregatorOne(IChannel ch, IRowCursor cursor, int col, bool returnMax)
                : base(ch, cursor, col)
            {
                ReturnMax = returnMax;
                if (ReturnMax)
                    _processValueDelegate = ProcessValueMax;
                else
                    _processValueDelegate = ProcessValueMin;
            }

            protected override void ProcessRow(in TValue val)
            {
                _processValueDelegate(in val);
            }

            public override object GetStat()
            {
                return Stat;
            }

            protected abstract void ProcessValueMin(in TValue val);
            protected abstract void ProcessValueMax(in TValue val);
        }

        private abstract class MinMaxAggregatorAcrossSlots<TItem, TStat> : StatAggregatorAcrossSlots<TItem, TStat>
        {
            protected readonly bool ReturnMax;
            protected delegate void ProcessValueDelegate(in TItem val);
            protected readonly ProcessValueDelegate ProcValueDelegate;
            // The count of the number of times ProcessValue has been called (used for tracking sparsity).
            private long _valuesProcessed;

            /// <summary>
            /// Returns the number of times that ProcessValue has been called.
            /// </summary>
            public long ValuesProcessed { get { return _valuesProcessed; } }

            protected MinMaxAggregatorAcrossSlots(IChannel ch, IRowCursor cursor, int col, bool returnMax)
                : base(ch, cursor, col)
            {
                ReturnMax = returnMax;
                if (ReturnMax)
                    ProcValueDelegate = ProcessValueMax;
                else
                    ProcValueDelegate = ProcessValueMin;
            }

            protected override void ProcessValue(in TItem val)
            {
                _valuesProcessed = _valuesProcessed + 1;
                ProcValueDelegate(in val);
            }

            protected abstract void ProcessValueMin(in TItem val);
            protected abstract void ProcessValueMax(in TItem val);
        }

        private abstract class MinMaxAggregatorBySlot<TItem, TStatItem> : StatAggregatorBySlot<TItem, TStatItem>
        {
            protected readonly bool ReturnMax;
            protected delegate void ProcessValueDelegate(in TItem val, int slot);
            protected readonly ProcessValueDelegate ProcValueDelegate;
            // The count of the number of times ProcessValue has been called on a specific slot (used for tracking sparsity).
            private readonly long[] _valuesProcessed;

            protected MinMaxAggregatorBySlot(IChannel ch, ColumnType type, IRowCursor cursor, int col, bool returnMax)
                : base(ch, type, cursor, col)
            {
                Ch.AssertValue(type);

                ReturnMax = returnMax;
                if (ReturnMax)
                    ProcValueDelegate = ProcessValueMax;
                else
                    ProcValueDelegate = ProcessValueMin;

                _valuesProcessed = new long[type.VectorSize];
            }

            protected override void ProcessValue(in TItem val, int slot)
            {
                Ch.Assert(0 <= slot && slot < Stat.Length);
                _valuesProcessed[slot]++;
                ProcValueDelegate(in val, slot);
            }

            protected long GetValuesProcessed(int slot)
            {
                return _valuesProcessed[slot];
            }

            protected abstract void ProcessValueMin(in TItem val, int slot);
            protected abstract void ProcessValueMax(in TItem val, int slot);
        }

        /// <summary>
        /// This is a mutable struct (so is evil). However, its scope is restricted
        /// and the only instances are in a field or an array, so the mutation does
        /// the right thing.
        /// </summary>
        private struct MeanStatDouble
        {
            // The number of non-finite numbers that have been processed so far.
            private long _cna;
            // The number of non-zero (finite) values processed.
            private long _cnz;
            // The current mean estimate for the _cnz values we've processed.
            private Double _cur;

            public void Update(Double val)
            {
                Contracts.Assert(Double.MinValue <= _cur && _cur <= Double.MaxValue);

                if (val == 0)
                    return;

                if (!FloatUtils.IsFinite(val))
                {
                    _cna++;
                    return;
                }

                _cnz++;

                // The sign compatibility test protects against overflow.
                if ((_cur > 0) ^ (val > 0))
                    _cur = _cur - _cur / _cnz + val / _cnz;
                else
                    _cur += (val - _cur) / _cnz;

                Contracts.Assert(Double.MinValue <= _cur && _cur <= Double.MaxValue);
            }

            public Double GetCurrentValue(IChannel ch, long count)
            {
                Contracts.Assert(Double.MinValue <= _cur && _cur <= Double.MaxValue);
                Contracts.Assert(_cnz >= 0 && _cna >= 0);
                Contracts.Assert(count >= _cna);
                Contracts.Assert(count - _cna >= _cnz);

                // If all values in the column are NAs, emit a warning and return 0.
                if (count == _cna)
                {
                    ch.Warning("All values in this column are NAs, using default value for imputation");
                    return 0;
                }

                // Fold in the zeros.
                Double stat = _cur * ((Double)_cnz / (count - _cna));
                Contracts.Assert(Double.MinValue <= stat && stat <= Double.MaxValue);
                return stat;
            }

            public Double GetCurrentValue(IChannel ch, UInt128 count)
            {
                Contracts.Assert(Double.MinValue <= _cur && _cur <= Double.MaxValue);
                Contracts.Assert(_cnz >= 0 && _cna >= 0);
                Contracts.Assert(count.Hi != 0 || count.Lo >= (ulong)_cna);

                // If all values in the column are NAs, emit a warning and return 0.
                // Is this what we want to do or should an error be thrown?
                if (count == (ulong)_cna)
                {
                    ch.Warning("All values in this column are NAs, using default value for imputation");
                    return 0;
                }

                // Fold in the zeros.
                Double stat = _cur * ((Double)_cnz / (Double)(count - (ulong)_cna));
                Contracts.Assert(Double.MinValue <= stat && stat <= Double.MaxValue);
                return stat;
            }
        }

        /// <summary>
        /// A mutable struct for keeping the appropriate statistics for mean calculations for IX types, TS, and DT,
        /// whose scope is restricted and only exists as an instance in a field or an array, utilizing the mutation
        /// of the struct correctly.
        /// </summary>
        private struct MeanStatInt
        {
            // The number of NAs seen.
            private long _cna;
            // The 128-bit 2's complement sum.
            private ulong _sumLo;
            private ulong _sumHi;

            [Conditional("DEBUG")]
            private void AssertValid(long valMax)
            {
                // Assert valMax is one less than a power of 2.
                Contracts.Assert(valMax > 0 && ((valMax + 1) & valMax) == 0);
                Contracts.Assert(_cna >= 0);
            }

            public void Update(long? val, long valMax)
            {
                AssertValid(valMax);
                Contracts.Assert(!val.HasValue || -valMax <= val && val <= valMax);

                if (!val.HasValue)
                    _cna++;
                else if (val >= 0)
                    IntUtils.Add(ref _sumHi, ref _sumLo, (ulong)val);
                else
                    IntUtils.Sub(ref _sumHi, ref _sumLo, (ulong)(-val));

                AssertValid(valMax);
            }

            public long GetCurrentValue(IChannel ch, long count, long valMax)
            {
                // Note: This method should not modify any fields.
                AssertValid(valMax);
                Contracts.Assert(count >= _cna);

                // If the sum is zero, return zero.
                if ((_sumHi | _sumLo) == 0)
                {
                    // If all values in a given column are NAs issue a warning.
                    if (count == _cna)
                        ch.Warning("All values in this column are NAs, using default value for imputation");
                    return 0;
                }

                Contracts.Assert(count > _cna);
                count -= _cna;
                Contracts.Assert(count > 0);

                ulong sumHi = _sumHi;
                ulong sumLo = _sumLo;

                bool neg = (long)sumHi < 0;

                if (neg)
                {
                    // Negative value. Work with the absolute value.
                    sumHi = ~sumHi;
                    sumLo = ~sumLo;
                    IntUtils.Add(ref sumHi, ref sumLo, 1);
                }

                // If this assert triggers, the caller did something wrong. Update should have been
                // called at most count times and each value should have been within the range of
                // a ulong, so the absolute value of the sum can't possibly be so large that sumHi
                // reaches or exceeds count. This assert implies that the Div part of the DivRound
                // call won't throw.
                Contracts.Assert(sumHi < (ulong)count);

                ulong res = IntUtils.DivRound(sumLo, sumHi, (ulong)count);
                Contracts.Assert(0 <= res && res <= (ulong)valMax);
                return neg ? -(long)res : (long)res;
            }

            public long GetCurrentValue(IChannel ch, UInt128 count, long valMax)
            {
                AssertValid(valMax);
                Contracts.Assert(count >= (ulong)_cna);

                // If the sum is zero, return zero.
                if ((_sumHi | _sumLo) == 0)
                {
                    // If all values in a given column are NAs issue a warning.
                    if (count == (ulong)_cna)
                        ch.Warning("All values in this column are NAs, using default value for imputation");
                    return 0;
                }

                Contracts.Assert(count > (ulong)_cna);
                count -= (ulong)_cna;
                Contracts.Assert(count > 0);

                ulong sumHi = _sumHi;
                ulong sumLo = _sumLo;

                bool neg = (long)sumHi < 0;

                if (neg)
                {
                    // Negative value. Work with the absolute value.
                    sumHi = ~sumHi;
                    sumLo = ~sumLo;
                    IntUtils.Add(ref sumHi, ref sumLo, 1);
                }

                // If this assert triggers, the caller did something wrong. Update should have been
                // called at most count times and each value should have been within the range of
                // a ulong, so the absolute value of the sum can't possibly be so large that sumHi
                // reaches or exceeds count. This assert implies that the Div part of the DivRound
                // call won't throw.
                Contracts.Assert(count > sumHi);

                ulong res = IntUtils.DivRound(sumLo, sumHi, count.Lo, count.Hi);
                Contracts.Assert(0 <= res && res <= (ulong)valMax);
                return neg ? -(long)res : (long)res;
            }
        }

        private static class R4
        {
            // Utilizes MeanStatDouble for the mean aggregators, a struct that holds _stat as a double, despite the fact that its
            // value should always be within the range of a valid Single after processing each value as it is representative of the
            // mean of a set of Single values. Conversion to Single happens in GetStat.
            public sealed class MeanAggregatorOne : StatAggregator<Single, MeanStatDouble>
            {
                public MeanAggregatorOne(IChannel ch, IRowCursor cursor, int col)
                    : base(ch, cursor, col)
                {
                }

                protected override void ProcessRow(in Single val)
                {
                    Stat.Update(val);
                }

                public override object GetStat()
                {
                    Double val = Stat.GetCurrentValue(Ch, RowCount);
                    Ch.Assert(Single.MinValue <= val && val <= Single.MaxValue);
                    return (Single)val;
                }
            }

            public sealed class MeanAggregatorAcrossSlots : StatAggregatorAcrossSlots<Single, MeanStatDouble>
            {
                public MeanAggregatorAcrossSlots(IChannel ch, IRowCursor cursor, int col)
                    : base(ch, cursor, col)
                {
                }

                protected override void ProcessValue(in Single val)
                {
                    Stat.Update(val);
                }

                public override object GetStat()
                {
                    Double val = Stat.GetCurrentValue(Ch, ValueCount);
                    Ch.Assert(Single.MinValue <= val && val <= Single.MaxValue);
                    return (Single)val;
                }
            }

            public sealed class MeanAggregatorBySlot : StatAggregatorBySlot<Single, MeanStatDouble>
            {
                public MeanAggregatorBySlot(IChannel ch, ColumnType type, IRowCursor cursor, int col)
                    : base(ch, type, cursor, col)
                {
                }

                protected override void ProcessValue(in Single val, int slot)
                {
                    Ch.Assert(0 <= slot && slot < Stat.Length);
                    Stat[slot].Update(val);
                }

                public override object GetStat()
                {
                    Single[] stat = new Single[Stat.Length];
                    for (int slot = 0; slot < stat.Length; slot++)
                    {
                        Double val = Stat[slot].GetCurrentValue(Ch, RowCount);
                        Ch.Assert(Single.MinValue <= val && val <= Single.MaxValue);
                        stat[slot] = (Single)val;
                    }
                    return stat;
                }
            }

            public sealed class MinMaxAggregatorOne : MinMaxAggregatorOne<Single, Single>
            {
                public MinMaxAggregatorOne(IChannel ch, IRowCursor cursor, int col, bool returnMax)
                    : base(ch, cursor, col, returnMax)
                {
                    Stat = ReturnMax ? Single.NegativeInfinity : Single.PositiveInfinity;
                }

                protected override void ProcessValueMin(in Single val)
                {
                    if (val < Stat)
                        Stat = val;
                }

                protected override void ProcessValueMax(in Single val)
                {
                    if (val > Stat)
                        Stat = val;
                }
            }

            public sealed class MinMaxAggregatorAcrossSlots : MinMaxAggregatorAcrossSlots<Single, Single>
            {
                public MinMaxAggregatorAcrossSlots(IChannel ch, IRowCursor cursor, int col, bool returnMax)
                    : base(ch, cursor, col, returnMax)
                {
                    Stat = ReturnMax ? Single.NegativeInfinity : Single.PositiveInfinity;
                }

                protected override void ProcessValueMin(in Single val)
                {
                    if (val < Stat)
                        Stat = val;
                }

                protected override void ProcessValueMax(in Single val)
                {
                    if (val > Stat)
                        Stat = val;
                }

                public override object GetStat()
                {
                    // If sparsity occurred, fold in a zero.
                    if (ValueCount > (ulong)ValuesProcessed)
                    {
                        Single def = 0;
                        ProcValueDelegate(in def);
                    }
                    return (Single)Stat;
                }
            }

            public sealed class MinMaxAggregatorBySlot : MinMaxAggregatorBySlot<Single, Single>
            {
                public MinMaxAggregatorBySlot(IChannel ch, ColumnType type, IRowCursor cursor, int col, bool returnMax)
                    : base(ch, type, cursor, col, returnMax)
                {
                    Single bound = ReturnMax ? Single.NegativeInfinity : Single.PositiveInfinity;
                    for (int i = 0; i < Stat.Length; i++)
                        Stat[i] = bound;
                }

                protected override void ProcessValueMin(in Single val, int slot)
                {
                    Ch.Assert(0 <= slot && slot < Stat.Length);
                    if (val < Stat[slot])
                        Stat[slot] = val;
                }

                protected override void ProcessValueMax(in Single val, int slot)
                {
                    Ch.Assert(0 <= slot && slot < Stat.Length);
                    if (val > Stat[slot])
                        Stat[slot] = val;
                }

                public override object GetStat()
                {
                    // Account for defaults resulting from sparsity.
                    for (int slot = 0; slot < Stat.Length; slot++)
                    {
                        if (GetValuesProcessed(slot) < RowCount)
                        {
                            Single def = 0;
                            ProcValueDelegate(in def, slot);
                        }
                    }
                    return Stat;
                }
            }
        }

        private static class R8
        {
            public sealed class MeanAggregatorOne : StatAggregator<Double, MeanStatDouble>
            {
                public MeanAggregatorOne(IChannel ch, IRowCursor cursor, int col)
                    : base(ch, cursor, col)
                {
                }

                protected override void ProcessRow(in Double val)
                {
                    Stat.Update(val);
                }

                public override object GetStat()
                {
                    return Stat.GetCurrentValue(Ch, RowCount);
                }
            }

            public sealed class MeanAggregatorAcrossSlots : StatAggregatorAcrossSlots<Double, MeanStatDouble>
            {
                public MeanAggregatorAcrossSlots(IChannel ch, IRowCursor cursor, int col)
                    : base(ch, cursor, col)
                {
                }

                protected override void ProcessValue(in Double val)
                {
                    Stat.Update(val);
                }

                public override object GetStat()
                {
                    return Stat.GetCurrentValue(Ch, ValueCount);
                }
            }

            public sealed class MeanAggregatorBySlot : StatAggregatorBySlot<Double, MeanStatDouble>
            {
                public MeanAggregatorBySlot(IChannel ch, ColumnType type, IRowCursor cursor, int col)
                    : base(ch, type, cursor, col)
                {
                }

                protected override void ProcessValue(in Double val, int slot)
                {
                    Ch.Assert(0 <= slot && slot < Stat.Length);
                    Stat[slot].Update(val);
                }

                public override object GetStat()
                {
                    Double[] stat = new Double[Stat.Length];
                    for (int slot = 0; slot < stat.Length; slot++)
                        stat[slot] = Stat[slot].GetCurrentValue(Ch, RowCount);
                    return stat;
                }
            }

            public sealed class MinMaxAggregatorOne : MinMaxAggregatorOne<Double, Double>
            {
                public MinMaxAggregatorOne(IChannel ch, IRowCursor cursor, int col, bool returnMax)
                    : base(ch, cursor, col, returnMax)
                {
                    Stat = ReturnMax ? Double.NegativeInfinity : Double.PositiveInfinity;
                }

                protected override void ProcessValueMin(in Double val)
                {
                    if (val < Stat)
                        Stat = val;
                }

                protected override void ProcessValueMax(in Double val)
                {
                    if (val > Stat)
                        Stat = val;
                }
            }

            public sealed class MinMaxAggregatorAcrossSlots : MinMaxAggregatorAcrossSlots<Double, Double>
            {
                public MinMaxAggregatorAcrossSlots(IChannel ch, IRowCursor cursor, int col, bool returnMax)
                    : base(ch, cursor, col, returnMax)
                {
                    Stat = ReturnMax ? Double.NegativeInfinity : Double.PositiveInfinity;
                }

                protected override void ProcessValueMin(in Double val)
                {
                    if (val < Stat)
                        Stat = val;
                }

                protected override void ProcessValueMax(in Double val)
                {
                    if (val > Stat)
                        Stat = val;
                }

                public override object GetStat()
                {
                    // If sparsity occurred, fold in a zero.
                    if (ValueCount > (ulong)ValuesProcessed)
                    {
                        Double def = 0;
                        ProcValueDelegate(in def);
                    }
                    return Stat;
                }
            }

            public sealed class MinMaxAggregatorBySlot : MinMaxAggregatorBySlot<Double, Double>
            {
                public MinMaxAggregatorBySlot(IChannel ch, ColumnType type, IRowCursor cursor, int col, bool returnMax)
                    : base(ch, type, cursor, col, returnMax)
                {
                    Double bound = ReturnMax ? Double.MinValue : Double.MaxValue;
                    for (int i = 0; i < Stat.Length; i++)
                        Stat[i] = bound;
                }

                protected override void ProcessValueMin(in Double val, int slot)
                {
                    Ch.Assert(0 <= slot && slot < Stat.Length);
                    if (FloatUtils.IsFinite(val))
                    {
                        if (val < Stat[slot])
                            Stat[slot] = val;
                    }
                }

                protected override void ProcessValueMax(in Double val, int slot)
                {
                    Ch.Assert(0 <= slot && slot < Stat.Length);
                    if (FloatUtils.IsFinite(val))
                    {
                        if (val > Stat[slot])
                            Stat[slot] = val;
                    }
                }

                public override object GetStat()
                {
                    // Account for defaults resulting from sparsity.
                    for (int slot = 0; slot < Stat.Length; slot++)
                    {
                        if (GetValuesProcessed(slot) < RowCount)
                        {
                            Double def = 0;
                            ProcValueDelegate(in def, slot);
                        }
                    }
                    return Stat;
                }
            }
        }
    }
}