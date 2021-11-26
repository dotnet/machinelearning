// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    using Conditional = System.Diagnostics.ConditionalAttribute;

    public sealed partial class MissingValueReplacingTransformer
    {
        private static StatAggregator CreateStatAggregator(IChannel ch, DataViewType type, ReplacementKind? kind, bool bySlot, DataViewRowCursor cursor, int col)
        {
            ch.Assert(type.GetItemType() is NumberDataViewType);
            if (!(type is VectorDataViewType vectorType))
            {
                // The type is a scalar.
                if (kind == ReplacementKind.Mean)
                {
                    if (type.RawType == typeof(float))
                        return new R4.MeanAggregatorOne(ch, cursor, col);
                    else if (type.RawType == typeof(double))
                        return new R8.MeanAggregatorOne(ch, cursor, col);
                }
                else if (kind == ReplacementKind.Min || kind == ReplacementKind.Max)
                {
                    if (type.RawType == typeof(float))
                        return new R4.MinMaxAggregatorOne(ch, cursor, col, kind == ReplacementKind.Max);
                    else if (type.RawType == typeof(double))
                        return new R8.MinMaxAggregatorOne(ch, cursor, col, kind == ReplacementKind.Max);
                }
                else if (kind == ReplacementKind.Mode)
                {
                    if (type.RawType == typeof(float))
                        return new R4.ModeAggregatorOne(ch, cursor, col);
                    else if (type.RawType == typeof(double))
                        return new R8.ModeAggregatorOne(ch, cursor, col);
                }
            }
            else if (bySlot)
            {
                // Imputation by slot.
                // REVIEW: It may be more appropriate to have a warning here instead.
                if (vectorType.Size == 0)
                    throw ch.Except("Imputation by slot is not allowed for vectors of unknown size.");

                if (kind == ReplacementKind.Mean)
                {
                    if (vectorType.ItemType.RawType == typeof(float))
                        return new R4.MeanAggregatorBySlot(ch, vectorType, cursor, col);
                    else if (vectorType.ItemType.RawType == typeof(double))
                        return new R8.MeanAggregatorBySlot(ch, vectorType, cursor, col);
                }
                else if (kind == ReplacementKind.Min || kind == ReplacementKind.Max)
                {
                    if (vectorType.ItemType.RawType == typeof(float))
                        return new R4.MinMaxAggregatorBySlot(ch, vectorType, cursor, col, kind == ReplacementKind.Max);
                    else if (vectorType.ItemType.RawType == typeof(double))
                        return new R8.MinMaxAggregatorBySlot(ch, vectorType, cursor, col, kind == ReplacementKind.Max);
                }
                else if (kind == ReplacementKind.Mode)
                {
                    if (vectorType.ItemType.RawType == typeof(float))
                        return new R4.ModeAggregatorBySlot(ch, vectorType, cursor, col);
                    else if (vectorType.ItemType.RawType == typeof(double))
                        return new R8.ModeAggregatorBySlot(ch, vectorType, cursor, col);
                }
            }
            else
            {
                // Imputation across slots.
                if (kind == ReplacementKind.Mean)
                {
                    if (vectorType.ItemType.RawType == typeof(float))
                        return new R4.MeanAggregatorAcrossSlots(ch, cursor, col);
                    else if (vectorType.ItemType.RawType == typeof(double))
                        return new R8.MeanAggregatorAcrossSlots(ch, cursor, col);
                }
                else if (kind == ReplacementKind.Min || kind == ReplacementKind.Max)
                {
                    if (vectorType.ItemType.RawType == typeof(float))
                        return new R4.MinMaxAggregatorAcrossSlots(ch, cursor, col, kind == ReplacementKind.Max);
                    else if (vectorType.ItemType.RawType == typeof(double))
                        return new R8.MinMaxAggregatorAcrossSlots(ch, cursor, col, kind == ReplacementKind.Max);
                }
                else if (kind == ReplacementKind.Mode)
                {
                    if (vectorType.ItemType.RawType == typeof(float))
                        return new R4.ModeAggregatorAcrossSlots(ch, cursor, col);
                    else if (vectorType.ItemType.RawType == typeof(double))
                        return new R8.ModeAggregatorAcrossSlots(ch, cursor, col);
                }
            }
            ch.Assert(false);
            throw ch.Except("Internal error, unrecognized imputation method ReplacementKind '{0}' or unrecognized type '{1}' " +
                "assigned in NAReplaceTransform.", kind, type);
        }

        private static DataViewRowId Add(DataViewRowId left, ulong right)
        {
            ulong resHi = left.High;
            ulong resLo = left.Low + right;
            if (resLo < right)
                resHi++;
            return new DataViewRowId(resLo, resHi);
        }

        private static DataViewRowId Subtract(DataViewRowId left, ulong right)
        {
            ulong resHi = left.High;
            ulong resLo = left.Low - right;
            if (resLo > left.Low)
                resHi--;
            return new DataViewRowId(resLo, resHi);
        }

        private static bool Equals(DataViewRowId left, ulong right)
        {
            return left.High == 0 && left.Low == right;
        }

        private static bool GreaterThanOrEqual(DataViewRowId left, ulong right)
        {
            return left.High > 0 || left.Low >= right;
        }

        private static bool GreaterThan(DataViewRowId left, ulong right)
        {
            return left.High > 0 || left.Low > right;
        }

        private static double ToDouble(DataViewRowId value)
        {
            // REVIEW: The 64-bit JIT has a bug where rounding might be not quite
            // correct when converting a ulong to double with the high bit set. Should we
            // care and compensate? See the DoubleParser code for a work-around.
            return value.High * ((double)(1UL << 32) * (1UL << 32)) + value.Low;
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

            protected StatAggregator(IChannel ch, DataViewRowCursor cursor, int col)
                : base(ch)
            {
                Ch.AssertValue(cursor);
                Ch.Assert(0 <= col);
                _getter = cursor.GetGetter<TValue>(cursor.Schema[col]);
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
            private DataViewRowId _valueCount;

            /// <summary>
            /// Returns the number of values that have been processed so far.
            /// </summary>
            public DataViewRowId ValueCount { get { return _valueCount; } }

            protected StatAggregatorAcrossSlots(IChannel ch, DataViewRowCursor cursor, int col)
                : base(ch, cursor, col)
            {
            }

            protected sealed override void ProcessRow(in VBuffer<TItem> src)
            {
                var srcValues = src.GetValues();
                var srcCount = srcValues.Length;

                for (int slot = 0; slot < srcCount; slot++)
                    ProcessValue(in srcValues[slot]);

                _valueCount = Add(_valueCount, (ulong)src.Length);
            }

            protected abstract void ProcessValue(in TItem val);
        }

        private abstract class StatAggregatorBySlot<TItem, TStatItem> : StatAggregator<VBuffer<TItem>, TStatItem[]>
        {
            protected StatAggregatorBySlot(IChannel ch, VectorDataViewType type, DataViewRowCursor cursor, int col)
                : base(ch, cursor, col)
            {
                Ch.AssertValue(type);
                Ch.Assert(type.IsKnownSize);
                Stat = new TStatItem[type.Size];
            }

            protected sealed override void ProcessRow(in VBuffer<TItem> src)
            {
                var srcValues = src.GetValues();
                var srcCount = srcValues.Length;
                if (src.IsDense)
                {
                    // The src vector is dense.
                    for (int slot = 0; slot < srcCount; slot++)
                        ProcessValue(in srcValues[slot], slot);
                }
                else
                {
                    // The src vector is sparse.
                    var srcIndices = src.GetIndices();
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

            protected MinMaxAggregatorOne(IChannel ch, DataViewRowCursor cursor, int col, bool returnMax)
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

            protected MinMaxAggregatorAcrossSlots(IChannel ch, DataViewRowCursor cursor, int col, bool returnMax)
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

            protected MinMaxAggregatorBySlot(IChannel ch, VectorDataViewType type, DataViewRowCursor cursor, int col, bool returnMax)
                : base(ch, type, cursor, col)
            {
                Ch.AssertValue(type);

                ReturnMax = returnMax;
                if (ReturnMax)
                    ProcValueDelegate = ProcessValueMax;
                else
                    ProcValueDelegate = ProcessValueMin;

                _valuesProcessed = new long[type.Size];
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
        /// A mutable struct for keeping the appropriate statistics for mode calculations, whose scope is restricted
        /// and only exists as an instance in a field or an array, utilizing the mutation of the struct correctly.
        /// </summary>
        private class ModeStat<TType>
        {
            // Delegate used to check if the value is valid. We use a delegate so that this class can support modes of all types.
            public delegate bool IsValid(in TType val);

            private TType _modeSoFar;
            private int _maxCount;
            private readonly Dictionary<TType, int> _valueCounts;
            private readonly IsValid _validityCheck;
            public ModeStat(IsValid valid)
            {
                _modeSoFar = default;
                _maxCount = 0;
                _valueCounts = new Dictionary<TType, int>();
                _validityCheck = valid;
            }

            public void Update(TType val)
            {
                // We don't include non finite values in the mode, so if its not finite then just return.
                if (!_validityCheck(val))
                    return;

                // If the key is already in the dictionary, we want to get the current count and increment it.
                // If the key is not in the dictionary, we want to set count to 1 so the count is correct.
                if (_valueCounts.TryGetValue(val, out int count))
                    count++;
                else
                    count = 1;

                _valueCounts[val] = count;

                if (count > _maxCount)
                {
                    _modeSoFar = val;
                    _maxCount = count;
                }
            }

            public TType GetCurrentValue()
            {
                return _modeSoFar;
            }
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
            private double _cur;

            public void Update(double val)
            {
                Contracts.Assert(double.MinValue <= _cur && _cur <= double.MaxValue);

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

                Contracts.Assert(double.MinValue <= _cur && _cur <= double.MaxValue);
            }

            public double GetCurrentValue(IChannel ch, long count)
            {
                Contracts.Assert(double.MinValue <= _cur && _cur <= double.MaxValue);
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
                double stat = _cur * ((double)_cnz / (count - _cna));
                Contracts.Assert(double.MinValue <= stat && stat <= double.MaxValue);
                return stat;
            }

            public double GetCurrentValue(IChannel ch, DataViewRowId count)
            {
                Contracts.Assert(double.MinValue <= _cur && _cur <= double.MaxValue);
                Contracts.Assert(_cnz >= 0 && _cna >= 0);
                Contracts.Assert(count.High != 0 || count.Low >= (ulong)_cna);

                // If all values in the column are NAs, emit a warning and return 0.
                // Is this what we want to do or should an error be thrown?
                if (Equals(count, (ulong)_cna))
                {
                    ch.Warning("All values in this column are NAs, using default value for imputation");
                    return 0;
                }

                // Fold in the zeros.
                double stat = _cur * ((double)_cnz / ToDouble(Subtract(count, (ulong)_cna)));
                Contracts.Assert(double.MinValue <= stat && stat <= double.MaxValue);
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
        }

        private static class R4
        {
            // Utilizes MeanStatDouble for the mean aggregators, a struct that holds _stat as a double, despite the fact that its
            // value should always be within the range of a valid Single after processing each value as it is representative of the
            // mean of a set of Single values. Conversion to Single happens in GetStat.
            public sealed class MeanAggregatorOne : StatAggregator<float, MeanStatDouble>
            {
                public MeanAggregatorOne(IChannel ch, DataViewRowCursor cursor, int col)
                    : base(ch, cursor, col)
                {
                }

                protected override void ProcessRow(in float val)
                {
                    Stat.Update(val);
                }

                public override object GetStat()
                {
                    double val = Stat.GetCurrentValue(Ch, RowCount);
                    Ch.Assert(float.MinValue <= val && val <= float.MaxValue);
                    return (float)val;
                }
            }

            public sealed class MeanAggregatorAcrossSlots : StatAggregatorAcrossSlots<float, MeanStatDouble>
            {
                public MeanAggregatorAcrossSlots(IChannel ch, DataViewRowCursor cursor, int col)
                    : base(ch, cursor, col)
                {
                }

                protected override void ProcessValue(in float val)
                {
                    Stat.Update(val);
                }

                public override object GetStat()
                {
                    double val = Stat.GetCurrentValue(Ch, ValueCount);
                    Ch.Assert(float.MinValue <= val && val <= float.MaxValue);
                    return (float)val;
                }
            }

            public sealed class MeanAggregatorBySlot : StatAggregatorBySlot<float, MeanStatDouble>
            {
                public MeanAggregatorBySlot(IChannel ch, VectorDataViewType type, DataViewRowCursor cursor, int col)
                    : base(ch, type, cursor, col)
                {
                }

                protected override void ProcessValue(in float val, int slot)
                {
                    Ch.Assert(0 <= slot && slot < Stat.Length);
                    Stat[slot].Update(val);
                }

                public override object GetStat()
                {
                    float[] stat = new float[Stat.Length];
                    for (int slot = 0; slot < stat.Length; slot++)
                    {
                        double val = Stat[slot].GetCurrentValue(Ch, RowCount);
                        Ch.Assert(float.MinValue <= val && val <= float.MaxValue);
                        stat[slot] = (float)val;
                    }
                    return stat;
                }
            }

            public sealed class MinMaxAggregatorOne : MinMaxAggregatorOne<float, float>
            {
                public MinMaxAggregatorOne(IChannel ch, DataViewRowCursor cursor, int col, bool returnMax)
                    : base(ch, cursor, col, returnMax)
                {
                    Stat = ReturnMax ? float.NegativeInfinity : float.PositiveInfinity;
                }

                protected override void ProcessValueMin(in float val)
                {
                    if (val < Stat)
                        Stat = val;
                }

                protected override void ProcessValueMax(in float val)
                {
                    if (val > Stat)
                        Stat = val;
                }
            }

            public sealed class MinMaxAggregatorAcrossSlots : MinMaxAggregatorAcrossSlots<float, float>
            {
                public MinMaxAggregatorAcrossSlots(IChannel ch, DataViewRowCursor cursor, int col, bool returnMax)
                    : base(ch, cursor, col, returnMax)
                {
                    Stat = ReturnMax ? float.NegativeInfinity : float.PositiveInfinity;
                }

                protected override void ProcessValueMin(in float val)
                {
                    if (val < Stat)
                        Stat = val;
                }

                protected override void ProcessValueMax(in float val)
                {
                    if (val > Stat)
                        Stat = val;
                }

                public override object GetStat()
                {
                    // If sparsity occurred, fold in a zero.
                    if (GreaterThan(ValueCount, (ulong)ValuesProcessed))
                    {
                        float def = 0;
                        ProcValueDelegate(in def);
                    }
                    return (float)Stat;
                }
            }

            public sealed class MinMaxAggregatorBySlot : MinMaxAggregatorBySlot<float, float>
            {
                public MinMaxAggregatorBySlot(IChannel ch, VectorDataViewType type, DataViewRowCursor cursor, int col, bool returnMax)
                    : base(ch, type, cursor, col, returnMax)
                {
                    float bound = ReturnMax ? float.NegativeInfinity : float.PositiveInfinity;
                    for (int i = 0; i < Stat.Length; i++)
                        Stat[i] = bound;
                }

                protected override void ProcessValueMin(in float val, int slot)
                {
                    Ch.Assert(0 <= slot && slot < Stat.Length);
                    if (val < Stat[slot])
                        Stat[slot] = val;
                }

                protected override void ProcessValueMax(in float val, int slot)
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
                            float def = 0;
                            ProcValueDelegate(in def, slot);
                        }
                    }
                    return Stat;
                }
            }

            public sealed class ModeAggregatorOne : StatAggregator<float, ModeStat<float>>
            {
                public ModeAggregatorOne(IChannel ch, DataViewRowCursor cursor, int col)
                    : base(ch, cursor, col)
                {
                    Stat = new ModeStat<float>((in float val) => FloatUtils.IsFinite(val));
                }

                public override object GetStat()
                {
                    return Stat.GetCurrentValue();
                }

                protected override void ProcessRow(in float val)
                {
                    Stat.Update(val);
                }
            }

            public sealed class ModeAggregatorAcrossSlots : StatAggregatorAcrossSlots<float, ModeStat<float>>
            {
                public ModeAggregatorAcrossSlots(IChannel ch, DataViewRowCursor cursor, int col)
                    : base(ch, cursor, col)
                {
                    Stat = new ModeStat<float>((in float val) => FloatUtils.IsFinite(val));
                }

                public override object GetStat()
                {
                    return Stat.GetCurrentValue();
                }

                protected override void ProcessValue(in float val)
                {
                    Stat.Update(val);
                }
            }

            public sealed class ModeAggregatorBySlot : StatAggregatorBySlot<float, ModeStat<float>>
            {
                public ModeAggregatorBySlot(IChannel ch, VectorDataViewType type, DataViewRowCursor cursor, int col)
                    : base(ch, type, cursor, col)
                {
                    for (int i = 0; i < Stat.Length; i++)
                    {
                        Stat[i] = new ModeStat<float>((in float val) => FloatUtils.IsFinite(val));
                    }
                }

                public override object GetStat()
                {
                    float[] stat = new float[Stat.Length];
                    for (int slot = 0; slot < stat.Length; slot++)
                    {
                        stat[slot] = Stat[slot].GetCurrentValue();
                    }
                    return stat;
                }

                protected override void ProcessValue(in float val, int slot)
                {
                    Stat[slot].Update(val);
                }
            }
        }

        private static class R8
        {
            public sealed class MeanAggregatorOne : StatAggregator<double, MeanStatDouble>
            {
                public MeanAggregatorOne(IChannel ch, DataViewRowCursor cursor, int col)
                    : base(ch, cursor, col)
                {
                }

                protected override void ProcessRow(in double val)
                {
                    Stat.Update(val);
                }

                public override object GetStat()
                {
                    return Stat.GetCurrentValue(Ch, RowCount);
                }
            }

            public sealed class MeanAggregatorAcrossSlots : StatAggregatorAcrossSlots<double, MeanStatDouble>
            {
                public MeanAggregatorAcrossSlots(IChannel ch, DataViewRowCursor cursor, int col)
                    : base(ch, cursor, col)
                {
                }

                protected override void ProcessValue(in double val)
                {
                    Stat.Update(val);
                }

                public override object GetStat()
                {
                    return Stat.GetCurrentValue(Ch, ValueCount);
                }
            }

            public sealed class MeanAggregatorBySlot : StatAggregatorBySlot<double, MeanStatDouble>
            {
                public MeanAggregatorBySlot(IChannel ch, VectorDataViewType type, DataViewRowCursor cursor, int col)
                    : base(ch, type, cursor, col)
                {
                }

                protected override void ProcessValue(in double val, int slot)
                {
                    Ch.Assert(0 <= slot && slot < Stat.Length);
                    Stat[slot].Update(val);
                }

                public override object GetStat()
                {
                    double[] stat = new double[Stat.Length];
                    for (int slot = 0; slot < stat.Length; slot++)
                        stat[slot] = Stat[slot].GetCurrentValue(Ch, RowCount);
                    return stat;
                }
            }

            public sealed class MinMaxAggregatorOne : MinMaxAggregatorOne<double, double>
            {
                public MinMaxAggregatorOne(IChannel ch, DataViewRowCursor cursor, int col, bool returnMax)
                    : base(ch, cursor, col, returnMax)
                {
                    Stat = ReturnMax ? double.NegativeInfinity : double.PositiveInfinity;
                }

                protected override void ProcessValueMin(in double val)
                {
                    if (val < Stat)
                        Stat = val;
                }

                protected override void ProcessValueMax(in double val)
                {
                    if (val > Stat)
                        Stat = val;
                }
            }

            public sealed class MinMaxAggregatorAcrossSlots : MinMaxAggregatorAcrossSlots<double, double>
            {
                public MinMaxAggregatorAcrossSlots(IChannel ch, DataViewRowCursor cursor, int col, bool returnMax)
                    : base(ch, cursor, col, returnMax)
                {
                    Stat = ReturnMax ? double.NegativeInfinity : double.PositiveInfinity;
                }

                protected override void ProcessValueMin(in double val)
                {
                    if (val < Stat)
                        Stat = val;
                }

                protected override void ProcessValueMax(in double val)
                {
                    if (val > Stat)
                        Stat = val;
                }

                public override object GetStat()
                {
                    // If sparsity occurred, fold in a zero.
                    if (GreaterThan(ValueCount, (ulong)ValuesProcessed))
                    {
                        double def = 0;
                        ProcValueDelegate(in def);
                    }
                    return Stat;
                }
            }

            public sealed class MinMaxAggregatorBySlot : MinMaxAggregatorBySlot<double, double>
            {
                public MinMaxAggregatorBySlot(IChannel ch, VectorDataViewType type, DataViewRowCursor cursor, int col, bool returnMax)
                    : base(ch, type, cursor, col, returnMax)
                {
                    double bound = ReturnMax ? double.MinValue : double.MaxValue;
                    for (int i = 0; i < Stat.Length; i++)
                        Stat[i] = bound;
                }

                protected override void ProcessValueMin(in double val, int slot)
                {
                    Ch.Assert(0 <= slot && slot < Stat.Length);
                    if (FloatUtils.IsFinite(val))
                    {
                        if (val < Stat[slot])
                            Stat[slot] = val;
                    }
                }

                protected override void ProcessValueMax(in double val, int slot)
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
                            double def = 0;
                            ProcValueDelegate(in def, slot);
                        }
                    }
                    return Stat;
                }
            }

            public sealed class ModeAggregatorOne : StatAggregator<double, ModeStat<double>>
            {
                public ModeAggregatorOne(IChannel ch, DataViewRowCursor cursor, int col)
                    : base(ch, cursor, col)
                {
                    Stat = new ModeStat<double>((in double val) => FloatUtils.IsFinite(val));
                }

                public override object GetStat()
                {
                    return Stat.GetCurrentValue();
                }

                protected override void ProcessRow(in double val)
                {
                    Stat.Update(val);
                }
            }

            public sealed class ModeAggregatorAcrossSlots : StatAggregatorAcrossSlots<double, ModeStat<double>>
            {
                public ModeAggregatorAcrossSlots(IChannel ch, DataViewRowCursor cursor, int col)
                    : base(ch, cursor, col)
                {
                    Stat = new ModeStat<double>((in double val) => FloatUtils.IsFinite(val));
                }

                public override object GetStat()
                {
                    return Stat.GetCurrentValue();
                }

                protected override void ProcessValue(in double val)
                {
                    Stat.Update(val);
                }
            }

            public sealed class ModeAggregatorBySlot : StatAggregatorBySlot<double, ModeStat<double>>
            {
                public ModeAggregatorBySlot(IChannel ch, VectorDataViewType type, DataViewRowCursor cursor, int col)
                    : base(ch, type, cursor, col)
                {
                    for (int i = 0; i < Stat.Length; i++)
                    {
                        Stat[i] = new ModeStat<double>((in double val) => FloatUtils.IsFinite(val));
                    }
                }

                public override object GetStat()
                {
                    double[] stat = new double[Stat.Length];
                    for (int slot = 0; slot < stat.Length; slot++)
                    {
                        stat[slot] = Stat[slot].GetCurrentValue();
                    }
                    return stat;
                }

                protected override void ProcessValue(in double val, int slot)
                {
                    Stat[slot].Update(val);
                }
            }
        }
    }
}
