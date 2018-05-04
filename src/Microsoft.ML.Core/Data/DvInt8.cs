// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.CompilerServices;

namespace Microsoft.ML.Runtime.Data
{
    using BL = DvBool;
    using I1 = DvInt1;
    using I2 = DvInt2;
    using I4 = DvInt4;
    using IX = DvInt8;
    using R4 = Single;
    using R8 = Double;
    using RawIX = Int64;

    public struct DvInt8 : IEquatable<IX>, IComparable<DvInt8>
    {
        public const RawIX RawNA = RawIX.MinValue;

        // Ideally this would be readonly. However, note that this struct has no
        // ctor, but instead only has conversion operators. The implicit conversion
        // operator from RawIX to DvIX performs better than an equivalent ctor,
        // and the conversion operator must assign the _value field.
        private RawIX _value;

        /// <summary>
        /// Property to return the raw value.
        /// </summary>
        public RawIX RawValue
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return _value; }
        }

        /// <summary>
        /// Static method to return the raw value. This is more convenient than the
        /// property in code-generation scenarios.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static RawIX GetRawBits(IX a)
        {
            return a._value;
        }

        public static IX NA
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return RawNA; }
        }

        public bool IsNA
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return _value == RawNA; }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator IX(RawIX value)
        {
            IX res;
            res._value = value;
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator IX(RawIX? value)
        {
            IX res;
            res._value = value ?? RawNA;
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator RawIX(IX value)
        {
            if (value._value == RawNA)
                throw Contracts.ExceptValue(nameof(value), "NA cast to long");
            return value._value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator RawIX?(IX value)
        {
            if (value._value == RawNA)
                return null;
            return value._value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator IX(BL a)
        {
            if (a.IsNA)
                return RawNA;
            return (RawIX)a.RawValue;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator IX(I1 a)
        {
            if (a.IsNA)
                return RawNA;
            return (RawIX)a.RawValue;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator IX(I2 a)
        {
            if (a.IsNA)
                return RawNA;
            return (RawIX)a.RawValue;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator IX(I4 a)
        {
            if (a.IsNA)
                return RawNA;
            return (RawIX)a.RawValue;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator IX(R4 a)
        {
            return (IX)(R8)a;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator R4(IX a)
        {
            if (a._value == RawNA)
                return R4.NaN;
            return (R4)a._value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator IX(R8 a)
        {
            const R8 lim = -(R8)RawIX.MinValue;
            if (-lim < a && a < lim)
            {
                RawIX n = (RawIX)a;
#if DEBUG
                Contracts.Assert(!a.IsNA());
                Contracts.Assert(n != RawNA);
                // Note that an R8 cannot represent long.MaxValue exactly so y + 1.0 below might be the same as y.
                R8 x = a;
                R8 y = n;
                if (a < 0)
                {
                    x = -x;
                    y = -y;
                }
                Contracts.Assert(y <= x);
                Contracts.Assert(x < y + 1.0 | y + 1.0 == y & x == y);
#endif
                return n;
            }

            return RawNA;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator R8(IX a)
        {
            if (a._value == RawNA)
                return R8.NaN;
            return (R8)a._value;
        }

        public override int GetHashCode()
        {
            return _value.GetHashCode();
        }

        public override bool Equals(object obj)
        {
            if (obj is IX)
                return _value == ((IX)obj)._value;
            return false;
        }

        public bool Equals(IX other)
        {
            return _value == other._value;
        }

        public int CompareTo(IX other)
        {
            if (_value == other._value)
                return 0;
            return _value < other._value ? -1 : 1;
        }

        public override string ToString()
        {
            if (_value == RawNA)
                return "NA";
            return _value.ToString();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL operator ==(IX a, IX b)
        {
            var av = a._value;
            var bv = b._value;
            if (av != RawNA && bv != RawNA)
                return av == bv ? BL.True : BL.False;
            return BL.NA;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL operator !=(IX a, IX b)
        {
            var av = a._value;
            var bv = b._value;
            if (av != RawNA && bv != RawNA)
                return av != bv ? BL.True : BL.False;
            return BL.NA;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL operator <(IX a, IX b)
        {
            var av = a._value;
            var bv = b._value;
            if (av != RawNA && bv != RawNA)
                return av < bv ? BL.True : BL.False;
            return BL.NA;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL operator <=(IX a, IX b)
        {
            var av = a._value;
            var bv = b._value;
            if (av != RawNA && bv != RawNA)
                return av <= bv ? BL.True : BL.False;
            return BL.NA;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL operator >=(IX a, IX b)
        {
            var av = a._value;
            var bv = b._value;
            if (av != RawNA && bv != RawNA)
                return av >= bv ? BL.True : BL.False;
            return BL.NA;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL operator >(IX a, IX b)
        {
            var av = a._value;
            var bv = b._value;
            if (av != RawNA && bv != RawNA)
                return av > bv ? BL.True : BL.False;
            return BL.NA;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static IX operator -(IX a)
        {
            return -a._value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static IX operator +(IX a, IX b)
        {
            var av = a._value;
            var bv = b._value;
            if (av != RawNA && bv != RawNA)
            {
                var res = av + bv;
                // Overflow happens iff the sign of the result is different than both source values.
                if ((av ^ res) >= 0)
                    return res;
                if ((bv ^ res) >= 0)
                    return res;
            }
            return RawNA;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static IX operator -(IX a, IX b)
        {
            var av = a._value;
            var bv = -b._value;
            if (av != RawNA && bv != RawNA)
            {
                var res = av + bv;
                // Overflow happens iff the sign of the result is different than both source values.
                if ((av ^ res) >= 0)
                    return res;
                if ((bv ^ res) >= 0)
                    return res;
            }
            return RawNA;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static IX operator *(IX a, IX b)
        {
            var av = a._value;
            var bv = b._value;
            bool neg = (av ^ bv) < 0;
            if (av < 0)
            {
                if (av == RawNA)
                    return RawNA;
                av = -av;
            }
            if (bv < 0)
            {
                if (bv == RawNA)
                    return RawNA;
                bv = -bv;
            }

            // Deal with the low 32 bits.
            ulong lo1 = (ulong)av & 0x00000000FFFFFFFF;
            ulong lo2 = (ulong)bv & 0x00000000FFFFFFFF;
            RawIX res = (RawIX)(lo1 * lo2);
            if (res < 0)
                return RawNA;

            // Get the high 32 bits, including cross terms.
            ulong hi1 = (ulong)av >> 32;
            ulong hi2 = (ulong)bv >> 32;
            if (hi1 != 0)
            {
                // If both high words are non-zero, overflow is guaranteed.
                if (hi2 != 0)
                    return RawNA;
                // Compute the cross term.
                ulong tmp = hi1 * lo2;
                if ((tmp & 0xFFFFFFFF80000000) != 0)
                    return RawNA;
                res += (long)(tmp << 32);
                if (res < 0)
                    return RawNA;
            }
            else if (hi2 != 0)
            {
                // Compute the cross term.
                ulong tmp = hi2 * lo1;
                if ((tmp & 0xFFFFFFFF80000000) != 0)
                    return RawNA;
                res += (long)(tmp << 32);
                if (res < 0)
                    return RawNA;
            }

            // Adjust the sign.
            if (neg)
                res = -res;
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static IX operator /(IX a, IX b)
        {
            var av = a._value;
            var bv = b._value;
            if (av != RawNA && bv != RawNA && bv != 0)
                return av / bv;
            return RawNA;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static IX operator %(IX a, IX b)
        {
            var av = a._value;
            var bv = b._value;
            if (av != RawNA && bv != RawNA && bv != 0)
                return av % bv;
            return RawNA;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static IX Abs(IX a)
        {
            // Can't use Math.Abs since it throws on the RawNA value.
            return a._value >= 0 ? a._value : -a._value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static IX Sign(IX a)
        {
            var val = a._value;
            var neg = -val;
            // This works for NA since -RawNA == RawNA.
            return val > neg ? +1 : val < neg ? -1 : val;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static IX Min(IX a, IX b)
        {
            var v1 = a._value;
            var v2 = b._value;
            // This works for NA since RawNA == RawIX.MinValue.
            return v1 <= v2 ? v1 : v2;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public IX Min(IX b)
        {
            var v1 = _value;
            var v2 = b._value;
            // This works for NA since RawNA == RawIX.MinValue.
            return v1 <= v2 ? v1 : v2;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static IX Max(IX a, IX b)
        {
            var v1 = a._value;
            var v2 = b._value;
            // This works for NA since RawNA - 1 == RawIX.MaxValue.
            return v1 - 1 >= v2 - 1 ? v1 : v2;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public IX Max(IX b)
        {
            var v1 = _value;
            var v2 = b._value;
            // This works for NA since RawNA - 1 == RawIX.MaxValue.
            return v1 - 1 >= v2 - 1 ? v1 : v2;
        }

        /// <summary>
        /// Raise a to the b power. Special cases:
        /// * 1^NA => 1
        /// * NA^0 => 1
        /// </summary>
        public static IX Pow(IX a, IX b)
        {
            var av = a.RawValue;
            var bv = b.RawValue;

            if (av == 1)
                return 1;
            switch (bv)
            {
            case 0:
                return 1;
            case 1:
                return av;
            case 2:
                return a * a;
            case RawNA:
                return RawNA;
            }
            if (av == -1)
                return (bv & 1) == 0 ? 1 : -1;
            if (bv < 0)
                return RawNA;
            if (av == RawNA)
                return RawNA;

            // Since the abs of the base is at least two, the exponent must be less than 63.
            if (bv >= 63)
                return RawNA;

            bool neg = false;
            if (av < 0)
            {
                av = -av;
                neg = (bv & 1) != 0;
            }
            Contracts.Assert(av >= 2);

            // Since the exponent is at least three, the base must be < 2^21.
            Contracts.Assert(bv >= 3);
            if (av >= (1L << 21))
                return RawNA;

            long res = 1;
            long x = av;
            // REVIEW: Is the catch too slow in the overflow case?
            try
            {
                checked
                {
                    for (; ; )
                    {
                        if ((bv & 1) != 0)
                            res *= x;
                        bv >>= 1;
                        if (bv == 0)
                            break;
                        x *= x;
                    }
                }
            }
            catch (OverflowException)
            {
                return RawNA;
            }
            Contracts.Assert(res > 0);

            if (neg)
                res = -res;
            return res;
        }
    }
}
