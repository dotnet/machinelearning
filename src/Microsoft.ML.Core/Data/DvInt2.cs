// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.CompilerServices;

namespace Microsoft.ML.Runtime.Data
{
    using BL = Boolean;
    using I1 = DvInt1;
    using I4 = DvInt4;
    using I8 = DvInt8;
    using IX = DvInt2;
    using R4 = Single;
    using R8 = Double;
    using RawI8 = Int64;
    using RawIX = Int16;

    public struct DvInt2 : IEquatable<IX>, IComparable<IX>
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
                throw Contracts.ExceptValue(nameof(value), "NA cast to short");
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
            return Convert.ToInt16(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator IX(I1 a)
        {
            if (a.IsNA)
                return RawNA;
            return (RawIX)a.RawValue;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator IX(I4 a)
        {
            RawIX res = (RawIX)a.RawValue;
            if (res != a.RawValue)
                return RawNA;
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator IX(I8 a)
        {
            RawIX res = (RawIX)a.RawValue;
            if (res != a.RawValue)
                return RawNA;
            return res;
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
                RawI8 nn = (RawI8)a;
                Contracts.Assert(nn == n);
                if (a >= 0)
                    Contracts.Assert(a - 1 < n & n <= a);
                else
                    Contracts.Assert(a <= n & n < a + 1);
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
    }
}
