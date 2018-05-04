// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.CompilerServices;

namespace Microsoft.ML.Runtime.Data
{
    using BL = DvBool;
    using R4 = Single;
    using R8 = Double;

    public struct DvBool : IEquatable<DvBool>, IComparable<DvBool>
    {
        private const byte _false = 0;
        private const byte _true = 1;
        private const byte _na = 128;
        public const byte RawNA = _na;

        private byte _value;

        public static BL False { get { BL res; res._value = _false; return res; } }
        public static BL True { get { BL res; res._value = _true; return res; } }
        public static BL NA { get { BL res; res._value = _na; return res; } }

        /// <summary>
        /// Property to return the raw value.
        /// </summary>
        public byte RawValue
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return _value; }
        }

        /// <summary>
        /// Static method to return the raw value. This is more convenient than the
        /// property in code-generation scenarios.
        /// </summary>
        public static byte GetRawBits(BL a)
        {
            return a._value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private DvBool(int value)
        {
            Contracts.Assert(value == _true || value == _false || value == _na);
            _value = (byte)value;
        }

        /// <summary>
        /// Returns whether this value is false.
        /// </summary>
        public bool IsFalse
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return _value == _false; }
        }

        /// <summary>
        /// Returns whether this value is true.
        /// </summary>
        public bool IsTrue
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return _value == _true; }
        }

        /// <summary>
        /// Returns whether this value is NA.
        /// </summary>
        public bool IsNA
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return _value > _true; }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator BL(bool value)
        {
            BL res;
            res._value = value ? _true : _false;
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator BL(bool? value)
        {
            BL res;
            res._value = value == null ? _na : value.GetValueOrDefault() ? _true : _false;
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator bool(BL value)
        {
            switch (value._value)
            {
            case _false:
                return false;
            case _true:
                return true;
            default:
                throw Contracts.ExceptValue(nameof(value), "NA cast to bool");
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator bool?(BL value)
        {
            switch (value._value)
            {
            case _false:
                return false;
            case _true:
                return true;
            default:
                return null;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator R4(BL value)
        {
            if (value._value <= _true)
                return value._value;
            return Single.NaN;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator R8(BL value)
        {
            if (value._value <= _true)
                return value._value;
            return Double.NaN;
        }

        public override int GetHashCode()
        {
            return _value.GetHashCode();
        }

        public override bool Equals(object obj)
        {
            if (obj is BL)
                return _value == ((BL)obj)._value;
            return false;
        }

        public bool Equals(BL other)
        {
            // Note that if one or both are "non-standard" NA values, this
            // could return false. Theoretically, that should never happen,
            // but unsafe code could cause it.
            return _value == other._value;
        }

        public int CompareTo(BL other)
        {
            // Note that if one or both are "non-standard" NA values, this could produce unexpected comparisons.
            // Theoretically, that should never happen, but unsafe code could cause it.
            Contracts.Assert(unchecked((sbyte)RawNA) < (sbyte)_false);
            if (_value == other._value)
                return 0;
            return (sbyte)_value < (sbyte)other._value ? -1 : 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL operator ==(BL a, BL b)
        {
            if (a._value <= _true && b._value <= _true)
                return a._value == b._value ? True : False;
            return NA;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL operator !=(BL a, BL b)
        {
            if (a._value <= _true && b._value <= _true)
                return a._value != b._value ? True : False;
            return NA;
        }

        public override string ToString()
        {
            if (_value == _false)
                return "False";
            if (_value == _true)
                return "True";
            return "NA";
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL operator !(BL a)
        {
            if (a._value <= _true)
                a._value ^= 1;
            return a;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL operator |(BL a, BL b)
        {
            if (a._value == _true)
                return a;
            if (b._value == _true)
                return b;
            if (a._value != _false)
                return a;
            return b;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL operator &(BL a, BL b)
        {
            if (a._value == _false)
                return a;
            if (b._value == _false)
                return b;
            if (a._value != _true)
                return a;
            return b;
        }
    }
}
