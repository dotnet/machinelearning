// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.CompilerServices;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// A structure serving as a sixteen-byte unsigned integer. It is used as the row id of <see cref="IDataView"/>.
    /// For datasets with millions of records, those IDs need to be unique, therefore the need for such a large structure to hold the values.
    /// Those Ids are derived from other Ids of the previous components of the pipelines, and dividing the structure in two: high order and low order of bits,
    /// and reduces the changes of those collisions even further.
    /// </summary>
    /// <seealso cref="Row.GetIdGetter"/>
    public readonly struct RowId : IComparable<RowId>, IEquatable<RowId>
    {
        ///<summary>The low order bits. Corresponds to H1 in the Murmur algorithms.</summary>
        public readonly ulong Low;

        ///<summary> The high order bits. Corresponds to H2 in the Murmur algorithms.</summary>
        public readonly ulong High;

        /// <summary>
        /// Initializes a new instance of <see cref="RowId"/>
        /// </summary>
        /// <param name="low">The low order <langword>ulong</langword>.</param>
        /// <param name="high">The high order <langword>ulong</langword>.</param>
        public RowId(ulong low, ulong high)
        {
            Low = low;
            High = high;
        }

        public override string ToString()
        {
            // Since H1 are the low order bits, they are printed second.
            return string.Format("{0:x16}{1:x16}", High, Low);
        }

        public int CompareTo(RowId other)
        {
            int result = High.CompareTo(other.High);
            return result == 0 ? Low.CompareTo(other.Low) : result;
        }

        public bool Equals(RowId other)
        {
            return Low == other.Low && High == other.High;
        }

        public override bool Equals(object obj)
        {
            if (obj != null && obj is RowId)
            {
                var item = (RowId)obj;
                return Equals(item);
            }
            return false;
        }

        public static RowId operator +(RowId first, ulong second)
        {
            ulong resHi = first.High;
            ulong resLo = first.Low + second;
            if (resLo < second)
                resHi++;
            return new RowId(resLo, resHi);
        }

        public static RowId operator -(RowId first, ulong second)
        {
            ulong resHi = first.High;
            ulong resLo = first.Low - second;
            if (resLo > first.Low)
                resHi--;
            return new RowId(resLo, resHi);
        }

        public static bool operator ==(RowId first, ulong second)
        {
            return first.High == 0 && first.Low == second;
        }

        public static bool operator !=(RowId first, ulong second)
        {
            return !(first == second);
        }

        public static bool operator <(RowId first, ulong second)
        {
            return first.High  == 0 && first.Low < second;
        }

        public static bool operator >(RowId first, ulong second)
        {
            return first.High > 0 || first.Low > second;
        }

        public static bool operator <=(RowId first, ulong second)
        {
            return first.High == 0 && first.Low <= second;
        }

        public static bool operator >=(RowId first, ulong second)
        {
            return first.High > 0 || first.Low >= second;
        }

        public static explicit operator double(RowId x)
        {
            // REVIEW: The 64-bit JIT has a bug where rounding might be not quite
            // correct when converting a ulong to double with the high bit set. Should we
            // care and compensate? See the DoubleParser code for a work-around.
            return x.High * ((double)(1UL << 32) * (1UL << 32)) + x.Low;
        }

        public override int GetHashCode()
        {
            return (int)(
                (uint)Low ^ (uint)(Low >> 32) ^
                (uint)(High << 7) ^ (uint)(High >> 57) ^ (uint)(High >> (57 - 32)));
        }

        #region Hashing style

        // This is adapted from reference Murmur3 128-bit implementation at
        // https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
        // The transformation operations do not mix the result, since it only necessary
        // that the result be unique, not that its bits have any sort of independence
        // or other distributional characteristics.

        private const ulong _c1 = 0x87c37b91114253d5;
        private const ulong _c2 = 0x4cf5ad432745937f;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void RotL(ref ulong x, int r)
        {
            Contracts.Assert(0 < r && r < 64);
            x = (x << r) | (x >> (64 - r));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong RotL(ulong x, int r)
        {
            Contracts.Assert(0 < r && r < 64);
            return (x << r) | (x >> (64 - r));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void FMix(ref ulong k)
        {
            k ^= k >> 33;
            k *= 0xff51afd7ed558ccd;
            k ^= k >> 33;
            k *= 0xc4ceb9fe1a85ec53;
            k ^= k >> 33;
        }

        private static void FinalMix(ref ulong h1, ref ulong h2, int len)
        {
            h1 ^= (ulong)len;
            h2 ^= (ulong)len;
            h1 += h2;
            h2 += h1;
            FMix(ref h1);
            FMix(ref h2);
            h1 += h2;
            h2 += h1;
        }

        /// <summary>
        /// An operation that treats the value as an unmixed Murmur3 128-bit hash state,
        /// and returns the hash state that would result if we hashed an addition 16 bytes
        /// that were all zeros, except for the last bit which is one.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public RowId Fork()
        {
            ulong h1 = Low;
            ulong h2 = High;
            // Here it's as if k1=1, k2=0.
            h1 = RotL(h1, 27);
            h1 += h2;
            h1 = h1 * 5 + 0x52dce729;
            h2 = RotL(h2, 31);
            h2 += h1;
            h2 = h2 * 5 + 0x38495ab5;
            h1 ^= RotL(_c1, 31) * _c2;
            return new RowId(h1, h2);
        }

        /// <summary>
        /// An operation that treats the value as an unmixed Murmur3 128-bit hash state,
        /// and returns the hash state that would result if we hashed an addition 16 bytes
        /// that were all zeros.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public RowId Next()
        {
            ulong h1 = Low;
            ulong h2 = High;
            // Here it's as if k1=0, k2=0.
            h1 = RotL(h1, 27);
            h1 += h2;
            h1 = h1 * 5 + 0x52dce729;
            h2 = RotL(h2, 31);
            h2 += h1;
            h2 = h2 * 5 + 0x38495ab5;
            return new RowId(h1, h2);
        }

        /// <summary>
        /// An operation that treats the value as an unmixed Murmur3 128-bit hash state,
        /// and returns the hash state that would result if we took <paramref name="other"/>,
        /// scrambled it using <see cref="Fork"/>, then hashed the result of that.
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public RowId Combine(RowId other)
        {
            var h1 = Low;
            var h2 = High;

            other = other.Fork();
            ulong k1 = other.Low; // First 8 bytes.
            ulong k2 = other.High; // Second 8 bytes.

            k1 *= _c1;
            k1 = RotL(k1, 31);
            k1 *= _c2;
            h1 ^= k1;
            h1 = RotL(h1, 27);
            h1 += h2;
            h1 = h1 * 5 + 0x52dce729;

            k2 *= _c2;
            k2 = RotL(k2, 33);
            k2 *= _c1;
            h2 ^= k2;
            h2 = RotL(h2, 31);
            h2 += h1;
            h2 = h2 * 5 + 0x38495ab5;

            return new RowId(h1, h2);
        }
        #endregion
    }
}
