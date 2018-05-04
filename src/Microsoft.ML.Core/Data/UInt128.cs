// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.CompilerServices;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// A sixteen-byte unsigned integer.
    /// </summary>
    public struct UInt128 : IComparable<UInt128>, IEquatable<UInt128>
    {
        // The low order bits. Corresponds to H1 in the Murmur algorithms.
        public readonly ulong Lo;
        // The high order bits. Corresponds to H2 in the Murmur algorithms.
        public readonly ulong Hi;

        public UInt128(ulong lo, ulong hi)
        {
            Lo = lo;
            Hi = hi;
        }

        public override string ToString()
        {
            // Since H1 are the low order bits, they are printed second.
            return string.Format("{0:x16}{1:x16}", Hi, Lo);
        }

        public int CompareTo(UInt128 other)
        {
            int result = Hi.CompareTo(other.Hi);
            return result == 0 ? Lo.CompareTo(other.Lo) : result;
        }

        public bool Equals(UInt128 other)
        {
            return Lo == other.Lo && Hi == other.Hi;
        }

        public override bool Equals(object obj)
        {
            if (obj != null && obj is UInt128)
            {
                var item = (UInt128)obj;
                return Equals(item);
            }
            return false;
        }

        public static UInt128 operator +(UInt128 first, ulong second)
        {
            ulong resHi = first.Hi;
            ulong resLo = first.Lo + second;
            if (resLo < second)
                resHi++;
            return new UInt128(resLo, resHi);
        }

        public static UInt128 operator -(UInt128 first, ulong second)
        {
            ulong resHi = first.Hi;
            ulong resLo = first.Lo - second;
            if (resLo > first.Lo)
                resHi--;
            return new UInt128(resLo, resHi);
        }

        public static bool operator ==(UInt128 first, ulong second)
        {
            return first.Hi == 0 && first.Lo == second;
        }

        public static bool operator !=(UInt128 first, ulong second)
        {
            return !(first == second);
        }

        public static bool operator <(UInt128 first, ulong second)
        {
            return first.Hi  == 0 && first.Lo < second;
        }

        public static bool operator >(UInt128 first, ulong second)
        {
            return first.Hi > 0 || first.Lo > second;
        }

        public static bool operator <=(UInt128 first, ulong second)
        {
            return first.Hi == 0 && first.Lo <= second;
        }

        public static bool operator >=(UInt128 first, ulong second)
        {
            return first.Hi > 0 || first.Lo >= second;
        }

        public static explicit operator double(UInt128 x)
        {
            // REVIEW: The 64-bit JIT has a bug where rounding might be not quite
            // correct when converting a ulong to double with the high bit set. Should we
            // care and compensate? See the DoubleParser code for a work-around.
            return x.Hi * ((double)(1UL << 32) * (1UL << 32)) + x.Lo;
        }

        public override int GetHashCode()
        {
            return (int)(
                (uint)Lo ^ (uint)(Lo >> 32) ^
                (uint)(Hi << 7) ^ (uint)(Hi >> 57) ^ (uint)(Hi >> (57 - 32)));
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
        public UInt128 Fork()
        {
            ulong h1 = Lo;
            ulong h2 = Hi;
            // Here it's as if k1=1, k2=0.
            h1 = RotL(h1, 27);
            h1 += h2;
            h1 = h1 * 5 + 0x52dce729;
            h2 = RotL(h2, 31);
            h2 += h1;
            h2 = h2 * 5 + 0x38495ab5;
            h1 ^= RotL(_c1, 31) * _c2;
            return new UInt128(h1, h2);
        }

        /// <summary>
        /// An operation that treats the value as an unmixed Murmur3 128-bit hash state,
        /// and returns the hash state that would result if we hashed an addition 16 bytes
        /// that were all zeros.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public UInt128 Next()
        {
            ulong h1 = Lo;
            ulong h2 = Hi;
            // Here it's as if k1=0, k2=0.
            h1 = RotL(h1, 27);
            h1 += h2;
            h1 = h1 * 5 + 0x52dce729;
            h2 = RotL(h2, 31);
            h2 += h1;
            h2 = h2 * 5 + 0x38495ab5;
            return new UInt128(h1, h2);
        }

        /// <summary>
        /// An operation that treats the value as an unmixed Murmur3 128-bit hash state,
        /// and returns the hash state that would result if we took <paramref name="other"/>,
        /// scrambled it using <see cref="Fork"/>, then hashed the result of that.
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public UInt128 Combine(UInt128 other)
        {
            var h1 = Lo;
            var h2 = Hi;

            other = other.Fork();
            ulong k1 = other.Lo; // First 8 bytes.
            ulong k2 = other.Hi; // Second 8 bytes.

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

            return new UInt128(h1, h2);
        }
        #endregion
    }
}
