// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Internal.Utilities
{
    [BestFriend]
    internal static class Hashing
    {
        private const uint _defaultSeed = (5381 << 16) + 5381;

        public static uint CombineHash(uint u1, uint u2)
        {
            return ((u1 << 7) | (u1 >> 25)) ^ u2;
        }

        public static int CombineHash(int n1, int n2)
        {
            return (int)CombineHash((uint)n1, (uint)n2);
        }

        /// <summary>
        /// Creates a combined hash of possibly heterogenously typed values.
        /// </summary>
        /// <param name="startHash">The leading hash, incorporated into the final hash</param>
        /// <param name="os">A variable list of objects, where null is a valid value</param>
        /// <returns>The combined hash incorpoating a starting hash, and the hash codes
        /// of all input values</returns>
        public static int CombinedHash(int startHash, params object[] os)
        {
            foreach (object o in os)
                startHash = CombineHash(startHash, o == null ? 0 : o.GetHashCode());
            return startHash;
        }

        /// <summary>
        /// Creates a combined hash of multiple homogenously typed non-null values.
        /// </summary>
        /// <typeparam name="T">The type of items to hash</typeparam>
        /// <param name="startHash">The leading hash, incorporated into the final hash</param>
        /// <param name="os">A variable list of non-null values</param>
        /// <returns>The combined hash incorpoating a starting hash, and the hash codes
        /// of all input values</returns>
        public static int CombinedHash<T>(int startHash, params T[] os)
        {
            foreach (T o in os)
                startHash = CombineHash(startHash, o.GetHashCode());
            return startHash;
        }

        public static uint HashUint(uint u)
        {
            ulong uu = (ulong)u * 0x7ff19519UL; // this number is prime.
            return Utils.GetLo(uu) + Utils.GetHi(uu);
        }

        public static int HashInt(int n)
        {
            return (int)HashUint((uint)n);
        }

        /// <summary>
        /// Hash the characters in a <see cref="ReadOnlySpan{T}"/> of <see cref="char"/>.
        /// This MUST produce the same result as the other overloads (with equivalent characters).
        /// </summary>
        public static uint HashString(ReadOnlySpan<char> str) => MurmurHash(_defaultSeed, str);

        /// <summary>
        /// Hash the characters in a string builder. This MUST produce the same result
        /// as HashString(sb.ToString()).
        /// </summary>
        public static uint HashString(StringBuilder sb)
        {
            Contracts.AssertValue(sb);
            return MurmurHash(_defaultSeed, sb, 0, sb.Length);
        }

        public static uint HashSequence(uint[] sequence, int min, int lim)
        {
            return MurmurHash(_defaultSeed, sequence, min, lim);
        }

        /// <summary>
        /// Combines the given hash value with a uint value, using the murmur hash 3 algorithm.
        /// Make certain to also use <see cref="MixHash"/> on the final hashed value, if you
        /// depend upon having distinct bits.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint MurmurRound(uint hash, uint chunk)
        {
            chunk *= 0xCC9E2D51;
            chunk = Rotate(chunk, 15);
            chunk *= 0x1B873593;

            hash ^= chunk;
            hash = Rotate(hash, 13);
            hash *= 5;
            hash += 0xE6546B64;

            return hash;
        }

        private static uint MurmurRoundSpanV2(uint hash, Span<byte> key, int len)
        {
            int nblocks = len / 4;
            byte[] data = key.ToArray();

            for (int i = nblocks; i >0; i--)
            {
                uint chunk = BitConverter.ToUInt32(data, (nblocks * 4 - i*4));
                chunk *= 0xCC9E2D51;
                chunk = Rotate(chunk, 15);
                chunk *= 0x1B873593;

                hash ^= chunk;
                hash = Rotate(hash, 13);
                hash *= 5;
                hash += 0xE6546B64;
            }

            byte[] tail = new byte[3]{data[len-3], data[len-2], data[len-1]};

            uint k1 = 0;

            switch (len & 3)
            {
                case 3:
                    k1 ^= (uint)tail[2] << 16;
                    goto case 2;
                case 2:
                    k1 ^= (uint)tail[1] << 8;
                    goto case 1;
                case 1:
                    k1 ^= tail[0];
                    k1 *= 0xCC9E2D51; k1 = Rotate(k1, 15);
                    k1 *= 0x1B873593;
                    hash ^= k1;
                    break;
            }

            return hash;
        }

        public static uint MurmurRoundFloat(uint hash, float chunk)
        {
            var floatBytes = BitConverter.GetBytes(chunk);
            Span<byte> key = new Span<byte>(floatBytes);
            return MurmurRoundSpanV2(hash, key, 4);
        }

        public static uint MurmurRoundDouble(uint hash, double chunk)
        {
            var doubleBytes = BitConverter.GetBytes(chunk);
            Span<byte> key = new Span<byte>(doubleBytes);
            return MurmurRoundSpanV2(hash, key, 8);
        }

        public static uint MurmurRoundText(uint hash, ReadOnlyMemory<char> chunk)
        {
            byte[] utf8Bytes = Encoding.UTF8.GetBytes(chunk.ToArray());
            var key = new Span<byte>(utf8Bytes);
            return MurmurRoundSpanV2(hash, key, chunk.Length);
        }

        public static uint MurmurRoundU1(uint hash, byte chunk)
        {
            Span<byte> key = new Span<byte>(new byte[] { chunk });
            return MurmurRoundSpanV2(hash, key, 1);
        }

        public static uint MurmurRoundU2(uint hash, ushort chunk)
        {
            var ushortBytes = BitConverter.GetBytes(chunk);
            Span<byte> key = new Span<byte>(ushortBytes);
            return MurmurRoundSpanV2(hash, key, 2);
        }

        public static uint MurmurRoundU4(uint hash, uint chunk)
        {
            var uintBytes = BitConverter.GetBytes(chunk);
            Span<byte> key = new Span<byte>(uintBytes);
            return MurmurRoundSpanV2(hash, key, 4);
        }

        public static uint MurmurRoundU8(uint hash, ulong chunk)
        {
            var ulongBytes = BitConverter.GetBytes(chunk);
            Span<byte> key = new Span<byte>(ulongBytes);
            return MurmurRoundSpanV2(hash, key, 8);
        }

        public static uint MurmurRoundI1(uint hash, sbyte chunk)
        {
            var sbyteBytes = BitConverter.GetBytes(chunk);
            Span<byte> key = new Span<byte>(sbyteBytes);
            return MurmurRoundSpanV2(hash, key, 1);
        }

        public static uint MurmurRoundI2(uint hash, short chunk)
        {
            var shortBytes = BitConverter.GetBytes(chunk);
            Span<byte> key = new Span<byte>(shortBytes);
            return MurmurRoundSpanV2(hash, key, 2);
        }

        public static uint MurmurRoundI4(uint hash, int chunk)
        {
            var intBytes = BitConverter.GetBytes(chunk);
            Span<byte> key = new Span<byte>(intBytes);
            return MurmurRoundSpanV2(hash, key, 4);
        }

        public static uint MurmurRoundI8(uint hash, long chunk)
        {
            var longBytes = BitConverter.GetBytes(chunk);
            Span<byte> key = new Span<byte>(longBytes);
            return MurmurRoundSpanV2(hash, key, 8);
        }

        /// <summary>
        /// Implements the murmur hash 3 algorithm, using a mock UTF-8 encoding.
        /// The UTF-8 conversion ignores the possibilities of unicode planes other than the 0th.
        /// That is, it simply converts char values to one, two, or three bytes according to
        /// the following rules:
        /// * 0x0000 to 0x007F : 0xxxxxxx
        /// * 0x0080 to 0x07FF : 110xxxxx 10xxxxxx
        /// * 0x0800 to 0xFFFF : 1110xxxx 10xxxxxx 10xxxxxx
        /// NOTE: This MUST match the StringBuilder version below.
        /// </summary>
        public static uint MurmurHash(uint hash, ReadOnlySpan<char> span, bool toUpper = false)
        {
            // Byte length (in pseudo UTF-8 form).
            int len = 0;

            // Current bits, value and count.
            ulong cur = 0;
            int bits = 0;
            for (int ich = 0; ich < span.Length; ich++)
            {
                Contracts.Assert((bits & 0x7) == 0);
                Contracts.Assert((uint)bits <= 24);
                Contracts.Assert(cur <= 0x00FFFFFF);

                uint ch = toUpper ? char.ToUpperInvariant(span[ich]) : span[ich];
                if (ch <= 0x007F)
                {
                    cur |= ch << bits;
                    bits += 8;
                }
                else if (ch <= 0x07FF)
                {
                    cur |= (ulong)((ch & 0x003F) | ((ch << 2) & 0x1F00) | 0xC080) << bits;
                    bits += 16;
                }
                else
                {
                    Contracts.Assert(ch <= 0xFFFF);
                    cur |= (ulong)((ch & 0x003F) | ((ch << 2) & 0x3F00) | ((ch << 4) & 0x0F0000) | 0xE08080) << bits;
                    bits += 24;
                }

                if (bits >= 32)
                {
                    hash = MurmurRound(hash, (uint)cur);
                    cur = cur >> 32;
                    bits -= 32;
                    len += 4;
                }
            }
            Contracts.Assert((bits & 0x7) == 0);
            Contracts.Assert((uint)bits <= 24);
            Contracts.Assert(cur <= 0x00FFFFFF);

            if (bits > 0)
            {
                hash = MurmurRound(hash, (uint)cur);
                len += bits / 8;
            }

            // Encode the length.
            hash = MurmurRound(hash, (uint)len);

            // Final mixing ritual for the hash.
            hash = MixHash(hash);

            return hash;
        }

        /// <summary>
        /// Implements the murmur hash 3 algorithm, using a mock UTF-8 encoding.
        /// The UTF-8 conversion ignores the possibilities of unicode planes other than the 0th.
        /// That is, it simply converts char values to one, two, or three bytes according to
        /// the following rules:
        /// * 0x0000 to 0x007F : 0xxxxxxx
        /// * 0x0080 to 0x07FF : 110xxxxx 10xxxxxx
        /// * 0x0800 to 0xFFFF : 1110xxxx 10xxxxxx 10xxxxxx
        /// NOTE: This MUST match the string version above.
        /// </summary>
        public static uint MurmurHash(uint hash, StringBuilder data, int ichMin, int ichLim, bool toUpper = false)
        {
            Contracts.Assert(0 <= ichMin & ichMin <= ichLim & ichLim <= Utils.Size(data));

            uint seed = hash;

            // Byte length (in pseudo UTF-8 form).
            int len = 0;

            // Current bits, value and count.
            ulong cur = 0;
            int bits = 0;
            for (int ich = ichMin; ich < ichLim; ich++)
            {
                Contracts.Assert((bits & 0x7) == 0);
                Contracts.Assert((uint)bits <= 24);
                Contracts.Assert(cur <= 0x00FFFFFF);

                uint ch = toUpper ? char.ToUpperInvariant(data[ich]) : data[ich];
                if (ch <= 0x007F)
                {
                    cur |= ch << bits;
                    bits += 8;
                }
                else if (ch <= 0x07FF)
                {
                    cur |= (ulong)((ch & 0x003F) | ((ch << 2) & 0x1F00) | 0xC080) << bits;
                    bits += 16;
                }
                else
                {
                    Contracts.Assert(ch <= 0xFFFF);
                    cur |= (ulong)((ch & 0x003F) | ((ch << 2) & 0x3F00) | ((ch << 4) & 0x0F0000) | 0xE08080) << bits;
                    bits += 24;
                }

                if (bits >= 32)
                {
                    hash = MurmurRound(hash, (uint)cur);
                    cur = cur >> 32;
                    bits -= 32;
                    len += 4;
                }
            }
            Contracts.Assert((bits & 0x7) == 0);
            Contracts.Assert((uint)bits <= 24);
            Contracts.Assert(cur <= 0x00FFFFFF);

            if (bits > 0)
            {
                hash = MurmurRound(hash, (uint)cur);
                len += bits / 8;
            }

            // Encode the length.
            hash = MurmurRound(hash, (uint)len);

            // Final mixing ritual for the hash.
            hash = MixHash(hash);

            Contracts.Assert(hash == MurmurHash(seed, data.ToString().AsSpan()));
            return hash;
        }

        /// <summary>
        /// Performs a MurmurRound on each int in the sequence
        /// </summary>
        public static uint MurmurHash(uint hash, uint[] data, int min, int lim)
        {
            Contracts.Check(0 <= min);
            Contracts.Check(min <= lim);
            Contracts.Check(lim <= Utils.Size(data));

            for (int i = min; i < lim; i++)
                hash = MurmurRound(hash, data[i]);

            hash = MurmurRound(hash, (uint)(lim - min));

            // Final mixing ritual for the hash.
            hash = MixHash(hash);

            return hash;
        }

        /// <summary>
        /// The final mixing ritual for the Murmur3 hashing algorithm. Most users of
        /// <see cref="MurmurRound"/> will want to close their progressive building of
        /// a hash with a call to this method.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint MixHash(uint hash)
        {
            hash ^= hash >> 16;
            hash *= 0x85ebca6b;
            hash ^= hash >> 13;
            hash *= 0xc2b2ae35;
            hash ^= hash >> 16;
            return hash;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint MixHashV2(uint hash, int len)
        {
            hash ^= (uint)len;
            hash ^= hash >> 16;
            hash *= 0x85ebca6b;
            hash ^= hash >> 13;
            hash *= 0xc2b2ae35;
            hash ^= hash >> 16;
            return hash;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint Rotate(uint x, int r)
        {
            return (x << r) | (x >> (32 - r));
        }
    }
}
