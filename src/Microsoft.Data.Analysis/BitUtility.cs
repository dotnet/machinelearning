// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Microsoft.Data.Analysis
{
    // License for BitUtility
    // --------------------------------------
    // This class is based on the code from Apache Arrow project 
    // https://github.com/apache/arrow/blob/main/csharp/src/Apache.Arrow/BitUtility.cs
    // that is available in the public domain inder Apache-2.0 license.
    // You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
    internal static class BitUtility
    {
        private static ReadOnlySpan<byte> PopcountTable => new byte[] {
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
            3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
        };

        private static ReadOnlySpan<byte> BitMask => new byte[] {
            1, 2, 4, 8, 16, 32, 64, 128
        };

        // Faster to use when we already have a span since it avoids indexing
        public static bool IsValid(ReadOnlySpan<byte> bitMapBufferSpan, int index)
        {
            var nullBitMapSpanIndex = index / 8;
            var thisBitMap = bitMapBufferSpan[nullBitMapSpanIndex];
            return IsBitSet(thisBitMap, index);
        }

        public static bool IsBitSet(byte curBitMap, int index)
        {
            return ((curBitMap >> (index & 7)) & 1) != 0;
        }

        public static bool IsBitClear(byte curBitMap, int index)
        {
            return ((curBitMap >> (index & 7)) & 1) == 0;
        }

        public static bool GetBit(byte data, int index) =>
           ((data >> index) & 1) != 0;

        public static bool GetBit(ReadOnlySpan<byte> data, int index) =>
            (data[index / 8] & BitMask[index % 8]) != 0;

        public static void ClearBit(Span<byte> data, int index)
        {
            data[index / 8] &= (byte)~BitMask[index % 8];
        }

        public static void SetBit(Span<byte> data, int index)
        {
            data[index / 8] |= BitMask[index % 8];
        }

        public static void SetBit(Span<byte> data, long index, bool value)
        {
            int idx = (int)(index / 8);
            int mod = (int)(index % 8);
            data[idx] = value
                ? (byte)(data[idx] | BitMask[mod])
                : (byte)(data[idx] & ~BitMask[mod]);
        }

        /// <summary>
        /// Set the number of bits in a span of bytes starting
        /// at a specific index, and limiting to length.
        /// </summary>
        /// <param name="data">Span to set bits value.</param>
        /// <param name="index">Bit index to start counting from.</param>
        /// <param name="length">Maximum of bits in the span to consider.</param>
        /// <param name="value">Bit value.</param>
        public static void SetBits(Span<byte> data, long index, long length, bool value)
        {
            if (length == 0)
                return;

            var endBitIndex = index + length - 1;

            // Use simpler method if there aren't many values
            if (length < 20)
            {
                for (var i = index; i <= endBitIndex; i++)
                {
                    SetBit(data, i, value);
                }
                return;
            }

            // Otherwise do the work to figure out how to copy whole bytes
            var startByteIndex = (int)(index / 8);
            var startBitOffset = (int)(index % 8);
            var endByteIndex = (int)(endBitIndex / 8);
            var endBitOffset = (int)(endBitIndex % 8);

            // If the starting index and ending index are not byte-aligned,
            // we'll need to set bits the slow way. If they are
            // byte-aligned, and for all other bytes in the 'middle', we
            // can use a faster byte-aligned set.
            var fullByteStartIndex = startBitOffset == 0 ? startByteIndex : startByteIndex + 1;
            var fullByteEndIndex = endBitOffset == 7 ? endByteIndex : endByteIndex - 1;

            // Bits we will be using to finish up the first byte
            if (startBitOffset != 0)
            {
                var slice = data.Slice(startByteIndex, 1);
                for (var i = startBitOffset; i <= 7; i++)
                    SetBit(slice, i, value);
            }

            if (fullByteEndIndex >= fullByteStartIndex)
            {
                var slice = data.Slice(fullByteStartIndex, fullByteEndIndex - fullByteStartIndex + 1);
                byte fill = (byte)(value ? 0xFF : 0x00);

                slice.Fill(fill);
            }

            if (endBitOffset != 7)
            {
                var slice = data.Slice(endByteIndex, 1);
                for (int i = 0; i <= endBitOffset; i++)
                    SetBit(slice, i, value);
            }
        }

        /// <summary>
        /// Returns the population count (number of bits set) in a span of bytes starting
        /// at 0 bit and limiting to length of bits.
        /// </summary>
        /// <param name="span"></param>
        /// <param name="length"></param>
        /// <returns></returns>
        public static long GetBitCount(ReadOnlySpan<byte> span, long length)
        {
            var endByteIndex = (int)(length / 8);

            Debug.Assert(span.Length >= endByteIndex);

            long count = 0;
            for (var i = 0; i < endByteIndex; i++)
                count += PopcountTable[span[i]];

            var endBitOffset = (int)(length % 8);

            if (endBitOffset != 0)
            {
                var partialByte = span[endByteIndex];
                for (var j = 0; j < endBitOffset; j++)
                {
                    count += GetBit(partialByte, j) ? 1 : 0;
                }
            }

            return count;
        }
    }
}
