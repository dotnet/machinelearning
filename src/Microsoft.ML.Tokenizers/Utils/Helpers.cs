// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;

namespace Microsoft.ML.Tokenizers
{
    internal static partial class Helpers
    {
        private const int UnicodeError = 0xFFFD;

        internal static void ArrayPoolGrow<T>(ref T[] arrayPoolArray, int requiredCapacity)
        {
            T[] tmp = ArrayPool<T>.Shared.Rent(Math.Max(arrayPoolArray.Length * 2, requiredCapacity));
            arrayPoolArray.CopyTo(tmp.AsSpan());
            ArrayPool<T>.Shared.Return(arrayPoolArray);
            arrayPoolArray = tmp;
        }

        internal static void ArrayPoolGrow<T>(ref Span<T> span, ref T[]? poolArray, int newSize)
        {
            Debug.Assert(span.Length <= newSize);

            T[] newPoolArray = ArrayPool<T>.Shared.Rent(newSize);
            span.CopyTo(newPoolArray);

            if (poolArray is not null)
            {
                ArrayPool<T>.Shared.Return(poolArray);
            }

            poolArray = newPoolArray;
            span = poolArray;
        }

        private static readonly int[] _oneCharLen = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };

        // Return length of a single UTF-8 source character
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int OneCharLen(byte src) => _oneCharLen[(src & 0xFF) >> 4];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int GetUtf16LengthFromUtf8Bytes(ReadOnlySpan<byte> utf8Bytes)
        {
            int length = 0;

            while (utf8Bytes.Length > 0)
            {
                int bytesLength = OneCharLen(utf8Bytes[0]);
                length += bytesLength == 4 ? 2 : 1;
                utf8Bytes = utf8Bytes.Slice(Math.Min(bytesLength, utf8Bytes.Length));
            }

            return length;
        }

        internal static int EncodeToUtf8(ReadOnlySpan<char> text, Span<byte> destination, Span<int> indexMapping)
        {
            Debug.Assert(!text.IsEmpty);
            Debug.Assert(Encoding.UTF8.GetMaxByteCount(text.Length) <= destination.Length);
            Debug.Assert(indexMapping.Length >= destination.Length);

            int targetIndex = 0;

            for (int i = 0; i < text.Length; i++)
            {
                uint c = (uint)text[i];
                if (c <= 0x7Fu)
                {
                    destination[targetIndex] = (byte)c;
                    indexMapping[targetIndex] = i;
                    targetIndex++;
                    continue;
                }

                if (c <= 0x7FFu)
                {
                    // Scalar 00000yyy yyxxxxxx -> bytes [ 110yyyyy 10xxxxxx ]
                    destination[targetIndex] = (byte)((c + (0b110u << 11)) >> 6);
                    destination[targetIndex + 1] = (byte)((c & 0x3Fu) + 0x80u);
                    indexMapping[targetIndex] = indexMapping[targetIndex + 1] = i;
                    targetIndex += 2;
                    continue;
                }

                if (i < text.Length - 1 && char.IsSurrogatePair((char)c, text[i + 1]))
                {
                    // Scalar 000uuuuu zzzzyyyy yyxxxxxx -> bytes [ 11110uuu 10uuzzzz 10yyyyyy 10xxxxxx ]
                    uint value = (uint)char.ConvertToUtf32((char)c, text[i + 1]);
                    destination[targetIndex] = (byte)((value + (0b11110 << 21)) >> 18);
                    destination[targetIndex + 1] = (byte)(((value & (0x3Fu << 12)) >> 12) + 0x80u);
                    destination[targetIndex + 2] = (byte)(((value & (0x3Fu << 6)) >> 6) + 0x80u);
                    destination[targetIndex + 3] = (byte)((value & 0x3Fu) + 0x80u);
                    indexMapping[targetIndex] = indexMapping[targetIndex + 1] = indexMapping[targetIndex + 2] = indexMapping[targetIndex + 3] = i;
                    i++;
                    targetIndex += 4;
                    continue;
                }

                // Scalar zzzzyyyy yyxxxxxx -> bytes [ 1110zzzz 10yyyyyy 10xxxxxx ]
                destination[targetIndex] = (byte)((c + (0b1110 << 16)) >> 12);
                destination[targetIndex + 1] = (byte)(((c & (0x3Fu << 6)) >> 6) + 0x80u);
                destination[targetIndex + 2] = (byte)((c & 0x3Fu) + 0x80u);
                indexMapping[targetIndex] = indexMapping[targetIndex + 1] = indexMapping[targetIndex + 2] = i;
                targetIndex += 3;
            }

            return targetIndex;
        }

        internal static int EncodeNextUtf8(ReadOnlySpan<char> text, Span<byte> destination)
        {
            Debug.Assert(!text.IsEmpty);
            Debug.Assert(destination.Length >= 4);

            uint c = (uint)text[0];
            if (c <= 0x7Fu)
            {
                destination[0] = (byte)c;
                return 1;
            }

            if (c <= 0x7FFu)
            {
                // Scalar 00000yyy yyxxxxxx -> bytes [ 110yyyyy 10xxxxxx ]
                destination[0] = (byte)((c + (0b110u << 11)) >> 6);
                destination[1] = (byte)((c & 0x3Fu) + 0x80u);
                return 2;
            }

            if (text.Length > 1 && char.IsSurrogatePair((char)c, text[1]))
            {
                // Scalar 000uuuuu zzzzyyyy yyxxxxxx -> bytes [ 11110uuu 10uuzzzz 10yyyyyy 10xxxxxx ]
                uint value = (uint)char.ConvertToUtf32((char)c, text[1]);
                destination[0] = (byte)((value + (0b11110 << 21)) >> 18);
                destination[1] = (byte)(((value & (0x3Fu << 12)) >> 12) + 0x80u);
                destination[2] = (byte)(((value & (0x3Fu << 6)) >> 6) + 0x80u);
                destination[3] = (byte)((value & 0x3Fu) + 0x80u);
                return 4;
            }

            // Scalar zzzzyyyy yyxxxxxx -> bytes [ 1110zzzz 10yyyyyy 10xxxxxx ]
            destination[0] = (byte)((c + (0b1110 << 16)) >> 12);
            destination[1] = (byte)(((c & (0x3Fu << 6)) >> 6) + 0x80u);
            destination[2] = (byte)((c & 0x3Fu) + 0x80u);
            return 3;
        }

        internal static int EncodeToUtf8AndTransform(ReadOnlySpan<char> text, Span<char> destination, Span<int> indexMapping)
        {
            Debug.Assert(!text.IsEmpty);
            Debug.Assert(Encoding.UTF8.GetMaxByteCount(text.Length) <= destination.Length);
            Debug.Assert(indexMapping.Length >= destination.Length);

            ByteToUnicodeEncoding byteToUnicodeEncoder = ByteToUnicodeEncoding.Instance;
            int targetIndex = 0;

            for (int i = 0; i < text.Length; i++)
            {
                uint c = (uint)text[i];
                if (c <= 0x7Fu)
                {
                    destination[targetIndex] = byteToUnicodeEncoder.ByteToUnicode[(char)c];
                    indexMapping[targetIndex] = i;
                    targetIndex++;
                    continue;
                }

                if (c <= 0x7FFu)
                {
                    // Scalar 00000yyy yyxxxxxx -> bytes [ 110yyyyy 10xxxxxx ]
                    destination[targetIndex] = byteToUnicodeEncoder.ByteToUnicode[(char)((c + (0b110u << 11)) >> 6)];
                    destination[targetIndex + 1] = byteToUnicodeEncoder.ByteToUnicode[(char)((c & 0x3Fu) + 0x80u)];
                    indexMapping[targetIndex] = indexMapping[targetIndex + 1] = i;
                    targetIndex += 2;
                    continue;
                }

                if (i < text.Length - 1 && char.IsSurrogatePair((char)c, text[i + 1]))
                {
                    // Scalar 000uuuuu zzzzyyyy yyxxxxxx -> bytes [ 11110uuu 10uuzzzz 10yyyyyy 10xxxxxx ]
                    uint value = (uint)char.ConvertToUtf32((char)c, text[i + 1]);
                    destination[targetIndex] = byteToUnicodeEncoder.ByteToUnicode[(char)((value + (0b11110 << 21)) >> 18)];
                    destination[targetIndex + 1] = byteToUnicodeEncoder.ByteToUnicode[(char)(((value & (0x3Fu << 12)) >> 12) + 0x80u)];
                    destination[targetIndex + 2] = byteToUnicodeEncoder.ByteToUnicode[(char)(((value & (0x3Fu << 6)) >> 6) + 0x80u)];
                    destination[targetIndex + 3] = byteToUnicodeEncoder.ByteToUnicode[(char)((value & 0x3Fu) + 0x80u)];
                    indexMapping[targetIndex] = indexMapping[targetIndex + 1] = indexMapping[targetIndex + 2] = indexMapping[targetIndex + 3] = i;
                    i++;
                    targetIndex += 4;
                    continue;
                }

                // Scalar zzzzyyyy yyxxxxxx -> bytes [ 1110zzzz 10yyyyyy 10xxxxxx ]
                destination[targetIndex] = byteToUnicodeEncoder.ByteToUnicode[(char)((c + (0b1110 << 16)) >> 12)];
                destination[targetIndex + 1] = byteToUnicodeEncoder.ByteToUnicode[(char)(((c & (0x3Fu << 6)) >> 6) + 0x80u)];
                destination[targetIndex + 2] = byteToUnicodeEncoder.ByteToUnicode[(char)((c & 0x3Fu) + 0x80u)];
                indexMapping[targetIndex] = indexMapping[targetIndex + 1] = indexMapping[targetIndex + 2] = i;
                targetIndex += 3;
            }

            return targetIndex;
        }

        public static bool ConvertUtf8ToUtf16(ReadOnlySpan<byte> utf8Bytes, Span<char> utf16Chars, out int bytesConsumed, out int charsWritten)
        {
            Debug.Assert(utf16Chars.Length >= GetUtf16LengthFromUtf8Bytes(utf8Bytes));

            int byteIndex = 0;
            int charIndex = 0;
            bytesConsumed = 0;
            charsWritten = 0;

            while (byteIndex < utf8Bytes.Length)
            {
                uint codePoint;
                int additionalBytes;

                byte firstByte = utf8Bytes[byteIndex];

                if ((firstByte & 0x80) == 0)
                {
                    // 1-byte sequence (ASCII)
                    codePoint = firstByte;
                    utf16Chars[charIndex++] = (char)firstByte;
                    charsWritten++;
                    bytesConsumed = ++byteIndex;
                    continue;
                }
                else if ((firstByte & 0xE0) == 0xC0)
                {
                    // 2-byte sequence
                    codePoint = (uint)(firstByte & 0x1F);
                    additionalBytes = 1;
                }
                else if ((firstByte & 0xF0) == 0xE0)
                {
                    // 3-byte sequence
                    codePoint = (uint)(firstByte & 0x0F);
                    additionalBytes = 2;
                }
                else if ((firstByte & 0xF8) == 0xF0)
                {
                    // 4-byte sequence
                    codePoint = (uint)(firstByte & 0x07);
                    additionalBytes = 3;
                }
                else
                {
                    return false;
                }

                if (byteIndex + additionalBytes >= utf8Bytes.Length)
                {
                    return true; // incomplete utf-8 sequence
                }

                for (int i = 1; i <= additionalBytes; i++)
                {
                    byte nextByte = utf8Bytes[byteIndex + i];
                    if ((nextByte & 0xC0) != 0x80)
                    {
                        return false;
                    }
                    codePoint = (codePoint << 6) | (uint)(nextByte & 0x3F);
                }

                byteIndex += additionalBytes + 1;
                bytesConsumed = byteIndex;

                if (codePoint <= 0xFFFF)
                {
                    utf16Chars[charIndex++] = (char)codePoint;
                }
                else
                {
                    codePoint -= 0x10000;
                    utf16Chars[charIndex++] = (char)((codePoint >> 10) + 0xD800);
                    utf16Chars[charIndex++] = (char)((codePoint & 0x3FF) + 0xDC00);
                }

                charsWritten = charIndex;
            }

            return true;
        }

        // encodedLength stores the number of bytes consumed after decoding.
        internal static int DecodeUtf8(ReadOnlySpan<byte> input, out int encodedLength)
        {
            Debug.Assert(input.Length > 0);

            if (input[0] < 0x80)
            {
                encodedLength = 1;
                return input[0];
            }
            else if (input.Length >= 2 && (input[0] & 0xE0) == 0xC0)
            {
                int cp = (((input[0] & 0x1F) << 6) | ((input[1] & 0x3F)));
                if (IsTrailByte(input[1]) && cp >= 0x0080 && IsValidCodepoint(cp))
                {
                    encodedLength = 2;
                    return cp;
                }
            }
            else if (input.Length >= 3 && (input[0] & 0xF0) == 0xE0)
            {
                int cp = (((input[0] & 0x0F) << 12) | ((input[1] & 0x3F) << 6) | ((input[2] & 0x3F)));
                if (IsTrailByte(input[1]) && IsTrailByte(input[2]) && cp >= 0x0800 && IsValidCodepoint(cp))
                {
                    encodedLength = 3;
                    return cp;
                }
            }
            else if (input.Length >= 4 && (input[0] & 0xf8) == 0xF0)
            {
                int cp = (((input[0] & 0x07) << 18) | ((input[1] & 0x3F) << 12) | ((input[2] & 0x3F) << 6) | ((input[3] & 0x3F)));
                if (IsTrailByte(input[1]) && IsTrailByte(input[2]) && IsTrailByte(input[3]) && cp >= 0x10000 && IsValidCodepoint(cp))
                {
                    encodedLength = 4;
                    return cp;
                }
            }

            // Invalid UTF-8.
            encodedLength = 1;
            return UnicodeError;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsTrailByte(byte x) => (sbyte)x < -0x40;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsValidCodepoint(int c) => ((uint)c) < 0xD800 || (c >= 0xE000 && c <= 0x10FFFF);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsValidDecodeUtf8(ReadOnlySpan<byte> input, out int encodedLength)
        {
            int c = DecodeUtf8(input, out encodedLength);
            return c != UnicodeError || encodedLength == 3;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static uint Swap32(uint x) => ((x & 0x000000FF) << 24) | ((x & 0x0000FF00) << 8) | ((x & 0x00FF0000) >> 8) | ((x & 0xFF000000) >> 24);
    }
}
