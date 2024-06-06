// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Diagnostics;
using System.Text;

namespace Microsoft.ML.Tokenizers
{
    internal static partial class Helpers
    {
        internal static void ArrayPoolGrow<T>(ref T[] arrayPoolArray, int requiredCapacity)
        {
            T[] tmp = ArrayPool<T>.Shared.Rent(Math.Max(arrayPoolArray.Length * 2, requiredCapacity));
            arrayPoolArray.CopyTo(tmp.AsSpan());
            ArrayPool<T>.Shared.Return(arrayPoolArray);
            arrayPoolArray = tmp;
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
    }
}
