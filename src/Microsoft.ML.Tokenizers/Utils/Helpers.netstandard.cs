// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

#if Test
namespace Microsoft.ML.Tokenizers.Tests
#else
namespace Microsoft.ML.Tokenizers
#endif // Test
{
    internal static partial class Helpers
    {
        public static ValueTask<string> ReadLineAsync(StreamReader reader, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return new ValueTask<string>(reader.ReadLineAsync());
        }

        public static async Task<Stream> GetStreamAsync(HttpClient client, string url, CancellationToken cancellationToken = default)
        {
            HttpResponseMessage response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsStreamAsync().ConfigureAwait(false);
        }

        public static Stream GetStream(HttpClient client, string url) => client.GetStreamAsync(url).GetAwaiter().GetResult();

        public static byte[] FromBase64String(string base64String, int offset, int length) => Convert.FromBase64String(base64String.Substring(offset, length));

        // Not support signed number
        internal static bool TryParseInt32(string s, int offset, out int result)
        {
            result = 0;
            if ((uint)offset >= s.Length)
            {
                return false;
            }

            for (int i = offset; i < s.Length; i++)
            {
                if ((uint)(s[i] - '0') > ('9' - '0'))
                {
                    return false;
                }

                result = result * 10 + (s[i] - '0');
            }

            return true;
        }

        internal static int GetHashCode(ReadOnlySpan<char> span)
        {
            int hash = 17;
            foreach (char c in span)
            {
                hash = hash * 31 + c;
            }

            return hash;
        }

        internal static unsafe int GetUtf8Bytes(ReadOnlySpan<char> source, Span<byte> destination)
        {
            fixed (char* sourcePtr = source)
            fixed (byte* destPtr = destination)
            {
                return Encoding.UTF8.GetBytes(sourcePtr, source.Length, destPtr, destination.Length);
            }
        }

        internal static unsafe bool TryGetUtf8Bytes(ReadOnlySpan<char> source, Span<byte> destination, out int bytesWritten)
        {
            fixed (char* sourcePtr = source)
            fixed (byte* destPtr = destination)
            {
                if (Encoding.UTF8.GetByteCount(sourcePtr, source.Length) <= destination.Length)
                {
                    bytesWritten = Encoding.UTF8.GetBytes(sourcePtr, source.Length, destPtr, destination.Length);
                    return true;
                }

                bytesWritten = 0;
                return false;
            }
        }

        internal static unsafe string GetString(ReadOnlySpan<byte> utf8Bytes)
        {
            fixed (byte* sourcePtr = utf8Bytes)
            {
                return Encoding.UTF8.GetString(sourcePtr, utf8Bytes.Length);
            }
        }

        internal static unsafe int GetChars(ReadOnlySpan<byte> bytes, Span<char> chars)
        {
            fixed (byte* bytesPtr = bytes)
            fixed (char* charsPtr = chars)
            {
                return Encoding.UTF8.GetChars(bytesPtr, bytes.Length, charsPtr, chars.Length);
            }
        }

        internal static void Replace(Span<char> span, char oldValue, char newValue)
        {
            for (int i = 0; i < span.Length; i++)
                if (span[i] == oldValue)
                    span[i] = newValue;
        }

        internal static void Replace(ReadOnlySpan<char> source, Span<char> destination, char oldValue, char newValue)
        {
            Debug.Assert(source.Length <= destination.Length);

            for (int i = 0; i < source.Length; i++)
            {
                destination[i] = source[i] == oldValue ? newValue : source[i];
            }
        }

        /// <summary>
        /// Encode the next code point in the text to UTF-8.
        /// </summary>
        /// <param name="text">The text to encode the first code point from.</param>
        /// <param name="textIndex">The index of the first code point to encode.</param>
        /// <param name="destination">The buffer to write the UTF-8 bytes to.</param>
        /// <param name="bytesIndex">The index in the buffer to write the UTF-8 encoded bytes to.</param>
        /// <returns>The number of characters consumed from the text.</returns>
        internal static int EncodeCodePointToUtf8(ReadOnlySpan<char> text, int textIndex, ref byte[] destination, ref int bytesIndex)
        {
            Debug.Assert(textIndex < text.Length);

            uint c = (uint)text[textIndex];
            if (c <= 0x7Fu)
            {
                if (bytesIndex + 1 > destination.Length)
                {
                    Helpers.ArrayPoolGrow(ref destination, destination.Length * 2);
                }
                destination[bytesIndex] = (byte)c;
                bytesIndex++;
                return 1;
            }

            if (c <= 0x7FFu)
            {
                // Scalar 00000yyy yyxxxxxx -> bytes [ 110yyyyy 10xxxxxx ]
                if (bytesIndex + 2 > destination.Length)
                {
                    Helpers.ArrayPoolGrow(ref destination, destination.Length * 2);
                }
                destination[bytesIndex] = (byte)((c + (0b110u << 11)) >> 6);
                destination[bytesIndex + 1] = (byte)((c & 0x3Fu) + 0x80u);
                bytesIndex += 2;
                return 1;
            }

            if (textIndex < text.Length - 1 && char.IsSurrogatePair((char)c, text[textIndex + 1]))
            {
                // Scalar 000uuuuu zzzzyyyy yyxxxxxx -> bytes [ 11110uuu 10uuzzzz 10yyyyyy 10xxxxxx ]
                if (bytesIndex + 4 > destination.Length)
                {
                    Helpers.ArrayPoolGrow(ref destination, Math.Max(destination.Length, 4) * 2);
                }

                uint value = (uint)char.ConvertToUtf32((char)c, text[textIndex + 1]);
                destination[bytesIndex] = (byte)((value + (0b11110 << 21)) >> 18);
                destination[bytesIndex + 1] = (byte)(((value & (0x3Fu << 12)) >> 12) + 0x80u);
                destination[bytesIndex + 2] = (byte)(((value & (0x3Fu << 6)) >> 6) + 0x80u);
                destination[bytesIndex + 3] = (byte)((value & 0x3Fu) + 0x80u);
                bytesIndex += 4;
                return 2;
            }

            if (bytesIndex + 3 > destination.Length)
            {
                Helpers.ArrayPoolGrow(ref destination, Math.Max(destination.Length, 3) * 2);
            }

            // Scalar zzzzyyyy yyxxxxxx -> bytes [ 1110zzzz 10yyyyyy 10xxxxxx ]
            destination[bytesIndex] = (byte)((c + (0b1110 << 16)) >> 12);
            destination[bytesIndex + 1] = (byte)(((c & (0x3Fu << 6)) >> 6) + 0x80u);
            destination[bytesIndex + 2] = (byte)((c & 0x3Fu) + 0x80u);
            bytesIndex += 3;
            return 1;
        }
    }
}

