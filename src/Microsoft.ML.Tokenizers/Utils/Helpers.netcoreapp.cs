// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers.Text;
using System.Diagnostics;
using System.Globalization;
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
        public static ValueTask<string?> ReadLineAsync(StreamReader reader, CancellationToken cancellationToken) =>
            reader.ReadLineAsync(cancellationToken);

        public static Task<Stream> GetStreamAsync(HttpClient client, string url, CancellationToken cancellationToken = default) =>
            client.GetStreamAsync(url, cancellationToken);

        public static Stream GetStream(HttpClient client, string url)
        {
            HttpResponseMessage response = client.Send(new HttpRequestMessage(HttpMethod.Get, url), HttpCompletionOption.ResponseHeadersRead);
            response.EnsureSuccessStatusCode();
            return response.Content.ReadAsStream();
        }

        public static byte[] FromBase64String(string base64String, int offset, int length)
        {
            if (!Base64.IsValid(base64String.AsSpan(offset, length), out int decodedLength))
            {
                throw new FormatException($"Invalid base64 string '{base64String.Substring(offset, length)}'");
            }

            byte[] bytes = new byte[decodedLength];
            bool success = Convert.TryFromBase64Chars(base64String.AsSpan(offset, length), bytes, out int bytesWritten);
            Debug.Assert(success);
            Debug.Assert(bytes.Length == bytesWritten);
            return bytes;
        }

        internal static bool TryParseInt32(string s, int offset, out int result)
            => int.TryParse(s.AsSpan().Slice(offset), NumberStyles.None, CultureInfo.InvariantCulture, out result);

        internal static int GetHashCode(ReadOnlySpan<char> span) => string.GetHashCode(span);

        internal static unsafe int GetUtf8Bytes(ReadOnlySpan<char> source, Span<byte> destination)
            => Encoding.UTF8.GetBytes(source, destination);

        internal static unsafe bool TryGetUtf8Bytes(ReadOnlySpan<char> source, Span<byte> destination, out int bytesWritten)
            => Encoding.UTF8.TryGetBytes(source, destination, out bytesWritten);

        internal static string GetString(ReadOnlySpan<byte> utf8Bytes)
            => Encoding.UTF8.GetString(utf8Bytes);

        internal static int GetChars(ReadOnlySpan<byte> bytes, Span<char> chars)
            => Encoding.UTF8.GetChars(bytes, chars);

        internal static void Replace(Span<char> span, char oldValue, char newValue) => span.Replace(oldValue, newValue);

        internal static void Replace(ReadOnlySpan<char> source, Span<char> destination, char oldValue, char newValue) => source.Replace(destination, oldValue, newValue);

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

            Rune.DecodeFromUtf16(text.Slice(textIndex), out Rune rune, out int charsConsumed);

            Span<byte> buffer = stackalloc byte[4]; // max number of bytes for a single code point utf-8 encoding.
            int bytesWritten = rune.EncodeToUtf8(buffer);
            if (bytesIndex + bytesWritten > destination.Length)
            {
                Helpers.ArrayPoolGrow(ref destination, destination.Length * 2);
            }

            buffer.Slice(0, bytesWritten).CopyTo(destination.AsSpan(bytesIndex));
            bytesIndex += bytesWritten;

            return charsConsumed;
        }
    }
}
