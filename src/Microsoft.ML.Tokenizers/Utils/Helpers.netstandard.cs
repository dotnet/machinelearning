// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Microsoft.ML.Tokenizers
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
    }
}

