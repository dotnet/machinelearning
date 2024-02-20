// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers.Text;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Threading.Tasks;
using System.Threading;
using System.Net.Http;

namespace Microsoft.ML.Tokenizers
{
    internal static class Helpers
    {
        public static ValueTask<string?> ReadLineAsync(StreamReader reader, CancellationToken cancellationToken) =>
            reader.ReadLineAsync(cancellationToken);

        public static Task<Stream> GetStreamAsync(HttpClient client, string url, CancellationToken cancellationToken) =>
            client.GetStreamAsync(url, cancellationToken);

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
    }
}
