// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Globalization;

namespace Microsoft.ML.Tokenizers
{
    internal static class Helpers
    {
        public static byte[] FromBase64String(string base64String, int offset, int length)
        {
            Span<byte> bytes = stackalloc byte[300];
            if (!Convert.TryFromBase64Chars(base64String.AsSpan().Slice(offset, length), bytes, out int bytesWritten))
            {
                throw new System.FormatException($"Invalid base64 string '{base64String.Substring(offset, length)}'");
            }
            return bytes.Slice(0, bytesWritten).ToArray();
        }

        internal static bool TryParseInt32(string s, int offset, out int result)
            => int.TryParse(s.AsSpan().Slice(offset), NumberStyles.None, CultureInfo.InvariantCulture, out result);
    }
}

