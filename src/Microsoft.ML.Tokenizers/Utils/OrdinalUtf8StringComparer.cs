// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Tokenizers
{
    internal class OrdinalUtf8StringComparer : IComparer<string>
    {
        internal static readonly OrdinalUtf8StringComparer Instance = new OrdinalUtf8StringComparer();
        public int Compare(string? x, string? y)
        {
            if (x == null || y == null)
            {
                return x == y ? 0 : (x == null ? -1 : 1);
            }

            Span<byte> buffer1 = stackalloc byte[520];
            Span<byte> buffer2 = stackalloc byte[520];

            int minLength = Math.Min(x.Length, y.Length);
            for (int i = 0; i < minLength; i++)
            {
                char c = x[i];
                char d = y[i];

                if (c == d)
                {
                    continue;
                }

                if (!Char.IsSurrogate(c) && !Char.IsSurrogate(d))
                {
                    return (int)x[i] - (int)y[i];
                }

                // Need to consider surrogate conversions to UTF-8 before comparing.

                while (i > 0 && (Char.IsSurrogate(x[i - 1]) || Char.IsSurrogate(y[i - 1])))
                {
                    i--;
                }

                int xLen = x.Length - i;
                int yLen = y.Length - i;

                byte[]? bytes1 = null;
                byte[]? bytes2 = null;

                int requiredLength1 = Encoding.UTF8.GetMaxByteCount(xLen);
                int requiredLength2 = Encoding.UTF8.GetMaxByteCount(yLen);

                if (requiredLength1 > buffer1.Length)
                {
                    bytes1 = ArrayPool<byte>.Shared.Rent(requiredLength1);
                    buffer1 = bytes1;
                }

                if (requiredLength2 > buffer2.Length)
                {
                    bytes2 = ArrayPool<byte>.Shared.Rent(requiredLength2);
                    buffer2 = bytes2;
                }

                xLen = Helpers.EncodeToUtf8(x.AsSpan(i), buffer1);
                yLen = Helpers.EncodeToUtf8(y.AsSpan(i), buffer2);

                int result = ByteArrayComparer.Instance.Compare(buffer1.Slice(0, xLen), buffer2.Slice(0, yLen));

                if (bytes1 != null)
                {
                    ArrayPool<byte>.Shared.Return(bytes1);
                }

                if (bytes2 != null)
                {
                    ArrayPool<byte>.Shared.Return(bytes2);
                }

                return result;
            }

            return x.Length - y.Length;
        }
    }
}
