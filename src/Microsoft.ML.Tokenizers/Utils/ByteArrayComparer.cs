// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Tokenizers
{
    internal sealed class ReadOnlyMemoryByteComparer : IEqualityComparer<ReadOnlyMemory<byte>>
    {
        public static ReadOnlyMemoryByteComparer Instance { get; } = new();

        public bool Equals(ReadOnlyMemory<byte> x, ReadOnlyMemory<byte> y) =>
            x.Span.SequenceEqual(y.Span);

        public int GetHashCode(ReadOnlyMemory<byte> x)
        {
            int hash = 17;
            foreach (byte b in x.Span)
            {
                hash = hash * 31 + b;
            }

            return hash;
        }
    }

    internal class ByteArrayComparer : IComparer<byte[]>
    {
        internal static readonly ByteArrayComparer Instance = new ByteArrayComparer();

        public int Compare(Span<byte> x, Span<byte> y)
        {
            int minLength = Math.Min(x.Length, y.Length);
            for (int i = 0; i < minLength; i++)
            {
                if (x[i] == y[i])
                {
                    continue;
                }

                return (int)x[i] - (int)y[i];
            }

            return x.Length - y.Length;
        }

        public int Compare(byte[]? x, byte[]? y)
        {
            if (x == null || y == null)
            {
                return x == y ? 0 : (x == null ? -1 : 1);
            }

            return Compare(x.AsSpan(), y.AsSpan());
        }
    }
}
