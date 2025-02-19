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
}
