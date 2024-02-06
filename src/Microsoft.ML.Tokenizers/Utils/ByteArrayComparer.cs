// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Tokenizers
{
    internal class ByteArrayComparer : IEqualityComparer<byte[]>
    {
        public bool Equals(byte[]? x, byte[]? y)
        {
            if (x is null || y is null)
            {
                return x == y;
            }

            return x.SequenceEqual(y);
        }

        public int GetHashCode(byte[] bytes)
        {
            if (bytes == null)
            {
                throw new ArgumentNullException(nameof(bytes));
            }

            int hash = 17;
            foreach (byte b in bytes)
            {
                hash = hash * 31 + b;
            }

            return hash;
        }
    }
}