// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>Used as a key in a dictionary to enable querying with either a string or a span.</summary>
    /// <remarks>
    /// This should only be used with a Ptr/Length for querying. For storing in a dictionary, this should
    /// always be used with a string.
    /// </remarks>
    internal unsafe readonly struct StringSpanOrdinalKey : IEquatable<StringSpanOrdinalKey>
    {
        public readonly char* Ptr;
        public readonly int Length;
        public readonly string? Data;

        public StringSpanOrdinalKey(char* ptr, int length)
        {
            Ptr = ptr;
            Length = length;
        }

        public StringSpanOrdinalKey(string data) =>
            Data = data;

        private ReadOnlySpan<char> Span => Ptr is not null ?
            new ReadOnlySpan<char>(Ptr, Length) :
            Data.AsSpan();

        public override bool Equals(object? obj) =>
            obj is StringSpanOrdinalKey wrapper && Equals(wrapper);

        public bool Equals(StringSpanOrdinalKey other) =>
            Span.SequenceEqual(other.Span);

        public override int GetHashCode()
        {
#if NET5_0_OR_GREATER
            return string.GetHashCode(Span);
#else
            int hash = 17;
            foreach (char c in Span)
            {
                hash = hash * 31 + c;
            }

            return hash;
#endif
        }
    }
}
