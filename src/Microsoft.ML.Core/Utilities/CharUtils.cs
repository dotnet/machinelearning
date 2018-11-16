// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using System;
using System.Runtime.CompilerServices;
using System.Threading;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    [BestFriend]
    internal static class CharUtils
    {
        private const int CharsCount = 0x10000;
        private static volatile char[] _lowerInvariantChars;
        private static volatile char[] _upperInvariantChars;

        private static char[] EnsureLowerInvariant()
        {
            var lower = new char[CharsCount];
            for (int i = 0; i < lower.Length; i++)
                lower[i] = char.ToLowerInvariant((char)i);
            Interlocked.CompareExchange(ref _lowerInvariantChars, lower, null);
            return _lowerInvariantChars;
        }

        private static char[] EnsureUpperInvariant()
        {
            var upper = new char[CharsCount];
            for (int i = 0; i < upper.Length; i++)
                upper[i] = char.ToUpperInvariant((char)i);
            Interlocked.CompareExchange(ref _upperInvariantChars, upper, null);
            return _upperInvariantChars;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static char ToLowerInvariant(char c)
        {
            var lower = _lowerInvariantChars ?? EnsureLowerInvariant();
            return lower[c];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static char ToUpperInvariant(char c)
        {
            var upper = _upperInvariantChars ?? EnsureUpperInvariant();
            return upper[c];
        }
    }
}
