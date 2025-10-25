// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;

namespace Microsoft.ML.Internal.Utilities
{
    /// <summary>
    /// Back-ports newer <see cref="Random"/> APIs for TFMs where they are unavailable.
    /// </summary>
    internal static class RandomShim
    {
#if NET6_0_OR_GREATER
        public static long NextInt64(Random random) => random.NextInt64();

        public static long NextInt64(Random random, long maxValue) => random.NextInt64(maxValue);

        public static long NextInt64(Random random, long minValue, long maxValue) => random.NextInt64(minValue, maxValue);

        public static void NextBytes(Random random, Span<byte> buffer) => random.NextBytes(buffer);
#else
        public static long NextInt64(Random random) => NextInt64(random, long.MinValue, long.MaxValue);

        public static long NextInt64(Random random, long maxValue)
        {
            if (maxValue <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxValue));

            return NextInt64(random, 0, maxValue);
        }

        public static long NextInt64(Random random, long minValue, long maxValue)
        {
            if (minValue >= maxValue)
                throw new ArgumentOutOfRangeException(nameof(minValue));

            ulong range = (ulong)(maxValue - minValue);

            while (true)
            {
                ulong sample = NextUInt64(random);
                ulong threshold = ulong.MaxValue - (ulong.MaxValue % range);
                if (sample < threshold)
                    return (long)(sample % range) + minValue;
            }
        }

        public static void NextBytes(Random random, Span<byte> buffer)
        {
            if (buffer.IsEmpty)
                return;

            byte[] rented = ArrayPool<byte>.Shared.Rent(buffer.Length);
            try
            {
                random.NextBytes(rented);
                rented.AsSpan(0, buffer.Length).CopyTo(buffer);
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(rented);
            }
        }

        private static ulong NextUInt64(Random random)
        {
            Span<byte> bytes = stackalloc byte[8];
            NextBytes(random, bytes);

            ulong value = 0;
            for (int i = 0; i < bytes.Length; i++)
            {
                value |= (ulong)bytes[i] << (i * 8);
            }

            return value;
        }
#endif
    }
}

