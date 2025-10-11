// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;

namespace Microsoft.ML.Internal.Utilities
{
    internal sealed class MersenneTwisterRandomSource : IRandomSource, IRandomBulkSource
    {
        private readonly MersenneTwister _twister;

        public MersenneTwisterRandomSource(int seed)
        {
            _twister = new MersenneTwister((uint)seed);
        }

        public int Next()
        {
            return (int)(_twister.NextTemperedUInt32() >> 1);
        }

        public int Next(int maxValue)
        {
            if (maxValue <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxValue));

            uint limit = (uint)maxValue;
            uint threshold = uint.MaxValue - (uint.MaxValue % limit);

            uint sample;
            do
            {
                sample = _twister.NextTemperedUInt32();
            }
            while (sample >= threshold);

            return (int)(sample % limit);
        }

        public int Next(int minValue, int maxValue)
        {
            if (minValue >= maxValue)
                throw new ArgumentOutOfRangeException(nameof(minValue));

            uint range = unchecked((uint)(maxValue - minValue));
            int offset = NextInt32InRange(range);
            return minValue + offset;
        }

        public long NextInt64()
        {
            return unchecked((long)NextUInt64());
        }

        public long NextInt64(long maxValue)
        {
            if (maxValue <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxValue));

            return (long)NextUInt64InRange((ulong)maxValue);
        }

        public long NextInt64(long minValue, long maxValue)
        {
            if (minValue >= maxValue)
                throw new ArgumentOutOfRangeException(nameof(minValue));

            ulong range = (ulong)(maxValue - minValue);
            long offset = (long)NextUInt64InRange(range);
            return minValue + offset;
        }

        public double NextDouble()
        {
            return _twister.NextDouble();
        }

        public float NextSingle()
        {
            uint word = _twister.NextTemperedUInt32();
            return (word >> 8) * (1.0f / (1u << 24));
        }

        public void NextBytes(Span<byte> buffer)
        {
            int offset = 0;

            Span<uint> word = stackalloc uint[1];
            while (offset < buffer.Length)
            {
                _twister.NextTemperedUInt32(word);
                uint value = word[0];

                int bytesToCopy = Math.Min(4, buffer.Length - offset);
                for (int i = 0; i < bytesToCopy; i++)
                {
                    buffer[offset++] = (byte)value;
                    value >>= 8;
                }
            }
        }

        public void NextDoubles(Span<double> destination)
        {
            _twister.NextDoubles(destination);
        }

        public void NextUInt32(Span<uint> destination)
        {
            _twister.NextTemperedUInt32(destination);
        }

        private ulong NextUInt64()
        {
            Span<uint> words = stackalloc uint[2];
            _twister.NextTemperedUInt32(words);
            return ((ulong)words[0] << 32) | words[1];
        }

        private int NextInt32InRange(uint range)
        {
            if (range == 0)
                throw new ArgumentOutOfRangeException(nameof(range));

            uint threshold = uint.MaxValue - (uint.MaxValue % range);

            uint sample;
            do
            {
                sample = _twister.NextTemperedUInt32();
            }
            while (sample >= threshold);

            return (int)(sample % range);
        }

        private ulong NextUInt64InRange(ulong range)
        {
            if (range == 0)
                throw new ArgumentOutOfRangeException(nameof(range));

            ulong threshold = ulong.MaxValue - (ulong.MaxValue % range);

            ulong sample;
            do
            {
                sample = NextUInt64();
            }
            while (sample >= threshold);

            return sample % range;
        }
    }
}
