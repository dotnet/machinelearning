// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;

namespace Microsoft.ML.Internal.Utilities
{
    internal sealed class RandomSourceAdapter : IRandomSource
    {
        private readonly Random _random;

        public RandomSourceAdapter(Random random)
        {
            _random = random ?? throw new ArgumentNullException(nameof(random));
        }

        public int Next() => _random.Next();

        public int Next(int maxValue) => _random.Next(maxValue);

        public int Next(int minValue, int maxValue) => _random.Next(minValue, maxValue);

        public long NextInt64()
        {
#if NET6_0_OR_GREATER
            return _random.NextInt64();
#else
            return RandomShim.NextInt64(_random);
#endif
        }

        public long NextInt64(long maxValue)
        {
#if NET6_0_OR_GREATER
            return _random.NextInt64(maxValue);
#else
            return RandomShim.NextInt64(_random, maxValue);
#endif
        }

        public long NextInt64(long minValue, long maxValue)
        {
#if NET6_0_OR_GREATER
            return _random.NextInt64(minValue, maxValue);
#else
            return RandomShim.NextInt64(_random, minValue, maxValue);
#endif
        }

        public double NextDouble() => _random.NextDouble();

        public float NextSingle()
        {
#if NET6_0_OR_GREATER
            return _random.NextSingle();
#else
            return RandomUtils.NextSingle(_random);
#endif
        }

        public void NextBytes(Span<byte> buffer)
        {
#if NET6_0_OR_GREATER
            _random.NextBytes(buffer);
#else
            RandomShim.NextBytes(_random, buffer);
#endif
        }
    }
}
