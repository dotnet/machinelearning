// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;

namespace Microsoft.ML.Internal.Utilities
{
    internal sealed class RandomFromRandomSource : Random
    {
        private readonly IRandomSource _source;

        public RandomFromRandomSource(IRandomSource source)
        {
            _source = source ?? throw new ArgumentNullException(nameof(source));
        }

        public override int Next() => _source.Next();

        public override int Next(int maxValue) => _source.Next(maxValue);

        public override int Next(int minValue, int maxValue) => _source.Next(minValue, maxValue);

        public override double NextDouble() => _source.NextDouble();

        public override void NextBytes(byte[] buffer)
        {
            if (buffer is null)
                throw new ArgumentNullException(nameof(buffer));

            _source.NextBytes(buffer);
        }

        protected override double Sample() => _source.NextDouble();

#if NET6_0_OR_GREATER
        public override void NextBytes(Span<byte> buffer) => _source.NextBytes(buffer);
        public override float NextSingle() => _source.NextSingle();
        public override long NextInt64() => _source.NextInt64();
        public override long NextInt64(long maxValue) => _source.NextInt64(maxValue);
        public override long NextInt64(long minValue, long maxValue) => _source.NextInt64(minValue, maxValue);
#endif
    }
}
