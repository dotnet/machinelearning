// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML
{
    /// <summary>
    /// Abstraction for RNG engines that expose the standard <see cref="Random"/> surface.
    /// </summary>
    public interface IRandomSource
    {
        int Next();
        int Next(int maxValue);
        int Next(int minValue, int maxValue);

        long NextInt64();
        long NextInt64(long maxValue);
        long NextInt64(long minValue, long maxValue);

        double NextDouble();
        float NextSingle();

        void NextBytes(Span<byte> buffer);
    }
}

