// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.RunTests;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Core.Tests.UnitTests
{
    public sealed class MersenneTwisterTests : BaseTestBaseline
    {
        public MersenneTwisterTests(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        [TestCategory("Utilities")]
        public void MixedApiCallsConsumeTemperedSequenceWithoutGaps()
        {
            const uint seed = 5489u;

            var baseline = new uint[32];
            var baselineTwister = new MersenneTwister(seed);
            baselineTwister.NextTemperedUInt32(baseline);

            var twister = new MersenneTwister(seed);
            var index = 0;

            Assert.Equal(baseline[index++], twister.NextTemperedUInt32());

            var firstDouble = twister.NextDouble();
            Assert.Equal(ToDoubleFromTempered(baseline[index], baseline[index + 1]), firstDouble);
            index += 2;

            var buffer = new uint[5];
            twister.NextTemperedUInt32(buffer);
            Assert.Equal(Slice(baseline, index, buffer.Length), buffer);
            index += buffer.Length;

            Assert.Equal(baseline[index++], twister.NextTemperedUInt32());

            var secondDouble = twister.NextDouble();
            Assert.Equal(ToDoubleFromTempered(baseline[index], baseline[index + 1]), secondDouble);
            index += 2;

            var secondBuffer = new uint[3];
            twister.NextTemperedUInt32(secondBuffer);
            Assert.Equal(Slice(baseline, index, secondBuffer.Length), secondBuffer);
            index += secondBuffer.Length;

            var thirdDouble = twister.NextDouble();
            Assert.Equal(ToDoubleFromTempered(baseline[index], baseline[index + 1]), thirdDouble);
            index += 2;

            var thirdBuffer = new uint[4];
            twister.NextTemperedUInt32(thirdBuffer);
            Assert.Equal(Slice(baseline, index, thirdBuffer.Length), thirdBuffer);
            index += thirdBuffer.Length;

            Assert.Equal(baseline[index++], twister.NextTemperedUInt32());

            Assert.True(index <= baseline.Length);
        }

        [Fact]
        [TestCategory("Utilities")]
        public void ProducesExpectedSequencesForDeterministicSeed()
        {
            const uint seed = 5489u;

            var expectedDoubles = new[]
            {
                0.8147236863931789,
                0.9057919370756192,
                0.12698681629350606,
                0.9133758561390194,
                0.6323592462254095,
                0.09754040499940952,
                0.2784982188670484,
                0.5468815192049838,
                0.9575068354342976,
                0.9648885351992765,
            };

            var expectedTempered = new uint[]
            {
                3499211612,
                581869302,
                3890346734,
                3586334585,
                545404204,
                4161255391,
                3922919429,
                949333985,
                2715962298,
                1323567403,
            };

            var doubleTwister = new MersenneTwister(seed);
            var actualDoubles = new double[expectedDoubles.Length];
            for (var i = 0; i < actualDoubles.Length; i++)
            {
                actualDoubles[i] = doubleTwister.NextDouble();
            }

            var uintTwister = new MersenneTwister(seed);
            var actualTempered = new uint[expectedTempered.Length];
            for (var i = 0; i < actualTempered.Length; i++)
            {
                actualTempered[i] = uintTwister.NextTemperedUInt32();
            }

            for (var i = 0; i < expectedDoubles.Length; i++)
            {
                Assert.Equal(expectedDoubles[i], actualDoubles[i], precision: 15);
            }

            Assert.Equal(expectedTempered, actualTempered);
        }

        private static uint[] Slice(uint[] source, int start, int length)
        {
            var result = new uint[length];
            Array.Copy(source, start, result, 0, length);
            return result;
        }

        private static double ToDoubleFromTempered(uint first, uint second)
        {
            var a = (ulong)(first >> 5);
            var b = (ulong)(second >> 6);
            var mantissa = (a << 26) | b;
            const double inverseTwo53 = 1.0 / 9007199254740992.0;
            return mantissa * inverseTwo53;
        }
    }
}
