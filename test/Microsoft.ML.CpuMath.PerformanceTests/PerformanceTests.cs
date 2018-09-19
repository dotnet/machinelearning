// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Microsoft.ML.Runtime.Internal.CpuMath;

namespace Microsoft.ML.CpuMath.PerformanceTests
{
    public abstract class PerformanceTests
    {
        private const int ExponentMax = 127;
        private const int ExponentMin = 0;
        private const int ExponentRange = ExponentMax / 8;

        protected const int IndexLength = 1000003;
        protected const int Length = 1000003;
        
        private const int DefaultSeed = 253421;
        protected const float DefaultScale = 1.11f;

        protected float[] src, dst, original, src1, src2, result;
        protected int[] idx;

        private int _seed = DefaultSeed;

        private float NextFloat(Random rand, int expRange)
        {
            double mantissa = (rand.NextDouble() * 2.0) - 1.0;
            double exponent = Math.Pow(2.0, rand.Next(-expRange + 1, expRange + 1));
            return (float)(mantissa * exponent);
        }

        private int GetSeed()
        {
            int seed = DefaultSeed;
            string CPUMATH_SEED = Environment.GetEnvironmentVariable("CPUMATH_SEED");

            if (CPUMATH_SEED != null)
            {
                if (!int.TryParse(CPUMATH_SEED, out seed))
                {
                    if (string.Equals(CPUMATH_SEED, "random", StringComparison.OrdinalIgnoreCase))
                    {
                        seed = new Random().Next();
                    }
                    else
                    {
                        seed = DefaultSeed;
                    }
                }
            }

            Console.WriteLine("Random seed: " + seed + "; set environment variable CPUMATH_SEED to this value to reproduce results");
            return seed;
        }

        [GlobalSetup]
        public void Setup()
        {
            src = new float[Length];
            dst = new float[Length];
            src1 = new float[Length];
            src2 = new float[Length];
            original = new float[Length];
            result = new float[Length];
            idx = new int[IndexLength];

            _seed = GetSeed();
            Random rand = new Random(_seed);

            for (int i = 0; i < Length; i++)
            {
                src[i] = NextFloat(rand, ExponentRange);
                dst[i] = NextFloat(rand, ExponentRange);
                original[i] = dst[i];
                result[i] = dst[i];
                src1[i] = NextFloat(rand, ExponentRange);
                src2[i] = NextFloat(rand, ExponentRange);
            }

            for (int i = 0; i < IndexLength; i++)
            {
                idx[i] = rand.Next(0, Length);
            }
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            original.CopyTo(dst, 0);
            original.CopyTo(result, 0);
        }
    }
}