// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Microsoft.ML.Runtime.Internal.CpuMath;

namespace Microsoft.ML.CpuMath.PerformanceTests
{
    public class AvxPerformanceTests
    {
        private const int EXP_MAX = 127;
        private const int EXP_MIN = 0;

        private const int IDXLEN = 1000003;
        private const int LEN = 1000003;
        private const int EXP_RANGE = EXP_MAX / 2;
        private const int DEFAULT_SEED = 253421;
        private const float DEFAULT_SCALE = 1.11f;
        private const int DEFAULT_CROW = 500;
        private const int DEFAULT_CCOL = 2000;
        private const bool ADD = true;

        private float[] src, dst, original, src1, src2, result;
        private int[] idx;
        private int seed = DEFAULT_SEED;

        private static float NextFloat(Random rand, int expRange)
        {
            double mantissa = (rand.NextDouble() * 2.0) - 1.0;
            double exponent = Math.Pow(2.0, rand.Next(-expRange + 1, expRange + 1));
            return (float)(mantissa * exponent);
        }

        private static int GetSeed()
        {
            int seed = DEFAULT_SEED;

            if (Environment.GetEnvironmentVariable("CPUMATH_SEED") != null)
            {
                string CPUMATH_SEED = Environment.GetEnvironmentVariable("CPUMATH_SEED");

                if (!int.TryParse(CPUMATH_SEED, out seed))
                {
                    if (string.Equals(CPUMATH_SEED, "random", StringComparison.OrdinalIgnoreCase))
                    {
                        seed = new Random().Next();
                    }
                    else
                    {
                        seed = DEFAULT_SEED;
                    }
                }
            }

            Console.WriteLine("Random seed: " + seed + "; set environment variable CPUMATH_SEED to this value to reproduce results");

            return seed;
        }

        [GlobalSetup]
        public void Setup()
        {
            src = new float[LEN];
            dst = new float[LEN];
            src1 = new float[LEN];
            src2 = new float[LEN];
            original = new float[LEN];
            result = new float[LEN];
            idx = new int[IDXLEN];

            seed = GetSeed();
            Random rand = new Random(seed);

            for (int i = 0; i < LEN; i++)
            {
                src[i] = NextFloat(rand, EXP_RANGE);
                dst[i] = NextFloat(rand, EXP_RANGE);
                original[i] = dst[i];
                result[i] = dst[i];
                src1[i] = NextFloat(rand, EXP_RANGE);
                src2[i] = NextFloat(rand, EXP_RANGE);
            }

            for (int i = 0; i < IDXLEN; i++)
            {
                idx[i] = rand.Next(0, LEN);
            }
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            original.CopyTo(dst, 0);
            original.CopyTo(result, 0);
        }

        [Benchmark]
        public unsafe void NativeScaleXPerf()
        {
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.ScaleX(DEFAULT_SCALE, pdst, LEN);
            }
        }

        [Benchmark]
        public void ManagedScaleXPerf() => CpuMathUtils.Scale(DEFAULT_SCALE, dst, LEN);

        [Benchmark]
        public unsafe void NativeAddScaleXPerf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.AddScaleX(DEFAULT_SCALE, psrc, pdst, LEN);
            }
        }

        [Benchmark]
        public void ManagedAddScaleXPerf() => CpuMathUtils.AddScale(DEFAULT_SCALE, src, dst, LEN);

        [Benchmark]
        public unsafe void NativeAddXPerf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.AddX(psrc, pdst, LEN);
            }
        }

        [Benchmark]
        public void ManagedAddXPerf() => CpuMathUtils.Add(src, dst, LEN);

        [Benchmark]
        public unsafe float NativeSumXPerf()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.SumX(psrc, LEN);
            }
        }

        [Benchmark]
        public float ManagedSumXPerf() => CpuMathUtils.Sum(src, LEN);
    }
}
