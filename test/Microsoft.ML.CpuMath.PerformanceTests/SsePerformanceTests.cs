// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Microsoft.ML.Runtime.Internal.CpuMath;

namespace Microsoft.ML.CpuMath.PerformanceTests
{
    public class SsePerformanceTests
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

        // Naming follows from SseIntrinsics.
        private const int CbAlign = 16;

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
                    if(string.Equals(CPUMATH_SEED, "random", StringComparison.OrdinalIgnoreCase))
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
        public unsafe float NativeAddScalarUPerf()
        {
            fixed (float* pdst = dst)
            {
                return CpuMathNativeUtils.AddScalarU(DEFAULT_SCALE, pdst, LEN);
            }
        }

        [Benchmark]
        public void ManagedAddScalarUPerf() => CpuMathUtils.Add(DEFAULT_SCALE, dst, LEN);

        [Benchmark]
        public unsafe void NativeScaleUPerf()
        {
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.ScaleU(DEFAULT_SCALE, pdst, LEN);
            }
        }

        [Benchmark]
        public void ManagedScaleUPerf() => CpuMathUtils.Scale(DEFAULT_SCALE, dst, LEN);

        [Benchmark]
        public unsafe void NativeScaleSrcUPerf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.ScaleSrcU(DEFAULT_SCALE, psrc, pdst, LEN);
            }
        }

        [Benchmark]
        public void ManagedScaleSrcUPerf() => CpuMathUtils.Scale(DEFAULT_SCALE, src, dst, LEN);

        [Benchmark]
        public unsafe void NativeScaleAddUPerf()
        {
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.ScaleAddU(DEFAULT_SCALE, DEFAULT_SCALE, pdst, LEN);
            }
        }

        [Benchmark]
        public void ManagedScaleAddUPerf() => CpuMathUtils.ScaleAdd(DEFAULT_SCALE, DEFAULT_SCALE, dst, LEN);

        [Benchmark]
        public unsafe void NativeAddScaleUPerf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.AddScaleU(DEFAULT_SCALE, psrc, pdst, LEN);
            }
        }

        [Benchmark]
        public void ManagedAddScaleUPerf() => CpuMathUtils.AddScale(DEFAULT_SCALE, src, dst, LEN);

        [Benchmark]
        public unsafe void NativeAddScaleSUPerf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (int* pidx = idx)
            {
                CpuMathNativeUtils.AddScaleSU(DEFAULT_SCALE, psrc, pidx, pdst, IDXLEN);
            }
        }

        [Benchmark]
        public void ManagedAddScaleSUPerf() => CpuMathUtils.AddScale(DEFAULT_SCALE, src, idx, dst, IDXLEN);

        [Benchmark]
        public unsafe void NativeAddScaleCopyUPerf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (float* pres = result)
            {
                CpuMathNativeUtils.AddScaleCopyU(DEFAULT_SCALE, psrc, pdst, pres, LEN);
            }
        }

        [Benchmark]
        public void ManagedAddScaleCopyUPerf() => CpuMathUtils.AddScaleCopy(DEFAULT_SCALE, src, dst, result, LEN);

        [Benchmark]
        public unsafe void NativeAddUPerf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.AddU(psrc, pdst, LEN);
            }
        }

        [Benchmark]
        public void ManagedAddUPerf() => CpuMathUtils.Add(src, dst, LEN);

        [Benchmark]
        public unsafe void NativeAddSUPerf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (int* pidx = idx)
            {
                CpuMathNativeUtils.AddSU(psrc, pidx, pdst, IDXLEN);
            }
        }

        [Benchmark]
        public void ManagedAddSUPerf() => CpuMathUtils.Add(src, idx, dst, IDXLEN);


        [Benchmark]
        public unsafe void NativeMulElementWiseUPerf()
        {
            fixed (float* psrc1 = src1)
            fixed (float* psrc2 = src2)
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.MulElementWiseU(psrc1, psrc2, pdst, LEN);
            }
        }

        [Benchmark]
        public void ManagedMulElementWiseUPerf() => CpuMathUtils.MulElementWise(src1, src2, dst, LEN);

        [Benchmark]
        public unsafe float NativeSumUPerf()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.SumU(psrc, LEN);
            }
        }

        [Benchmark]
        public float ManagedSumUPerf() => CpuMathUtils.Sum(src, LEN);

        [Benchmark]
        public unsafe float NativeSumSqUPerf()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.SumSqU(psrc, LEN);
            }
        }

        [Benchmark]
        public float ManagedSumSqUPerf() => CpuMathUtils.SumSq(src, LEN);

        [Benchmark]
        public unsafe float NativeSumSqDiffUPerf()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.SumSqDiffU(DEFAULT_SCALE, psrc, LEN);
            }
        }

        [Benchmark]
        public float ManagedSumSqDiffUPerf() => CpuMathUtils.SumSq(DEFAULT_SCALE, src, 0, LEN);

        [Benchmark]
        public unsafe float NativeSumAbsUPerf()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.SumAbsU(psrc, LEN);
            }
        }

        [Benchmark]
        public float ManagedSumAbsUPerf() => CpuMathUtils.SumAbs(src, LEN);

        [Benchmark]
        public unsafe float NativeSumAbsDiffUPerf()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.SumAbsDiffU(DEFAULT_SCALE, psrc, LEN);
            }
        }

        [Benchmark]
        public float ManagedSumAbsDiffUPerf() => CpuMathUtils.SumAbs(DEFAULT_SCALE, src, 0, LEN);

        [Benchmark]
        public unsafe float NativeMaxAbsUPerf()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.MaxAbsU(psrc, LEN);
            }
        }

        [Benchmark]
        public float ManagedMaxAbsUPerf() => CpuMathUtils.MaxAbs(src, LEN);

        [Benchmark]
        public unsafe float NativeMaxAbsDiffUPerf()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.MaxAbsDiffU(DEFAULT_SCALE, psrc, LEN);
            }
        }

        [Benchmark]
        public float ManagedMaxAbsDiffUPerf() => CpuMathUtils.MaxAbsDiff(DEFAULT_SCALE, src, LEN);
        // TODO: MaxAbsU!!!

        [Benchmark]
        public unsafe float NativeDotUPerf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                return CpuMathNativeUtils.DotU(psrc, pdst, LEN);
            }
        }

        [Benchmark]
        public float ManagedDotUPerf() => CpuMathUtils.DotProductDense(src, dst, LEN);

        [Benchmark]
        public unsafe float NativeDotSUPerf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (int* pidx = idx)
            {
                return CpuMathNativeUtils.DotSU(psrc, pdst, pidx, IDXLEN);
            }
        }

        [Benchmark]
        public float ManagedDotSUPerf() => CpuMathUtils.DotProductSparse(src, dst, idx, IDXLEN);

        [Benchmark]
        public unsafe float NativeDist2Perf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                return CpuMathNativeUtils.Dist2(psrc, pdst, LEN);
            }
        }

        [Benchmark]
        public float ManagedDist2Perf() => CpuMathUtils.L2DistSquared(src, dst, LEN);

        [Benchmark]
        public unsafe void NativeSdcaL1UpdateUPerf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (float* pres = result)
            {
                CpuMathNativeUtils.SdcaL1UpdateU(DEFAULT_SCALE, psrc, DEFAULT_SCALE, pdst, pres, LEN);
            }
        }

        [Benchmark]
        public void ManagedSdcaL1UpdateUPerf() => CpuMathUtils.SdcaL1UpdateDense(DEFAULT_SCALE, LEN, src, DEFAULT_SCALE, dst, result);

        [Benchmark]
        public unsafe void NativeSdcaL1UpdateSUPerf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (float* pres = result)
            fixed (int* pidx = idx)
            {
                CpuMathNativeUtils.SdcaL1UpdateSU(DEFAULT_SCALE, psrc, pidx, DEFAULT_SCALE, pdst, pres, IDXLEN);
            }
        }

        [Benchmark]
        public void ManagedSdcaL1UpdateSUPerf() => CpuMathUtils.SdcaL1UpdateSparse(DEFAULT_SCALE, LEN, src, idx, IDXLEN, DEFAULT_SCALE, dst, result);
    }
}
