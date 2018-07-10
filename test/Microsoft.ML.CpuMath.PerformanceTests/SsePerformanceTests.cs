using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Microsoft.ML.Runtime.Internal.CpuMath;

namespace Microsoft.ML.CpuMath.PerformanceTests
{
    public class SsePerformanceTests
    {
        internal const int EXP_MAX = 127;
        internal const int EXP_MIN = 0;
        internal static float NextFloat(Random rand, int expRange)
        {
            double mantissa = (rand.NextDouble() * 2.0) - 1.0;
            double exponent = Math.Pow(2.0, rand.Next(-expRange + 1, expRange + 1));
            return (float)(mantissa * exponent);
        }

        private float[] src, dst, original, src1, src2;
        private int[] idx;
        private readonly int idxlen = 1000003;
        private readonly int len = 1000003;
        private readonly int expRange = EXP_MAX / 2;
        private readonly int seed = 2;
        private readonly float scale = 1.11f;

        [GlobalSetup]
        public void Setup()
        {
            src = new float[len];
            dst = new float[len];
            src1 = new float[len];
            src2 = new float[len];
            original = new float[len];
            Random rand = new Random(seed);
            idx = new int[idxlen];

            for (int i = 0; i < len; i++)
            {
                src[i] = NextFloat(rand, expRange);
                dst[i] = NextFloat(rand, expRange);
                original[i] = dst[i];
                src1[i] = NextFloat(rand, expRange);
                src2[i] = NextFloat(rand, expRange);
            }

            for (int i = 0; i < idxlen; i++)
            {
                idx[i] = rand.Next(0, len);
            }
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            original.CopyTo(dst, 0);
        }

        [Benchmark]
        public unsafe float NativeDotUPerf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                return CpuMathNativeUtils.DotU(psrc, pdst, len);
            }
        }

        [Benchmark]
        public float MyDotUPerf() => CpuMathUtils.DotProductDense(src, dst, len);

        [Benchmark]
        public unsafe float NativeDotSUPerf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (int* pidx = idx)
            {
                return CpuMathNativeUtils.DotSU(psrc, pdst, pidx, idxlen);
            }
        }

        [Benchmark]
        public float MyDotSUPerf() => CpuMathUtils.DotProductSparse(src, dst, idx, idxlen);

        [Benchmark]
        public unsafe float NativeSumSqUPerf()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.SumSqU(psrc, len);
            }
        }

        [Benchmark]
        public float MySumSqUPerf() => CpuMathUtils.SumSq(src, len);

        [Benchmark]
        public unsafe void NativeAddUPerf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.AddU(psrc, pdst, len);
            }
        }

        [Benchmark]
        public void MyAddUPerf() => CpuMathUtils.Add(src, dst, len);

        [Benchmark]
        public unsafe void NativeAddSUPerf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (int* pidx = idx)
            {
                CpuMathNativeUtils.AddSU(psrc, pidx, pdst, idxlen);
            }
        }

        [Benchmark]
        public void MyAddSUPerf() => CpuMathUtils.Add(src, idx, dst, idxlen);

        [Benchmark]
        public unsafe void NativeAddScaleUPerf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.AddScaleU(scale, psrc, pdst, len);
            }
        }

        [Benchmark]
        public void MyAddScaleUPerf() => CpuMathUtils.AddScale(scale, src, dst, len);

        [Benchmark]
        public unsafe void NativeAddScaleSUPerf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (int* pidx = idx)
            {
                CpuMathNativeUtils.AddScaleSU(scale, psrc, pidx, pdst, idxlen);
            }
        }

        [Benchmark]
        public void MyAddScaleSUPerf() => CpuMathUtils.AddScale(scale, src, idx, dst, idxlen);

        [Benchmark]
        public unsafe void NativeScaleUPerf()
        {
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.ScaleU(scale, pdst, len);
            }
        }

        [Benchmark]
        public void MyScaleUPerf() => CpuMathUtils.Scale(scale, dst, len);

        [Benchmark]
        public unsafe float NativeDist2Perf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                return CpuMathNativeUtils.Dist2(psrc, pdst, len);
            }
        }

        [Benchmark]
        public float MyDist2Perf() => CpuMathUtils.L2DistSquared(src, dst, len);

        [Benchmark]
        public unsafe float NativeSumAbsUPerf()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.SumAbsU(psrc, len);
            }
        }

        [Benchmark]
        public float MySumAbsqUPerf() => CpuMathUtils.SumAbs(src, len);

        [Benchmark]
        public unsafe void NativeMulElementWiseUPerf()
        {
            fixed (float* psrc1 = src1)
            fixed (float* psrc2 = src2)
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.MulElementWiseU(psrc1, psrc2, pdst, len);
            }
        }

        [Benchmark]
        public void MyMulElementWiseUPerf() => CpuMathUtils.MulElementWise(src1, src2, dst, len);
    }
}
