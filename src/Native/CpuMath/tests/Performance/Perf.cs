using System;
using System.Runtime.InteropServices;
using System.Security;
using Intrinsics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

namespace PerformanceTests
{
    public class SsePerf
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

        public SsePerf()
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

        [DllImport("CpuMathNative", EntryPoint = "DotU"), SuppressUnmanagedCodeSecurity]
        internal unsafe static extern float NativeDotU(/*const*/ float* pa, /*const*/ float* pb, int c);

        [DllImport("CpuMathNative", EntryPoint = "DotSU"), SuppressUnmanagedCodeSecurity]
        internal unsafe static extern float NativeDotSU(/*const*/ float* pa, /*const*/ float* pb, /*const*/ int* pi, int c);

        [DllImport("CpuMathNative", EntryPoint = "SumSqU"), SuppressUnmanagedCodeSecurity]
        internal unsafe static extern float NativeSumSqU(/*const*/ float* ps, int c);

        [DllImport("CpuMathNative", EntryPoint = "AddU"), SuppressUnmanagedCodeSecurity]
        internal unsafe static extern void NativeAddU(/*_In_ const*/ float* ps, /*_Inout_*/ float* pd, int c);

        [DllImport("CpuMathNative", EntryPoint = "AddSU"), SuppressUnmanagedCodeSecurity]
        internal unsafe static extern void NativeAddSU(/*_In_ const*/ float* ps, /*_In_ const*/ int* pi, /*_Inout_*/ float* pd, int c);

        [DllImport("CpuMathNative", EntryPoint = "AddScaleU"), SuppressUnmanagedCodeSecurity]
        internal unsafe static extern void NativeAddScaleU(float a, /*_In_ const*/ float* ps, /*_Inout_*/ float* pd, int c);

        [DllImport("CpuMathNative", EntryPoint = "AddScaleSU"), SuppressUnmanagedCodeSecurity]
        internal unsafe static extern void NativeAddScaleSU(float a, /*_In_ const*/ float* ps, /*_In_ const*/ int* pi, /*_Inout_*/ float* pd, int c);

        [DllImport("CpuMathNative", EntryPoint = "ScaleU"), SuppressUnmanagedCodeSecurity]
        internal unsafe static extern void NativeScaleU(float a, /*_Inout_*/ float* pd, int c);

        [DllImport("CpuMathNative", EntryPoint = "Dist2"), SuppressUnmanagedCodeSecurity]
        internal unsafe static extern float NativeDist2(/*const*/ float* px, /*const*/ float* py, int c);

        [DllImport("CpuMathNative", EntryPoint = "SumAbsU"), SuppressUnmanagedCodeSecurity]
        internal unsafe static extern float NativeSumAbsU(/*const*/ float* ps, int c);

        [DllImport("CpuMathNative", EntryPoint = "MulElementWiseU"), SuppressUnmanagedCodeSecurity]
        internal unsafe static extern void NativeMulElementWiseU(/*_In_ const*/ float* ps1, /*_In_ const*/ float* ps2, /*_Inout_*/ float* pd, int c);

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
                return NativeDotU(psrc, pdst, len);
            }
        }

        [Benchmark]
        public float MyDotUPerf() => IntrinsicsUtils.DotU(src, dst);

        [Benchmark]
        public unsafe float NativeDotSUPerf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (int* pidx = idx)
            {
                return NativeDotSU(psrc, pdst, pidx, idxlen);
            }
        }

        [Benchmark]
        public float MyDotSUPerf() => IntrinsicsUtils.DotSU(src, dst, idx);

        [Benchmark]
        public unsafe float NativeSumSqUPerf()
        {
            fixed (float* psrc = src)
            {
                return NativeSumSqU(psrc, len);
            }
        }

        [Benchmark]
        public float MySumSqUPerf() => IntrinsicsUtils.SumSqU(src);

        [Benchmark]
        public unsafe void NativeAddUPerf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                NativeAddU(psrc, pdst, len);
            }
        }

        [Benchmark]
        public void MyAddUPerf() => IntrinsicsUtils.AddU(src, dst);

        [Benchmark]
        public unsafe void NativeAddSUPerf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (int* pidx = idx)
            {
                NativeAddSU(psrc, pidx, pdst, idxlen);
            }
        }

        [Benchmark]
        public void MyAddSUPerf() => IntrinsicsUtils.AddSU(src, idx, dst);

        [Benchmark]
        public unsafe void NativeAddScaleUPerf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                NativeAddScaleU(scale, psrc, pdst, len);
            }
        }

        [Benchmark]
        public void MyAddScaleUPerf() => IntrinsicsUtils.AddScaleU(scale, src, dst);

        [Benchmark]
        public unsafe void NativeAddScaleSUPerf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (int* pidx = idx)
            {
                NativeAddScaleSU(scale, psrc, pidx, pdst, idxlen);
            }
        }

        [Benchmark]
        public void MyAddScaleSUPerf() => IntrinsicsUtils.AddScaleSU(scale, src, idx, dst);

        [Benchmark]
        public unsafe void NativeScaleUPerf()
        {
            fixed (float* pdst = dst)
            {
                NativeScaleU(scale, pdst, len);
            }
        }

        [Benchmark]
        public void MyScaleUPerf() => IntrinsicsUtils.ScaleU(scale, dst);

        [Benchmark]
        public unsafe float NativeDist2Perf()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                return NativeDist2(psrc, pdst, len);
            }
        }

        [Benchmark]
        public float MyDist2Perf() => IntrinsicsUtils.Dist2(src, dst);

        [Benchmark]
        public unsafe float NativeSumAbsUPerf()
        {
            fixed (float* psrc = src)
            {
                return NativeSumAbsU(psrc, len);
            }
        }

        [Benchmark]
        public float MySumAbsqUPerf() => IntrinsicsUtils.SumAbsU(src);

        [Benchmark]
        public unsafe void NativeMulElementWiseUPerf()
        {
            fixed (float* psrc1 = src1)
            fixed (float* psrc2 = src2)
            fixed (float* pdst = dst)
            {
                NativeMulElementWiseU(psrc1, psrc2, pdst, len);
            }
        }

        [Benchmark]
        public void MyMulElementWiseUPerf() => IntrinsicsUtils.MulElementWiseU(src1, src2, dst);
    }

    public class Perf
    {
        public static void Main(string[] args)
        {
            var summary = BenchmarkRunner.Run<SsePerf>();
        }
    }
}
