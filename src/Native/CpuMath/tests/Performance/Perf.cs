using System;
using System.Runtime.InteropServices;
using System.Security;
using Intrinsics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

namespace PerformanceTests
{
    public unsafe class SsePerf
    {
        internal const int EXP_MAX = 127;
        internal const int EXP_MIN = 0;
        internal static float NextFloat(Random rand, int expRange)
        {
            double mantissa = (rand.NextDouble() * 2.0) - 1.0;
            double exponent = Math.Pow(2.0, rand.Next(-expRange + 1, expRange + 1));
            return (float)(mantissa * exponent);
        }

        private float[] src, dst;
        private int[] idx;
        private readonly int idxlen = 23;
        private readonly int len = 23;
        private readonly int expRange = EXP_MAX / 2;
        private readonly int seed = 2;
        private unsafe float* psrc, pdst;
        private unsafe int* pidx;


        public SsePerf()
        {
            src = new float[len];
            dst = new float[len];
            Random rand = new Random(seed);
            idx = new int[idxlen];

            for (int i = 0; i < len; i++)
            {
                src[i] = NextFloat(rand, expRange);
                dst[i] = NextFloat(rand, expRange);
            }

            for (int i = 0; i < idxlen; i++)
            {
                idx[i] = rand.Next(0, len);
            }

            fixed (float* a = &src[0])
            fixed (float* b = &dst[0])
            fixed (int* c = &idx[0])
            {
                psrc = a;
                pdst = b;
                pidx = c;
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

        [Benchmark]
        public unsafe float NativeDotUPerf() => NativeDotU(psrc, pdst, len);

        [Benchmark]
        public float MyDotUPerf() => IntrinsicsUtils.DotU(src, dst);

        [Benchmark]
        public unsafe float NativeDotSUPerf() => NativeDotSU(psrc, pdst, pidx, idxlen);

        [Benchmark]
        public float MyDotSUPerf() => IntrinsicsUtils.DotSU(src, dst, idx);

        [Benchmark]
        public unsafe float NativeSumSqUPerf() => NativeSumSqU(psrc, len);

        [Benchmark]
        public float MySumSqUPerf() => IntrinsicsUtils.SumSqU(src);

        [Benchmark]
        public unsafe float NativeDist2Perf() => NativeDist2(psrc, pdst, len);

        [Benchmark]
        public float MyDist2Perf() => IntrinsicsUtils.Dist2(src, dst);

        [Benchmark]
        public unsafe float NativeSumAbsUPerf() => NativeSumAbsU(psrc, len);

        [Benchmark]
        public float MySumAbsqUPerf() => IntrinsicsUtils.SumAbsU(src);
    }

    public class Perf
    {
        public static void Main(string[] args)
        {
            var summary = BenchmarkRunner.Run<SsePerf>();
        }
    }
}
