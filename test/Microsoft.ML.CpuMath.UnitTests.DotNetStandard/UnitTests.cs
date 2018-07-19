using System;
using System.Runtime.InteropServices;
using System.Security;
using Xunit;
using Xunit.Abstractions;
using Microsoft.ML.Runtime.Internal.CpuMath;

namespace Microsoft.ML.CpuMath.UnitTests.DotNetStandard
{
    public class UnitTests
    {
        private readonly ITestOutputHelper output;

        public UnitTests(ITestOutputHelper output)
        {
            this.output = output;
        }

        internal const int EXP_MAX = 127;
        internal const int EXP_MIN = 0;
        internal static float NextFloat(Random rand, int expRange)
        {
            double mantissa = (rand.NextDouble() * 2.0) - 1.0;
            double exponent = Math.Pow(2.0, rand.Next(-expRange + 1, expRange + 1));
            return (float)(mantissa * exponent);
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

        [Theory]
        [InlineData(10, 0, EXP_MAX)]
        [InlineData(20, 0, EXP_MAX)]
        [InlineData(20, 1, EXP_MAX / 2)]
        [InlineData(30, 2, EXP_MAX / 2)]
        public unsafe void DotUTest(int len, int seed, int expRange)
        {
            // arrange
            float[] src = new float[len];
            float[] dst = new float[len];
            Random rand = new Random(seed);

            for (int i = 0; i < len; i++)
            {
                src[i] = NextFloat(rand, expRange);
                dst[i] = NextFloat(rand, expRange);
            }

            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                // act
                var nativeOutput = NativeDotU(psrc, pdst, len);
                var managedOutput = CpuMathUtils.DotProductDense(src, dst, len);
                output.WriteLine($"{nativeOutput} == {managedOutput}");

                // assert
                Assert.True(nativeOutput == managedOutput);
            }
        }

        [Theory]
        [InlineData(10, 0, EXP_MAX / 2)]
        [InlineData(20, 0, EXP_MAX / 2)]
        [InlineData(20, 1, EXP_MAX / 2)]
        [InlineData(30, 2, EXP_MAX / 2)]
        public unsafe void DotSUTest(int len, int seed, int expRange)
        {
            // arrange
            float[] src = new float[len];
            float[] dst = new float[len];
            Random rand = new Random(seed);
            int idxlen = rand.Next(1, len + 1);
            int[] idx = new int[idxlen];

            for (int i = 0; i < len; i++)
            {
                src[i] = NextFloat(rand, expRange);
                dst[i] = NextFloat(rand, expRange);
            }

            for (int i = 0; i < idxlen; i++)
            {
                idx[i] = rand.Next(0, len);
            }

            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (int* pidx = idx)
            {
                // act
                var nativeOutput = NativeDotSU(psrc, pdst, pidx, idxlen);
                var managedOutput = CpuMathUtils.DotProductSparse(src, dst, idx, idxlen);
                output.WriteLine($"{nativeOutput} == {managedOutput}");

                // assert
                Assert.True(nativeOutput == managedOutput);
            }
        }

        [Theory]
        [InlineData(10, 0, EXP_MAX / 2)]
        [InlineData(20, 0, EXP_MAX / 2)]
        [InlineData(20, 1, EXP_MAX / 2)]
        [InlineData(30, 2, EXP_MAX / 2)]
        public unsafe void SumSqUTest(int len, int seed, int expRange)
        {
            // arrange
            float[] src = new float[len];
            Random rand = new Random(seed);

            for (int i = 0; i < len; i++)
            {
                src[i] = NextFloat(rand, expRange);
            }

            fixed (float* psrc = src)
            {
                // act
                var nativeOutput = NativeSumSqU(psrc, len);
                var managedOutput = CpuMathUtils.SumSq(src, len);
                output.WriteLine($"{nativeOutput} == {managedOutput}");

                // assert
                Assert.True(nativeOutput == managedOutput);
            }
        }

        [Theory]
        [InlineData(10, 0, EXP_MAX / 2)]
        [InlineData(20, 0, EXP_MAX / 2)]
        [InlineData(20, 1, EXP_MAX / 2)]
        [InlineData(30, 2, EXP_MAX / 2)]
        public unsafe void AddUTest(int len, int seed, int expRange)
        {
            // arrange
            float[] src = new float[len];
            float[] dst = new float[len];
            Random rand = new Random(seed);

            for (int i = 0; i < len; i++)
            {
                src[i] = NextFloat(rand, expRange);
                dst[i] = NextFloat(rand, expRange);
            }

            float[] nativeOutput = dst;

            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (float* pnative = &nativeOutput[0])
            {
                // act
                NativeAddU(psrc, pnative, len);

                CpuMathUtils.Add(src, dst, len);
                var managedOutput = dst;
                output.WriteLine($"{nativeOutput} == {managedOutput}");

                // assert
                Assert.Equal(managedOutput, nativeOutput);
            }
        }

        [Theory]
        [InlineData(10, 0, EXP_MAX / 2)]
        [InlineData(20, 0, EXP_MAX / 2)]
        [InlineData(20, 1, EXP_MAX / 2)]
        [InlineData(30, 2, EXP_MAX / 2)]
        public unsafe void AddSUTest(int len, int seed, int expRange)
        {
            // arrange
            float[] src = new float[len];
            float[] dst = new float[len];
            Random rand = new Random(seed);
            int idxlen = rand.Next(1, len + 1);
            int[] idx = new int[idxlen];

            for (int i = 0; i < len; i++)
            {
                src[i] = NextFloat(rand, expRange);
                dst[i] = NextFloat(rand, expRange);
            }

            for (int i = 0; i < idxlen; i++)
            {
                idx[i] = rand.Next(0, len);
            }

            float[] nativeOutput = dst;

            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (int* pidx = idx)
            fixed (float* pnative = &nativeOutput[0])
            {
                // act
                NativeAddSU(psrc, pidx, pnative, idxlen);

                CpuMathUtils.Add(src, idx, dst, idxlen);
                var managedOutput = dst;
                output.WriteLine($"{nativeOutput} == {managedOutput}");

                // assert
                Assert.Equal(managedOutput, nativeOutput);
            }
        }

        [Theory]
        [InlineData(10, 0, EXP_MAX / 2, 0.3)]
        [InlineData(20, 0, EXP_MAX / 2, 1)]
        [InlineData(20, 1, EXP_MAX / 2, 22.7)]
        [InlineData(30, 2, EXP_MAX / 2, 111.468)]
        public unsafe void AddScaleUTest(int len, int seed, int expRange, float scale)
        {
            // arrange
            float[] src = new float[len];
            float[] dst = new float[len];
            Random rand = new Random(seed);

            for (int i = 0; i < len; i++)
            {
                src[i] = NextFloat(rand, expRange);
                dst[i] = NextFloat(rand, expRange);
            }

            float[] nativeOutput = dst;

            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (float* pnative = &nativeOutput[0])
            {
                // act
                NativeAddScaleU(scale, psrc, pnative, len);

                CpuMathUtils.AddScale(scale, src, dst, len);
                var managedOutput = dst;
                output.WriteLine($"{nativeOutput} == {managedOutput}");

                // assert
                Assert.Equal(managedOutput, nativeOutput);
            }
        }

        [Theory]
        [InlineData(10, 0, EXP_MAX / 2, 0.3)]
        [InlineData(20, 0, EXP_MAX / 2, 1)]
        [InlineData(20, 1, EXP_MAX / 2, 22.7)]
        [InlineData(30, 2, EXP_MAX / 2, 111.468)]
        public unsafe void AddScaleSUTest(int len, int seed, int expRange, float scale)
        {
            // arrange
            float[] src = new float[len];
            float[] dst = new float[len];
            Random rand = new Random(seed);
            int idxlen = rand.Next(1, len + 1);
            int[] idx = new int[idxlen];

            for (int i = 0; i < len; i++)
            {
                src[i] = NextFloat(rand, expRange);
                dst[i] = NextFloat(rand, expRange);
            }

            for (int i = 0; i < idxlen; i++)
            {
                idx[i] = rand.Next(0, len);
            }

            float[] nativeOutput = dst;

            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (int* pidx = idx)
            fixed (float* pnative = &nativeOutput[0])
            {
                // act
                NativeAddScaleSU(scale, psrc, pidx, pnative, idxlen);

                CpuMathUtils.AddScale(scale, src, idx, dst, idxlen);
                var managedOutput = dst;
                output.WriteLine($"{nativeOutput} == {managedOutput}");

                // assert
                Assert.Equal(managedOutput, nativeOutput);
            }
        }

        [Theory]
        [InlineData(10, 0, EXP_MAX / 2, 0.3)]
        [InlineData(20, 0, EXP_MAX / 2, 1)]
        [InlineData(20, 1, EXP_MAX / 2, 22.7)]
        [InlineData(30, 2, EXP_MAX / 2, 111.468)]
        public unsafe void ScaleUTest(int len, int seed, int expRange, float scale)
        {
            // arrange
            float[] dst = new float[len];
            Random rand = new Random(seed);

            for (int i = 0; i < len; i++)
            {
                dst[i] = NextFloat(rand, expRange);
            }

            float[] nativeOutput = dst;

            fixed (float* pdst = dst)
            fixed (float* pnative = &nativeOutput[0])
            {
                // act
                NativeScaleU(scale, pnative, len);

                CpuMathUtils.Scale(scale, dst, len);
                var managedOutput = dst;
                output.WriteLine($"{nativeOutput} == {managedOutput}");

                // assert
                Assert.Equal(managedOutput, nativeOutput);
            }
        }

        [Theory]
        [InlineData(10, 0, EXP_MAX / 2)]
        [InlineData(20, 0, EXP_MAX / 2)]
        [InlineData(20, 1, EXP_MAX / 2)]
        [InlineData(30, 2, EXP_MAX / 2)]
        public unsafe void Dist2Test(int len, int seed, int expRange)
        {
            // arrange
            float[] src = new float[len];
            float[] dst = new float[len];
            Random rand = new Random(seed);

            for (int i = 0; i < len; i++)
            {
                src[i] = NextFloat(rand, expRange);
                dst[i] = NextFloat(rand, expRange);
            }

            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                // act
                var nativeOutput = NativeDist2(psrc, pdst, len);
                var managedOutput = CpuMathUtils.L2DistSquared(src, dst, len);
                output.WriteLine($"{nativeOutput} == {managedOutput}");

                // assert
                Assert.Equal(managedOutput, nativeOutput);
            }
        }

        [Theory]
        [InlineData(10, 0, EXP_MAX / 2)]
        [InlineData(20, 0, EXP_MAX / 2)]
        [InlineData(20, 1, EXP_MAX / 2)]
        [InlineData(30, 2, EXP_MAX / 2)]
        public unsafe void SumAbsUTest(int len, int seed, int expRange)
        {
            // arrange
            float[] src = new float[len];
            Random rand = new Random(seed);

            for (int i = 0; i < len; i++)
            {
                src[i] = NextFloat(rand, expRange);
            }

            fixed (float* psrc = src)
            {
                // act
                var nativeOutput = NativeSumAbsU(psrc, len);
                var managedOutput = CpuMathUtils.SumAbs(src, len);
                output.WriteLine($"{nativeOutput} == {managedOutput}");

                // assert
                Assert.Equal(managedOutput, nativeOutput);
            }
        }

        [Theory]
        [InlineData(10, 0, EXP_MAX / 2)]
        [InlineData(20, 0, EXP_MAX / 2)]
        [InlineData(20, 1, EXP_MAX / 2)]
        [InlineData(30, 2, EXP_MAX / 2)]
        public unsafe void MulElementWiseUTest(int len, int seed, int expRange)
        {
            // arrange
            float[] src1 = new float[len];
            float[] src2 = new float[len];
            float[] dst = new float[len];
            Random rand = new Random(seed);

            for (int i = 0; i < len; i++)
            {
                src1[i] = NextFloat(rand, expRange);
                src2[i] = NextFloat(rand, expRange);
                dst[i] = NextFloat(rand, expRange);
            }

            float[] nativeOutput = dst;

            fixed (float* psrc1 = &src1[0])
            fixed (float* psrc2 = &src2[0])
            fixed (float* pdst = dst)
            fixed (float* pnative = &nativeOutput[0])
            {
                // act
                NativeMulElementWiseU(psrc1, psrc2, pnative, len);

                CpuMathUtils.MulElementWise(src1, src2, dst, len);
                var managedOutput = dst;
                output.WriteLine($"{nativeOutput} == {managedOutput}");

                // assert
                Assert.Equal(managedOutput, nativeOutput);
            }
        }
    }
}
