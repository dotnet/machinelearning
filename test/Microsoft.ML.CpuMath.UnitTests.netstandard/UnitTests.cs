// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Xunit;
using Microsoft.ML.Runtime.Internal.CpuMath;

namespace Microsoft.ML.CpuMath.UnitTests
{
    public class CpuMathUtilsUnitTests
    {
        private readonly float[][] testArrays;
        private readonly int[] testIndexArray;
        private const float DEFAULT_SCALE = 1.7f;
        private FloatEqualityComparer comparer;

        public CpuMathUtilsUnitTests()
        {
            // padded array whose length is a multiple of 4
            float[] testArray1 = new float[8] { 1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f, 3.29f };
            // unpadded array whose length is not a multiple of 4.
            float[] testArray2 = new float[7] { 1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f };
            testArrays = new float[][] { testArray1, testArray2 };
            testIndexArray = new int[4] { 0, 2, 5, 6 };
            comparer = new FloatEqualityComparer();
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public unsafe void DotUTest(int test)
        {
            float[] src = (float[]) testArrays[test].Clone();
            float[] dst = (float[]) src.Clone();
            
            for (int i = 0; i < dst.Length; i++)
            {
                dst[i] += 1;
            }

            var managedOutput = CpuMathUtils.DotProductDense(src, dst, dst.Length);

            if (test == 0)
            {
                Assert.Equal(13306.0376f, managedOutput, 4);
            }
            else
            {
                Assert.Equal(13291.9235f, managedOutput, 2);
            }
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public unsafe void DotSUTest(int test)
        {
            float[] src = (float[])testArrays[test].Clone();
            float[] dst = (float[])src.Clone();
            int[] idx = testIndexArray;

            for (int i = 0; i < dst.Length; i++)
            {
                dst[i] += 1;
            }

            var managedOutput = CpuMathUtils.DotProductSparse(src, dst, idx, idx.Length);
            Assert.Equal(736.7352f, managedOutput, 4);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public unsafe void SumSqUTest(int test)
        {
            float[] src = (float[])testArrays[test].Clone();

            var managedOutput = CpuMathUtils.SumSq(src, src.Length);

            if (test == 0)
            {
                Assert.Equal(13399.9376f, managedOutput, 2);
            }
            else
            {
                Assert.Equal(13389.1135f, managedOutput, 2);
            }
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public unsafe void AddUTest(int test)
        {
            float[] src = (float[])testArrays[test].Clone();
            float[] dst = (float[])src.Clone();
            float[] expectedOutput = (float[])src.Clone();

            for (int i = 0; i < dst.Length; i++)
            {
                dst[i] += 1;
            }

            for (int i = 0; i < expectedOutput.Length; i++)
            {
                expectedOutput[i] = 2 * expectedOutput[i] + 1;
            }

            CpuMathUtils.Add(src, dst, dst.Length);
            var managedOutput = dst;
            Assert.Equal(expectedOutput, managedOutput, comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public unsafe void AddSUTest(int test)
        {
            float[] src = (float[])testArrays[test].Clone();
            float[] dst = (float[])src.Clone();
            int[] idx = testIndexArray;
            float[] expectedOutput = (float[])dst.Clone();

            expectedOutput[0] = 3.92f;
            expectedOutput[2] = -12.14f;
            expectedOutput[5] = -36.69f;
            expectedOutput[6] = 46.29f;

            CpuMathUtils.Add(src, idx, dst, idx.Length);
            var managedOutput = dst;
            Assert.Equal(expectedOutput, managedOutput, comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public unsafe void AddScaleUTest(int test)
        {
            float[] src = (float[])testArrays[test].Clone();
            float[] dst = (float[])src.Clone();
            float[] expectedOutput = (float[])dst.Clone();

            for (int i = 0; i < expectedOutput.Length; i++)
            {
                expectedOutput[i] *= (1 + DEFAULT_SCALE);
            }

            CpuMathUtils.AddScale(DEFAULT_SCALE, src, dst, dst.Length);
            var managedOutput = dst;
            Assert.Equal(expectedOutput, managedOutput, comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public unsafe void AddScaleSUTest(int test)
        {
            float[] src = (float[])testArrays[test].Clone();
            float[] dst = (float[])src.Clone();
            int[] idx = testIndexArray;
            float[] expectedOutput = (float[])dst.Clone();

            expectedOutput[0] = 5.292f;
            expectedOutput[2] = -13.806f;
            expectedOutput[5] = -43.522f;
            expectedOutput[6] = 55.978f;

            CpuMathUtils.AddScale(DEFAULT_SCALE, src, idx, dst, idx.Length);
            var managedOutput = dst;
            Assert.Equal(expectedOutput, managedOutput, comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public unsafe void ScaleUTest(int test)
        {
            float[] dst = (float[])testArrays[test].Clone();
            float[] expectedOutput = (float[])dst.Clone();

            for (int i = 0; i < expectedOutput.Length; i++)
            {
                expectedOutput[i] *= DEFAULT_SCALE;
            }

            CpuMathUtils.Scale(DEFAULT_SCALE, dst, dst.Length);
            var managedOutput = dst;
            Assert.Equal(expectedOutput, managedOutput, comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public unsafe void Dist2Test(int test)
        {
            float[] src = (float[])testArrays[test].Clone();
            float[] dst = (float[])src.Clone();

            for (int i = 0; i < dst.Length; i++)
            {
                dst[i] += 1;
            }

            var managedOutput = CpuMathUtils.L2DistSquared(src, dst, dst.Length);

            if (test == 0)
            {
                Assert.Equal(8.0f, managedOutput, 0);
            }
            else
            {
                Assert.Equal(7.0f, managedOutput, 0);
            }
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public unsafe void SumAbsUTest(int test)
        {
            float[] src = (float[])testArrays[test].Clone();

            var managedOutput = CpuMathUtils.SumAbs(src, src.Length);

            if (test == 0)
            {
                Assert.Equal(196.98f, managedOutput, 2);
            }
            else
            {
                Assert.Equal(193.69f, managedOutput, 2);
            }
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public unsafe void MulElementWiseUTest(int test)
        {
            float[] src1 = (float[])testArrays[test].Clone();
            float[] src2 = (float[])src1.Clone();
            float[] dst = (float[])src1.Clone();

            for (int i = 0; i < src2.Length; i++)
            {
                src2[i] += 1;
            }

            float[] expectedOutput = (float[])src1.Clone();

            for (int i = 0; i < expectedOutput.Length; i++)
            {
                expectedOutput[i] *= (1 + expectedOutput[i]);
            }

            CpuMathUtils.MulElementWise(src1, src2, dst, dst.Length);
            var managedOutput = dst;
            Assert.Equal(expectedOutput, managedOutput, comparer);
        }
    }

    internal class FloatEqualityComparer : IEqualityComparer<float>
    {
        public bool Equals(float a, float b)
        {
            return Math.Abs(a - b) < 1e-5f;
        }

        public int GetHashCode(float a)
        {
            throw new NotImplementedException();
        }
    }
}
