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
            // Padded array whose length is a multiple of 4
            float[] testArray1 = new float[8] { 1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f, 3.29f };
            // Unpadded array whose length is not a multiple of 4.
            float[] testArray2 = new float[7] { 1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f };
            testArrays = new float[][] { testArray1, testArray2 };
            testIndexArray = new int[4] { 0, 2, 5, 6 };
            comparer = new FloatEqualityComparer();
        }

        [Theory]
        [InlineData(0, 13306.0376f)]
        [InlineData(1, 13291.9235f)]
        public void DotUTest(int test, float expected)
        {
            float[] src = (float[]) testArrays[test].Clone();
            float[] dst = (float[]) src.Clone();
            
            for (int i = 0; i < dst.Length; i++)
            {
                dst[i] += 1;
            }

            var actual = CpuMathUtils.DotProductDense(src, dst, dst.Length);
            Assert.Equal(expected, actual, 2);
        }

        [Theory]
        [InlineData(0, 736.7352f)]
        [InlineData(1, 736.7352f)]
        public void DotSUTest(int test, float expected)
        {
            float[] src = (float[])testArrays[test].Clone();
            float[] dst = (float[])src.Clone();
            int[] idx = testIndexArray;

            // Ensures src and dst are different arrays
            for (int i = 0; i < dst.Length; i++)
            {
                dst[i] += 1;
            }

            var actual = CpuMathUtils.DotProductSparse(src, dst, idx, idx.Length);
            Assert.Equal(expected, actual, 4);
        }

        [Theory]
        [InlineData(0, 13399.9376f)]
        [InlineData(1, 13389.1135f)]
        public void SumSqUTest(int test, float expected)
        {
            float[] src = (float[])testArrays[test].Clone();
            var actual = CpuMathUtils.SumSq(src, src.Length);
            Assert.Equal(expected, actual, 2);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void AddUTest(int test)
        {
            float[] src = (float[])testArrays[test].Clone();
            float[] dst = (float[])src.Clone();
            float[] expected = (float[])src.Clone();

            // Ensures src and dst are different arrays
            for (int i = 0; i < dst.Length; i++)
            {
                dst[i] += 1;
            }

            for (int i = 0; i < expected.Length; i++)
            {
                expected[i] = 2 * expected[i] + 1;
            }

            CpuMathUtils.Add(src, dst, dst.Length);
            var actual = dst;
            Assert.Equal(expected, actual, comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void AddSUTest(int test)
        {
            float[] src = (float[])testArrays[test].Clone();
            float[] dst = (float[])src.Clone();
            int[] idx = testIndexArray;
            float[] expected = (float[])dst.Clone();

            expected[0] = 3.92f;
            expected[2] = -12.14f;
            expected[5] = -36.69f;
            expected[6] = 46.29f;

            CpuMathUtils.Add(src, idx, dst, idx.Length);
            var actual = dst;
            Assert.Equal(expected, actual, comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void AddScaleUTest(int test)
        {
            float[] src = (float[])testArrays[test].Clone();
            float[] dst = (float[])src.Clone();
            float[] expected = (float[])dst.Clone();

            for (int i = 0; i < expected.Length; i++)
            {
                expected[i] *= (1 + DEFAULT_SCALE);
            }

            CpuMathUtils.AddScale(DEFAULT_SCALE, src, dst, dst.Length);
            var actual = dst;
            Assert.Equal(expected, actual, comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void AddScaleSUTest(int test)
        {
            float[] src = (float[])testArrays[test].Clone();
            float[] dst = (float[])src.Clone();
            int[] idx = testIndexArray;
            float[] expected = (float[])dst.Clone();

            expected[0] = 5.292f;
            expected[2] = -13.806f;
            expected[5] = -43.522f;
            expected[6] = 55.978f;

            CpuMathUtils.AddScale(DEFAULT_SCALE, src, idx, dst, idx.Length);
            var actual = dst;
            Assert.Equal(expected, actual, comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void ScaleUTest(int test)
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
        [InlineData(0, 8.0f)]
        [InlineData(1, 7.0f)]
        public void Dist2Test(int test, float expected)
        {
            float[] src = (float[])testArrays[test].Clone();
            float[] dst = (float[])src.Clone();

            // Ensures src and dst are different arrays
            for (int i = 0; i < dst.Length; i++)
            {
                dst[i] += 1;
            }

            var actual = CpuMathUtils.L2DistSquared(src, dst, dst.Length);
            Assert.Equal(expected, actual, 0);
        }

        [Theory]
        [InlineData(0, 196.98f)]
        [InlineData(1, 193.69f)]
        public void SumAbsUTest(int test, float expected)
        {
            float[] src = (float[])testArrays[test].Clone();
            var actual = CpuMathUtils.SumAbs(src, src.Length);
            Assert.Equal(expected, actual, 2);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void MulElementWiseUTest(int test)
        {
            float[] src1 = (float[])testArrays[test].Clone();
            float[] src2 = (float[])src1.Clone();
            float[] dst = (float[])src1.Clone();

            // Ensures src1 and src2 are different arrays
            for (int i = 0; i < src2.Length; i++)
            {
                src2[i] += 1;
            }

            float[] expected = (float[])src1.Clone();

            for (int i = 0; i < expected.Length; i++)
            {
                expected[i] *= (1 + expected[i]);
            }

            CpuMathUtils.MulElementWise(src1, src2, dst, dst.Length);
            var actual = dst;
            Assert.Equal(expected, actual, comparer);
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
