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
        private readonly AlignedArray[] testMatrices;
        private readonly AlignedArray[] testSrcVectors;
        private readonly AlignedArray[] testDstVectors;
        private const float DEFAULT_SCALE = 1.7f;
        private const int SseCbAlign = 16;
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

            // Padded matrices whose dimensions are multiples of 4
            float[] testMatrix1 = new float[4 * 4] { 1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f, 3.29f,
                1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f, 3.29f };
            float[] testMatrix2 = new float[4 * 8];

            for (int i = 0; i < testMatrix2.Length; i++)
            {
                testMatrix2[i] = i + 1;
            }

            AlignedArray testMatrixAligned1 = new AlignedArray(4 * 4, SseCbAlign);
            AlignedArray testMatrixAligned2 = new AlignedArray(4 * 8, SseCbAlign);
            testMatrixAligned1.CopyFrom(testMatrix1, 0, testMatrix1.Length);
            testMatrixAligned2.CopyFrom(testMatrix2, 0, testMatrix2.Length);

            testMatrices = new AlignedArray[] { testMatrixAligned1, testMatrixAligned2 };

            // Padded source vectors whose dimensions are multiples of 4
            float[] testSrcVector1 = new float[4] { 1f, 2f, 3f, 4f };
            float[] testSrcVector2 = new float[8] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f };

            AlignedArray testSrcVectorAligned1 = new AlignedArray(4, SseCbAlign);
            AlignedArray testSrcVectorAligned2 = new AlignedArray(8, SseCbAlign);
            testSrcVectorAligned1.CopyFrom(testSrcVector1, 0, testSrcVector1.Length);
            testSrcVectorAligned2.CopyFrom(testSrcVector2, 0, testSrcVector2.Length);

            testSrcVectors = new AlignedArray[] { testSrcVectorAligned1, testSrcVectorAligned2 };

            // Padded destination vectors whose dimensions are multiples of 4
            float[] testDstVector1 = new float[4] { 0f, 1f, 2f, 3f };
            float[] testDstVector2 = new float[8] { 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f };

            AlignedArray testDstVectorAligned1 = new AlignedArray(4, SseCbAlign);
            AlignedArray testDstVectorAligned2 = new AlignedArray(8, SseCbAlign);
            testDstVectorAligned1.CopyFrom(testDstVector1, 0, testDstVector1.Length);
            testDstVectorAligned2.CopyFrom(testDstVector2, 0, testDstVector2.Length);

            testDstVectors = new AlignedArray[] { testDstVectorAligned1, testDstVectorAligned2 };
        }

        [Theory]
        [InlineData(0, 0, 0, new float[] { 23.28f, -49.72f, 23.28f, -49.72f })]
        [InlineData(1, 1, 0, new float[] { 204f, 492f, 780f, 1068f })]
        [InlineData(1, 0, 1, new float[] { 30f, 70f, 110f, 150f, 190f, 230f, 270f, 310f })]
        public void MatMulATest(int matTest, int srcTest, int dstTest, float[] expected)
        {
            AlignedArray mat = testMatrices[matTest];
            AlignedArray src = testSrcVectors[srcTest];
            AlignedArray dst = testDstVectors[dstTest];

            CpuMathUtils.MatTimesSrc(false, false, mat, src, dst, dst.Size);
            float[] actual = new float[dst.Size];
            dst.CopyTo(actual, 0, dst.Size);
            Assert.Equal(expected, actual, comparer);
        }

        [Theory]
        [InlineData(0, 0, 0, new float[] { 23.28f, -48.72f, 25.28f, -46.72f })]
        [InlineData(1, 1, 0, new float[] { 204f, 493f, 782f, 1071f })]
        [InlineData(1, 0, 1, new float[] { 30f, 71f, 112f, 153f, 194f, 235f, 276f, 317f })]
        public void MatMulAAddTest(int matTest, int srcTest, int dstTest, float[] expected)
        {
            AlignedArray mat = testMatrices[matTest];
            AlignedArray src = testSrcVectors[srcTest];
            AlignedArray dst = testDstVectors[dstTest];

            CpuMathUtils.MatTimesSrc(false, true, mat, src, dst, dst.Size);
            float[] actual = new float[dst.Size];
            dst.CopyTo(actual, 0, dst.Size);
            Assert.Equal(expected, actual, comparer);
        }

        [Theory]
        [InlineData(0, 0, 0, new float[] { -630.38f, -171.1f, 155.66f, 75.1f })]
        [InlineData(1, 0, 1, new float[] { 170f, 180f, 190f, 200f, 210f, 220f, 230f, 240f })]
        [InlineData(1, 1, 0, new float[] { 708f, 744f, 780f, 816f })]
        public void MatMulTranATest(int matTest, int srcTest, int dstTest, float[] expected)
        {
            AlignedArray mat = testMatrices[matTest];
            AlignedArray src = testSrcVectors[srcTest];
            AlignedArray dst = testDstVectors[dstTest];

            CpuMathUtils.MatTimesSrc(true, false, mat, src, dst, src.Size);
            float[] actual = new float[dst.Size];
            dst.CopyTo(actual, 0, dst.Size);
            Assert.Equal(expected, actual, comparer);
        }

        [Theory]
        [InlineData(0, 0, 0, new float[] { -630.38f, -170.1f, 157.66f, 78.1f })]
        [InlineData(1, 0, 1, new float[] { 170f, 181f, 192f, 203f, 214f, 225f, 236f, 247f })]
        [InlineData(1, 1, 0, new float[] { 708f, 745f, 782f, 819f })]
        public void MatMulTranAAddTest(int matTest, int srcTest, int dstTest, float[] expected)
        {
            AlignedArray mat = testMatrices[matTest];
            AlignedArray src = testSrcVectors[srcTest];
            AlignedArray dst = testDstVectors[dstTest];

            CpuMathUtils.MatTimesSrc(true, true, mat, src, dst, src.Size);
            float[] actual = new float[dst.Size];
            dst.CopyTo(actual, 0, dst.Size);
            Assert.Equal(expected, actual, comparer);
        }

        [Theory]
        [InlineData(0, 0, 0, new float[] { -27.32f, -9.02f, -27.32f, -9.02f })]
        [InlineData(1, 1, 0, new float[] { 95f, 231f, 367f, 503f })]
        [InlineData(1, 0, 1, new float[] { 10f, 26f, 42f, 58f, 74f, 90f, 106f, 122f })]
        public void MatMulPATest(int matTest, int srcTest, int dstTest, float[] expected)
        {
            AlignedArray mat = testMatrices[matTest];
            AlignedArray src = testSrcVectors[srcTest];
            AlignedArray dst = testDstVectors[dstTest];
            int[] idx = testIndexArray;

            CpuMathUtils.MatTimesSrc(false, false, mat, idx, src, 0, 0, 2 + 2 * srcTest, dst, dst.Size);
            float[] actual = new float[dst.Size];
            dst.CopyTo(actual, 0, dst.Size);
            Assert.Equal(expected, actual, comparer);
        }

        [Theory]
        [InlineData(0, 0, 0, new float[] { -27.32f, -8.02f, -25.32f, -6.02f })]
        [InlineData(1, 1, 0, new float[] { 95f, 232f, 369f, 506f })]
        [InlineData(1, 0, 1, new float[] { 10f, 27f, 44f, 61f, 78f, 95f, 112f, 129f })]
        public void MatMulPAAddTest(int matTest, int srcTest, int dstTest, float[] expected)
        {
            AlignedArray mat = testMatrices[matTest];
            AlignedArray src = testSrcVectors[srcTest];
            AlignedArray dst = testDstVectors[dstTest];
            int[] idx = testIndexArray;

            CpuMathUtils.MatTimesSrc(false, true, mat, idx, src, 0, 0, 2 + 2 * srcTest, dst, dst.Size);
            float[] actual = new float[dst.Size];
            dst.CopyTo(actual, 0, dst.Size);
            Assert.Equal(expected, actual, comparer);
        }

        [Theory]
        [InlineData(0, 0, 0, new float[] { 7.84f, -9.52f, -39.04f, 55.36f })]
        [InlineData(1, 0, 1, new float[] { 52f, 56f, 60f, 64f, 68f, 72f, 76f, 80f })]
        [InlineData(1, 1, 0, new float[] { 329f, 346f, 363f, 380f })]
        public void MatMulTranPATest(int matTest, int srcTest, int dstTest, float[] expected)
        {
            AlignedArray mat = testMatrices[matTest];
            AlignedArray src = testSrcVectors[srcTest];
            AlignedArray dst = testDstVectors[dstTest];
            int[] idx = testIndexArray;

            CpuMathUtils.MatTimesSrc(true, false, mat, idx, src, 0, 0, 2 + 2 * srcTest, dst, src.Size);
            float[] actual = new float[dst.Size];
            dst.CopyTo(actual, 0, dst.Size);
            Assert.Equal(expected, actual, comparer);
        }

        [Theory]
        [InlineData(0, 0, 0, new float[] { 7.84f, -8.52f, -37.04f, 58.36f })]
        [InlineData(1, 0, 1, new float[] { 52f, 57f, 62f, 67f, 72f, 77f, 82f, 87f })]
        [InlineData(1, 1, 0, new float[] { 329f, 347f, 365f, 383f })]
        public void MatMulTranPAAddTest(int matTest, int srcTest, int dstTest, float[] expected)
        {
            AlignedArray mat = testMatrices[matTest];
            AlignedArray src = testSrcVectors[srcTest];
            AlignedArray dst = testDstVectors[dstTest];
            int[] idx = testIndexArray;

            CpuMathUtils.MatTimesSrc(true, true, mat, idx, src, 0, 0, 2 + 2 * srcTest, dst, src.Size);
            float[] actual = new float[dst.Size];
            dst.CopyTo(actual, 0, dst.Size);
            Assert.Equal(expected, actual, comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void AddScalarUTest(int test)
        {
            float[] dst = (float[])testArrays[test].Clone();
            float[] expected = (float[])dst.Clone();

            for (int i = 0; i < expected.Length; i++)
            {
                expected[i] += DEFAULT_SCALE;
            }

            CpuMathUtils.Add(DEFAULT_SCALE, dst, dst.Length);
            var actual = dst;
            Assert.Equal(expected, actual, comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void ScaleUTest(int test)
        {
            float[] dst = (float[])testArrays[test].Clone();
            float[] expected = (float[])dst.Clone();

            for (int i = 0; i < expected.Length; i++)
            {
                expected[i] *= DEFAULT_SCALE;
            }

            CpuMathUtils.Scale(DEFAULT_SCALE, dst, dst.Length);
            var actual = dst;
            Assert.Equal(expected, actual, comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void ScaleSrcUTest(int test)
        {
            float[] src = (float[])testArrays[test].Clone();
            float[] dst = (float[])src.Clone();
            float[] expected = (float[])dst.Clone();

            for (int i = 0; i < expected.Length; i++)
            {
                expected[i] *= DEFAULT_SCALE;
            }

            CpuMathUtils.Scale(DEFAULT_SCALE, src, dst, dst.Length);
            var actual = dst;
            Assert.Equal(expected, actual, comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void ScaleAddUTest(int test)
        {
            float[] dst = (float[])testArrays[test].Clone();
            float[] expected = (float[])dst.Clone();

            for (int i = 0; i < expected.Length; i++)
            {
                expected[i] = DEFAULT_SCALE * (dst[i] + DEFAULT_SCALE);
            }

            CpuMathUtils.ScaleAdd(DEFAULT_SCALE, DEFAULT_SCALE, dst, dst.Length);
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
        public void AddScaleCopyUTest(int test)
        {
            float[] src = (float[])testArrays[test].Clone();
            float[] dst = (float[])src.Clone();
            float[] result = (float[])dst.Clone();
            float[] expected = (float[])dst.Clone();

            for (int i = 0; i < expected.Length; i++)
            {
                expected[i] *= (1 + DEFAULT_SCALE);
            }

            CpuMathUtils.AddScaleCopy(DEFAULT_SCALE, src, dst, result, dst.Length);
            var actual = result;
            Assert.Equal(expected, actual, comparer);
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

        [Theory]
        [InlineData(0, -93.9f)]
        [InlineData(1, -97.19f)]
        public void SumUTest(int test, float expected)
        {
            float[] src = (float[])testArrays[test].Clone();
            var actual = CpuMathUtils.Sum(src, src.Length);
            Assert.Equal(expected, actual, 2);
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
        [InlineData(0, 13742.3176f)]
        [InlineData(1, 13739.7895f)]
        public void SumSqDiffUTest(int test, float expected)
        {
            float[] src = (float[])testArrays[test].Clone();
            var actual = CpuMathUtils.SumSq(DEFAULT_SCALE, src, 0, src.Length);
            Assert.Equal(expected, actual, 2);
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
        [InlineData(0, 196.98f)]
        [InlineData(1, 195.39f)]
        public void SumAbsDiffUTest(int test, float expected)
        {
            float[] src = (float[])testArrays[test].Clone();
            var actual = CpuMathUtils.SumAbs(DEFAULT_SCALE, src, 0, src.Length);
            Assert.Equal(expected, actual, 2);
        }

        [Theory]
        [InlineData(0, 106.37f)]
        [InlineData(1, 106.37f)]
        public void MaxAbsUTest(int test, float expected)
        {
            float[] src = (float[])testArrays[test].Clone();
            var actual = CpuMathUtils.MaxAbs(src, src.Length);
            Assert.Equal(expected, actual, 2);
        }

        [Theory]
        [InlineData(0, 108.07f)]
        [InlineData(1, 108.07f)]
        public void MaxAbsDiffUTest(int test, float expected)
        {
            float[] src = (float[])testArrays[test].Clone();
            var actual = CpuMathUtils.MaxAbsDiff(DEFAULT_SCALE, src, src.Length);
            Assert.Equal(expected, actual, 2);
        }

        [Theory]
        [InlineData(0, 13306.0376f)]
        [InlineData(1, 13291.9235f)]
        public void DotUTest(int test, float expected)
        {
            float[] src = (float[])testArrays[test].Clone();
            float[] dst = (float[])src.Clone();

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
        [InlineData(0, new int[] { 0, 2 }, new float[] { 0f, 2f, 0f, 4f })]
        [InlineData(1, new int[] { 0, 2, 5, 6 }, new float[] { 0f, 2f, 0f, 4f, 5f, 0f, 0f, 8f })]
        public void ZeroItemsUTest(int test, int[] idx, float[] expected)
        {
            AlignedArray src = new AlignedArray(4 + 4 * test, SseCbAlign);
            src.CopyFrom(testSrcVectors[test]);

            CpuMathUtils.ZeroMatrixItems(src, src.Size, src.Size, idx);
            float[] actual = new float[src.Size];
            src.CopyTo(actual, 0, src.Size);
            Assert.Equal(expected, actual, comparer);
        }

        [Theory]
        [InlineData(0, new int[] { 0, 1 }, new float[] { 0f, 2f, 0f, 4f })]
        [InlineData(1, new int[] { 0, 2, 4 }, new float[] { 0f, 2f, 0f, 4f, 5f, 0f, 7f, 8f })]
        public void ZeroMatrixItemsCoreTest(int test, int[] idx, float[] expected)
        {
            AlignedArray src = new AlignedArray(4 + 4 * test, SseCbAlign);
            src.CopyFrom(testSrcVectors[test]);

            CpuMathUtils.ZeroMatrixItems(src, src.Size / 2 - 1, src.Size / 2, idx);
            float[] actual = new float[src.Size];
            src.CopyTo(actual, 0, src.Size);
            Assert.Equal(expected, actual, comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void SdcaL1UpdateUTest(int test)
        {
            float[] src = (float[])testArrays[test].Clone();
            float[] v = (float[])src.Clone();
            float[] w = (float[])src.Clone();
            float[] expected = (float[])w.Clone();

            for (int i = 0; i < expected.Length; i++)
            {
                float value = src[i] * (1 + DEFAULT_SCALE);
                expected[i] = Math.Abs(value) > DEFAULT_SCALE ? (value > 0 ? value - DEFAULT_SCALE : value + DEFAULT_SCALE) : 0;
            }

            CpuMathUtils.SdcaL1UpdateDense(DEFAULT_SCALE, src.Length, src, DEFAULT_SCALE, v, w);
            var actual = w;
            Assert.Equal(expected, actual, comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void SdcaL1UpdateSUTest(int test)
        {
            float[] src = (float[])testArrays[test].Clone();
            float[] v = (float[])src.Clone();
            float[] w = (float[])src.Clone();
            int[] idx = testIndexArray;
            float[] expected = (float[])w.Clone();

            for (int i = 0; i < idx.Length; i++)
            {
                int index = idx[i];
                float value = v[index] + src[i] * DEFAULT_SCALE;
                expected[index] = Math.Abs(value) > DEFAULT_SCALE ? (value > 0 ? value - DEFAULT_SCALE : value + DEFAULT_SCALE) : 0;
            }

            CpuMathUtils.SdcaL1UpdateSparse(DEFAULT_SCALE, src.Length, src, idx, idx.Length, DEFAULT_SCALE, v, w);
            var actual = w;
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
