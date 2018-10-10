// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Xunit;
using Xunit.Abstractions;
using Microsoft.ML.Runtime.Internal.CpuMath;

namespace Microsoft.ML.CpuMath.UnitTests
{
    public class CpuMathUtilsUnitTests
    {
        private readonly float[][] _testArrays;
        private readonly int[] _testIndexArray;
        private readonly AlignedArray[] _testMatrices;
        private readonly AlignedArray[] _testSrcVectors;
        private readonly AlignedArray[] _testDstVectors;
        private readonly int _vectorAlignment = CpuMathUtils.GetVectorAlignment();
        private readonly FloatEqualityComparer _comparer;
        private readonly FloatEqualityComparerForMatMul _matMulComparer;

        private const float DefaultScale = 1.7f;

        public CpuMathUtilsUnitTests()
        {
            // Padded array whose length is a multiple of 4
            float[] testArray1 = new float[16] { 1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f, 3.29f, 1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f, 3.29f };
            // Unpadded array whose length is not a multiple of 4.
            float[] testArray2 = new float[15] { 1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f, 3.29f, 1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f };
            _testArrays = new float[][] { testArray1, testArray2 };
            _testIndexArray = new int[9] { 0, 2, 5, 6, 8, 11, 12, 13, 14 };
            _comparer = new FloatEqualityComparer();
            _matMulComparer = new FloatEqualityComparerForMatMul();

            // Padded matrices whose dimensions are multiples of 8
            float[] testMatrix1 = new float[8 * 8] { 1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f, 3.29f,
                                                        1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f, 3.29f,
                                                        1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f, 3.29f,
                                                        1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f, 3.29f,
                                                        1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f, 3.29f,
                                                        1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f, 3.29f,
                                                        1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f, 3.29f,
                                                        1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f, 3.29f };
            float[] testMatrix2 = new float[8 * 16];

            for (int i = 0; i < testMatrix2.Length; i++)
            {
                testMatrix2[i] = i + 1;
            }

            AlignedArray testMatrixAligned1 = new AlignedArray(8 * 8, _vectorAlignment);
            AlignedArray testMatrixAligned2 = new AlignedArray(8 * 16, _vectorAlignment);
            testMatrixAligned1.CopyFrom(testMatrix1, 0, testMatrix1.Length);
            testMatrixAligned2.CopyFrom(testMatrix2, 0, testMatrix2.Length);

            _testMatrices = new AlignedArray[] { testMatrixAligned1, testMatrixAligned2 };

            // Padded source vectors whose dimensions are multiples of 8
            float[] testSrcVector1 = new float[8] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f };
            float[] testSrcVector2 = new float[16] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f };

            AlignedArray testSrcVectorAligned1 = new AlignedArray(8, _vectorAlignment);
            AlignedArray testSrcVectorAligned2 = new AlignedArray(16, _vectorAlignment);
            testSrcVectorAligned1.CopyFrom(testSrcVector1, 0, testSrcVector1.Length);
            testSrcVectorAligned2.CopyFrom(testSrcVector2, 0, testSrcVector2.Length);

            _testSrcVectors = new AlignedArray[] { testSrcVectorAligned1, testSrcVectorAligned2 };

            // Padded destination vectors whose dimensions are multiples of 8
            float[] testDstVector1 = new float[8] { 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f };
            float[] testDstVector2 = new float[16] { 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f };

            AlignedArray testDstVectorAligned1 = new AlignedArray(8, _vectorAlignment);
            AlignedArray testDstVectorAligned2 = new AlignedArray(16, _vectorAlignment);
            testDstVectorAligned1.CopyFrom(testDstVector1, 0, testDstVector1.Length);
            testDstVectorAligned2.CopyFrom(testDstVector2, 0, testDstVector2.Length);

            _testDstVectors = new AlignedArray[] { testDstVectorAligned1, testDstVectorAligned2 };
        }

        [Theory]
        [InlineData(0, 0, 0, new float[] { -416.6801f, -416.6801f, -416.6801f, -416.6801f, -416.6801f, -416.6801f, -416.6801f, -416.6801f })]
        [InlineData(1, 1, 0, new float[] { 1496f, 3672f, 5848f, 8024f, 10200f, 12376f, 14552f, 16728f })]
        [InlineData(1, 0, 1, new float[] { 204f, 492f, 780f, 1068f, 1356f, 1644f, 1932f, 2220f, 2508f, 2796f, 3084f, 3372f, 3660f, 3948f, 4236f, 4524f })]
        public void MatMulATest(int matTest, int srcTest, int dstTest, float[] expected)
        {
            AlignedArray mat = _testMatrices[matTest];
            AlignedArray src = _testSrcVectors[srcTest];
            AlignedArray dst = _testDstVectors[dstTest];

            CpuMathUtils.MatTimesSrc(false, false, mat, src, dst, dst.Size);
            float[] actual = new float[dst.Size];
            dst.CopyTo(actual, 0, dst.Size);
            Assert.Equal(expected, actual, _matMulComparer);
        }

        [Theory]
        [InlineData(0, 0, 0, new float[] { -416.6801f, -415.6801f, -414.6801f, -413.6801f, -412.6801f, -411.6801f, -410.6801f, -409.6801f })]
        [InlineData(1, 1, 0, new float[] { 1496f, 3673f, 5850f, 8027f, 10204f, 12381f, 14558f, 16735f })]
        [InlineData(1, 0, 1, new float[] { 204f, 493f, 782f, 1071f, 1360f, 1649f, 1938f, 2227f, 2516f, 2805f, 3094f, 3383f, 3672f, 3961f, 4250f, 4539f })]
        public void MatMulAAddTest(int matTest, int srcTest, int dstTest, float[] expected)
        {
            AlignedArray mat = _testMatrices[matTest];
            AlignedArray src = _testSrcVectors[srcTest];
            AlignedArray dst = _testDstVectors[dstTest];

            CpuMathUtils.MatTimesSrc(false, true, mat, src, dst, dst.Size);
            float[] actual = new float[dst.Size];
            dst.CopyTo(actual, 0, dst.Size);
            Assert.Equal(expected, actual, _matMulComparer);
        }

        [Theory]
        [InlineData(0, 0, 0, new float[] { 70.56001f, -85.68f, -351.36f, 498.24f, -3829.32f, -969.48f, 1168.2f, 118.44f })]
        [InlineData(1, 0, 1, new float[] { 2724f, 2760f, 2796f, 2832f, 2868f, 2904f, 2940f, 2976f, 3012f, 3048f, 3084f, 3120f, 3156f, 3192f, 3228f, 3264f })]
        [InlineData(1, 1, 0, new float[] { 11016f, 11152f, 11288f, 11424f, 11560f, 11696f, 11832f, 11968f })]
        public void MatMulTranATest(int matTest, int srcTest, int dstTest, float[] expected)
        {
            AlignedArray mat = _testMatrices[matTest];
            AlignedArray src = _testSrcVectors[srcTest];
            AlignedArray dst = _testDstVectors[dstTest];

            CpuMathUtils.MatTimesSrc(true, false, mat, src, dst, src.Size);
            float[] actual = new float[dst.Size];
            dst.CopyTo(actual, 0, dst.Size);
            Assert.Equal(expected, actual, _matMulComparer);
        }

        [Theory]
        [InlineData(0, 0, 0, new float[] { 70.56001f, -84.68f, -349.36f, 501.24f, -3825.32f, -964.48f, 1174.2f, 125.44f })]
        [InlineData(1, 0, 1, new float[] { 2724f, 2761f, 2798f, 2835f, 2872f, 2909f, 2946f, 2983f, 3020f, 3057f, 3094f, 3131f, 3168f, 3205f, 3242f, 3279f })]
        [InlineData(1, 1, 0, new float[] { 11016f, 11153f, 11290f, 11427f, 11564f, 11701f, 11838f, 11975f })]
        public void MatMulTranAAddTest(int matTest, int srcTest, int dstTest, float[] expected)
        {
            AlignedArray mat = _testMatrices[matTest];
            AlignedArray src = _testSrcVectors[srcTest];
            AlignedArray dst = _testDstVectors[dstTest];

            CpuMathUtils.MatTimesSrc(true, true, mat, src, dst, src.Size);
            float[] actual = new float[dst.Size];
            dst.CopyTo(actual, 0, dst.Size);
            Assert.Equal(expected, actual, _matMulComparer);
        }

        [Theory]
        [InlineData(0, 0, 0, new float[] { 38.25002f, 38.25002f, 38.25002f, 38.25002f, 38.25002f, 38.25002f, 38.25002f, 38.25002f })]
        [InlineData(1, 1, 0, new float[] { 910f, 2190f, 3470f, 4750f, 6030f, 7310f, 8590f, 9870f })]
        [InlineData(1, 0, 1, new float[] { 95f, 231f, 367f, 503f, 639f, 775f, 911f, 1047f, 1183f, 1319f, 1455f, 1591f, 1727f, 1863f, 1999f, 2135f })]
        public void MatMulPATest(int matTest, int srcTest, int dstTest, float[] expected)
        {
            AlignedArray mat = _testMatrices[matTest];
            AlignedArray src = _testSrcVectors[srcTest];
            AlignedArray dst = _testDstVectors[dstTest];
            int[] idx = _testIndexArray;

            CpuMathUtils.MatTimesSrc(false, false, mat, idx, src, 0, 0, (srcTest == 0) ? 4 : 9, dst, dst.Size);
            float[] actual = new float[dst.Size];
            dst.CopyTo(actual, 0, dst.Size);
            Assert.Equal(expected, actual, _matMulComparer);
        }

        [Theory]
        [InlineData(0, 0, 0, new float[] { 38.25002f, 39.25002f, 40.25002f, 41.25002f, 42.25002f, 43.25002f, 44.25002f, 45.25002f })]
        [InlineData(1, 1, 0, new float[] { 910f, 2191f, 3472f, 4753f, 6034f, 7315f, 8596f, 9877f })]
        [InlineData(1, 0, 1, new float[] { 95f, 232f, 369f, 506f, 643f, 780f, 917f, 1054f, 1191f, 1328f, 1465f, 1602f, 1739f, 1876f, 2013f, 2150f })]
        public void MatMulPAAddTest(int matTest, int srcTest, int dstTest, float[] expected)
        {
            AlignedArray mat = _testMatrices[matTest];
            AlignedArray src = _testSrcVectors[srcTest];
            AlignedArray dst = _testDstVectors[dstTest];
            int[] idx = _testIndexArray;

            CpuMathUtils.MatTimesSrc(false, true, mat, idx, src, 0, 0, (srcTest == 0) ? 4 : 9, dst, dst.Size);
            float[] actual = new float[dst.Size];
            dst.CopyTo(actual, 0, dst.Size);
            Assert.Equal(expected, actual, _matMulComparer);
        }

        [Theory]
        [InlineData(0, 0, 0, new float[] { 33.32f, -40.46f, -165.92f, 235.28f, -1808.29f, -457.81f, 551.65f, 55.93f })]
        [InlineData(1, 0, 1, new float[] { 1265f, 1282f, 1299f, 1316f, 1333f, 1350f, 1367f, 1384f, 1401f, 1418f, 1435f, 1452f, 1469f, 1486f, 1503f, 1520f })]
        [InlineData(1, 1, 0, new float[] { 6720f, 6800f, 6880f, 6960f, 7040f, 7120f, 7200f, 7280f })]
        public void MatMulTranPATest(int matTest, int srcTest, int dstTest, float[] expected)
        {
            AlignedArray mat = _testMatrices[matTest];
            AlignedArray src = _testSrcVectors[srcTest];
            AlignedArray dst = _testDstVectors[dstTest];
            int[] idx = _testIndexArray;

            CpuMathUtils.MatTimesSrc(true, false, mat, idx, src, 0, 0, (srcTest == 0) ? 4 : 9, dst, src.Size);
            float[] actual = new float[dst.Size];
            dst.CopyTo(actual, 0, dst.Size);
            Assert.Equal(expected, actual, _matMulComparer);
        }

        [Theory]
        [InlineData(0, 0, 0, new float[] { 33.32f, -39.46f, -163.92f, 238.28f, -1804.29f, -452.81f, 557.65f, 62.93f })]
        [InlineData(1, 0, 1, new float[] { 1265f, 1283f, 1301f, 1319f, 1337f, 1355f, 1373f, 1391f, 1409f, 1427f, 1445f, 1463f, 1481f, 1499f, 1517f, 1535f })]
        [InlineData(1, 1, 0, new float[] { 6720f, 6801f, 6882f, 6963f, 7044f, 7125f, 7206f, 7287f })]
        public void MatMulTranPAAddTest(int matTest, int srcTest, int dstTest, float[] expected)
        {
            AlignedArray mat = _testMatrices[matTest];
            AlignedArray src = _testSrcVectors[srcTest];
            AlignedArray dst = _testDstVectors[dstTest];
            int[] idx = _testIndexArray;

            CpuMathUtils.MatTimesSrc(true, true, mat, idx, src, 0, 0, (srcTest == 0) ? 4 : 9, dst, src.Size);
            float[] actual = new float[dst.Size];
            dst.CopyTo(actual, 0, dst.Size);
            Assert.Equal(expected, actual, _matMulComparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void AddScalarUTest(int test)
        {
            float[] dst = (float[])_testArrays[test].Clone();
            float[] expected = (float[])dst.Clone();

            for (int i = 0; i < expected.Length; i++)
            {
                expected[i] += DefaultScale;
            }

            CpuMathUtils.Add(DefaultScale, dst, dst.Length);
            var actual = dst;
            Assert.Equal(expected, actual, _comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void ScaleUTest(int test)
        {
            float[] dst = (float[])_testArrays[test].Clone();
            float[] expected = (float[])dst.Clone();

            for (int i = 0; i < expected.Length; i++)
            {
                expected[i] *= DefaultScale;
            }

            CpuMathUtils.Scale(DefaultScale, dst, dst.Length);
            var actual = dst;
            Assert.Equal(expected, actual, _comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void ScaleSrcUTest(int test)
        {
            float[] src = (float[])_testArrays[test].Clone();
            float[] dst = (float[])src.Clone();
            float[] expected = (float[])dst.Clone();

            for (int i = 0; i < expected.Length; i++)
            {
                expected[i] *= DefaultScale;
            }

            CpuMathUtils.Scale(DefaultScale, src, dst, dst.Length);
            var actual = dst;
            Assert.Equal(expected, actual, _comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void ScaleAddUTest(int test)
        {
            float[] dst = (float[])_testArrays[test].Clone();
            float[] expected = (float[])dst.Clone();

            for (int i = 0; i < expected.Length; i++)
            {
                expected[i] = DefaultScale * (dst[i] + DefaultScale);
            }

            CpuMathUtils.ScaleAdd(DefaultScale, DefaultScale, dst, dst.Length);
            var actual = dst;
            Assert.Equal(expected, actual, _comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void AddScaleUTest(int test)
        {
            float[] src = (float[])_testArrays[test].Clone();
            float[] dst = (float[])src.Clone();
            float[] expected = (float[])dst.Clone();

            for (int i = 0; i < expected.Length; i++)
            {
                expected[i] *= (1 + DefaultScale);
            }

            CpuMathUtils.AddScale(DefaultScale, src, dst, dst.Length);
            var actual = dst;
            Assert.Equal(expected, actual, _comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void AddScaleSUTest(int test)
        {
            float[] src = (float[])_testArrays[test].Clone();
            float[] dst = (float[])src.Clone();
            int[] idx = _testIndexArray;
            float[] expected = (float[])dst.Clone();

            expected[0] = 5.292f;
            expected[2] = -13.806f;
            expected[5] = -43.522f;
            expected[6] = 55.978f;
            expected[8] = -178.869f;
            expected[11] = -31.941f;
            expected[12] = -51.205f;
            expected[13] = -21.337f;
            expected[14] = 35.782f;

            CpuMathUtils.AddScale(DefaultScale, src, idx, dst, idx.Length);
            var actual = dst;
            Assert.Equal(expected, actual, _comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void AddScaleCopyUTest(int test)
        {
            float[] src = (float[])_testArrays[test].Clone();
            float[] dst = (float[])src.Clone();
            float[] result = (float[])dst.Clone();
            float[] expected = (float[])dst.Clone();

            for (int i = 0; i < expected.Length; i++)
            {
                expected[i] *= (1 + DefaultScale);
            }

            CpuMathUtils.AddScaleCopy(DefaultScale, src, dst, result, dst.Length);
            var actual = result;
            Assert.Equal(expected, actual, _comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void AddUTest(int test)
        {
            float[] src = (float[])_testArrays[test].Clone();
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
            Assert.Equal(expected, actual, _comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void AddSUTest(int test)
        {
            float[] src = (float[])_testArrays[test].Clone();
            float[] dst = (float[])src.Clone();
            int[] idx = _testIndexArray;
            float[] expected = (float[])dst.Clone();

            expected[0] = 3.92f;
            expected[2] = -12.14f;
            expected[5] = -36.69f;
            expected[6] = 46.29f;
            expected[8] = -104.41f;
            expected[11] = -13.09f;
            expected[12] = -73.92f;
            expected[13] = -23.64f;
            expected[14] = 34.41f;

            CpuMathUtils.Add(src, idx, dst, idx.Length);
            var actual = dst;
            Assert.Equal(expected, actual, _comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void MulElementWiseUTest(int test)
        {
            float[] src1 = (float[])_testArrays[test].Clone();
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
            Assert.Equal(expected, actual, _comparer);
        }

        [Theory]
        [InlineData(0, -187.8f)]
        [InlineData(1, -191.09f)]
        public void SumUTest(int test, float expected)
        {
            float[] src = (float[])_testArrays[test].Clone();
            var actual = CpuMathUtils.Sum(src, src.Length);
            Assert.Equal(expected, actual, 2);
        }

        [Theory]
        [InlineData(0, 26799.8752f)]
        [InlineData(1, 26789.0511f)]
        public void SumSqUTest(int test, float expected)
        {
            float[] src = (float[])_testArrays[test].Clone();
            var actual = CpuMathUtils.SumSq(src, src.Length);
            Assert.Equal(expected, actual, 2);
        }

        [Theory]
        [InlineData(0, 27484.6352f)]
        [InlineData(1, 27482.1071f)]
        public void SumSqDiffUTest(int test, float expected)
        {
            float[] src = (float[])_testArrays[test].Clone();
            var actual = CpuMathUtils.SumSq(DefaultScale, src, 0, src.Length);
            Assert.Equal(expected, actual, 2);
        }

        [Theory]
        [InlineData(0, 393.96f)]
        [InlineData(1, 390.67f)]
        public void SumAbsUTest(int test, float expected)
        {
            float[] src = (float[])_testArrays[test].Clone();
            var actual = CpuMathUtils.SumAbs(src, src.Length);
            Assert.Equal(expected, actual, 2);
        }

        [Theory]
        [InlineData(0, 393.96f)]
        [InlineData(1, 392.37f)]
        public void SumAbsDiffUTest(int test, float expected)
        {
            float[] src = (float[])_testArrays[test].Clone();
            var actual = CpuMathUtils.SumAbs(DefaultScale, src, 0, src.Length);
            Assert.Equal(expected, actual, 2);
        }

        [Theory]
        [InlineData(0, 106.37f)]
        [InlineData(1, 106.37f)]
        public void MaxAbsUTest(int test, float expected)
        {
            float[] src = (float[])_testArrays[test].Clone();
            var actual = CpuMathUtils.MaxAbs(src, src.Length);
            Assert.Equal(expected, actual, 2);
        }

        [Theory]
        [InlineData(0, 108.07f)]
        [InlineData(1, 108.07f)]
        public void MaxAbsDiffUTest(int test, float expected)
        {
            float[] src = (float[])_testArrays[test].Clone();
            var actual = CpuMathUtils.MaxAbsDiff(DefaultScale, src, src.Length);
            Assert.Equal(expected, actual, 2);
        }

        [Theory]
        [InlineData(0, 26612.0752f)]
        [InlineData(1, 26597.9611f)]
        public void DotUTest(int test, float expected)
        {
            float[] src = (float[])_testArrays[test].Clone();
            float[] dst = (float[])src.Clone();

            for (int i = 0; i < dst.Length; i++)
            {
                dst[i] += 1;
            }

            var actual = CpuMathUtils.DotProductDense(src, dst, dst.Length);
            Assert.Equal(expected, actual, 1);
        }

        [Theory]
        [InlineData(0, -3406.2154f)]
        [InlineData(1, -3406.2154f)]
        public void DotSUTest(int test, float expected)
        {
            float[] src = (float[])_testArrays[test].Clone();
            float[] dst = (float[])src.Clone();
            int[] idx = _testIndexArray;

            // Ensures src and dst are different arrays
            for (int i = 0; i < dst.Length; i++)
            {
                dst[i] += 1;
            }

            var actual = CpuMathUtils.DotProductSparse(src, dst, idx, idx.Length);
            Assert.Equal(expected, actual, 2);
        }

        [Theory]
        [InlineData(0, 16.0f)]
        [InlineData(1, 15.0f)]
        public void Dist2Test(int test, float expected)
        {
            float[] src = (float[])_testArrays[test].Clone();
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
        [InlineData(0, new int[] { 0, 2, 5, 6 }, new float[] { 0f, 2f, 0f, 4f, 5f, 0f, 0f, 8f })]
        [InlineData(1, new int[] { 0, 2, 5, 6, 8, 11, 12, 13, 14 }, new float[] { 0f, 2f, 0f, 4f, 5f, 0f, 0f, 8f, 0f, 10f, 11f, 0f, 0f, 0f, 0f, 16f })]
        public void ZeroItemsUTest(int test, int[] idx, float[] expected)
        {
            AlignedArray src = new AlignedArray(8 + 8 * test, _vectorAlignment);
            src.CopyFrom(_testSrcVectors[test]);

            CpuMathUtils.ZeroMatrixItems(src, src.Size, src.Size, idx);
            float[] actual = new float[src.Size];
            src.CopyTo(actual, 0, src.Size);
            Assert.Equal(expected, actual, _comparer);
        }

        [Theory]
        [InlineData(0, new int[] { 0, 2, 5 }, new float[] { 0f, 2f, 0f, 4f, 5f, 6f, 0f, 8f })]
        [InlineData(1, new int[] { 0, 2, 5, 6, 8, 11, 12, 13 }, new float[] { 0f, 2f, 0f, 4f, 5f, 0f, 0f, 8f, 9f, 0f, 11f, 12f, 0f, 0f, 0f, 16f })]
        public void ZeroMatrixItemsCoreTest(int test, int[] idx, float[] expected)
        {
            AlignedArray src = new AlignedArray(8 + 8 * test, _vectorAlignment);
            src.CopyFrom(_testSrcVectors[test]);

            CpuMathUtils.ZeroMatrixItems(src, src.Size / 2 - 1, src.Size / 2, idx);
            float[] actual = new float[src.Size];
            src.CopyTo(actual, 0, src.Size);
            Assert.Equal(expected, actual, _comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void SdcaL1UpdateUTest(int test)
        {
            float[] src = (float[])_testArrays[test].Clone();
            float[] v = (float[])src.Clone();
            float[] w = (float[])src.Clone();
            float[] expected = (float[])w.Clone();

            for (int i = 0; i < expected.Length; i++)
            {
                float value = src[i] * (1 + DefaultScale);
                expected[i] = Math.Abs(value) > DefaultScale ? (value > 0 ? value - DefaultScale : value + DefaultScale) : 0;
            }

            CpuMathUtils.SdcaL1UpdateDense(DefaultScale, src.Length, src, DefaultScale, v, w);
            var actual = w;
            Assert.Equal(expected, actual, _comparer);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        public void SdcaL1UpdateSUTest(int test)
        {
            float[] src = (float[])_testArrays[test].Clone();
            float[] v = (float[])src.Clone();
            float[] w = (float[])src.Clone();
            int[] idx = _testIndexArray;
            float[] expected = (float[])w.Clone();

            for (int i = 0; i < idx.Length; i++)
            {
                int index = idx[i];
                float value = v[index] + src[i] * DefaultScale;
                expected[index] = Math.Abs(value) > DefaultScale ? (value > 0 ? value - DefaultScale : value + DefaultScale) : 0;
            }

            CpuMathUtils.SdcaL1UpdateSparse(DefaultScale, src.Length, src, idx, idx.Length, DefaultScale, v, w);
            var actual = w;
            Assert.Equal(expected, actual, _comparer);
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

    internal class FloatEqualityComparerForMatMul : IEqualityComparer<float>
    {
        public bool Equals(float a, float b)
        {
            return Math.Abs(a - b) < 1e-3f;
        }

        public int GetHashCode(float a)
        {
            throw new NotImplementedException();
        }
    }
}