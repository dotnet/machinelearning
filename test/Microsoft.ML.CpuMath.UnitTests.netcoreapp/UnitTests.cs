// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.TestFramework;
using Xunit;

namespace Microsoft.ML.CpuMath.UnitTests
{
    public class CpuMathUtilsUnitTests
    {
        private static readonly float[][] _testArrays;
        private static readonly int[] _testIndexArray;
        private static readonly AlignedArray[] _testMatrices;
        private static readonly AlignedArray[] _testSrcVectors;
        private static readonly AlignedArray[] _testDstVectors;
        private static readonly int _vectorAlignment = CpuMathUtils.GetVectorAlignment();
        private static readonly FloatEqualityComparer _comparer;
        private static readonly FloatEqualityComparerForMatMul _matMulComparer;
        private static readonly string defaultMode = "defaultMode";
#if NETCOREAPP3_0
        private static Dictionary<string, string> DisableAvxEnvironmentVariables;
        private static Dictionary<string, string> DisableAvxAndSseEnvironmentVariables;
        private static readonly string disableAvx = "COMPlus_EnableAVX";
        private static readonly string disableSse = "COMPlus_EnableSSE";
        private static readonly string disableAvxAndSse = "COMPlus_EnableHWIntrinsic";
#endif

        static CpuMathUtilsUnitTests()
        {
            // Padded array whose length is a multiple of 4
            float[] testArray1 = new float[32] { 1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f, 3.29f, 1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f, 3.29f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f };
            // Unpadded array whose length is not a multiple of 4.
            float[] testArray2 = new float[30] { 1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f, 3.29f, 1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f };
            // Small Input Size Array
            float[] testArray3 = new float[15] { 1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f, 3.29f, 1.96f, -2.38f, -9.76f, 13.84f, -106.37f, -26.93f, 32.45f };
            _testArrays = new float[][] { testArray1, testArray2, testArray3 };
            _testIndexArray = new int[18] { 0, 2, 5, 6, 8, 11, 12, 13, 14, 16, 18, 21, 22, 24, 26, 27, 28, 29};
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
            testMatrixAligned1.CopyFrom(testMatrix1);
            testMatrixAligned2.CopyFrom(testMatrix2);

            _testMatrices = new AlignedArray[] { testMatrixAligned1, testMatrixAligned2 };

            // Padded source vectors whose dimensions are multiples of 8
            float[] testSrcVector1 = new float[8] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f };
            float[] testSrcVector2 = new float[16] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f };

            AlignedArray testSrcVectorAligned1 = new AlignedArray(8, _vectorAlignment);
            AlignedArray testSrcVectorAligned2 = new AlignedArray(16, _vectorAlignment);
            testSrcVectorAligned1.CopyFrom(testSrcVector1);
            testSrcVectorAligned2.CopyFrom(testSrcVector2);

            _testSrcVectors = new AlignedArray[] { testSrcVectorAligned1, testSrcVectorAligned2 };

            // Padded destination vectors whose dimensions are multiples of 8
            float[] testDstVector1 = new float[8] { 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f };
            float[] testDstVector2 = new float[16] { 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f };

            AlignedArray testDstVectorAligned1 = new AlignedArray(8, _vectorAlignment);
            AlignedArray testDstVectorAligned2 = new AlignedArray(16, _vectorAlignment);
            testDstVectorAligned1.CopyFrom(testDstVector1);
            testDstVectorAligned2.CopyFrom(testDstVector2);

            _testDstVectors = new AlignedArray[] { testDstVectorAligned1, testDstVectorAligned2 };

#if NETCOREAPP3_0
            DisableAvxEnvironmentVariables = new Dictionary<string, string>()
            {
                { disableAvx , "0" }
            };

            DisableAvxAndSseEnvironmentVariables = new Dictionary<string, string>()
            {
                { disableAvx , "0" },
                { disableSse , "0" }
            };
#endif
        }

        private static void CheckProperFlag(string mode)
        {
#if NETCOREAPP3_0
            if (mode == defaultMode)
            {
                Assert.True(System.Runtime.Intrinsics.X86.Avx.IsSupported);
                Assert.True(System.Runtime.Intrinsics.X86.Sse.IsSupported);
            }
            else if (mode == disableAvx)
            {
                Assert.False(System.Runtime.Intrinsics.X86.Avx.IsSupported);
                Assert.True(System.Runtime.Intrinsics.X86.Sse.IsSupported);
            }
            else if (mode == disableAvxAndSse)
            {
                Assert.False(System.Runtime.Intrinsics.X86.Avx.IsSupported);
                Assert.False(System.Runtime.Intrinsics.X86.Sse.IsSupported);
            }
#endif
        }

        public static TheoryData<string, string, Dictionary<string, string>> AddData() => new TheoryData<string, string, Dictionary<string, string>>()
        {
            {  defaultMode, "0", null },
            {  defaultMode, "1", null },
            {  defaultMode, "2", null },
#if NETCOREAPP3_0
            { disableAvx, "0", DisableAvxEnvironmentVariables },
            { disableAvx, "1", DisableAvxEnvironmentVariables },

            { disableAvxAndSse, "0", DisableAvxAndSseEnvironmentVariables },
            { disableAvxAndSse, "1", DisableAvxAndSseEnvironmentVariables },
#endif
        };

        public static TheoryData<string, string, string, Dictionary<string, string>> AddScaleData() => new TheoryData<string, string, string, Dictionary<string, string>>()
        {
            {  defaultMode, "0", "1.7", null },
            {  defaultMode, "1", "1.7", null },
            {  defaultMode, "2", "1.7", null },
            {  defaultMode, "0", "-1.7", null },
            {  defaultMode, "1", "-1.7", null },
            {  defaultMode, "2", "-1.7", null },
#if NETCOREAPP3_0
            {  disableAvx, "0", "1.7", DisableAvxEnvironmentVariables },
            {  disableAvx, "1", "1.7", DisableAvxEnvironmentVariables },
            {  disableAvx, "0", "-1.7", DisableAvxEnvironmentVariables },
            {  disableAvx, "1", "-1.7", DisableAvxEnvironmentVariables },

            { disableAvxAndSse, "0", "1.7", DisableAvxAndSseEnvironmentVariables },
            { disableAvxAndSse, "1", "1.7", DisableAvxAndSseEnvironmentVariables },
            { disableAvxAndSse, "0", "-1.7", DisableAvxAndSseEnvironmentVariables },
            { disableAvxAndSse, "1", "-1.7", DisableAvxAndSseEnvironmentVariables },
#endif
        };

        public static TheoryData<string, string, string, string, Dictionary<string, string>> MatMulData => new TheoryData<string, string, string, string, Dictionary<string, string>>()
        {
            { defaultMode, "0", "0", "0", null },
            { defaultMode, "1", "1", "0", null },
            { defaultMode, "1", "0", "1", null },
#if NETCOREAPP3_0
            { disableAvx, "0", "0", "0", DisableAvxEnvironmentVariables },
            { disableAvx, "1", "1", "0", DisableAvxEnvironmentVariables },
            { disableAvx, "1", "0", "1", DisableAvxEnvironmentVariables },

            { disableAvxAndSse , "0", "0", "0", DisableAvxAndSseEnvironmentVariables },
            { disableAvxAndSse , "1", "1", "0", DisableAvxAndSseEnvironmentVariables },
            { disableAvxAndSse , "1", "0", "1", DisableAvxAndSseEnvironmentVariables },
#endif
        };

        [Theory]
        [MemberData(nameof(MatMulData))]
        public void MatMulTest(string mode, string matTest, string srcTest, string dstTest, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1, arg2, arg3) =>
            {
                CheckProperFlag(arg0);
                AlignedArray mat = _testMatrices[int.Parse(arg1)];
                AlignedArray src = _testSrcVectors[int.Parse(arg2)];
                AlignedArray dst = _testDstVectors[int.Parse(arg3)];

                float[] expected = new float[dst.Size];
                for (int i = 0; i < dst.Size; i++)
                {
                    float dotProduct = 0;
                    for (int j = 0; j < src.Size; j++)
                    {
                        dotProduct += mat[i * src.Size + j] * src[j];
                    }                    
                    expected[i] = dotProduct;
                }

                CpuMathUtils.MatrixTimesSource(false, mat, src, dst, dst.Size);
                float[] actual = new float[dst.Size];
                dst.CopyTo(actual, 0, dst.Size);
                Assert.Equal(expected, actual, _matMulComparer);
                return RemoteExecutor.SuccessExitCode;
            }, mode, matTest, srcTest, dstTest, new RemoteInvokeOptions(environmentVariables));
        }

        [Theory]
        [MemberData(nameof(MatMulData))]
        public void MatMulTranTest(string mode, string matTest, string srcTest, string dstTest, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1, arg2, arg3) =>
            {
                CheckProperFlag(arg0);
                AlignedArray mat = _testMatrices[int.Parse(arg1)];
                AlignedArray src = _testSrcVectors[int.Parse(arg2)];
                AlignedArray dst = _testDstVectors[int.Parse(arg3)];

                float[] expected = new float[dst.Size];
                for (int i = 0; i < dst.Size; i++)
                {
                    float dotProduct = 0;
                    for (int j = 0; j < src.Size; j++)
                    {
                        dotProduct += mat[j * dst.Size + i] * src[j];
                    }
                    expected[i] = dotProduct;
                }

                CpuMathUtils.MatrixTimesSource(true, mat, src, dst, src.Size);
                float[] actual = new float[dst.Size];
                dst.CopyTo(actual, 0, dst.Size);
                Assert.Equal(expected, actual, _matMulComparer);
                return RemoteExecutor.SuccessExitCode;
            }, mode, matTest, srcTest, dstTest, new RemoteInvokeOptions(environmentVariables));
        }

        [Theory]
        [MemberData(nameof(MatMulData))]
        public void MatTimesSrcSparseTest(string mode, string matTest, string srcTest, string dstTest, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1, arg2, arg3) =>
            {
                CheckProperFlag(arg0);
                AlignedArray mat = _testMatrices[int.Parse(arg1)];
                AlignedArray src = _testSrcVectors[int.Parse(arg2)];
                AlignedArray dst = _testDstVectors[int.Parse(arg3)];
                int[] idx = _testIndexArray;

                float[] expected = new float[dst.Size];
                int limit = (int.Parse(arg2) == 0) ? 4 : 9;
                for (int i = 0; i < dst.Size; i++)
                {
                    float dotProduct = 0;
                    for (int j = 0; j < limit; j++)
                    {
                        int col = idx[j];
                        dotProduct += mat[i * src.Size + col] * src[col];
                    }
                    expected[i] = dotProduct;
                }

                CpuMathUtils.MatrixTimesSource(mat, idx, src, 0, 0, limit, dst, dst.Size);
                float[] actual = new float[dst.Size];
                dst.CopyTo(actual, 0, dst.Size);
                Assert.Equal(expected, actual, _matMulComparer);
                return RemoteExecutor.SuccessExitCode;

            }, mode, matTest, srcTest, dstTest, new RemoteInvokeOptions(environmentVariables));
        }

        [Theory]
        [MemberData(nameof(AddScaleData))]
        public void AddScalarUTest(string mode, string test, string scale, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1, arg2) =>
            {
                CheckProperFlag(arg0);
                float defaultScale = float.Parse(arg2, CultureInfo.InvariantCulture);
                float[] dst = (float[])_testArrays[int.Parse(arg1)].Clone();
                float[] expected = (float[])dst.Clone();

                for (int i = 0; i < expected.Length; i++)
                {
                    expected[i] += defaultScale;
                }

                CpuMathUtils.Add(defaultScale, dst);
                var actual = dst;
                Assert.Equal(expected, actual, _comparer);
                return RemoteExecutor.SuccessExitCode;
            }, mode, test, scale, new RemoteInvokeOptions(environmentVariables));
        }

        [Theory]
        [MemberData(nameof(AddScaleData))]
        public void ScaleTest(string mode, string test, string scale, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1, arg2) =>
            {
                CheckProperFlag(arg0);
                float defaultScale = float.Parse(arg2, CultureInfo.InvariantCulture);
                float[] dst = (float[])_testArrays[int.Parse(arg1)].Clone();
                float[] expected = (float[])dst.Clone();

                for (int i = 0; i < expected.Length; i++)
                {
                    expected[i] *= defaultScale;
                }

                CpuMathUtils.Scale(defaultScale, dst);
                var actual = dst;
                Assert.Equal(expected, actual, _comparer);
                return RemoteExecutor.SuccessExitCode;
            }, mode, test, scale, new RemoteInvokeOptions(environmentVariables));
        }

        [Theory]
        [MemberData(nameof(AddScaleData))]
        public void ScaleSrcUTest(string mode, string test, string scale, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1, arg2) =>
            {
                CheckProperFlag(arg0);
                float defaultScale = float.Parse(arg2, CultureInfo.InvariantCulture);
                float[] src = (float[])_testArrays[int.Parse(arg1)].Clone();
                float[] dst = (float[])src.Clone();
                float[] expected = (float[])dst.Clone();

                for (int i = 0; i < expected.Length; i++)
                {
                    expected[i] *= defaultScale;
                }

                CpuMathUtils.Scale(defaultScale, src, dst, dst.Length);
                var actual = dst;
                Assert.Equal(expected, actual, _comparer);
                return RemoteExecutor.SuccessExitCode;
            }, mode, test, scale, new RemoteInvokeOptions(environmentVariables));
        }

        [Theory]
        [MemberData(nameof(AddScaleData))]
        public void ScaleAddUTest(string mode, string test, string scale, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1, arg2) =>
            {
                CheckProperFlag(arg0);
                float defaultScale = float.Parse(arg2, CultureInfo.InvariantCulture);
                float[] dst = (float[])_testArrays[int.Parse(arg1)].Clone();
                float[] expected = (float[])dst.Clone();

                for (int i = 0; i < expected.Length; i++)
                {
                    expected[i] = defaultScale * (dst[i] + defaultScale);
                }

                CpuMathUtils.ScaleAdd(defaultScale, defaultScale, dst);
                var actual = dst;
                Assert.Equal(expected, actual, _comparer);
                return RemoteExecutor.SuccessExitCode;
            }, mode, test, scale, new RemoteInvokeOptions(environmentVariables));

        }

        [Theory]
        [MemberData(nameof(AddScaleData))]
        public void AddScaleUTest(string mode, string test, string scale, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1, arg2) =>
            {
                CheckProperFlag(arg0);
                float defaultScale = float.Parse(arg2, CultureInfo.InvariantCulture);
                float[] src = (float[])_testArrays[int.Parse(arg1)].Clone();
                float[] dst = (float[])src.Clone();
                float[] expected = (float[])dst.Clone();

                for (int i = 0; i < expected.Length; i++)
                {
                    expected[i] *= (1 + defaultScale);
                }

                CpuMathUtils.AddScale(defaultScale, src, dst, dst.Length);
                var actual = dst;
                Assert.Equal(expected, actual, _comparer);
                return RemoteExecutor.SuccessExitCode;
            }, mode, test, scale, new RemoteInvokeOptions(environmentVariables));
        }

        [Theory]
        [MemberData(nameof(AddScaleData))]
        public void AddScaleSUTest(string mode, string test, string scale, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1, arg2) =>
            {
                CheckProperFlag(arg0);
                float defaultScale = float.Parse(arg2, CultureInfo.InvariantCulture);
                float[] src = (float[])_testArrays[int.Parse(arg1)].Clone();
                float[] dst = (float[])src.Clone();
                int[] idx = _testIndexArray;
                int limit = int.Parse(arg1) == 2 ? 9 : 18;
                float[] expected = (float[])dst.Clone();

                CpuMathUtils.AddScale(defaultScale, src, idx, dst, limit);
                for (int i = 0; i < limit; i++)
                {
                    int index = idx[i];
                    expected[index] += defaultScale * src[i];
                }

                Assert.Equal(expected, dst, _comparer);
                return RemoteExecutor.SuccessExitCode;
            }, mode, test, scale, new RemoteInvokeOptions(environmentVariables));
        }

        [Theory]
        [MemberData(nameof(AddScaleData))]
        public void AddScaleCopyUTest(string mode, string test, string scale, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1, arg2) =>
            {
                CheckProperFlag(arg0);
                float defaultScale = float.Parse(arg2, CultureInfo.InvariantCulture);
                float[] src = (float[])_testArrays[int.Parse(arg1)].Clone();
                float[] dst = (float[])src.Clone();
                float[] result = (float[])dst.Clone();
                float[] expected = (float[])dst.Clone();

                for (int i = 0; i < expected.Length; i++)
                {
                    expected[i] *= (1 + defaultScale);
                }

                CpuMathUtils.AddScaleCopy(defaultScale, src, dst, result, dst.Length);
                var actual = result;
                Assert.Equal(expected, actual, _comparer);
                return RemoteExecutor.SuccessExitCode;
            }, mode, test, scale, new RemoteInvokeOptions(environmentVariables));
        }

        [Theory]
        [MemberData(nameof(AddData))]
        public void AddUTest(string mode, string test, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1) => 
            {
                CheckProperFlag(arg0);
                float[] src = (float[])_testArrays[int.Parse(arg1)].Clone();
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
                return RemoteExecutor.SuccessExitCode;
            }, mode, test, new RemoteInvokeOptions (environmentVariables));
        }

        [Theory]
        [MemberData(nameof(AddData))]
        public void AddSUTest(string mode, string test, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1) =>
            {
                CheckProperFlag(arg0);
                float[] src = (float[])_testArrays[int.Parse(arg1)].Clone();
                float[] dst = (float[])src.Clone();
                int[] idx = _testIndexArray;
                int limit = int.Parse(arg1) == 2 ? 9 : 18;
                float[] expected = (float[])dst.Clone();

                for (int i = 0; i < limit; i++)
                {
                    int index = idx[i];
                    expected[index] += src[i];
                }

                CpuMathUtils.Add(src, idx, dst, limit);
                var actual = dst;
                Assert.Equal(expected, actual, _comparer);
                return RemoteExecutor.SuccessExitCode;
            }, mode, test, new RemoteInvokeOptions(environmentVariables));
        }

        [Theory]
        [MemberData(nameof(AddData))]
        public void MulElementWiseUTest(string mode, string test, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1) =>
            {
                CheckProperFlag(arg1);
                float[] src1 = (float[])_testArrays[int.Parse(arg1)].Clone();
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
                return RemoteExecutor.SuccessExitCode; 
            }, mode, test, new RemoteInvokeOptions(environmentVariables));
        }

        [Theory]
        [MemberData(nameof(AddData))]
        public void SumTest(string mode, string test, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1) =>
            {
                CheckProperFlag(arg0);
                float[] src = (float[])_testArrays[int.Parse(arg1)].Clone();
                float expected = 0;
                for (int i = 0; i < src.Length; i++)
                {
                    expected += src[i];
                }

                var actual = CpuMathUtils.Sum(src);
                Assert.Equal(expected, actual, 2);
                return RemoteExecutor.SuccessExitCode;
            }, mode, test, new RemoteInvokeOptions(environmentVariables));
        }

        [Theory]
        [MemberData(nameof(AddData))]
        public void SumSqUTest(string mode, string test, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1) =>
            {
                CheckProperFlag(arg0);
                float[] src = (float[])_testArrays[int.Parse(arg1)].Clone();
                float expected = 0;
                for (int i = 0; i < src.Length; i++)
                {
                    expected += src[i] * src[i];
                }

                var actual = CpuMathUtils.SumSq(src);
                Assert.Equal(expected, actual, 2);
                return RemoteExecutor.SuccessExitCode;
            }, mode, test, new RemoteInvokeOptions(environmentVariables));
        }

        [Theory]
        [MemberData(nameof(AddScaleData))]
        public void SumSqDiffUTest(string mode, string test, string scale, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1, arg2) =>
            {
                CheckProperFlag(arg0);
                float defaultScale = float.Parse(arg2, CultureInfo.InvariantCulture);
                float[] src = (float[])_testArrays[int.Parse(arg1)].Clone();
                var actual = CpuMathUtils.SumSq(defaultScale, src);

                float expected = 0;
                for (int i = 0; i < src.Length; i++)
                {
                    expected += (src[i] - defaultScale) * (src[i] - defaultScale);
                }

                Assert.Equal(expected, actual, 1);
                return RemoteExecutor.SuccessExitCode;
            }, mode, test, scale, new RemoteInvokeOptions(environmentVariables));
        }

        [Theory]
        [MemberData(nameof(AddData))]
        public void SumAbsUTest(string mode, string test, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1) =>
            {
                CheckProperFlag(arg0);
                float[] src = (float[])_testArrays[int.Parse(arg1)].Clone();
                float expected = 0;
                for (int i = 0; i < src.Length; i++)
                {
                    expected += Math.Abs(src[i]);
                }

                var actual = CpuMathUtils.SumAbs(src);
                Assert.Equal(expected, actual, 2);
                return RemoteExecutor.SuccessExitCode;
            }, mode, test, new RemoteInvokeOptions(environmentVariables));
        }

        [Theory]
        [MemberData(nameof(AddScaleData))]
        public void SumAbsDiffUTest(string mode, string test, string scale, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1, arg2) =>
            {
                CheckProperFlag(arg0);
                float defaultScale = float.Parse(arg2, CultureInfo.InvariantCulture);
                float[] src = (float[])_testArrays[int.Parse(arg1)].Clone();
                var actual = CpuMathUtils.SumAbs(defaultScale, src);

                float expected = 0;
                for (int i = 0; i < src.Length; i++)
                {
                    expected += Math.Abs(src[i] - defaultScale);
                }

                Assert.Equal(expected, actual, 2);
                return RemoteExecutor.SuccessExitCode;
            }, mode, test, scale, new RemoteInvokeOptions(environmentVariables));
        }

        [Theory]
        [MemberData(nameof(AddData))]
        public void MaxAbsUTest(string mode, string test, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1) =>
            {
                CheckProperFlag(arg0);
                float[] src = (float[])_testArrays[int.Parse(arg1)].Clone();
                var actual = CpuMathUtils.MaxAbs(src);

                float expected = 0;
                for (int i = 0; i < src.Length; i++)
                {
                    float abs = Math.Abs(src[i]);
                    if (abs > expected)
                    {
                        expected = abs;
                    }
                }

                Assert.Equal(expected, actual, 2);
                return RemoteExecutor.SuccessExitCode;
            }, mode, test, new RemoteInvokeOptions(environmentVariables));
        }

        [Theory]
        [MemberData(nameof(AddScaleData))]
        public void MaxAbsDiffUTest(string mode, string test, string scale, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1, arg2) =>
            {
                CheckProperFlag(arg0);
                float defaultScale = float.Parse(arg2, CultureInfo.InvariantCulture);
                float[] src = (float[])_testArrays[int.Parse(arg1)].Clone();
                var actual = CpuMathUtils.MaxAbsDiff(defaultScale, src);

                float expected = 0;
                for (int i = 0; i < src.Length; i++)
                {
                    float abs = Math.Abs(src[i] - defaultScale);
                    if (abs > expected)
                    {
                        expected = abs;
                    }
                }
                Assert.Equal(expected, actual, 2);
                return RemoteExecutor.SuccessExitCode;
            }, mode, test, scale, new RemoteInvokeOptions(environmentVariables));
        }

        [Theory]
        [MemberData(nameof(AddData))]
        public void DotUTest(string mode, string test, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1) =>
            {
                CheckProperFlag(arg0);
                float[] src = (float[])_testArrays[int.Parse(arg1)].Clone();
                float[] dst = (float[])src.Clone();

                for (int i = 0; i < dst.Length; i++)
                {
                    dst[i] += 1;
                }

                float expected = 0;
                for (int i = 0; i < dst.Length; i++)
                {
                    expected += src[i] * dst[i];
                }

                var actual = CpuMathUtils.DotProductDense(src, dst, dst.Length);
                Assert.Equal(expected, actual, 1);
                return RemoteExecutor.SuccessExitCode;
            }, mode, test, new RemoteInvokeOptions(environmentVariables));
        }

        [Theory]
        [MemberData(nameof(AddData))]
        public void DotSUTest(string mode, string test, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1) =>
            {
                CheckProperFlag(arg0);
                float[] src = (float[])_testArrays[int.Parse(arg1)].Clone();
                float[] dst = (float[])src.Clone();
                int[] idx = _testIndexArray;
                int limit = int.Parse(arg1) == 2 ? 9 : 18;

                // Ensures src and dst are different arrays
                for (int i = 0; i < dst.Length; i++)
                {
                    dst[i] += 1;
                }

                float expected = 0;
                for (int i = 0; i < limit; i++)
                {
                    int index = idx[i];
                    expected += src[index] * dst[i];
                }

                var actual = CpuMathUtils.DotProductSparse(src, dst, idx, limit);
                Assert.Equal(expected, actual, 2);
                return RemoteExecutor.SuccessExitCode;
            }, mode, test, new RemoteInvokeOptions(environmentVariables));
        }

        [Theory]
        [MemberData(nameof(AddData))]
        public void Dist2Test(string mode, string test, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1) =>
            {
                CheckProperFlag(arg0);
                float[] src = (float[])_testArrays[int.Parse(arg1)].Clone();
                float[] dst = (float[])src.Clone();

                // Ensures src and dst are different arrays
                for (int i = 0; i < dst.Length; i++)
                {
                    dst[i] += 1;
                }

                float expected = 0;
                for (int i = 0; i < dst.Length; i++)
                {
                    float distance = src[i] - dst[i];
                    expected += distance * distance;
                }

                var actual = CpuMathUtils.L2DistSquared(src, dst, dst.Length);
                Assert.Equal(expected, actual, 0);
                return RemoteExecutor.SuccessExitCode;
            }, mode, test, new RemoteInvokeOptions(environmentVariables));
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
        [MemberData(nameof(AddScaleData))]
        public void SdcaL1UpdateUTest(string mode, string test, string scale, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1, arg2) =>
            {
                CheckProperFlag(arg0);
                float defaultScale = float.Parse(arg2, CultureInfo.InvariantCulture);
                float[] src = (float[])_testArrays[int.Parse(arg1)].Clone();
                float[] v = (float[])src.Clone();
                float[] w = (float[])src.Clone();
                float[] expected = (float[])w.Clone();

                for (int i = 0; i < expected.Length; i++)
                {
                    float value = src[i] * (1 + defaultScale);
                    expected[i] = Math.Abs(value) > defaultScale ? (value > 0 ? value - defaultScale : value + defaultScale) : 0;
                }

                CpuMathUtils.SdcaL1UpdateDense(defaultScale, src.Length, src, defaultScale, v, w);
                var actual = w;
                Assert.Equal(expected, actual, _comparer);
                return RemoteExecutor.SuccessExitCode;
            }, mode, test, scale, new RemoteInvokeOptions(environmentVariables));
        }


        [Theory]
        [MemberData(nameof(AddScaleData))]
        public void SdcaL1UpdateSUTest(string mode, string test, string scale, Dictionary<string, string> environmentVariables)
        {
            RemoteExecutor.RemoteInvoke((arg0, arg1, arg2) =>
            {
                CheckProperFlag(arg0);
                float defaultScale = float.Parse(arg2, CultureInfo.InvariantCulture);
                float[] src = (float[])_testArrays[int.Parse(arg1)].Clone();
                float[] v = (float[])src.Clone();
                float[] w = (float[])src.Clone();
                int[] idx = _testIndexArray;
                int limit = int.Parse(arg1) == 2 ? 9 : 18;
                float[] expected = (float[])w.Clone();

                for (int i = 0; i < limit; i++)
                {
                    int index = idx[i];
                    float value = v[index] + src[i] * defaultScale;
                    expected[index] = Math.Abs(value) > defaultScale ? (value > 0 ? value - defaultScale : value + defaultScale) : 0;
                }

                CpuMathUtils.SdcaL1UpdateSparse(defaultScale, limit, src, idx, defaultScale, v, w);
                var actual = w;
                Assert.Equal(expected, actual, _comparer);
                return RemoteExecutor.SuccessExitCode;
            }, mode, test, scale, new RemoteInvokeOptions(environmentVariables));
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