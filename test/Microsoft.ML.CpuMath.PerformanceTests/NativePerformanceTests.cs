// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.CpuMath.Core;

namespace Microsoft.ML.CpuMath.PerformanceTests
{
    public class NativePerformanceTests : PerformanceTests
    {
        private const int CbAlign = 16;

        private static unsafe float* Ptr(AlignedArray a, float* p)
        {
            Contracts.AssertValue(a);
            float* q = p + a.GetBase((long)p);
            Contracts.Assert(((long)q & (CbAlign - 1)) == 0);
            return q;
        }

        [Benchmark]
        public unsafe void AddScalarU()
        {
            fixed (float* pdst = dst)
            {
                Thunk.AddScalarU(DefaultScale, pdst, Length);
            }
        }

        [Benchmark]
        public unsafe void Scale()
        {
            fixed (float* pdst = dst)
            {
                Thunk.Scale(DefaultScale, pdst, Length);
            }
        }

        [Benchmark]
        public unsafe void ScaleSrcU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                Thunk.ScaleSrcU(DefaultScale, psrc, pdst, Length);
            }
        }

        [Benchmark]
        public unsafe void ScaleAddU()
        {
            fixed (float* pdst = dst)
            {
                Thunk.ScaleAddU(DefaultScale, DefaultScale, pdst, Length);
            }
        }

        [Benchmark]
        public unsafe void AddScaleU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                Thunk.AddScaleU(DefaultScale, psrc, pdst, Length);
            }
        }

        [Benchmark]
        public unsafe void AddScaleSU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (int* pidx = idx)
            {
                Thunk.AddScaleSU(DefaultScale, psrc, pidx, pdst, IndexLength);
            }
        }

        [Benchmark]
        public unsafe void AddScaleCopyU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (float* pres = result)
            {
                Thunk.AddScaleCopyU(DefaultScale, psrc, pdst, pres, Length);
            }
        }

        [Benchmark]
        public unsafe void AddU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                Thunk.AddU(psrc, pdst, Length);
            }
        }

        [Benchmark]
        public unsafe void AddSU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (int* pidx = idx)
            {
                Thunk.AddSU(psrc, pidx, pdst, IndexLength);
            }
        }

        [Benchmark]
        public unsafe void MulElementWiseU()
        {
            fixed (float* psrc1 = src1)
            fixed (float* psrc2 = src2)
            fixed (float* pdst = dst)
            {
                Thunk.MulElementWiseU(psrc1, psrc2, pdst, Length);
            }
        }

        [Benchmark]
        public unsafe float Sum()
        {
            fixed (float* psrc = src)
            {
                return Thunk.Sum(psrc, Length);
            }
        }

        [Benchmark]
        public unsafe float SumSqU()
        {
            fixed (float* psrc = src)
            {
                return Thunk.SumSqU(psrc, Length);
            }
        }

        [Benchmark]
        public unsafe float SumSqDiffU()
        {
            fixed (float* psrc = src)
            {
                return Thunk.SumSqDiffU(DefaultScale, psrc, Length);
            }
        }

        [Benchmark]
        public unsafe float SumAbsU()
        {
            fixed (float* psrc = src)
            {
                return Thunk.SumAbsU(psrc, Length);
            }
        }

        [Benchmark]
        public unsafe float SumAbsDiffU()
        {
            fixed (float* psrc = src)
            {
                return Thunk.SumAbsDiffU(DefaultScale, psrc, Length);
            }
        }

        [Benchmark]
        public unsafe float MaxAbsU()
        {
            fixed (float* psrc = src)
            {
                return Thunk.MaxAbsU(psrc, Length);
            }
        }

        [Benchmark]
        public unsafe float MaxAbsDiffU()
        {
            fixed (float* psrc = src)
            {
                return Thunk.MaxAbsDiffU(DefaultScale, psrc, Length);
            }
        }

        [Benchmark]
        public unsafe float DotU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                return Thunk.DotU(psrc, pdst, Length);
            }
        }

        [Benchmark]
        public unsafe float DotSU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (int* pidx = idx)
            {
                return Thunk.DotSU(psrc, pdst, pidx, IndexLength);
            }
        }

        [Benchmark]
        public unsafe float Dist2()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                return Thunk.Dist2(psrc, pdst, Length);
            }
        }

        [Benchmark]
        public unsafe void SdcaL1UpdateU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (float* pres = result)
            {
                Thunk.SdcaL1UpdateU(DefaultScale, psrc, DefaultScale, pdst, pres, Length);
            }
        }

        [Benchmark]
        public unsafe void SdcaL1UpdateSU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (float* pres = result)
            fixed (int* pidx = idx)
            {
                Thunk.SdcaL1UpdateSU(DefaultScale, psrc, pidx, DefaultScale, pdst, pres, IndexLength);
            }
        }

        [Benchmark]
        public unsafe void MatMul()
        {
            fixed (float* pmat = &testMatrixAligned.Items[0])
            fixed (float* psrc = &testSrcVectorAligned.Items[0])
            fixed (float* pdst = &testDstVectorAligned.Items[0])
                Thunk.MatMul(Ptr(testMatrixAligned, pmat), Ptr(testSrcVectorAligned, psrc), Ptr(testDstVectorAligned, pdst), matrixLength, testSrcVectorAligned.Size);
        }

        [Benchmark]
        public unsafe void MatMulTran()
        {
            fixed (float* pmat = &testMatrixAligned.Items[0])
            fixed (float* psrc = &testSrcVectorAligned.Items[0])
            fixed (float* pdst = &testDstVectorAligned.Items[0])
                Thunk.MatMulTran(Ptr(testMatrixAligned, pmat), Ptr(testSrcVectorAligned, psrc), Ptr(testDstVectorAligned, pdst), testDstVectorAligned.Size, matrixLength);
        }

        [Benchmark]
        public unsafe void MatMulP()
        {
            fixed (float* pmat = &testMatrixAligned.Items[0])
            fixed (float* psrc = &testSrcVectorAligned.Items[0])
            fixed (float* pdst = &testDstVectorAligned.Items[0])
            fixed (int* ppossrc = &matrixIdx[0])
                Thunk.MatMulP(Ptr(testMatrixAligned, pmat), ppossrc, Ptr(testSrcVectorAligned, psrc), 0, 0, MatrixIndexLength, Ptr(testDstVectorAligned, pdst), matrixLength, testSrcVectorAligned.Size);
        }
    }
}
