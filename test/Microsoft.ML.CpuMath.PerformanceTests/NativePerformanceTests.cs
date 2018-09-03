// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Microsoft.ML.Runtime.Internal.CpuMath;

namespace Microsoft.ML.CpuMath.PerformanceTests
{
    public class NativePerformanceTests : PerformanceTests
    {
        [Benchmark]
        public unsafe void AddScalarU()
        {
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.AddScalarU(DEFAULT_SCALE, pdst, LEN);
            }
        }
        
        [Benchmark]
        public unsafe void ScaleU()
        {
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.ScaleU(DEFAULT_SCALE, pdst, LEN);
            }
        }
        
        [Benchmark]
        public unsafe void ScaleSrcU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.ScaleSrcU(DEFAULT_SCALE, psrc, pdst, LEN);
            }
        }
        
        [Benchmark]
        public unsafe void ScaleAddU()
        {
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.ScaleAddU(DEFAULT_SCALE, DEFAULT_SCALE, pdst, LEN);
            }
        }
        
        [Benchmark]
        public unsafe void AddScaleU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.AddScaleU(DEFAULT_SCALE, psrc, pdst, LEN);
            }
        }
        
        [Benchmark]
        public unsafe void AddScaleSU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (int* pidx = idx)
            {
                CpuMathNativeUtils.AddScaleSU(DEFAULT_SCALE, psrc, pidx, pdst, IDXLEN);
            }
        }
        
        [Benchmark]
        public unsafe void AddScaleCopyU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (float* pres = result)
            {
                CpuMathNativeUtils.AddScaleCopyU(DEFAULT_SCALE, psrc, pdst, pres, LEN);
            }
        }
        
        [Benchmark]
        public unsafe void AddU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.AddU(psrc, pdst, LEN);
            }
        }
        
        [Benchmark]
        public unsafe void AddSU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (int* pidx = idx)
            {
                CpuMathNativeUtils.AddSU(psrc, pidx, pdst, IDXLEN);
            }
        }
        
        [Benchmark]
        public unsafe void MulElementWiseU()
        {
            fixed (float* psrc1 = src1)
            fixed (float* psrc2 = src2)
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.MulElementWiseU(psrc1, psrc2, pdst, LEN);
            }
        }
        
        [Benchmark]
        public unsafe float SumU()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.SumU(psrc, LEN);
            }
        }
        
        [Benchmark]
        public unsafe float SumSqU()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.SumSqU(psrc, LEN);
            }
        }
        
        [Benchmark]
        public unsafe float SumSqDiffU()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.SumSqDiffU(DEFAULT_SCALE, psrc, LEN);
            }
        }
        
        [Benchmark]
        public unsafe float SumAbsU()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.SumAbsU(psrc, LEN);
            }
        }
        
        [Benchmark]
        public unsafe float SumAbsDiffU()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.SumAbsDiffU(DEFAULT_SCALE, psrc, LEN);
            }
        }
        
        [Benchmark]
        public unsafe float MaxAbsU()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.MaxAbsU(psrc, LEN);
            }
        }
        
        [Benchmark]
        public unsafe float MaxAbsDiffU()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.MaxAbsDiffU(DEFAULT_SCALE, psrc, LEN);
            }
        }
        
        [Benchmark]
        public unsafe float DotU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                return CpuMathNativeUtils.DotU(psrc, pdst, LEN);
            }
        }
        
        [Benchmark]
        public unsafe float DotSU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (int* pidx = idx)
            {
                return CpuMathNativeUtils.DotSU(psrc, pdst, pidx, IDXLEN);
            }
        }

        [Benchmark]
        public unsafe float Dist2()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                return CpuMathNativeUtils.Dist2(psrc, pdst, LEN);
            }
        }

        [Benchmark]
        public unsafe void SdcaL1UpdateU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (float* pres = result)
            {
                CpuMathNativeUtils.SdcaL1UpdateU(DEFAULT_SCALE, psrc, DEFAULT_SCALE, pdst, pres, LEN);
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
                CpuMathNativeUtils.SdcaL1UpdateSU(DEFAULT_SCALE, psrc, pidx, DEFAULT_SCALE, pdst, pres, IDXLEN);
            }
        }
    }
}
