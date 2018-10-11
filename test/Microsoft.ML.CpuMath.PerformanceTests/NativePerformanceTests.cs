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
                CpuMathNativeUtils.AddScalarU(DefaultScale, pdst, Length);
            }
        }
        
        [Benchmark]
        public unsafe void Scale()
        {
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.Scale(DefaultScale, pdst, Length);
            }
        }
        
        [Benchmark]
        public unsafe void ScaleSrcU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.ScaleSrcU(DefaultScale, psrc, pdst, Length);
            }
        }
        
        [Benchmark]
        public unsafe void ScaleAddU()
        {
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.ScaleAddU(DefaultScale, DefaultScale, pdst, Length);
            }
        }
        
        [Benchmark]
        public unsafe void AddScaleU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.AddScaleU(DefaultScale, psrc, pdst, Length);
            }
        }
        
        [Benchmark]
        public unsafe void AddScaleSU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (int* pidx = idx)
            {
                CpuMathNativeUtils.AddScaleSU(DefaultScale, psrc, pidx, pdst, IndexLength);
            }
        }
        
        [Benchmark]
        public unsafe void AddScaleCopyU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (float* pres = result)
            {
                CpuMathNativeUtils.AddScaleCopyU(DefaultScale, psrc, pdst, pres, Length);
            }
        }
        
        [Benchmark]
        public unsafe void AddU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.AddU(psrc, pdst, Length);
            }
        }
        
        [Benchmark]
        public unsafe void AddSU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (int* pidx = idx)
            {
                CpuMathNativeUtils.AddSU(psrc, pidx, pdst, IndexLength);
            }
        }
        
        [Benchmark]
        public unsafe void MulElementWiseU()
        {
            fixed (float* psrc1 = src1)
            fixed (float* psrc2 = src2)
            fixed (float* pdst = dst)
            {
                CpuMathNativeUtils.MulElementWiseU(psrc1, psrc2, pdst, Length);
            }
        }
        
        [Benchmark]
        public unsafe float SumU()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.SumU(psrc, Length);
            }
        }
        
        [Benchmark]
        public unsafe float SumSqU()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.SumSqU(psrc, Length);
            }
        }
        
        [Benchmark]
        public unsafe float SumSqDiffU()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.SumSqDiffU(DefaultScale, psrc, Length);
            }
        }
        
        [Benchmark]
        public unsafe float SumAbsU()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.SumAbsU(psrc, Length);
            }
        }
        
        [Benchmark]
        public unsafe float SumAbsDiffU()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.SumAbsDiffU(DefaultScale, psrc, Length);
            }
        }
        
        [Benchmark]
        public unsafe float MaxAbsU()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.MaxAbsU(psrc, Length);
            }
        }
        
        [Benchmark]
        public unsafe float MaxAbsDiffU()
        {
            fixed (float* psrc = src)
            {
                return CpuMathNativeUtils.MaxAbsDiffU(DefaultScale, psrc, Length);
            }
        }
        
        [Benchmark]
        public unsafe float DotU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                return CpuMathNativeUtils.DotU(psrc, pdst, Length);
            }
        }
        
        [Benchmark]
        public unsafe float DotSU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (int* pidx = idx)
            {
                return CpuMathNativeUtils.DotSU(psrc, pdst, pidx, IndexLength);
            }
        }

        [Benchmark]
        public unsafe float Dist2()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                return CpuMathNativeUtils.Dist2(psrc, pdst, Length);
            }
        }

        [Benchmark]
        public unsafe void SdcaL1UpdateU()
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (float* pres = result)
            {
                CpuMathNativeUtils.SdcaL1UpdateU(DefaultScale, psrc, DefaultScale, pdst, pres, Length);
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
                CpuMathNativeUtils.SdcaL1UpdateSU(DefaultScale, psrc, pidx, DefaultScale, pdst, pres, IndexLength);
            }
        }
    }
}
