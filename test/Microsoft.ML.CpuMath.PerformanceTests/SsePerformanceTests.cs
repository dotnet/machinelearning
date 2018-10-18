// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Microsoft.ML.Runtime.Internal.CpuMath;

namespace Microsoft.ML.CpuMath.PerformanceTests
{
    public class SsePerformanceTests : PerformanceTests
    {
        [Benchmark]
        public void AddScalarU()
            => SseIntrinsics.AddScalarU(DefaultScale, new Span<float>(dst, 0, Length));
        
        [Benchmark]
        public void Scale()
            => SseIntrinsics.Scale(DefaultScale, new Span<float>(dst, 0, Length));
        
        [Benchmark]
        public void ScaleSrcU()
            => SseIntrinsics.ScaleSrcU(DefaultScale, src, dst, Length);

        [Benchmark]
        public void ScaleAddU()
            => SseIntrinsics.ScaleAddU(DefaultScale, DefaultScale, new Span<float>(dst, 0, Length));
        
        [Benchmark]
        public void AddScaleU()
            => SseIntrinsics.AddScaleU(DefaultScale, src, dst, Length);

        [Benchmark]
        public void AddScaleSU()
            => SseIntrinsics.AddScaleSU(DefaultScale, src, idx, dst, IndexLength);

        [Benchmark]
        public void AddScaleCopyU()
            => SseIntrinsics.AddScaleCopyU(DefaultScale, src, dst, result, Length);

        [Benchmark]
        public void AddU()
            => SseIntrinsics.AddU(src, dst, Length);

        [Benchmark]
        public void AddSU()
            => SseIntrinsics.AddSU(src, idx, dst, IndexLength);

        [Benchmark]
        public void MulElementWiseU()
            => SseIntrinsics.MulElementWiseU(src1, src2, dst, Length);

        [Benchmark]
        public float SumU()
            => SseIntrinsics.SumU(new Span<float>(src, 0, Length));

        [Benchmark]
        public float SumSqU()
            => SseIntrinsics.SumSqU(new Span<float>(src, 0, Length));
        
        [Benchmark]
        public float SumSqDiffU()
            => SseIntrinsics.SumSqDiffU(DefaultScale, new Span<float>(src, 0, Length));
        
        [Benchmark]
        public float SumAbsU()
            => SseIntrinsics.SumAbsU(new Span<float>(src, 0, Length));

        [Benchmark]
        public float SumAbsDiffU()
            => SseIntrinsics.SumAbsDiffU(DefaultScale, new Span<float>(src, 0, Length));
        
        [Benchmark]
        public float MaxAbsU()
            => SseIntrinsics.MaxAbsU(new Span<float>(src, 0, Length));
        
        [Benchmark]
        public float MaxAbsDiffU()
            => SseIntrinsics.MaxAbsDiffU(DefaultScale, new Span<float>(src, 0, Length));
        
        [Benchmark]
        public float DotU()
            => SseIntrinsics.DotU(src, dst, Length);
        
        [Benchmark]
        public float DotSU()
            => SseIntrinsics.DotSU(src, dst, idx, IndexLength);
        
        [Benchmark]
        public float Dist2()
            => SseIntrinsics.Dist2(src, dst, Length);

        [Benchmark]
        public void SdcaL1UpdateU()
            => SseIntrinsics.SdcaL1UpdateU(DefaultScale, Length, src, DefaultScale, dst, result);

        [Benchmark]
        public void SdcaL1UpdateSU()
            => SseIntrinsics.SdcaL1UpdateSU(DefaultScale, IndexLength, src, idx, DefaultScale, dst, result);

        [Benchmark]
        public void MatMulX()
            => SseIntrinsics.MatMul(src, src1, dst, 1000, 1000);

        [Benchmark]
        public void MatMulTranX()
            => SseIntrinsics.MatMulTran(src, src1, dst, 1000, 1000);
    }
}
