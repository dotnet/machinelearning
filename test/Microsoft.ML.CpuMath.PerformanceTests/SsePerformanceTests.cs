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
            => SseIntrinsics.AddScalarU(DEFAULT_SCALE, new Span<float>(dst, 0, LEN));
        
        [Benchmark]
        public void ScaleU()
            => SseIntrinsics.ScaleU(DEFAULT_SCALE, new Span<float>(dst, 0, LEN));
        
        [Benchmark]
        public void ScaleSrcU()
            => SseIntrinsics.ScaleSrcU(DEFAULT_SCALE, new Span<float>(src, 0, LEN), new Span<float>(dst, 0, LEN));

        [Benchmark]
        public void ScaleAddU()
            => SseIntrinsics.ScaleAddU(DEFAULT_SCALE, DEFAULT_SCALE, new Span<float>(dst, 0, LEN));
        
        [Benchmark]
        public void AddScaleU()
            => SseIntrinsics.AddScaleU(DEFAULT_SCALE, new Span<float>(src, 0, LEN), new Span<float>(dst, 0, LEN));

        [Benchmark]
        public void AddScaleSU()
            => SseIntrinsics.AddScaleSU(DEFAULT_SCALE, new Span<float>(src), new Span<int>(idx, 0, IDXLEN), new Span<float>(dst));

        [Benchmark]
        public void AddScaleCopyU()
            => SseIntrinsics.AddScaleCopyU(DEFAULT_SCALE, new Span<float>(src, 0, LEN), new Span<float>(dst, 0, LEN), new Span<float>(result, 0, LEN));

        [Benchmark]
        public void AddU()
            => SseIntrinsics.AddU(new Span<float>(src, 0, LEN), new Span<float>(dst, 0, LEN));

        [Benchmark]
        public void AddSU()
            => SseIntrinsics.AddSU(new Span<float>(src), new Span<int>(idx, 0, IDXLEN), new Span<float>(dst));

        [Benchmark]
        public void MulElementWiseU()
            => SseIntrinsics.MulElementWiseU(new Span<float>(src1, 0, LEN), new Span<float>(src2, 0, LEN),
                                            new Span<float>(dst, 0, LEN));

        [Benchmark]
        public float SumU()
            => SseIntrinsics.SumU(new Span<float>(src, 0, LEN));

        [Benchmark]
        public float SumSqU()
            => SseIntrinsics.SumSqU(new Span<float>(src, 0, LEN));
        
        [Benchmark]
        public float SumSqDiffU()
            => SseIntrinsics.SumSqDiffU(DEFAULT_SCALE, new Span<float>(src, 0, LEN));
        
        [Benchmark]
        public float SumAbsU()
            => SseIntrinsics.SumAbsU(new Span<float>(src, 0, LEN));

        [Benchmark]
        public float SumAbsDiffU()
            => SseIntrinsics.SumAbsDiffU(DEFAULT_SCALE, new Span<float>(src, 0, LEN));
        
        [Benchmark]
        public float MaxAbsU()
            => SseIntrinsics.MaxAbsU(new Span<float>(src, 0, LEN));
        
        [Benchmark]
        public float MaxAbsDiffU()
            => SseIntrinsics.MaxAbsDiffU(DEFAULT_SCALE, new Span<float>(src, 0, LEN));
        
        [Benchmark]
        public float DotU()
            => SseIntrinsics.DotU(new Span<float>(src, 0, LEN), new Span<float>(dst, 0, LEN));
        
        [Benchmark]
        public float DotSU()
            => SseIntrinsics.DotSU(new Span<float>(src), new Span<float>(dst), new Span<int>(idx, 0, IDXLEN));
        
        [Benchmark]
        public float Dist2()
            => SseIntrinsics.Dist2(new Span<float>(src, 0, LEN), new Span<float>(dst, 0, LEN));

        [Benchmark]
        public void SdcaL1UpdateU()
            => SseIntrinsics.SdcaL1UpdateU(DEFAULT_SCALE, new Span<float>(src, 0, LEN), DEFAULT_SCALE, new Span<float>(dst, 0, LEN), new Span<float>(result, 0, LEN));

        [Benchmark]
        public void SdcaL1UpdateSU()
            => SseIntrinsics.SdcaL1UpdateSU(DEFAULT_SCALE, new Span<float>(src, 0, IDXLEN), new Span<int>(idx, 0, IDXLEN), DEFAULT_SCALE, new Span<float>(dst), new Span<float>(result));
    }
}
