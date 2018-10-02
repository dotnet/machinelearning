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
        public void ScaleU()
            => SseIntrinsics.ScaleU(DefaultScale, new Span<float>(dst, 0, Length));
        
        [Benchmark]
        public void ScaleSrcU()
            => SseIntrinsics.ScaleSrcU(DefaultScale, new Span<float>(src, 0, Length), new Span<float>(dst, 0, Length));

        [Benchmark]
        public void ScaleAddU()
            => SseIntrinsics.ScaleAddU(DefaultScale, DefaultScale, new Span<float>(dst, 0, Length));
        
        [Benchmark]
        public void AddScaleU()
            => SseIntrinsics.AddScaleU(DefaultScale, new Span<float>(src, 0, Length), new Span<float>(dst, 0, Length));

        [Benchmark]
        public void AddScaleSU()
            => SseIntrinsics.AddScaleSU(DefaultScale, new Span<float>(src), new Span<int>(idx, 0, IndexLength), new Span<float>(dst));

        [Benchmark]
        public void AddScaleCopyU()
            => SseIntrinsics.AddScaleCopyU(DefaultScale, new Span<float>(src, 0, Length), new Span<float>(dst, 0, Length), new Span<float>(result, 0, Length));

        [Benchmark]
        public void AddU()
            => SseIntrinsics.AddU(new Span<float>(src, 0, Length), new Span<float>(dst, 0, Length));

        [Benchmark]
        public void AddSU()
            => SseIntrinsics.AddSU(new Span<float>(src), new Span<int>(idx, 0, IndexLength), new Span<float>(dst));

        [Benchmark]
        public void MulElementWiseU()
            => SseIntrinsics.MulElementWiseU(new Span<float>(src1, 0, Length), new Span<float>(src2, 0, Length),
                                            new Span<float>(dst, 0, Length));

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
            => SseIntrinsics.DotU(new Span<float>(src, 0, Length), new Span<float>(dst, 0, Length));
        
        [Benchmark]
        public float DotSU()
            => SseIntrinsics.DotSU(new Span<float>(src), new Span<float>(dst), new Span<int>(idx, 0, IndexLength));
        
        [Benchmark]
        public float Dist2()
            => SseIntrinsics.Dist2(new Span<float>(src, 0, Length), new Span<float>(dst, 0, Length));

        [Benchmark]
        public void SdcaL1UpdateU()
            => SseIntrinsics.SdcaL1UpdateU(DefaultScale, new Span<float>(src, 0, Length), DefaultScale, new Span<float>(dst, 0, Length), new Span<float>(result, 0, Length));

        [Benchmark]
        public void SdcaL1UpdateSU()
            => SseIntrinsics.SdcaL1UpdateSU(DefaultScale, new Span<float>(src, 0, IndexLength), new Span<int>(idx, 0, IndexLength), DefaultScale, new Span<float>(dst), new Span<float>(result));
    }
}
