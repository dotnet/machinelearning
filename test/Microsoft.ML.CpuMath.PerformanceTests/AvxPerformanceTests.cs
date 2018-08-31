// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Microsoft.ML.Runtime.Internal.CpuMath;

namespace Microsoft.ML.CpuMath.PerformanceTests
{
    public class AvxPerformanceTests : PerformanceTests
    {
        [Benchmark]
        public void AddScalarU()
            => AvxIntrinsics.AddScalarU(DEFAULT_SCALE, new Span<float>(dst, 0, LEN));

        [Benchmark]
        public void ScaleU()
            => AvxIntrinsics.ScaleU(DEFAULT_SCALE, new Span<float>(dst, 0, LEN));

        [Benchmark]
        public void ScaleSrcU()
            => AvxIntrinsics.ScaleSrcU(DEFAULT_SCALE, new Span<float>(src, 0, LEN), new Span<float>(dst, 0, LEN));

        [Benchmark]
        public void ScaleAddU()
            => AvxIntrinsics.ScaleAddU(DEFAULT_SCALE, DEFAULT_SCALE, new Span<float>(dst, 0, LEN));

        [Benchmark]
        public void AddScaleU()
            => AvxIntrinsics.AddScaleU(DEFAULT_SCALE, new Span<float>(src, 0, LEN), new Span<float>(dst, 0, LEN));

        [Benchmark]
        public void AddScaleSU()
            => AvxIntrinsics.AddScaleSU(DEFAULT_SCALE, new Span<float>(src), new Span<int>(idx, 0, IDXLEN), new Span<float>(dst));

        [Benchmark]
        public void AddScaleCopyU()
            => AvxIntrinsics.AddScaleCopyU(DEFAULT_SCALE, new Span<float>(src, 0, LEN), new Span<float>(dst, 0, LEN), new Span<float>(result, 0, LEN));

        [Benchmark]
        public void AddU()
            => AvxIntrinsics.AddU(new Span<float>(src, 0, LEN), new Span<float>(dst, 0, LEN));

        [Benchmark]
        public void AddSU()
            => AvxIntrinsics.AddSU(new Span<float>(src), new Span<int>(idx, 0, IDXLEN), new Span<float>(dst));

        [Benchmark]
        public void MulElementWiseU()
            => AvxIntrinsics.MulElementWiseU(new Span<float>(src1, 0, LEN), new Span<float>(src2, 0, LEN),
                                            new Span<float>(dst, 0, LEN));

        [Benchmark]
        public float SumU()
            => AvxIntrinsics.SumU(new Span<float>(src, 0, LEN));

        [Benchmark]
        public float SumSqU()
            => AvxIntrinsics.SumSqU(new Span<float>(src, 0, LEN));

        [Benchmark]
        public float SumSqDiffU()
            => AvxIntrinsics.SumSqDiffU(DEFAULT_SCALE, new Span<float>(src, 0, LEN));

       [Benchmark]
        public float SumAbsU()
            => AvxIntrinsics.SumAbsU(new Span<float>(src, 0, LEN));

        [Benchmark]
        public float SumAbsDiffU()
            => AvxIntrinsics.SumAbsDiffU(DEFAULT_SCALE, new Span<float>(src, 0, LEN));

        [Benchmark]
        public float MaxAbsU()
            => AvxIntrinsics.MaxAbsU(new Span<float>(src, 0, LEN));

        [Benchmark]
        public float MaxAbsDiffU()
            => AvxIntrinsics.MaxAbsDiffU(DEFAULT_SCALE, new Span<float>(src, 0, LEN));

        [Benchmark]
        public float DotU()
            => AvxIntrinsics.DotU(new Span<float>(src, 0, LEN), new Span<float>(dst, 0, LEN));

        [Benchmark]
        public float DotSU()
            => AvxIntrinsics.DotSU(new Span<float>(src), new Span<float>(dst), new Span<int>(idx, 0, IDXLEN));

        [Benchmark]
        public float Dist2()
            => AvxIntrinsics.Dist2(new Span<float>(src, 0, LEN), new Span<float>(dst, 0, LEN));

        [Benchmark]
        public void SdcaL1UpdateU()
            => AvxIntrinsics.SdcaL1UpdateU(DEFAULT_SCALE, new Span<float>(src, 0, LEN), DEFAULT_SCALE, new Span<float>(dst, 0, LEN), new Span<float>(result, 0, LEN));

        [Benchmark]
        public void SdcaL1UpdateSU()
            => AvxIntrinsics.SdcaL1UpdateSU(DEFAULT_SCALE, new Span<float>(src, 0, IDXLEN), new Span<int>(idx, 0, IDXLEN), DEFAULT_SCALE, new Span<float>(dst), new Span<float>(result));
    }
}
