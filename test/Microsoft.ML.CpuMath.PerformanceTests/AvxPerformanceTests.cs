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
            => AvxIntrinsics.AddScalarU(DefaultScale, new Span<float>(dst, 0, Length));

        [Benchmark]
        public void Scale()
            => AvxIntrinsics.Scale(DefaultScale, new Span<float>(dst, 0, Length));

        [Benchmark]
        public void ScaleSrcU()
            => AvxIntrinsics.ScaleSrcU(DefaultScale, new Span<float>(src, 0, Length), new Span<float>(dst, 0, Length));

        [Benchmark]
        public void ScaleAddU()
            => AvxIntrinsics.ScaleAddU(DefaultScale, DefaultScale, new Span<float>(dst, 0, Length));

        [Benchmark]
        public void AddScaleU()
            => AvxIntrinsics.AddScaleU(DefaultScale, new Span<float>(src, 0, Length), new Span<float>(dst, 0, Length));

        [Benchmark]
        public void AddScaleSU()
            => AvxIntrinsics.AddScaleSU(DefaultScale, new Span<float>(src), new Span<int>(idx, 0, IndexLength), new Span<float>(dst));

        [Benchmark]
        public void AddScaleCopyU()
            => AvxIntrinsics.AddScaleCopyU(DefaultScale, new Span<float>(src, 0, Length), new Span<float>(dst, 0, Length), new Span<float>(result, 0, Length));

        [Benchmark]
        public void AddU()
            => AvxIntrinsics.AddU(new Span<float>(src, 0, Length), new Span<float>(dst, 0, Length));

        [Benchmark]
        public void AddSU()
            => AvxIntrinsics.AddSU(new Span<float>(src), new Span<int>(idx, 0, IndexLength), new Span<float>(dst));

        [Benchmark]
        public void MulElementWiseU()
            => AvxIntrinsics.MulElementWiseU(new Span<float>(src1, 0, Length), new Span<float>(src2, 0, Length),
                                            new Span<float>(dst, 0, Length));

        [Benchmark]
        public float SumU()
            => AvxIntrinsics.SumU(new Span<float>(src, 0, Length));

        [Benchmark]
        public float SumSqU()
            => AvxIntrinsics.SumSqU(new Span<float>(src, 0, Length));

        [Benchmark]
        public float SumSqDiffU()
            => AvxIntrinsics.SumSqDiffU(DefaultScale, new Span<float>(src, 0, Length));

       [Benchmark]
        public float SumAbsU()
            => AvxIntrinsics.SumAbsU(new Span<float>(src, 0, Length));

        [Benchmark]
        public float SumAbsDiffU()
            => AvxIntrinsics.SumAbsDiffU(DefaultScale, new Span<float>(src, 0, Length));

        [Benchmark]
        public float MaxAbsU()
            => AvxIntrinsics.MaxAbsU(new Span<float>(src, 0, Length));

        [Benchmark]
        public float MaxAbsDiffU()
            => AvxIntrinsics.MaxAbsDiffU(DefaultScale, new Span<float>(src, 0, Length));

        [Benchmark]
        public float DotU()
            => AvxIntrinsics.DotU(new Span<float>(src, 0, Length), new Span<float>(dst, 0, Length));

        [Benchmark]
        public float DotSU()
            => AvxIntrinsics.DotSU(new Span<float>(src), new Span<float>(dst), new Span<int>(idx, 0, IndexLength));

        [Benchmark]
        public float Dist2()
            => AvxIntrinsics.Dist2(new Span<float>(src, 0, Length), new Span<float>(dst, 0, Length));

        [Benchmark]
        public void SdcaL1UpdateU()
            => AvxIntrinsics.SdcaL1UpdateU(DefaultScale, new Span<float>(src, 0, Length), DefaultScale, new Span<float>(dst, 0, Length), new Span<float>(result, 0, Length));

        [Benchmark]
        public void SdcaL1UpdateSU()
            => AvxIntrinsics.SdcaL1UpdateSU(DefaultScale, new Span<float>(src, 0, IndexLength), new Span<int>(idx, 0, IndexLength), DefaultScale, new Span<float>(dst), new Span<float>(result));

        [Benchmark]
        public void MatMulX()
            => AvxIntrinsics.MatMulX(true, src, src1, dst, 1000, 1000);

        [Benchmark]
        public void MatMulTranX()
            => AvxIntrinsics.MatMulTranX(true, src, src1, dst, 1000, 1000);
    }
}
