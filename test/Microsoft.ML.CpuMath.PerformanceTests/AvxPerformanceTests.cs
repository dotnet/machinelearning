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
            => AvxIntrinsics.ScaleSrcU(DefaultScale, src, dst, Length);

        [Benchmark]
        public void ScaleAddU()
            => AvxIntrinsics.ScaleAddU(DefaultScale, DefaultScale, new Span<float>(dst, 0, Length));

        [Benchmark]
        public void AddScaleU()
            => AvxIntrinsics.AddScaleU(DefaultScale, src, dst, Length);

        [Benchmark]
        public void AddScaleSU()
            => AvxIntrinsics.AddScaleSU(DefaultScale, src, idx, dst, IndexLength);

        [Benchmark]
        public void AddScaleCopyU()
            => AvxIntrinsics.AddScaleCopyU(DefaultScale, src, dst, result, Length);

        [Benchmark]
        public void AddU()
            => AvxIntrinsics.AddU(src, dst, Length);

        [Benchmark]
        public void AddSU()
            => AvxIntrinsics.AddSU(src, idx, dst, IndexLength);

        [Benchmark]
        public void MulElementWiseU()
            => AvxIntrinsics.MulElementWiseU(src1, src2, dst, Length);

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
            => AvxIntrinsics.DotU(src, dst, Length);

        [Benchmark]
        public float DotSU()
            => AvxIntrinsics.DotSU(src, dst, idx, IndexLength);

        [Benchmark]
        public float Dist2()
            => AvxIntrinsics.Dist2(src, dst, Length);

        [Benchmark]
        public void SdcaL1UpdateU()
            => AvxIntrinsics.SdcaL1UpdateU(DefaultScale, Length, src, DefaultScale, dst, result);

        [Benchmark]
        public void SdcaL1UpdateSU()
            => AvxIntrinsics.SdcaL1UpdateSU(DefaultScale, IndexLength, src, idx, DefaultScale, dst, result);
        [Benchmark]
        public void MatMulX()
            => AvxIntrinsics.MatMulX(src, src1, dst, 1000, 1000);

        [Benchmark]
        public void MatMulTranX()
            => AvxIntrinsics.MatMulTranX(src, src1, dst, 1000, 1000);
    }
}
