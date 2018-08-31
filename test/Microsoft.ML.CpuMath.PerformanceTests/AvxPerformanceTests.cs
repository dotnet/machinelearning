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
        public void ManagedAddScalarUPerf()
        {
            AvxIntrinsics.AddScalarU(DEFAULT_SCALE, new Span<float>(dst, 0, LEN));
        }

        [Benchmark]
        public void ManagedScaleUPerf()
        {
            AvxIntrinsics.ScaleU(DEFAULT_SCALE, new Span<float>(dst, 0, LEN));
        }

        [Benchmark]
        public void ManagedScaleSrcUPerf()
        {
            AvxIntrinsics.ScaleSrcU(DEFAULT_SCALE, new Span<float>(src, 0, LEN), new Span<float>(dst, 0, LEN));
        }

        [Benchmark]
        public void ManagedScaleAddUPerf()
        {
            AvxIntrinsics.ScaleAddU(DEFAULT_SCALE, DEFAULT_SCALE, new Span<float>(dst, 0, LEN));
        }

        [Benchmark]
        public void ManagedAddScaleUPerf()
        {
            AvxIntrinsics.AddScaleU(DEFAULT_SCALE, new Span<float>(src, 0, LEN), new Span<float>(dst, 0, LEN));
        }

        [Benchmark]
        public void ManagedAddScaleSUPerf()
        {
            AvxIntrinsics.AddScaleSU(DEFAULT_SCALE, new Span<float>(src), new Span<int>(idx, 0, IDXLEN), new Span<float>(dst));
        }

        [Benchmark]
        public void ManagedAddScaleCopyUPerf()
        {
            AvxIntrinsics.AddScaleCopyU(DEFAULT_SCALE, new Span<float>(src, 0, LEN), new Span<float>(dst, 0, LEN), new Span<float>(result, 0, LEN));
        }

        [Benchmark]
        public void ManagedAddUPerf()
        {
            AvxIntrinsics.AddU(new Span<float>(src, 0, LEN), new Span<float>(dst, 0, LEN));
        }

        [Benchmark]
        public void ManagedAddSUPerf()
        {
            AvxIntrinsics.AddSU(new Span<float>(src), new Span<int>(idx, 0, IDXLEN), new Span<float>(dst));
        }


        [Benchmark]
        public void ManagedMulElementWiseUPerf()
        {
            AvxIntrinsics.MulElementWiseU(new Span<float>(src1, 0, LEN), new Span<float>(src2, 0, LEN),
                                            new Span<float>(dst, 0, LEN));
        }

        [Benchmark]
        public float ManagedSumUPerf()
        {
            return AvxIntrinsics.SumU(new Span<float>(src, 0, LEN));
        }

        [Benchmark]
        public float ManagedSumSqUPerf()
        {
            return AvxIntrinsics.SumSqU(new Span<float>(src, 0, LEN));
        }

        [Benchmark]
        public float ManagedSumSqDiffUPerf()
        {
            return AvxIntrinsics.SumSqDiffU(DEFAULT_SCALE, new Span<float>(src, 0, LEN));
        }

       [Benchmark]
        public float ManagedSumAbsUPerf()
        {
            return AvxIntrinsics.SumAbsU(new Span<float>(src, 0, LEN));
        }

        [Benchmark]
        public float ManagedSumAbsDiffUPerf()
        {
            return AvxIntrinsics.SumAbsDiffU(DEFAULT_SCALE, new Span<float>(src, 0, LEN));
        }

        [Benchmark]
        public float ManagedMaxAbsUPerf()
        {
            return AvxIntrinsics.MaxAbsU(new Span<float>(src, 0, LEN));
        }

        [Benchmark]
        public float ManagedMaxAbsDiffUPerf()
        {
            return AvxIntrinsics.MaxAbsDiffU(DEFAULT_SCALE, new Span<float>(src, 0, LEN));
        }

        [Benchmark]
        public float ManagedDotUPerf()
        {
            return AvxIntrinsics.DotU(new Span<float>(src, 0, LEN), new Span<float>(dst, 0, LEN));
        }

        [Benchmark]
        public float ManagedDotSUPerf()
        {
            return AvxIntrinsics.DotSU(new Span<float>(src), new Span<float>(dst), new Span<int>(idx, 0, IDXLEN));
        }

        [Benchmark]
        public float ManagedDist2Perf()
        {
            return AvxIntrinsics.Dist2(new Span<float>(src, 0, LEN), new Span<float>(dst, 0, LEN));
        }

        [Benchmark]
        public void ManagedSdcaL1UpdateUPerf()
        {
            AvxIntrinsics.SdcaL1UpdateU(DEFAULT_SCALE, new Span<float>(src, 0, LEN), DEFAULT_SCALE, new Span<float>(dst, 0, LEN), new Span<float>(result, 0, LEN));
        }

        [Benchmark]
        public void ManagedSdcaL1UpdateSUPerf()
        {
            AvxIntrinsics.SdcaL1UpdateSU(DEFAULT_SCALE, new Span<float>(src, 0, IDXLEN), new Span<int>(idx, 0, IDXLEN), DEFAULT_SCALE, new Span<float>(dst), new Span<float>(result));
        }
    }
}
