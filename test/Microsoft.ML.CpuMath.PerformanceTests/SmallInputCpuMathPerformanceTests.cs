// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using BenchmarkDotNet.Attributes;
using Microsoft.ML.Internal.CpuMath;

namespace Microsoft.ML.CpuMath.PerformanceTests
{
    public class SmallInputCpuMathPerformanceTests : PerformanceTests
    {
        [Params(256)]
        public int arrayLength;

        [Benchmark]
        public void AddScalarU()
            => CpuMathUtils.Add(DefaultScale, dst.AsSpan(0, arrayLength));

        [Benchmark]
        public void Scale()
            => CpuMathUtils.Scale(DefaultScale, dst.AsSpan(0, arrayLength));

        [Benchmark]
        public void ScaleSrcU()
            => CpuMathUtils.Scale(DefaultScale, src, dst, arrayLength);

        [Benchmark]
        public void ScaleAddU()
            => CpuMathUtils.ScaleAdd(DefaultScale, DefaultScale, dst.AsSpan(0, arrayLength));

        [Benchmark]
        public void AddScaleU()
            => CpuMathUtils.AddScale(DefaultScale, src, dst, arrayLength);

        [Benchmark]
        public void AddScaleSU()
            => CpuMathUtils.AddScale(DefaultScale, src, idx, dst, arrayLength);

        [Benchmark]
        public void AddScaleCopyU()
            => CpuMathUtils.AddScaleCopy(DefaultScale, src, dst, result, arrayLength);

        [Benchmark]
        public void AddU()
            => CpuMathUtils.Add(src, dst, arrayLength);

        [Benchmark]
        public void AddSU()
            => CpuMathUtils.Add(src, idx, dst, arrayLength);

        [Benchmark]
        public void MulElementWiseU()
            => CpuMathUtils.MulElementWise(src1, src2, dst, arrayLength);

        [Benchmark]
        public float Sum()
            => CpuMathUtils.Sum(new Span<float>(src, 0, arrayLength));

        [Benchmark]
        public float SumSqU()
            => CpuMathUtils.SumSq(new Span<float>(src, 0, arrayLength));

        [Benchmark]
        public float SumSqDiffU()
            => CpuMathUtils.SumSq(DefaultScale, src.AsSpan(0, arrayLength));

        [Benchmark]
        public float SumAbsU()
             => CpuMathUtils.SumAbs(src.AsSpan(0, arrayLength));

        [Benchmark]
        public float SumAbsDiffU()
            => CpuMathUtils.SumAbs(DefaultScale, src.AsSpan(0, arrayLength));

        [Benchmark]
        public float MaxAbsU()
            => CpuMathUtils.MaxAbs(src.AsSpan(0, arrayLength));

        [Benchmark]
        public float MaxAbsDiffU()
            => CpuMathUtils.MaxAbsDiff(DefaultScale, src.AsSpan(0, arrayLength));

        [Benchmark]
        public float DotU()
            => CpuMathUtils.DotProductDense(src, dst, arrayLength);

        [Benchmark]
        public float DotSU()
            => CpuMathUtils.DotProductSparse(src, dst, idx, arrayLength);

        [Benchmark]
        public float Dist2()
            => CpuMathUtils.L2DistSquared(src, dst, arrayLength);

        [Benchmark]
        public void SdcaL1UpdateU()
            => CpuMathUtils.SdcaL1UpdateDense(DefaultScale, arrayLength, src, DefaultScale, dst, result);

        [Benchmark]
        public void SdcaL1UpdateSU()
            => CpuMathUtils.SdcaL1UpdateSparse(DefaultScale, arrayLength, src, idx, DefaultScale, dst, result);
    }
}
