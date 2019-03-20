// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using BenchmarkDotNet.Attributes;
using Microsoft.ML.Internal.CpuMath;

namespace Microsoft.ML.CpuMath.PerformanceTests
{
    public class SmallInputCpuMathPerformanceTests: PerformanceTests
    {
        private int _smallInputLength = 10;

        [Benchmark]
        public void AddScalarU()
            => CpuMathUtils.Add(DefaultScale, dst.AsSpan(0, _smallInputLength));

        [Benchmark]
        public void Scale()
            => CpuMathUtils.Scale(DefaultScale, dst.AsSpan(0, _smallInputLength));

        [Benchmark]
        public void ScaleSrcU()
            => CpuMathUtils.Scale(DefaultScale, src, dst, _smallInputLength);

        [Benchmark]
        public void ScaleAddU()
            => CpuMathUtils.ScaleAdd(DefaultScale, DefaultScale, dst.AsSpan(0, _smallInputLength));

        [Benchmark]
        public void AddScaleU()
            => CpuMathUtils.AddScale(DefaultScale, src, dst, _smallInputLength);

        [Benchmark]
        public void AddScaleSU()
            => CpuMathUtils.AddScale(DefaultScale, src, idx, dst, _smallInputLength);

        [Benchmark]
        public void AddScaleCopyU()
            => CpuMathUtils.AddScaleCopy(DefaultScale, src, dst, result, _smallInputLength);

        [Benchmark]
        public void AddU()
            => CpuMathUtils.Add(src, dst, _smallInputLength);

        [Benchmark]
        public void AddSU()
            => CpuMathUtils.Add(src, idx, dst, _smallInputLength);

        [Benchmark]
        public void MulElementWiseU()
            => CpuMathUtils.MulElementWise(src1, src2, dst, _smallInputLength);

        [Benchmark]
        public float Sum()
            => CpuMathUtils.Sum(new Span<float>(src, 0, _smallInputLength));

        [Benchmark]
        public float SumSqU()
            => CpuMathUtils.SumSq(new Span<float>(src, 0, _smallInputLength));

        [Benchmark]
        public float SumSqDiffU()
            => CpuMathUtils.SumSq(DefaultScale, src.AsSpan(0, _smallInputLength));

        [Benchmark]
        public float SumAbsU()
             => CpuMathUtils.SumAbs(src.AsSpan(0, _smallInputLength));

        [Benchmark]
        public float SumAbsDiffU()
            => CpuMathUtils.SumAbs(DefaultScale, src.AsSpan(0, _smallInputLength));

        [Benchmark]
        public float MaxAbsU()
            => CpuMathUtils.MaxAbs(src.AsSpan(0, _smallInputLength));

        [Benchmark]
        public float MaxAbsDiffU()
            => CpuMathUtils.MaxAbsDiff(DefaultScale, src.AsSpan(0, _smallInputLength));

        [Benchmark]
        public float DotU()
            => CpuMathUtils.DotProductDense(src, dst, _smallInputLength);

        [Benchmark]
        public float DotSU()
            => CpuMathUtils.DotProductSparse(src, dst, idx, _smallInputLength);

        [Benchmark]
        public float Dist2()
            => CpuMathUtils.L2DistSquared(src, dst, _smallInputLength);

        [Benchmark]
        public void SdcaL1UpdateU()
            => CpuMathUtils.SdcaL1UpdateDense(DefaultScale, _smallInputLength, src, DefaultScale, dst, result);

        [Benchmark]
        public void SdcaL1UpdateSU()
            => CpuMathUtils.SdcaL1UpdateSparse(DefaultScale, _smallInputLength, src, idx, DefaultScale, dst, result);
    }
}
