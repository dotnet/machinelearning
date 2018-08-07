﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    public static partial class CpuMathUtils
    {
        public static void Scale(float a, float[] dst, int count) => SseUtils.Scale(a, dst, count);

        public static void Scale(float a, float[] dst, int offset, int count) => SseUtils.Scale(a, dst, offset, count);

        public static void AddScale(float a, float[] src, float[] dst, int count) => SseUtils.AddScale(a, src, dst, count);

        public static void AddScale(float a, float[] src, float[] dst, int dstOffset, int count) => SseUtils.AddScale(a, src, dst, dstOffset, count);

        public static void AddScale(float a, float[] src, int[] indices, float[] dst, int count) => SseUtils.AddScale(a, src, indices, dst, count);

        public static void AddScale(float a, float[] src, int[] indices, float[] dst, int dstOffset, int count) => SseUtils.AddScale(a, src, indices, dst, dstOffset, count);

        public static void Add(float[] src, float[] dst, int count) => SseUtils.Add(src, dst, count);

        public static void Add(float[] src, int[] indices, float[] dst, int count) => SseUtils.Add(src, indices, dst, count);

        public static void Add(float[] src, int[] indices, float[] dst, int dstOffset, int count) => SseUtils.Add(src, indices, dst, dstOffset, count);

        public static void MulElementWise(float[] src1, float[] src2, float[] dst, int count) => SseUtils.MulElementWise(src1, src2, dst, count);

        public static float SumSq(float[] src, int count) => SseUtils.SumSq(src, count);

        public static float SumSq(float[] src, int offset, int count) => SseUtils.SumSq(src, offset, count);

        public static float SumAbs(float[] src, int count) => SseUtils.SumAbs(src, count);

        public static float SumAbs(float[] src, int offset, int count) => SseUtils.SumAbs(src, offset, count);

        public static float DotProductDense(float[] a, float[] b, int count) => SseUtils.DotProductDense(a, b, count);

        public static float DotProductDense(float[] a, int offset, float[] b, int count) => SseUtils.DotProductDense(a, offset, b, count);

        public static float DotProductSparse(float[] a, float[] b, int[] indices, int count) => SseUtils.DotProductSparse(a, b, indices, count);

        public static float DotProductSparse(float[] a, int offset, float[] b, int[] indices, int count) => SseUtils.DotProductSparse(a, offset, b, indices, count);

        public static float L2DistSquared(float[] a, float[] b, int count) => SseUtils.L2DistSquared(a, b, count);
    }
}
