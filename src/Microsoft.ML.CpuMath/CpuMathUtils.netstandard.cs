// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.CompilerServices;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    public static partial class CpuMathUtils
    {
        // The count of bytes in Vector128<T>, corresponding to _cbAlign in AlignedArray
        private const int Vector128Alignment = 16;

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static int GetVectorAlignment()
            => Vector128Alignment;

        public static void MatTimesSrc(bool tran, bool add, AlignedArray mat, AlignedArray src, AlignedArray dst, int crun) => SseUtils.MatTimesSrc(tran, add, mat, src, dst, crun);

        public static void MatTimesSrc(bool tran, bool add, AlignedArray mat, int[] rgposSrc, AlignedArray srcValues,
            int posMin, int iposMin, int iposLim, AlignedArray dst, int crun) => SseUtils.MatTimesSrc(tran, add, mat, rgposSrc, srcValues, posMin, iposMin, iposLim, dst, crun);

        public static void Add(float a, Span<float> dst) => SseUtils.Add(a, dst);

        public static void Scale(float a, Span<float> dst) => SseUtils.Scale(a, dst);

        public static void Scale(float a, ReadOnlySpan<float> src, Span<float> dst, int count) => SseUtils.Scale(a, src, dst, count);

        public static void ScaleAdd(float a, float b, Span<float> dst) => SseUtils.ScaleAdd(a, b, dst);

        public static void AddScale(float a, ReadOnlySpan<float> src, Span<float> dst, int count) => SseUtils.AddScale(a, src, dst, count);

        public static void AddScale(float a, ReadOnlySpan<float> src, ReadOnlySpan<int> indices, Span<float> dst, int count) => SseUtils.AddScale(a, src, indices, dst, count);

        public static void AddScaleCopy(float a, ReadOnlySpan<float> src, ReadOnlySpan<float> dst, Span<float> res, int count) => SseUtils.AddScaleCopy(a, src, dst, res, count);

        public static void Add(ReadOnlySpan<float> src, Span<float> dst, int count) => SseUtils.Add(src, dst, count);

        public static void Add(ReadOnlySpan<float> src, ReadOnlySpan<int> indices, Span<float> dst, int count) => SseUtils.Add(src, indices, dst, count);

        public static void MulElementWise(ReadOnlySpan<float> src1, ReadOnlySpan<float> src2, Span<float> dst, int count) => SseUtils.MulElementWise(src1, src2, dst, count);

        public static float Sum(ReadOnlySpan<float> src) => SseUtils.Sum(src);

        public static float SumSq(ReadOnlySpan<float> src) => SseUtils.SumSq(src);

        public static float SumSq(float mean, ReadOnlySpan<float> src) => SseUtils.SumSq(mean, src);

        public static float SumAbs(ReadOnlySpan<float> src) => SseUtils.SumAbs(src);

        public static float SumAbs(float mean, ReadOnlySpan<float> src) => SseUtils.SumAbs(mean, src);

        public static float MaxAbs(ReadOnlySpan<float> src) => SseUtils.MaxAbs(src);

        public static float MaxAbsDiff(float mean, ReadOnlySpan<float> src) => SseUtils.MaxAbsDiff(mean, src);

        public static float DotProductDense(ReadOnlySpan<float> a, ReadOnlySpan<float> b, int count) => SseUtils.DotProductDense(a, b, count);

        public static float DotProductSparse(ReadOnlySpan<float> a, ReadOnlySpan<float> b, ReadOnlySpan<int> indices, int count) => SseUtils.DotProductSparse(a, b, indices, count);

        public static float L2DistSquared(ReadOnlySpan<float> a, ReadOnlySpan<float> b, int count) => SseUtils.L2DistSquared(a, b, count);

        public static void ZeroMatrixItems(AlignedArray dst, int ccol, int cfltRow, int[] indices) => SseUtils.ZeroMatrixItems(dst, ccol, cfltRow, indices);

        public static void SdcaL1UpdateDense(float primalUpdate, int count, ReadOnlySpan<float> src, float threshold, Span<float> v, Span<float> w)
            => SseUtils.SdcaL1UpdateDense(primalUpdate, count, src, threshold, v, w);

        public static void SdcaL1UpdateSparse(float primalUpdate, int count, ReadOnlySpan<float> src, ReadOnlySpan<int> indices, float threshold, Span<float> v, Span<float> w)
            => SseUtils.SdcaL1UpdateSparse(primalUpdate, count, src, indices, threshold, v, w);
    }
}
