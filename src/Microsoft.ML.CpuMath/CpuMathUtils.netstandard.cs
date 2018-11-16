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

        public static void MatrixTimesSource(bool transpose, AlignedArray matrix, AlignedArray source, AlignedArray destination, int stride) => SseUtils.MatTimesSrc(transpose, matrix, source, destination, stride);

        public static void MatrixTimesSource(AlignedArray matrix, int[] rgposSrc, AlignedArray sourceValues,
            int posMin, int iposMin, int iposLimit, AlignedArray destination, int stride) => SseUtils.MatTimesSrc(matrix, rgposSrc, sourceValues, posMin, iposMin, iposLimit, destination, stride);

        public static void Add(float value, Span<float> destination) => SseUtils.Add(value, destination);

        public static void Scale(float value, Span<float> destination) => SseUtils.Scale(value, destination);

        public static void Scale(float value, ReadOnlySpan<float> source, Span<float> destination, int count) => SseUtils.Scale(value, source, destination, count);

        public static void ScaleAdd(float value, float addend, Span<float> destination) => SseUtils.ScaleAdd(value, addend, destination);

        public static void AddScale(float value, ReadOnlySpan<float> source, Span<float> destination, int count) => SseUtils.AddScale(value, source, destination, count);

        public static void AddScale(float value, ReadOnlySpan<float> source, ReadOnlySpan<int> indices, Span<float> destination, int count) => SseUtils.AddScale(value, source, indices, destination, count);

        public static void AddScaleCopy(float value, ReadOnlySpan<float> source, ReadOnlySpan<float> destination, Span<float> res, int count) => SseUtils.AddScaleCopy(value, source, destination, res, count);

        public static void Add(ReadOnlySpan<float> source, Span<float> destination, int count) => SseUtils.Add(source, destination, count);

        public static void Add(ReadOnlySpan<float> source, ReadOnlySpan<int> indices, Span<float> destination, int count) => SseUtils.Add(source, indices, destination, count);

        public static void MulElementWise(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> destination, int count) => SseUtils.MulElementWise(left, right, destination, count);

        public static float Sum(ReadOnlySpan<float> source) => SseUtils.Sum(source);

        public static float SumSq(ReadOnlySpan<float> source) => SseUtils.SumSq(source);

        public static float SumSq(float mean, ReadOnlySpan<float> source) => SseUtils.SumSq(mean, source);

        public static float SumAbs(ReadOnlySpan<float> source) => SseUtils.SumAbs(source);

        public static float SumAbs(float mean, ReadOnlySpan<float> source) => SseUtils.SumAbs(mean, source);

        public static float MaxAbs(ReadOnlySpan<float> source) => SseUtils.MaxAbs(source);

        public static float MaxAbsDiff(float mean, ReadOnlySpan<float> source) => SseUtils.MaxAbsDiff(mean, source);

        public static float DotProductDense(ReadOnlySpan<float> left, ReadOnlySpan<float> right, int count) => SseUtils.DotProductDense(left, right, count);

        public static float DotProductSparse(ReadOnlySpan<float> left, ReadOnlySpan<float> right, ReadOnlySpan<int> indices, int count) => SseUtils.DotProductSparse(left, right, indices, count);

        public static float L2DistSquared(ReadOnlySpan<float> left, ReadOnlySpan<float> right, int count) => SseUtils.L2DistSquared(left, right, count);

        public static void ZeroMatrixItems(AlignedArray destination, int ccol, int cfltRow, int[] indices) => SseUtils.ZeroMatrixItems(destination, ccol, cfltRow, indices);

        public static void SdcaL1UpdateDense(float primalUpdate, int count, ReadOnlySpan<float> source, float threshold, Span<float> v, Span<float> w)
            => SseUtils.SdcaL1UpdateDense(primalUpdate, count, source, threshold, v, w);

        public static void SdcaL1UpdateSparse(float primalUpdate, int count, ReadOnlySpan<float> source, ReadOnlySpan<int> indices, float threshold, Span<float> v, Span<float> w)
            => SseUtils.SdcaL1UpdateSparse(primalUpdate, count, source, indices, threshold, v, w);
    }
}
