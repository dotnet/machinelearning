﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using Microsoft.ML.Internal.CpuMath.Core;

namespace Microsoft.ML.Internal.CpuMath
{
    internal static partial class CpuMathUtils
    {
        // The count of bytes in Vector128<T>, corresponding to _cbAlign in AlignedArray
        private const int Vector128Alignment = 16;

        // The count of bytes in Vector256<T>, corresponding to _cbAlign in AlignedArray
        private const int Vector256Alignment = 32;

        // The count of bytes in a 32-bit float, corresponding to _cbAlign in AlignedArray
        private const int FloatAlignment = 4;

        // If neither AVX nor SSE is supported, return basic alignment for a 4-byte float.
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static int GetVectorAlignment()
            => Avx.IsSupported ? Vector256Alignment : (Sse.IsSupported ? Vector128Alignment : FloatAlignment);

        /// <summary>
        /// Multiplies a matrix times a source.
        /// </summary>
        /// <param name="transpose"><see langword="true"/> to transpose the matrix; otherwise <see langword="false"/>.</param>
        /// <param name="matrix">The input matrix.</param>
        /// <param name="source">The source matrix.</param>
        /// <param name="destination">The destination matrix.</param>
        /// <param name="stride">The column stride.</param>
        public static void MatrixTimesSource(bool transpose, AlignedArray matrix, AlignedArray source, AlignedArray destination, int stride)
        {
            Contracts.Assert(matrix.Size == destination.Size * source.Size);
            Contracts.Assert(stride >= 0);

            if (Avx.IsSupported)
            {
                if (!transpose)
                {
                    Contracts.Assert(stride <= destination.Size);
                    AvxIntrinsics.MatMul(matrix, source, destination, stride, source.Size);
                }
                else
                {
                    Contracts.Assert(stride <= source.Size);
                    AvxIntrinsics.MatMulTran(matrix, source, destination, destination.Size, stride);
                }
            }
            else if (Sse.IsSupported)
            {
                if (!transpose)
                {
                    Contracts.Assert(stride <= destination.Size);
                    SseIntrinsics.MatMul(matrix, source, destination, stride, source.Size);
                }
                else
                {
                    Contracts.Assert(stride <= source.Size);
                    SseIntrinsics.MatMulTran(matrix, source, destination, destination.Size, stride);
                }
            }
            else
            {
                if (!transpose)
                {
                    Contracts.Assert(stride <= destination.Size);
                    for (int i = 0; i < stride; i++)
                    {
                        float dotProduct = 0;
                        for (int j = 0; j < source.Size; j++)
                        {
                            dotProduct += matrix[i * source.Size + j] * source[j];
                        }

                        destination[i] = dotProduct;
                    }
                }
                else
                {
                    Contracts.Assert(stride <= source.Size);
                    for (int i = 0; i < destination.Size; i++)
                    {
                        float dotProduct = 0;
                        for (int j = 0; j < stride; j++)
                        {
                            dotProduct += matrix[j * destination.Size + i] * source[j];
                        }

                        destination[i] = dotProduct;
                    }
                }
            }
        }

        /// <summary>
        /// Multiplies a matrix times a source.
        /// </summary>
        /// <param name="matrix">The input matrix.</param>
        /// <param name="rgposSrc">The source positions.</param>
        /// <param name="sourceValues">The source values.</param>
        /// <param name="posMin">The minimum position.</param>
        /// <param name="iposMin">The minimum position index.</param>
        /// <param name="iposLimit">The position limit.</param>
        /// <param name="destination">The destination matrix.</param>
        /// <param name="stride">The column stride.</param>
        public static void MatrixTimesSource(AlignedArray matrix, ReadOnlySpan<int> rgposSrc, AlignedArray sourceValues,
            int posMin, int iposMin, int iposLimit, AlignedArray destination, int stride)
        {
            Contracts.Assert(iposMin >= 0);
            Contracts.Assert(iposMin <= iposLimit);
            Contracts.Assert(iposLimit <= rgposSrc.Length);
            Contracts.Assert(matrix.Size == destination.Size * sourceValues.Size);

            if (iposMin >= iposLimit)
            {
                destination.ZeroItems();
                return;
            }

            Contracts.AssertNonEmpty(rgposSrc);
            Contracts.Assert(stride >= 0);

            if (Avx.IsSupported)
            {
                Contracts.Assert(stride <= destination.Size);
                AvxIntrinsics.MatMulP(matrix, rgposSrc, sourceValues, posMin, iposMin, iposLimit, destination, stride, sourceValues.Size);
            }
            else if (Sse.IsSupported)
            {
                Contracts.Assert(stride <= destination.Size);
                SseIntrinsics.MatMulP(matrix, rgposSrc, sourceValues, posMin, iposMin, iposLimit, destination, stride, sourceValues.Size);
            }
            else
            {
                Contracts.Assert(stride <= destination.Size);
                for (int i = 0; i < stride; i++)
                {
                    float dotProduct = 0;
                    for (int j = iposMin; j < iposLimit; j++)
                    {
                        int col = rgposSrc[j] - posMin;
                        dotProduct += matrix[i * sourceValues.Size + col] * sourceValues[col];
                    }
                    destination[i] = dotProduct;
                }
            }
        }

        /// <summary>
        /// Adds a value to a destination.
        /// </summary>
        /// <param name="value">The value to add.</param>
        /// <param name="destination">The destination to add the value to.</param>
        public static void Add(float value, Span<float> destination)
        {
            Contracts.AssertNonEmpty(destination);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.AddScalarU(value, destination);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.AddScalarU(value, destination);
            }
            else
            {
                for (int i = 0; i < destination.Length; i++)
                {
                    destination[i] += value;
                }
            }
        }

        /// <summary>
        /// Scales a value to a destination.
        /// </summary>
        /// <param name="value">The value to add.</param>
        /// <param name="destination">The destination to add the value to.</param>
        public static void Scale(float value, Span<float> destination)
        {
            Contracts.AssertNonEmpty(destination);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.Scale(value, destination);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.Scale(value, destination);
            }
            else
            {
                for (int i = 0; i < destination.Length; i++)
                {
                    destination[i] *= value;
                }
            }
        }

        /// <summary>
        /// Scales a values by a source to a destination.
        /// destination = value * source
        /// </summary>
        /// <param name="value">The value to scale by.</param>
        /// <param name="source">The source values.</param>
        /// <param name="destination">The destination.</param>
        /// <param name="count">The count of items.</param>
        public static void Scale(float value, ReadOnlySpan<float> source, Span<float> destination, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.AssertNonEmpty(destination);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);
            Contracts.Assert(count <= destination.Length);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.ScaleSrcU(value, source, destination, count);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.ScaleSrcU(value, source, destination, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    destination[i] = value * source[i];
                }
            }
        }

        /// <summary>
        /// Add to the destination by scale with an addend value.
        /// </summary>
        /// <code>
        /// destination[i] = scale * (destination[i] + addend)
        /// </code>
        /// <param name="scale">The scale to add by.</param>
        /// <param name="addend">The added value.</param>
        /// <param name="destination">The destination.</param>
        public static void ScaleAdd(float scale, float addend, Span<float> destination)
        {
            Contracts.AssertNonEmpty(destination);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.ScaleAddU(scale, addend, destination);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.ScaleAddU(scale, addend, destination);
            }
            else
            {
                for (int i = 0; i < destination.Length; i++)
                {
                    destination[i] = scale * (destination[i] + addend);
                }
            }
        }

        /// <summary>
        /// Add to the destination from the source by scale.
        /// </summary>
        /// <param name="scale">The scale to add by.</param>
        /// <param name="source">The source values.</param>
        /// <param name="destination">The destination values.</param>
        /// <param name="count">The count of items.</param>
        public static void AddScale(float scale, ReadOnlySpan<float> source, Span<float> destination, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.AssertNonEmpty(destination);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);
            Contracts.Assert(count <= destination.Length);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.AddScaleU(scale, source, destination, count);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.AddScaleU(scale, source, destination, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    destination[i] += scale * source[i];
                }
            }
        }

        /// <summary>
        /// Add to the destination by scale and source with indices.
        /// </summary>
        /// <param name="scale">The scale to add by.</param>
        /// <param name="source">The source values.</param>
        /// <param name="indices">The indices of value collection.</param>
        /// <param name="destination">The destination values.</param>
        /// <param name="count">The count of items.</param>
        public static void AddScale(float scale, ReadOnlySpan<float> source, ReadOnlySpan<int> indices, Span<float> destination, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.AssertNonEmpty(indices);
            Contracts.AssertNonEmpty(destination);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);
            Contracts.Assert(count <= indices.Length);
            Contracts.Assert(count < destination.Length);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.AddScaleSU(scale, source, indices, destination, count);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.AddScaleSU(scale, source, indices, destination, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    int index = indices[i];
                    destination[index] += scale * source[i];
                }
            }
        }

        /// <summary>
        /// Add to the destination by scale and source into a new result.
        /// </summary>
        /// <param name="scale">The scale to add by.</param>
        /// <param name="source">The source values.</param>
        /// <param name="destination">The destination values.</param>
        /// <param name="result">A new collection of values to be returned.</param>
        /// <param name="count">The count of items.</param>
        public static void AddScaleCopy(float scale, ReadOnlySpan<float> source, ReadOnlySpan<float> destination, Span<float> result, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.AssertNonEmpty(destination);
            Contracts.AssertNonEmpty(result);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);
            Contracts.Assert(count <= destination.Length);
            Contracts.Assert(count <= result.Length);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.AddScaleCopyU(scale, source, destination, result, count);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.AddScaleCopyU(scale, source, destination, result, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    result[i] = scale * source[i] + destination[i];
                }
            }
        }

        /// <summary>
        /// Add from a source to a destination.
        /// </summary>
        /// <param name="source">The source values.</param>
        /// <param name="destination">The destination values.</param>
        /// <param name="count">The count of items.</param>
        public static void Add(ReadOnlySpan<float> source, Span<float> destination, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.AssertNonEmpty(destination);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);
            Contracts.Assert(count <= destination.Length);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.AddU(source, destination, count);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.AddU(source, destination, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    destination[i] += source[i];
                }
            }
        }

        /// <summary>
        /// Add from a source to a destination with indices.
        /// </summary>
        /// <param name="source">The source values.</param>
        /// <param name="indices"></param>
        /// <param name="destination">The destination values.</param>
        /// <param name="count">The count of items.</param>
        public static void Add(ReadOnlySpan<float> source, ReadOnlySpan<int> indices, Span<float> destination, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.AssertNonEmpty(indices);
            Contracts.AssertNonEmpty(destination);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);
            Contracts.Assert(count <= indices.Length);
            Contracts.Assert(count < destination.Length);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.AddSU(source, indices, destination, count);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.AddSU(source, indices, destination, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    int index = indices[i];
                    destination[index] += source[i];
                }
            }
        }

        /// <summary>
        /// Multiply each element with left and right elements.
        /// </summary>
        /// <param name="left">The left element.</param>
        /// <param name="right">The right element.</param>
        /// <param name="destination">The destination values.</param>
        /// <param name="count">The count of items.</param>
        public static void MulElementWise(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> destination, int count)
        {
            Contracts.AssertNonEmpty(left);
            Contracts.AssertNonEmpty(right);
            Contracts.AssertNonEmpty(destination);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= left.Length);
            Contracts.Assert(count <= right.Length);
            Contracts.Assert(count <= destination.Length);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.MulElementWiseU(left, right, destination, count);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.MulElementWiseU(left, right, destination, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    destination[i] = left[i] * right[i];
                }
            }
        }

        /// <summary>
        /// Sum the values in the source.
        /// </summary>
        /// <param name="source">The source values.</param>
        /// <returns>The sum of all items in <paramref name="source"/>.</returns>
        public static float Sum(ReadOnlySpan<float> source)
        {
            Contracts.AssertNonEmpty(source);

            if (Avx.IsSupported)
            {
                return AvxIntrinsics.Sum(source);
            }
            else if (Sse.IsSupported)
            {
                return SseIntrinsics.Sum(source);
            }
            else
            {
                float sum = 0;
                for (int i = 0; i < source.Length; i++)
                {
                    sum += source[i];
                }
                return sum;
            }
        }

        /// <summary>
        /// Sum the squares of each item in the source.
        /// </summary>
        /// <param name="source">The source values.</param>
        /// <returns>The sum of the squares of all items in <paramref name="source"/>.</returns>
        public static float SumSq(ReadOnlySpan<float> source)
        {
            Contracts.AssertNonEmpty(source);

            if (Avx.IsSupported)
            {
                return AvxIntrinsics.SumSqU(source);
            }
            else if (Sse.IsSupported)
            {
                return SseIntrinsics.SumSqU(source);
            }
            else
            {
                float result = 0;
                for (int i = 0; i < source.Length; i++)
                {
                    result += source[i] * source[i];
                }
                return result;
            }
        }

        /// <summary>
        /// Sum the square of each item in the source and subtract the mean.
        /// </summary>
        /// <param name="mean">The mean value.</param>
        /// <param name="source">The source values.</param>
        /// <returns>The sum of all items in <paramref name="source"/> by <paramref name="mean"/>.</returns>
        public static float SumSq(float mean, ReadOnlySpan<float> source)
        {
            Contracts.AssertNonEmpty(source);

            if (Avx.IsSupported)
            {
                return (mean == 0) ? AvxIntrinsics.SumSqU(source) : AvxIntrinsics.SumSqDiffU(mean, source);
            }
            else if (Sse.IsSupported)
            {
                return (mean == 0) ? SseIntrinsics.SumSqU(source) : SseIntrinsics.SumSqDiffU(mean, source);
            }
            else
            {
                float result = 0;
                for (int i = 0; i < source.Length; i++)
                {
                    result += (source[i] - mean) * (source[i] - mean);
                }
                return result;
            }
        }

        /// <summary>
        /// Sum the absolute value of each item in the source.
        /// </summary>
        /// <param name="source">The source values.</param>
        /// <returns>The sum of all absolute value of the items in <paramref name="source"/>.</returns>
        public static float SumAbs(ReadOnlySpan<float> source)
        {
            Contracts.AssertNonEmpty(source);

            if (Avx.IsSupported)
            {
                return AvxIntrinsics.SumAbsU(source);
            }
            else if (Sse.IsSupported)
            {
                return SseIntrinsics.SumAbsU(source);
            }
            else
            {
                float sum = 0;
                for (int i = 0; i < source.Length; i++)
                {
                    sum += Math.Abs(source[i]);
                }
                return sum;
            }
        }

        /// <summary>
        /// Sum the absolute value of each item in the source and subtract the mean.
        /// </summary>
        /// <param name="mean">The mean value.</param>
        /// <param name="source">The source values.</param>
        /// <returns>The sum of all items by absolute value in <paramref name="source"/> subtracted by <paramref name="mean"/>.</returns>
        public static float SumAbs(float mean, ReadOnlySpan<float> source)
        {
            Contracts.AssertNonEmpty(source);

            if (Avx.IsSupported)
            {
                return (mean == 0) ? AvxIntrinsics.SumAbsU(source) : AvxIntrinsics.SumAbsDiffU(mean, source);
            }
            else if (Sse.IsSupported)
            {
                return (mean == 0) ? SseIntrinsics.SumAbsU(source) : SseIntrinsics.SumAbsDiffU(mean, source);
            }
            else
            {
                float sum = 0;
                for (int i = 0; i < source.Length; i++)
                {
                    sum += Math.Abs(source[i] - mean);
                }
                return sum;
            }
        }

        /// <summary>
        /// Take the maximum absolute value within the source.
        /// </summary>
        /// <param name="source">The source values.</param>
        /// <returns>The max of all absolute value items in <paramref name="source"/>.</returns>
        public static float MaxAbs(ReadOnlySpan<float> source)
        {
            Contracts.AssertNonEmpty(source);

            if (Avx.IsSupported)
            {
                return AvxIntrinsics.MaxAbsU(source);
            }
            else if (Sse.IsSupported)
            {
                return SseIntrinsics.MaxAbsU(source);
            }
            else
            {
                float max = 0;
                for (int i = 0; i < source.Length; i++)
                {
                    float abs = Math.Abs(source[i]);
                    if (abs > max)
                    {
                        max = abs;
                    }
                }
                return max;
            }
        }

        /// <summary>
        /// Take the maximum absolute value within the source and subtract the mean.
        /// </summary>
        /// <param name="mean">The mean value.</param>
        /// <param name="source">The source values.</param>
        /// <returns>The sum of all absolute value items in <paramref name="source"/> subtracted by <paramref name="mean"/>.</returns>
        public static float MaxAbsDiff(float mean, ReadOnlySpan<float> source)
        {
            Contracts.AssertNonEmpty(source);

            if (Avx.IsSupported)
            {
                return AvxIntrinsics.MaxAbsDiffU(mean, source);
            }
            else if (Sse.IsSupported)
            {
                return SseIntrinsics.MaxAbsDiffU(mean, source);
            }
            else
            {
                float max = 0;
                for (int i = 0; i < source.Length; i++)
                {
                    float abs = Math.Abs(source[i] - mean);
                    if (abs > max)
                    {
                        max = abs;
                    }
                }
                return max;
            }
        }

        /// <summary>
        /// Returns the dot product of each item in the left and right spans.
        /// </summary>
        /// <param name="left">The left span.</param>
        /// <param name="right">The right span.</param>
        /// <param name="count">The count of items.</param>
        /// <returns>The dot product.</returns>
        public static float DotProductDense(ReadOnlySpan<float> left, ReadOnlySpan<float> right, int count)
        {
            Contracts.AssertNonEmpty(left);
            Contracts.AssertNonEmpty(right);
            Contracts.Assert(count > 0);
            Contracts.Assert(left.Length >= count);
            Contracts.Assert(right.Length >= count);

            if (Avx.IsSupported)
            {
                return AvxIntrinsics.DotU(left, right, count);
            }
            else if (Sse.IsSupported)
            {
                return SseIntrinsics.DotU(left, right, count);
            }
            else
            {
                float result = 0;
                for (int i = 0; i < count; i++)
                {
                    result += left[i] * right[i];
                }
                return result;
            }
        }

        /// <summary>
        /// Returns the dot product of each item by index in the left and right spans.
        /// </summary>
        /// <param name="left">The left span.</param>
        /// <param name="right">The right span.</param>
        /// <param name="indices">The indicies of the left span.</param>
        /// <param name="count">The count of items.</param>
        /// <returns>The dot product.</returns>
        public static float DotProductSparse(ReadOnlySpan<float> left, ReadOnlySpan<float> right, ReadOnlySpan<int> indices, int count)
        {
            Contracts.AssertNonEmpty(left);
            Contracts.AssertNonEmpty(right);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count > 0);
            Contracts.Assert(count < left.Length);
            Contracts.Assert(count <= right.Length);
            Contracts.Assert(count <= indices.Length);

            if (Avx.IsSupported)
            {
                return AvxIntrinsics.DotSU(left, right, indices, count);
            }
            else if (Sse.IsSupported)
            {
                return SseIntrinsics.DotSU(left, right, indices, count);
            }
            else
            {
                float result = 0;
                for (int i = 0; i < count; i++)
                {
                    int index = indices[i];
                    result += left[index] * right[i];
                }
                return result;
            }
        }

        /// <summary>
        /// Returns the sum of the squared distance between the left and right spans.
        /// </summary>
        /// <param name="left">The left span.</param>
        /// <param name="right">The right span.</param>
        /// <param name="count">The count of items.</param>
        /// <returns>The squared distance value.</returns>
        public static float L2DistSquared(ReadOnlySpan<float> left, ReadOnlySpan<float> right, int count)
        {
            Contracts.AssertNonEmpty(left);
            Contracts.AssertNonEmpty(right);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= left.Length);
            Contracts.Assert(count <= right.Length);

            if (Avx.IsSupported)
            {
                return AvxIntrinsics.Dist2(left, right, count);
            }
            else if (Sse.IsSupported)
            {
                return SseIntrinsics.Dist2(left, right, count);
            }
            else
            {
                float norm = 0;
                for (int i = 0; i < count; i++)
                {
                    float distance = left[i] - right[i];
                    norm += distance * distance;
                }
                return norm;
            }
        }

        /// <summary>
        /// Sets the matrix items to zero.
        /// </summary>
        /// <param name="destination">The destination values.</param>
        /// <param name="ccol">The stride column.</param>
        /// <param name="cfltRow">The row to use.</param>
        /// <param name="indices">The indicies.</param>
        public static void ZeroMatrixItems(AlignedArray destination, int ccol, int cfltRow, int[] indices)
        {
            Contracts.Assert(ccol > 0);
            Contracts.Assert(ccol <= cfltRow);

            if (ccol == cfltRow)
            {
                ZeroItemsU(destination, destination.Size, indices, indices.Length);
            }
            else
            {
                ZeroMatrixItemsCore(destination, destination.Size, ccol, cfltRow, indices, indices.Length);
            }
        }

        private static unsafe void ZeroItemsU(AlignedArray destination, int c, int[] indices, int cindices)
        {
            fixed (float* pdst = &destination.Items[0])
            fixed (int* pidx = &indices[0])
            {
                for (int i = 0; i < cindices; ++i)
                {
                    int index = pidx[i];
                    Contracts.Assert(index >= 0);
                    Contracts.Assert(index < c);
                    pdst[index] = 0;
                }
            }
        }

        private static unsafe void ZeroMatrixItemsCore(AlignedArray destination, int c, int ccol, int cfltRow, int[] indices, int cindices)
        {
            fixed (float* pdst = &destination.Items[0])
            fixed (int* pidx = &indices[0])
            {
                int ivLogMin = 0;
                int ivLogLim = ccol;
                int ivPhyMin = 0;

                for (int i = 0; i < cindices; ++i)
                {
                    int index = pidx[i];
                    Contracts.Assert(index >= 0);
                    Contracts.Assert(index < c);

                    int col = index - ivLogMin;
                    if ((uint)col >= (uint)ccol)
                    {
                        Contracts.Assert(ivLogMin > index || index >= ivLogLim);

                        int row = index / ccol;
                        ivLogMin = row * ccol;
                        ivLogLim = ivLogMin + ccol;
                        ivPhyMin = row * cfltRow;

                        Contracts.Assert(index >= ivLogMin);
                        Contracts.Assert(index < ivLogLim);
                        col = index - ivLogMin;
                    }

                    pdst[ivPhyMin + col] = 0;
                }
            }
        }

        /// <summary>
        /// Updates span items with threshold.
        /// </summary>
        /// <param name="primalUpdate">The primal update value.</param>
        /// <param name="count">The count of items.</param>
        /// <param name="source">The source values.</param>
        /// <param name="threshold">The threshold value.</param>
        /// <param name="v">The v span.</param>
        /// <param name="w">The w span.</param>
        public static void SdcaL1UpdateDense(float primalUpdate, int count, ReadOnlySpan<float> source, float threshold, Span<float> v, Span<float> w)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.AssertNonEmpty(v);
            Contracts.AssertNonEmpty(w);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);
            Contracts.Assert(count <= v.Length);
            Contracts.Assert(count <= w.Length);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.SdcaL1UpdateU(primalUpdate, count, source, threshold, v, w);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.SdcaL1UpdateU(primalUpdate, count, source, threshold, v, w);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    v[i] += source[i] * primalUpdate;
                    float value = v[i];
                    w[i] = Math.Abs(value) > threshold ? (value > 0 ? value - threshold : value + threshold) : 0;
                }
            }
        }

        /// <summary>
        /// Updates span items with threshold by indices.
        /// </summary>
        /// <param name="primalUpdate">The primal update value.</param>
        /// <param name="count">The count of items.</param>
        /// <param name="source">The source values.</param>
        /// <param name="indices">The indicies of the source span.</param>
        /// <param name="threshold">The threshold.</param>
        /// <param name="v">The v span.</param>
        /// <param name="w">The w span.</param>
        public static void SdcaL1UpdateSparse(float primalUpdate, int count, ReadOnlySpan<float> source, ReadOnlySpan<int> indices, float threshold, Span<float> v, Span<float> w)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.AssertNonEmpty(indices);
            Contracts.AssertNonEmpty(v);
            Contracts.AssertNonEmpty(w);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);
            Contracts.Assert(count <= indices.Length);
            Contracts.Assert(count <= v.Length);
            Contracts.Assert(count <= w.Length);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.SdcaL1UpdateSU(primalUpdate, count, source, indices, threshold, v, w);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.SdcaL1UpdateSU(primalUpdate, count, source, indices, threshold, v, w);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    int index = indices[i];
                    v[index] += source[i] * primalUpdate;
                    float value = v[index];
                    w[index] = Math.Abs(value) > threshold ? (value > 0 ? value - threshold : value + threshold) : 0;
                }
            }
        }
    }
}
