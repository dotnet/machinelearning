// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    public static partial class CpuMathUtils
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
                            dotProduct += matrix[j * source.Size + i] * source[j];
                        }

                        destination[i] = dotProduct;
                    }
                }
            }
        }

        public static void MatTimesSrc(AlignedArray matrix, int[] rgposSrc, AlignedArray sourceValues,
            int posMin, int iposMin, int iposLimit, AlignedArray destination, int stride)
        {
            Contracts.AssertValue(rgposSrc);
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

        // destination = value * source
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

        // destination[i] = scale * (destination[i] + addend)
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
