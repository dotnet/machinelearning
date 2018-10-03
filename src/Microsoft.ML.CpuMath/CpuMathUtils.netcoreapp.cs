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

        public static void MatrixTimesSource(bool transpose, bool add, AlignedArray matrix, AlignedArray source, AlignedArray destination, int stride)
        {
            Contracts.Assert(matrix.Size == destination.Size * source.Size);
            Contracts.Assert(stride >= 0);

            if (Avx.IsSupported)
            {
                if (!transpose)
                {
                    Contracts.Assert(stride <= destination.Size);
                    AvxIntrinsics.MatMulX(add, matrix, source, destination, stride, source.Size);
                }
                else
                {
                    Contracts.Assert(stride <= source.Size);
                    AvxIntrinsics.MatMulTranX(add, matrix, source, destination, destination.Size, stride);
                }
            }
            else if (Sse.IsSupported)
            {
                if (!transpose)
                {
                    Contracts.Assert(stride <= destination.Size);
                    SseIntrinsics.MatMulA(add, matrix, source, destination, stride, source.Size);
                }
                else
                {
                    Contracts.Assert(stride <= source.Size);
                    SseIntrinsics.MatMulTranA(add, matrix, source, destination, destination.Size, stride);
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

                        if (add)
                        {
                            destination[i] += dotProduct;
                        }
                        else
                        {
                            destination[i] = dotProduct;
                        }
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

                        if (add)
                        {
                            destination[i] += dotProduct;
                        }
                        else
                        {
                            destination[i] = dotProduct;
                        }
                    }
                }
            }
        }

        public static void MatrixTimesSource(bool transpose, bool add, AlignedArray matrix, int[] posSource, AlignedArray sourceValues,
            int posMin, int iposMin, int iposLimit, AlignedArray destination, int stride)
        {
            Contracts.AssertValue(posSource);
            Contracts.Assert(iposMin >= 0);
            Contracts.Assert(iposMin <= iposLimit);
            Contracts.Assert(iposLimit <= posSource.Length);
            Contracts.Assert(matrix.Size == destination.Size * sourceValues.Size);

            if (iposMin >= iposLimit)
            {
                if (!add)
                    destination.ZeroItems();
                return;
            }

            Contracts.AssertNonEmpty(posSource);
            Contracts.Assert(stride >= 0);

            if (Avx.IsSupported)
            {
                if (!transpose)
                {
                    Contracts.Assert(stride <= destination.Size);
                    AvxIntrinsics.MatMulPX(add, matrix, posSource, sourceValues, posMin, iposMin, iposLimit, destination, stride, sourceValues.Size);
                }
                else
                {
                    Contracts.Assert(stride <= sourceValues.Size);
                    AvxIntrinsics.MatMulTranPX(add, matrix, posSource, sourceValues, posMin, iposMin, iposLimit, destination, destination.Size);
                }
            }
            else if (Sse.IsSupported)
            {
                if (!transpose)
                {
                    Contracts.Assert(stride <= destination.Size);
                    SseIntrinsics.MatMulPA(add, matrix, posSource, sourceValues, posMin, iposMin, iposLimit, destination, stride, sourceValues.Size);
                }
                else
                {
                    Contracts.Assert(stride <= sourceValues.Size);
                    SseIntrinsics.MatMulTranPA(add, matrix, posSource, sourceValues, posMin, iposMin, iposLimit, destination, destination.Size);
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
                        for (int j = iposMin; j < iposLimit; j++)
                        {
                            int col = posSource[j] - posMin;
                            dotProduct += matrix[i * sourceValues.Size + col] * sourceValues[col];
                        }

                        if (add)
                        {
                            destination[i] += dotProduct;
                        }
                        else
                        {
                            destination[i] = dotProduct;
                        }
                    }
                }
                else
                {
                    Contracts.Assert(stride <= sourceValues.Size);
                    for (int i = 0; i < destination.Size; i++)
                    {
                        float dotProduct = 0;
                        for (int j = iposMin; j < iposLimit; j++)
                        {
                            int col = posSource[j] - posMin;
                            dotProduct += matrix[col * destination.Size + i] * sourceValues[col];
                        }

                        if (add)
                        {
                            destination[i] += dotProduct;
                        }
                        else
                        {
                            destination[i] = dotProduct;
                        }
                    }

                }
            }
        }

        public static void Add(float value, float[] destination, int count)
        {
            Contracts.AssertNonEmpty(destination);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= destination.Length);

            Add(value, new Span<float>(destination, 0, count));
        }

        private static void Add(float value, Span<float> destination)
        {
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

        public static void Scale(float value, float[] destination, int count)
        {
            Contracts.AssertNonEmpty(destination);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= destination.Length);

            Scale(value, new Span<float>(destination, 0, count));
        }

        public static void Scale(float value, float[] destination, int offset, int count)
        {
            Contracts.AssertNonEmpty(destination);
            Contracts.Assert(count > 0);
            Contracts.Assert(offset >= 0);
            Contracts.Assert(offset < (destination.Length - count));

            Scale(value, new Span<float>(destination, offset, count));
        }

        private static void Scale(float value, Span<float> destination)
        {
            if (Avx.IsSupported)
            {
                AvxIntrinsics.ScaleU(value, destination);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.ScaleU(value, destination);
            }
            else
            {
                for (int i = 0; i < destination.Length; i++)
                {
                    destination[i] *= value;
                }
            }
        }

        // destination = a * source
        public static void Scale(float value, float[] source, float[] destination, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.AssertNonEmpty(destination);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);
            Contracts.Assert(count <= destination.Length);

            Scale(value, new Span<float>(source, 0, count), new Span<float>(destination, 0, count));
        }

        private static void Scale(float value, Span<float> source, Span<float> destination)
        {
            if (Avx.IsSupported)
            {
                AvxIntrinsics.ScaleSrcU(value, source, destination);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.ScaleSrcU(value, source, destination);
            }
            else
            {
                for (int i = 0; i < destination.Length; i++)
                {
                    destination[i] = value * source[i];
                }
            }
        }

        // destination[i] = a * (destination[i] + b)
        public static void ScaleAdd(float scale, float addend, float[] destination, int count)
        {
            Contracts.AssertNonEmpty(destination);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= destination.Length);

            ScaleAdd(scale, addend, new Span<float>(destination, 0, count));
        }

        private static void ScaleAdd(float scale, float addend, Span<float> destination)
        {
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

        public static void AddScale(float scale, float[] source, float[] destination, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.AssertNonEmpty(destination);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);
            Contracts.Assert(count <= destination.Length);

            AddScale(scale, new Span<float>(source, 0, count), new Span<float>(destination, 0, count));
        }

        public static void AddScale(float scale, float[] source, float[] destination, int dstOffset, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.AssertNonEmpty(destination);
            Contracts.Assert(dstOffset >= 0);
            Contracts.Assert(dstOffset < destination.Length);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);
            Contracts.Assert(count <= (destination.Length - dstOffset));

            AddScale(scale, new Span<float>(source, 0, count), new Span<float>(destination, dstOffset, count));
        }

        private static void AddScale(float scale, Span<float> source, Span<float> destination)
        {
            if (Avx.IsSupported)
            {
                AvxIntrinsics.AddScaleU(scale, source, destination);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.AddScaleU(scale, source, destination);
            }
            else
            {
                for (int i = 0; i < destination.Length; i++)
                {
                    destination[i] += scale * source[i];
                }
            }
        }

        public static void AddScale(float scale, float[] source, int[] indices, float[] destination, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.AssertNonEmpty(indices);
            Contracts.AssertNonEmpty(destination);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);
            Contracts.Assert(count <= indices.Length);
            Contracts.Assert(count < destination.Length);

            AddScale(scale, new Span<float>(source), new Span<int>(indices, 0, count), new Span<float>(destination));
        }

        public static void AddScale(float scale, float[] source, int[] indices, float[] destination, int dstOffset, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.AssertNonEmpty(indices);
            Contracts.AssertNonEmpty(destination);
            Contracts.Assert(dstOffset >= 0);
            Contracts.Assert(dstOffset < destination.Length);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);
            Contracts.Assert(count <= indices.Length);
            Contracts.Assert(count < (destination.Length - dstOffset));

            AddScale(scale, new Span<float>(source), new Span<int>(indices, 0, count),
                    new Span<float>(destination, dstOffset, destination.Length - dstOffset));
        }

        private static void AddScale(float scale, Span<float> source, Span<int> indices, Span<float> destination)
        {
            if (Avx.IsSupported)
            {
                AvxIntrinsics.AddScaleSU(scale, source, indices, destination);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.AddScaleSU(scale, source, indices, destination);
            }
            else
            {
                for (int i = 0; i < indices.Length; i++)
                {
                    int index = indices[i];
                    destination[index] += scale * source[i];
                }
            }
        }

        public static void AddScaleCopy(float scale, float[] source, float[] destination, float[] res, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.AssertNonEmpty(destination);
            Contracts.AssertNonEmpty(res);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);
            Contracts.Assert(count <= destination.Length);
            Contracts.Assert(count <= res.Length);

            AddScaleCopy(scale, new Span<float>(source, 0, count), new Span<float>(destination, 0, count), new Span<float>(res, 0, count));
        }

        private static void AddScaleCopy(float scale, Span<float> source, Span<float> destination, Span<float> res)
        {
            if (Avx.IsSupported)
            {
                AvxIntrinsics.AddScaleCopyU(scale, source, destination, res);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.AddScaleCopyU(scale, source, destination, res);
            }
            else
            {
                for (int i = 0; i < res.Length; i++)
                {
                    res[i] = scale * source[i] + destination[i];
                }
            }
        }

        public static void Add(float[] source, float[] destination, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.AssertNonEmpty(destination);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);
            Contracts.Assert(count <= destination.Length);

            Add(new Span<float>(source, 0, count), new Span<float>(destination, 0, count));
        }

        private static void Add(Span<float> source, Span<float> destination)
        {
            if (Avx.IsSupported)
            {
                AvxIntrinsics.AddU(source, destination);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.AddU(source, destination);
            }
            else
            {
                for (int i = 0; i < destination.Length; i++)
                {
                    destination[i] += source[i];
                }
            }
        }

        public static void Add(float[] source, int[] indices, float[] destination, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.AssertNonEmpty(indices);
            Contracts.AssertNonEmpty(destination);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);
            Contracts.Assert(count <= indices.Length);
            Contracts.Assert(count < destination.Length);

            Add(new Span<float>(source), new Span<int>(indices, 0, count), new Span<float>(destination));
        }

        public static void Add(float[] source, int[] indices, float[] destination, int dstOffset, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.AssertNonEmpty(indices);
            Contracts.AssertNonEmpty(destination);
            Contracts.Assert(dstOffset >= 0);
            Contracts.Assert(dstOffset < destination.Length);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);
            Contracts.Assert(count <= indices.Length);
            Contracts.Assert(count <= (destination.Length - dstOffset));

            Add(new Span<float>(source), new Span<int>(indices, 0, count),
                new Span<float>(destination, dstOffset, destination.Length - dstOffset));
        }

        private static void Add(Span<float> source, Span<int> indices, Span<float> destination)
        {
            if (Avx.IsSupported)
            {
                AvxIntrinsics.AddSU(source, indices, destination);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.AddSU(source, indices, destination);
            }
            else
            {
                for (int i = 0; i < indices.Length; i++)
                {
                    int index = indices[i];
                    destination[index] += source[i];
                }
            }
        }

        public static void MulElementWise(float[] left, float[] right, float[] destination, int count)
        {
            Contracts.AssertNonEmpty(left);
            Contracts.AssertNonEmpty(right);
            Contracts.AssertNonEmpty(destination);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= left.Length);
            Contracts.Assert(count <= right.Length);

            MulElementWise(new Span<float>(left, 0, count), new Span<float>(right, 0, count),
                            new Span<float>(destination, 0, count));
        }

        private static void MulElementWise(Span<float> left, Span<float> right, Span<float> destination)
        {
            if (Avx.IsSupported)
            {
                AvxIntrinsics.MulElementWiseU(left, right, destination);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.MulElementWiseU(left, right, destination);
            }
            else
            {
                for (int i = 0; i < destination.Length; i++)
                {
                    destination[i] = left[i] * right[i];
                }
            }
        }

        public static float Sum(float[] source, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);

            return Sum(new Span<float>(source, 0, count));
        }

        public static float Sum(float[] source, int offset, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.Assert(count > 0);
            Contracts.Assert(offset >= 0);
            Contracts.Assert(offset <= (source.Length - count));

            return Sum(new Span<float>(source, offset, count));
        }

        private static float Sum(Span<float> source)
        {
            if (Avx.IsSupported)
            {
                return AvxIntrinsics.SumU(source);
            }
            else if (Sse.IsSupported)
            {
                return SseIntrinsics.SumU(source);
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

        public static float SumSq(float[] source, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);

            return SumSq(new Span<float>(source, 0, count));
        }

        public static float SumSq(float[] source, int offset, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.Assert(count > 0);
            Contracts.Assert(offset >= 0);
            Contracts.Assert(offset <= (source.Length - count));

            return SumSq(new Span<float>(source, offset, count));
        }

        private static float SumSq(Span<float> source)
        {
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

        public static float SumSq(float mean, float[] source, int offset, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.Assert(count > 0);
            Contracts.Assert(offset >= 0);
            Contracts.Assert(offset <= (source.Length - count));

            return SumSq(mean, new Span<float>(source, offset, count));
        }

        private static float SumSq(float mean, Span<float> source)
        {
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

        public static float SumAbs(float[] source, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);

            return SumAbs(new Span<float>(source, 0, count));
        }

        public static float SumAbs(float[] source, int offset, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.Assert(count > 0);
            Contracts.Assert(offset >= 0);
            Contracts.Assert(offset <= (source.Length - count));

            return SumAbs(new Span<float>(source, offset, count));
        }

        private static float SumAbs(Span<float> source)
        {
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

        public static float SumAbs(float mean, float[] source, int offset, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.Assert(count > 0);
            Contracts.Assert(offset >= 0);
            Contracts.Assert(offset <= (source.Length - count));

            return SumAbs(mean, new Span<float>(source, offset, count));
        }

        private static float SumAbs(float mean, Span<float> source)
        {
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

        public static float MaxAbs(float[] source, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);

            return MaxAbs(new Span<float>(source, 0, count));
        }

        public static float MaxAbs(float[] source, int offset, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.Assert(count > 0);
            Contracts.Assert(offset >= 0);
            Contracts.Assert(offset <= (source.Length - count));

            return MaxAbs(new Span<float>(source, offset, count));
        }

        private static float MaxAbs(Span<float> source)
        {
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

        public static float MaxAbsDiff(float mean, float[] source, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);

            return MaxAbsDiff(mean, new Span<float>(source, 0, count));
        }

        private static float MaxAbsDiff(float mean, Span<float> source)
        {
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

        public static float DotProductDense(float[] a, float[] b, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(count > 0);
            Contracts.Assert(a.Length >= count);
            Contracts.Assert(b.Length >= count);

            return DotProductDense(new Span<float>(a, 0, count), new Span<float>(b, 0, count));
        }

        public static float DotProductDense(float[] a, int offset, float[] b, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= b.Length);
            Contracts.Assert(offset >= 0);
            Contracts.Assert(offset <= (a.Length - count));

            return DotProductDense(new Span<float>(a, offset, count), new Span<float>(b, 0, count));
        }

        private static float DotProductDense(Span<float> a, Span<float> b)
        {
            if (Avx.IsSupported)
            {
                return AvxIntrinsics.DotU(a, b);
            }
            else if (Sse.IsSupported)
            {
                return SseIntrinsics.DotU(a, b);
            }
            else
            {
                float result = 0;
                for (int i = 0; i < b.Length; i++)
                {
                    result += a[i] * b[i];
                }
                return result;
            }
        }

        public static float DotProductSparse(float[] value, float[] b, int[] indices, int count)
        {
            Contracts.AssertNonEmpty(value);
            Contracts.AssertNonEmpty(b);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count > 0);
            Contracts.Assert(count < value.Length);
            Contracts.Assert(count <= b.Length);
            Contracts.Assert(count <= indices.Length);

            return DotProductSparse(new Span<float>(value), new Span<float>(b),
                                    new Span<int>(indices, 0, count));
        }

        public static float DotProductSparse(float[] value, int offset, float[] b, int[] indices, int count)
        {
            Contracts.AssertNonEmpty(value);
            Contracts.AssertNonEmpty(b);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count > 0);
            Contracts.Assert(count < (value.Length - offset));
            Contracts.Assert(count <= b.Length);
            Contracts.Assert(count <= indices.Length);
            Contracts.Assert(offset >= 0);
            Contracts.Assert(offset < value.Length);

            return DotProductSparse(new Span<float>(value, offset, value.Length - offset),
                                    new Span<float>(b), new Span<int>(indices, 0, count));
        }

        private static float DotProductSparse(Span<float> value, Span<float> b, Span<int> indices)
        {
            if (Avx.IsSupported)
            {
                return AvxIntrinsics.DotSU(value, b, indices);
            }
            else if (Sse.IsSupported)
            {
                return SseIntrinsics.DotSU(value, b, indices);
            }
            else
            {
                float result = 0;
                for (int i = 0; i < indices.Length; i++)
                {
                    int index = indices[i];
                    result += value[index] * b[i];
                }
                return result;
            }
        }

        public static float L2DistSquared(float[] left, float[] right, int count)
        {
            Contracts.AssertNonEmpty(left);
            Contracts.AssertNonEmpty(right);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= left.Length);
            Contracts.Assert(count <= right.Length);

            return L2DistSquared(new Span<float>(left, 0, count), new Span<float>(right, 0, count));
        }

        private static float L2DistSquared(Span<float> left, Span<float> right)
        {
            if (Avx.IsSupported)
            {
                return AvxIntrinsics.Dist2(left, right);
            }
            else if (Sse.IsSupported)
            {
                return SseIntrinsics.Dist2(left, right);
            }
            else
            {
                float norm = 0;
                for (int i = 0; i < right.Length; i++)
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

        public static void SdcaL1UpdateDense(float primalUpdate, int length, float[] source, float threshold, float[] v, float[] w)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.AssertNonEmpty(v);
            Contracts.AssertNonEmpty(w);
            Contracts.Assert(length > 0);
            Contracts.Assert(length <= source.Length);
            Contracts.Assert(length <= v.Length);
            Contracts.Assert(length <= w.Length);

            SdcaL1UpdateDense(primalUpdate, new Span<float>(source, 0, length), threshold, new Span<float>(v, 0, length), new Span<float>(w, 0, length));
        }

        private static void SdcaL1UpdateDense(float primalUpdate, Span<float> source, float threshold, Span<float> v, Span<float> w)
        {
            if (Avx.IsSupported)
            {
                AvxIntrinsics.SdcaL1UpdateU(primalUpdate, source, threshold, v, w);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.SdcaL1UpdateU(primalUpdate, source, threshold, v, w);
            }
            else
            {
                for (int i = 0; i < source.Length; i++)
                {
                    v[i] += source[i] * primalUpdate;
                    float value = v[i];
                    w[i] = Math.Abs(value) > threshold ? (value > 0 ? value - threshold : value + threshold) : 0;
                }
            }
        }

        // REVIEW NEEDED: The second argument "length" is unused even in the existing code.
        public static void SdcaL1UpdateSparse(float primalUpdate, int length, float[] source, int[] indices, int count, float threshold, float[] v, float[] w)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.AssertNonEmpty(indices);
            Contracts.AssertNonEmpty(v);
            Contracts.AssertNonEmpty(w);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);
            Contracts.Assert(count <= indices.Length);
            Contracts.Assert(length <= v.Length);
            Contracts.Assert(length <= w.Length);

            SdcaL1UpdateSparse(primalUpdate, new Span<float>(source, 0, count), new Span<int>(indices, 0, count), threshold, new Span<float>(v), new Span<float>(w));
        }

        private static void SdcaL1UpdateSparse(float primalUpdate, Span<float> source, Span<int> indices, float threshold, Span<float> v, Span<float> w)
        {
            if (Avx.IsSupported)
            {
                AvxIntrinsics.SdcaL1UpdateSU(primalUpdate, source, indices, threshold, v, w);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.SdcaL1UpdateSU(primalUpdate, source, indices, threshold, v, w);
            }
            else
            {
                for (int i = 0; i < indices.Length; i++)
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
