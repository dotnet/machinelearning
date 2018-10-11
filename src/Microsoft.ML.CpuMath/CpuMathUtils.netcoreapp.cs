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

        public static void MatTimesSrc(bool tran, bool add, AlignedArray mat, AlignedArray src, AlignedArray dst, int crun)
        {
            Contracts.Assert(mat.Size == dst.Size * src.Size);
            Contracts.Assert(crun >= 0);

            if (Avx.IsSupported)
            {
                if (!tran)
                {
                    Contracts.Assert(crun <= dst.Size);
                    AvxIntrinsics.MatMulX(add, mat, src, dst, crun, src.Size);
                }
                else
                {
                    Contracts.Assert(crun <= src.Size);
                    AvxIntrinsics.MatMulTranX(add, mat, src, dst, dst.Size, crun);
                }
            }
            else if (Sse.IsSupported)
            {
                if (!tran)
                {
                    Contracts.Assert(crun <= dst.Size);
                    SseIntrinsics.MatMulA(add, mat, src, dst, crun, src.Size);
                }
                else
                {
                    Contracts.Assert(crun <= src.Size);
                    SseIntrinsics.MatMulTranA(add, mat, src, dst, dst.Size, crun);
                }
            }
            else
            {
                if (!tran)
                {
                    Contracts.Assert(crun <= dst.Size);
                    for (int i = 0; i < crun; i++)
                    {
                        float dotProduct = 0;
                        for (int j = 0; j < src.Size; j++)
                        {
                            dotProduct += mat[i * src.Size + j] * src[j];
                        }

                        if (add)
                        {
                            dst[i] += dotProduct;
                        }
                        else
                        {
                            dst[i] = dotProduct;
                        }
                    }
                }
                else
                {
                    Contracts.Assert(crun <= src.Size);
                    for (int i = 0; i < dst.Size; i++)
                    {
                        float dotProduct = 0;
                        for (int j = 0; j < crun; j++)
                        {
                            dotProduct += mat[j * src.Size + i] * src[j];
                        }

                        if (add)
                        {
                            dst[i] += dotProduct;
                        }
                        else
                        {
                            dst[i] = dotProduct;
                        }
                    }
                }
            }
        }

        public static void MatTimesSrc(bool tran, bool add, AlignedArray mat, int[] rgposSrc, AlignedArray srcValues,
            int posMin, int iposMin, int iposLim, AlignedArray dst, int crun)
        {
            Contracts.AssertValue(rgposSrc);
            Contracts.Assert(iposMin >= 0);
            Contracts.Assert(iposMin <= iposLim);
            Contracts.Assert(iposLim <= rgposSrc.Length);
            Contracts.Assert(mat.Size == dst.Size * srcValues.Size);

            if (iposMin >= iposLim)
            {
                if (!add)
                    dst.ZeroItems();
                return;
            }

            Contracts.AssertNonEmpty(rgposSrc);
            Contracts.Assert(crun >= 0);

            if (Avx.IsSupported)
            {
                if (!tran)
                {
                    Contracts.Assert(crun <= dst.Size);
                    AvxIntrinsics.MatMulPX(add, mat, rgposSrc, srcValues, posMin, iposMin, iposLim, dst, crun, srcValues.Size);
                }
                else
                {
                    Contracts.Assert(crun <= srcValues.Size);
                    AvxIntrinsics.MatMulTranPX(add, mat, rgposSrc, srcValues, posMin, iposMin, iposLim, dst, dst.Size);
                }
            }
            else if (Sse.IsSupported)
            {
                if (!tran)
                {
                    Contracts.Assert(crun <= dst.Size);
                    SseIntrinsics.MatMulPA(add, mat, rgposSrc, srcValues, posMin, iposMin, iposLim, dst, crun, srcValues.Size);
                }
                else
                {
                    Contracts.Assert(crun <= srcValues.Size);
                    SseIntrinsics.MatMulTranPA(add, mat, rgposSrc, srcValues, posMin, iposMin, iposLim, dst, dst.Size);
                }
            }
            else
            {
                if (!tran)
                {
                    Contracts.Assert(crun <= dst.Size);
                    for (int i = 0; i < crun; i++)
                    {
                        float dotProduct = 0;
                        for (int j = iposMin; j < iposLim; j++)
                        {
                            int col = rgposSrc[j] - posMin;
                            dotProduct += mat[i * srcValues.Size + col] * srcValues[col];
                        }

                        if (add)
                        {
                            dst[i] += dotProduct;
                        }
                        else
                        {
                            dst[i] = dotProduct;
                        }
                    }
                }
                else
                {
                    Contracts.Assert(crun <= srcValues.Size);
                    for (int i = 0; i < dst.Size; i++)
                    {
                        float dotProduct = 0;
                        for (int j = iposMin; j < iposLim; j++)
                        {
                            int col = rgposSrc[j] - posMin;
                            dotProduct += mat[col * dst.Size + i] * srcValues[col];
                        }

                        if (add)
                        {
                            dst[i] += dotProduct;
                        }
                        else
                        {
                            dst[i] = dotProduct;
                        }
                    }

                }
            }
        }

        public static void Add(float a, Span<float> dst)
        {
            Contracts.AssertNonEmpty(dst);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.AddScalarU(a, dst);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.AddScalarU(a, dst);
            }
            else
            {
                for (int i = 0; i < dst.Length; i++)
                {
                    dst[i] += a;
                }
            }
        }

        public static void Scale(float a, Span<float> dst)
        {
            Contracts.AssertNonEmpty(dst);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.Scale(a, dst);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.Scale(a, dst);
            }
            else
            {
                for (int i = 0; i < dst.Length; i++)
                {
                    dst[i] *= a;
                }
            }
        }

        // dst = a * src
        public static void Scale(float a, ReadOnlySpan<float> src, Span<float> dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= dst.Length);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.ScaleSrcU(a, src, dst, count);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.ScaleSrcU(a, src, dst, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    dst[i] = a * src[i];
                }
            }
        }

        // dst[i] = a * (dst[i] + b)
        public static void ScaleAdd(float a, float b, Span<float> dst)
        {
            Contracts.AssertNonEmpty(dst);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.ScaleAddU(a, b, dst);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.ScaleAddU(a, b, dst);
            }
            else
            {
                for (int i = 0; i < dst.Length; i++)
                {
                    dst[i] = a * (dst[i] + b);
                }
            }
        }

        public static void AddScale(float a, ReadOnlySpan<float> src, Span<float> dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= dst.Length);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.AddScaleU(a, src, dst, count);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.AddScaleU(a, src, dst, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    dst[i] += a * src[i];
                }
            }
        }

        public static void AddScale(float a, ReadOnlySpan<float> src, ReadOnlySpan<int> indices, Span<float> dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.AssertNonEmpty(indices);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= indices.Length);
            Contracts.Assert(count < dst.Length);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.AddScaleSU(a, src, indices, dst, count);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.AddScaleSU(a, src, indices, dst, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    int index = indices[i];
                    dst[index] += a * src[i];
                }
            }
        }

        public static void AddScaleCopy(float a, ReadOnlySpan<float> src, ReadOnlySpan<float> dst, Span<float> res, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.AssertNonEmpty(dst);
            Contracts.AssertNonEmpty(res);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= dst.Length);
            Contracts.Assert(count <= res.Length);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.AddScaleCopyU(a, src, dst, res, count);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.AddScaleCopyU(a, src, dst, res, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    res[i] = a * src[i] + dst[i];
                }
            }
        }

        public static void Add(ReadOnlySpan<float> src, Span<float> dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= dst.Length);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.AddU(src, dst, count);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.AddU(src, dst, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    dst[i] += src[i];
                }
            }
        }

        public static void Add(ReadOnlySpan<float> src, ReadOnlySpan<int> indices, Span<float> dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.AssertNonEmpty(indices);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= indices.Length);
            Contracts.Assert(count < dst.Length);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.AddSU(src, indices, dst, count);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.AddSU(src, indices, dst, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    int index = indices[i];
                    dst[index] += src[i];
                }
            }
        }

        public static void MulElementWise(ReadOnlySpan<float> src1, ReadOnlySpan<float> src2, Span<float> dst, int count)
        {
            Contracts.AssertNonEmpty(src1);
            Contracts.AssertNonEmpty(src2);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src1.Length);
            Contracts.Assert(count <= src2.Length);
            Contracts.Assert(count <= dst.Length);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.MulElementWiseU(src1, src2, dst, count);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.MulElementWiseU(src1, src2, dst, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    dst[i] = src1[i] * src2[i];
                }
            }
        }

        public static float Sum(ReadOnlySpan<float> src)
        {
            Contracts.AssertNonEmpty(src);

            if (Avx.IsSupported)
            {
                return AvxIntrinsics.SumU(src);
            }
            else if (Sse.IsSupported)
            {
                return SseIntrinsics.SumU(src);
            }
            else
            {
                float sum = 0;
                for (int i = 0; i < src.Length; i++)
                {
                    sum += src[i];
                }
                return sum;
            }
        }

        public static float SumSq(ReadOnlySpan<float> src)
        {
            Contracts.AssertNonEmpty(src);

            if (Avx.IsSupported)
            {
                return AvxIntrinsics.SumSqU(src);
            }
            else if (Sse.IsSupported)
            {
                return SseIntrinsics.SumSqU(src);
            }
            else
            {
                float result = 0;
                for (int i = 0; i < src.Length; i++)
                {
                    result += src[i] * src[i];
                }
                return result;
            }
        }

        public static float SumSq(float mean, ReadOnlySpan<float> src)
        {
            Contracts.AssertNonEmpty(src);

            if (Avx.IsSupported)
            {
                return (mean == 0) ? AvxIntrinsics.SumSqU(src) : AvxIntrinsics.SumSqDiffU(mean, src);
            }
            else if (Sse.IsSupported)
            {
                return (mean == 0) ? SseIntrinsics.SumSqU(src) : SseIntrinsics.SumSqDiffU(mean, src);
            }
            else
            {
                float result = 0;
                for (int i = 0; i < src.Length; i++)
                {
                    result += (src[i] - mean) * (src[i] - mean);
                }
                return result;
            }
        }

        public static float SumAbs(ReadOnlySpan<float> src)
        {
            Contracts.AssertNonEmpty(src);

            if (Avx.IsSupported)
            {
                return AvxIntrinsics.SumAbsU(src);
            }
            else if (Sse.IsSupported)
            {
                return SseIntrinsics.SumAbsU(src);
            }
            else
            {
                float sum = 0;
                for (int i = 0; i < src.Length; i++)
                {
                    sum += Math.Abs(src[i]);
                }
                return sum;
            }
        }

        public static float SumAbs(float mean, ReadOnlySpan<float> src)
        {
            Contracts.AssertNonEmpty(src);

            if (Avx.IsSupported)
            {
                return (mean == 0) ? AvxIntrinsics.SumAbsU(src) : AvxIntrinsics.SumAbsDiffU(mean, src);
            }
            else if (Sse.IsSupported)
            {
                return (mean == 0) ? SseIntrinsics.SumAbsU(src) : SseIntrinsics.SumAbsDiffU(mean, src);
            }
            else
            {
                float sum = 0;
                for (int i = 0; i < src.Length; i++)
                {
                    sum += Math.Abs(src[i] - mean);
                }
                return sum;
            }
        }

        public static float MaxAbs(ReadOnlySpan<float> src)
        {
            Contracts.AssertNonEmpty(src);

            if (Avx.IsSupported)
            {
                return AvxIntrinsics.MaxAbsU(src);
            }
            else if (Sse.IsSupported)
            {
                return SseIntrinsics.MaxAbsU(src);
            }
            else
            {
                float max = 0;
                for (int i = 0; i < src.Length; i++)
                {
                    float abs = Math.Abs(src[i]);
                    if (abs > max)
                    {
                        max = abs;
                    }
                }
                return max;
            }
        }

        public static float MaxAbsDiff(float mean, ReadOnlySpan<float> src)
        {
            Contracts.AssertNonEmpty(src);

            if (Avx.IsSupported)
            {
                return AvxIntrinsics.MaxAbsDiffU(mean, src);
            }
            else if (Sse.IsSupported)
            {
                return SseIntrinsics.MaxAbsDiffU(mean, src);
            }
            else
            {
                float max = 0;
                for (int i = 0; i < src.Length; i++)
                {
                    float abs = Math.Abs(src[i] - mean);
                    if (abs > max)
                    {
                        max = abs;
                    }
                }
                return max;
            }
        }

        public static float DotProductDense(ReadOnlySpan<float> a, ReadOnlySpan<float> b, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(count > 0);
            Contracts.Assert(a.Length >= count);
            Contracts.Assert(b.Length >= count);

            if (Avx.IsSupported)
            {
                return AvxIntrinsics.DotU(a, b, count);
            }
            else if (Sse.IsSupported)
            {
                return SseIntrinsics.DotU(a, b, count);
            }
            else
            {
                float result = 0;
                for (int i = 0; i < count; i++)
                {
                    result += a[i] * b[i];
                }
                return result;
            }
        }

        public static float DotProductSparse(ReadOnlySpan<float> a, ReadOnlySpan<float> b, ReadOnlySpan<int> indices, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.AssertNonEmpty(b);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count > 0);
            Contracts.Assert(count < a.Length);
            Contracts.Assert(count <= b.Length);
            Contracts.Assert(count <= indices.Length);

            if (Avx.IsSupported)
            {
                return AvxIntrinsics.DotSU(a, b, indices, count);
            }
            else if (Sse.IsSupported)
            {
                return SseIntrinsics.DotSU(a, b, indices, count);
            }
            else
            {
                float result = 0;
                for (int i = 0; i < count; i++)
                {
                    int index = indices[i];
                    result += a[index] * b[i];
                }
                return result;
            }
        }

        public static float L2DistSquared(ReadOnlySpan<float> a, ReadOnlySpan<float> b, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= a.Length);
            Contracts.Assert(count <= b.Length);

            if (Avx.IsSupported)
            {
                return AvxIntrinsics.Dist2(a, b, count);
            }
            else if (Sse.IsSupported)
            {
                return SseIntrinsics.Dist2(a, b, count);
            }
            else
            {
                float norm = 0;
                for (int i = 0; i < count; i++)
                {
                    float distance = a[i] - b[i];
                    norm += distance * distance;
                }
                return norm;
            }
        }

        public static void ZeroMatrixItems(AlignedArray dst, int ccol, int cfltRow, int[] indices)
        {
            Contracts.Assert(ccol > 0);
            Contracts.Assert(ccol <= cfltRow);

            if (ccol == cfltRow)
            {
                ZeroItemsU(dst, dst.Size, indices, indices.Length);
            }
            else
            {
                ZeroMatrixItemsCore(dst, dst.Size, ccol, cfltRow, indices, indices.Length);
            }
        }

        private static unsafe void ZeroItemsU(AlignedArray dst, int c, int[] indices, int cindices)
        {
            fixed (float* pdst = &dst.Items[0])
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

        private static unsafe void ZeroMatrixItemsCore(AlignedArray dst, int c, int ccol, int cfltRow, int[] indices, int cindices)
        {
            fixed (float* pdst = &dst.Items[0])
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

        public static void SdcaL1UpdateDense(float primalUpdate, int count, ReadOnlySpan<float> src, float threshold, Span<float> v, Span<float> w)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.AssertNonEmpty(v);
            Contracts.AssertNonEmpty(w);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= v.Length);
            Contracts.Assert(count <= w.Length);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.SdcaL1UpdateU(primalUpdate, count, src, threshold, v, w);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.SdcaL1UpdateU(primalUpdate, count, src, threshold, v, w);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    v[i] += src[i] * primalUpdate;
                    float value = v[i];
                    w[i] = Math.Abs(value) > threshold ? (value > 0 ? value - threshold : value + threshold) : 0;
                }
            }
        }

        public static void SdcaL1UpdateSparse(float primalUpdate, int count, ReadOnlySpan<float> src, ReadOnlySpan<int> indices, float threshold, Span<float> v, Span<float> w)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.AssertNonEmpty(indices);
            Contracts.AssertNonEmpty(v);
            Contracts.AssertNonEmpty(w);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= indices.Length);
            Contracts.Assert(count <= v.Length);
            Contracts.Assert(count <= w.Length);

            if (Avx.IsSupported)
            {
                AvxIntrinsics.SdcaL1UpdateSU(primalUpdate, count, src, indices, threshold, v, w);
            }
            else if (Sse.IsSupported)
            {
                SseIntrinsics.SdcaL1UpdateSU(primalUpdate, count, src, indices, threshold, v, w);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    int index = indices[i];
                    v[index] += src[i] * primalUpdate;
                    float value = v[index];
                    w[index] = Math.Abs(value) > threshold ? (value > 0 ? value - threshold : value + threshold) : 0;
                }
            }
        }
    }
}
