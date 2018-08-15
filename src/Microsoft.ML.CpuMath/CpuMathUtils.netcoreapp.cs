// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.Intrinsics.X86;
using System;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    public static partial class CpuMathUtils
    {
        // The count of bytes in Vector128<T>, corresponding to _cbAlign in AlignedArray
        public const int Vector128Alignment = 16;

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

        public static void Add(float a, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= dst.Length);

            Add(a, new Span<float>(dst, 0, count));
        }

        private static void Add(float a, Span<float> dst)
        {
            if (Sse.IsSupported)
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

        public static void Scale(float a, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= dst.Length);

            Scale(a, new Span<float>(dst, 0, count));
        }

        public static void Scale(float a, float[] dst, int offset, int count)
        {
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count > 0);
            Contracts.Assert(offset >= 0);
            Contracts.Assert(offset < (dst.Length - count));

            Scale(a, new Span<float>(dst, offset, count));
        }

        private static void Scale(float a, Span<float> dst)
        {
            if (Sse.IsSupported)
            {
                SseIntrinsics.ScaleU(a, dst);
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
        public static void Scale(float a, float[] src, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= dst.Length);

            Scale(a, new Span<float>(src, 0, count), new Span<float>(dst, 0, count));
        }

        private static void Scale(float a, Span<float> src, Span<float> dst)
        {
            if (Sse.IsSupported)
            {
                SseIntrinsics.ScaleSrcU(a, src, dst);
            }
            else
            {
                for (int i = 0; i < dst.Length; i++)
                {
                    dst[i] = a * src[i];
                }
            }
        }

        // dst[i] = a * (dst[i] + b)
        public static void ScaleAdd(float a, float b, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= dst.Length);

            ScaleAdd(a, b, new Span<float>(dst, 0, count));
        }

        private static void ScaleAdd(float a, float b, Span<float> dst)
        {
            if (Sse.IsSupported)
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

        public static void AddScale(float a, float[] src, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= dst.Length);

            AddScale(a, new Span<float>(src, 0, count), new Span<float>(dst, 0, count));
        }

        public static void AddScale(float a, float[] src, float[] dst, int dstOffset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(dstOffset >= 0);
            Contracts.Assert(dstOffset < dst.Length);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= (dst.Length - dstOffset));

            AddScale(a, new Span<float>(src, 0, count), new Span<float>(dst, dstOffset, count));
        }

        private static void AddScale(float a, Span<float> src, Span<float> dst)
        {
            if (Sse.IsSupported)
            {
                SseIntrinsics.AddScaleU(a, src, dst);
            }
            else
            {
                for (int i = 0; i < dst.Length; i++)
                {
                    dst[i] += a * src[i];
                }
            }
        }

        public static void AddScale(float a, float[] src, int[] indices, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.AssertNonEmpty(indices);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= indices.Length);
            Contracts.Assert(count < dst.Length);

            AddScale(a, new Span<float>(src), new Span<int>(indices, 0, count), new Span<float>(dst));
        }

        public static void AddScale(float a, float[] src, int[] indices, float[] dst, int dstOffset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.AssertNonEmpty(indices);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(dstOffset >= 0);
            Contracts.Assert(dstOffset < dst.Length);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= indices.Length);
            Contracts.Assert(count < (dst.Length - dstOffset));

            AddScale(a, new Span<float>(src), new Span<int>(indices, 0, count),
                    new Span<float>(dst, dstOffset, dst.Length - dstOffset));
        }

        private static void AddScale(float a, Span<float> src, Span<int> indices, Span<float> dst)
        {
            if (Sse.IsSupported)
            {
                SseIntrinsics.AddScaleSU(a, src, indices, dst);
            }
            else
            {
                for (int i = 0; i < indices.Length; i++)
                {
                    int index = indices[i];
                    dst[index] += a * src[i];
                }
            }
        }

        public static void AddScaleCopy(float a, float[] src, float[] dst, float[] res, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.AssertNonEmpty(dst);
            Contracts.AssertNonEmpty(res);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= dst.Length);
            Contracts.Assert(count <= res.Length);

            AddScaleCopy(a, new Span<float>(src, 0, count), new Span<float>(dst, 0, count), new Span<float>(res, 0, count));
        }

        private static void AddScaleCopy(float a, Span<float> src, Span<float> dst, Span<float> res)
        {
            if (Sse.IsSupported)
            {
                SseIntrinsics.AddScaleCopyU(a, src, dst, res);
            }
            else
            {
                for (int i = 0; i < res.Length; i++)
                {
                    res[i] = a * src[i] + dst[i];
                }
            }
        }

        public static void Add(float[] src, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= dst.Length);

            Add(new Span<float>(src, 0, count), new Span<float>(dst, 0, count));
        }

        private static void Add(Span<float> src, Span<float> dst)
        {
            if (Sse.IsSupported)
            {
                SseIntrinsics.AddU(src, dst);
            }
            else
            {
                for (int i = 0; i < dst.Length; i++)
                {
                    dst[i] += src[i];
                }
            }
        }

        public static void Add(float[] src, int[] indices, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.AssertNonEmpty(indices);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= indices.Length);
            Contracts.Assert(count < dst.Length);

            Add(new Span<float>(src), new Span<int>(indices, 0, count), new Span<float>(dst));
        }

        public static void Add(float[] src, int[] indices, float[] dst, int dstOffset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.AssertNonEmpty(indices);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(dstOffset >= 0);
            Contracts.Assert(dstOffset < dst.Length);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= indices.Length);
            Contracts.Assert(count <= (dst.Length - dstOffset));

            Add(new Span<float>(src), new Span<int>(indices, 0, count),
                new Span<float>(dst, dstOffset, dst.Length - dstOffset));
        }

        private static void Add(Span<float> src, Span<int> indices, Span<float> dst)
        {
            if (Sse.IsSupported)
            {
                SseIntrinsics.AddSU(src, indices, dst);
            }
            else
            {
                for (int i = 0; i < indices.Length; i++)
                {
                    int index = indices[i];
                    dst[index] += src[i];
                }
            }
        }

        public static void MulElementWise(float[] src1, float[] src2, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src1);
            Contracts.AssertNonEmpty(src2);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src1.Length);
            Contracts.Assert(count <= src2.Length);

            MulElementWise(new Span<float>(src1, 0, count), new Span<float>(src2, 0, count),
                            new Span<float>(dst, 0, count));
        }

        private static void MulElementWise(Span<float> src1, Span<float> src2, Span<float> dst)
        {
            if (Sse.IsSupported)
            {
                SseIntrinsics.MulElementWiseU(src1, src2, dst);
            }
            else
            {
                for (int i = 0; i < dst.Length; i++)
                {
                    dst[i] = src1[i] * src2[i];
                }
            }
        }

        public static float Sum(float[] src, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src.Length);

            return Sum(new Span<float>(src, 0, count));
        }

        public static float Sum(float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(count > 0);
            Contracts.Assert(offset >= 0);
            Contracts.Assert(offset <= (src.Length - count));

            return Sum(new Span<float>(src, offset, count));
        }

        private static float Sum(Span<float> src)
        {
            if (Sse.IsSupported)
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

        public static float SumSq(float[] src, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src.Length);

            return SumSq(new Span<float>(src, 0, count));
        }

        public static float SumSq(float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(count > 0);
            Contracts.Assert(offset >= 0);
            Contracts.Assert(offset <= (src.Length - count));

            return SumSq(new Span<float>(src, offset, count));
        }

        private static float SumSq(Span<float> src)
        {
            if (Sse.IsSupported)
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

        public static float SumSq(float mean, float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(count > 0);
            Contracts.Assert(offset >= 0);
            Contracts.Assert(offset <= (src.Length - count));

            return SumSq(mean, new Span<float>(src, offset, count));
        }

        private static float SumSq(float mean, Span<float> src)
        {
            if (Sse.IsSupported)
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

        public static float SumAbs(float[] src, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src.Length);

            return SumAbs(new Span<float>(src, 0, count));
        }

        public static float SumAbs(float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(count > 0);
            Contracts.Assert(offset >= 0);
            Contracts.Assert(offset <= (src.Length - count));

            return SumAbs(new Span<float>(src, offset, count));
        }

        private static float SumAbs(Span<float> src)
        {
            if (Sse.IsSupported)
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

        public static float SumAbs(float mean, float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(count > 0);
            Contracts.Assert(offset >= 0);
            Contracts.Assert(offset <= (src.Length - count));

            return SumAbs(mean, new Span<float>(src, offset, count));
        }

        private static float SumAbs(float mean, Span<float> src)
        {
            if (Sse.IsSupported)
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

        public static float MaxAbs(float[] src, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src.Length);

            return MaxAbs(new Span<float>(src, 0, count));
        }

        public static float MaxAbs(float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(count > 0);
            Contracts.Assert(offset >= 0);
            Contracts.Assert(offset <= (src.Length - count));

            return MaxAbs(new Span<float>(src, offset, count));
        }

        private static float MaxAbs(Span<float> src)
        {
            if (Sse.IsSupported)
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

        public static float MaxAbsDiff(float mean, float[] src, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src.Length);

            return MaxAbsDiff(mean, new Span<float>(src, 0, count));
        }

        private static float MaxAbsDiff(float mean, Span<float> src)
        {
            if (Sse.IsSupported)
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
            if (Sse.IsSupported)
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

        public static float DotProductSparse(float[] a, float[] b, int[] indices, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.AssertNonEmpty(b);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count > 0);
            Contracts.Assert(count < a.Length);
            Contracts.Assert(count <= b.Length);
            Contracts.Assert(count <= indices.Length);

            return DotProductSparse(new Span<float>(a), new Span<float>(b),
                                    new Span<int>(indices, 0, count));
        }

        public static float DotProductSparse(float[] a, int offset, float[] b, int[] indices, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.AssertNonEmpty(b);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count > 0);
            Contracts.Assert(count < (a.Length - offset));
            Contracts.Assert(count <= b.Length);
            Contracts.Assert(count <= indices.Length);
            Contracts.Assert(offset >= 0);
            Contracts.Assert(offset < a.Length);

            return DotProductSparse(new Span<float>(a, offset, a.Length - offset),
                                    new Span<float>(b), new Span<int>(indices, 0, count));
        }

        private static float DotProductSparse(Span<float> a, Span<float> b, Span<int> indices)
        {
            if (Sse.IsSupported)
            {
                return SseIntrinsics.DotSU(a, b, indices);
            }
            else
            {
                float result = 0;
                for (int i = 0; i < indices.Length; i++)
                {
                    int index = indices[i];
                    result += a[index] * b[i];
                }
                return result;
            }
        }

        public static float L2DistSquared(float[] a, float[] b, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= a.Length);
            Contracts.Assert(count <= b.Length);

            return L2DistSquared(new Span<float>(a, 0, count), new Span<float>(b, 0, count));
        }

        private static float L2DistSquared(Span<float> a, Span<float> b)
        {
            if (Sse.IsSupported)
            {
                return SseIntrinsics.Dist2(a, b);
            }
            else
            {
                float norm = 0;
                for (int i = 0; i < b.Length; i++)
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

        public static void SdcaL1UpdateDense(float primalUpdate, int length, float[] src, float threshold, float[] v, float[] w)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.AssertNonEmpty(v);
            Contracts.AssertNonEmpty(w);
            Contracts.Assert(length > 0);
            Contracts.Assert(length <= src.Length);
            Contracts.Assert(length <= v.Length);
            Contracts.Assert(length <= w.Length);

            SdcaL1UpdateDense(primalUpdate, new Span<float>(src, 0, length), threshold, new Span<float>(v, 0, length), new Span<float>(w, 0, length));
        }

        private static void SdcaL1UpdateDense(float primalUpdate, Span<float> src, float threshold, Span<float> v, Span<float> w)
        {
            if (Sse.IsSupported)
            {
                SseIntrinsics.SdcaL1UpdateU(primalUpdate, src, threshold, v, w);
            }
            else
            {
                for (int i = 0; i < src.Length; i++)
                {
                    v[i] += src[i] * primalUpdate;
                    float value = v[i];
                    w[i] = Math.Abs(value) > threshold ? (value > 0 ? value - threshold : value + threshold) : 0;
                }
            }
        }

        // REVIEW NEEDED: The second argument "length" is unused even in the existing code.
        public static void SdcaL1UpdateSparse(float primalUpdate, int length, float[] src, int[] indices, int count, float threshold, float[] v, float[] w)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.AssertNonEmpty(indices);
            Contracts.AssertNonEmpty(v);
            Contracts.AssertNonEmpty(w);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= indices.Length);
            Contracts.Assert(count < length);
            Contracts.Assert(length <= v.Length);
            Contracts.Assert(length <= w.Length);

            SdcaL1UpdateSparse(primalUpdate, new Span<float>(src, 0, count), new Span<int>(indices, 0, count), threshold, new Span<float>(v), new Span<float>(w));
        }

        private static void SdcaL1UpdateSparse(float primalUpdate, Span<float> src, Span<int> indices, float threshold, Span<float> v, Span<float> w)
        {
            if (Sse.IsSupported)
            {
                SseIntrinsics.SdcaL1UpdateSU(primalUpdate, src, indices, threshold, v, w);
            }
            else
            {
                for (int i = 0; i < indices.Length; i++)
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
