// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.Intrinsics.X86;
using System;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    public static partial class CpuMathUtils
    {
        public static void MatTimesSrc(bool tran, bool add, AlignedArray mat, AlignedArray src, AlignedArray dst, int crun)
        {
            Contracts.Assert(mat.Size == dst.Size * src.Size);

            if (Sse.IsSupported)
            {
                if (!tran)
                {
                    Contracts.Assert(0 <= crun && crun <= dst.Size);
                    SseIntrinsics.MatMulA(add, mat, src, dst, crun, src.Size);
                }
                else
                {
                    Contracts.Assert(0 <= crun && crun <= src.Size);
                    SseIntrinsics.MatMulTranA(add, mat, src, dst, dst.Size, crun);
                }
            }
            else
            {
                if (!tran)
                {
                    Contracts.Assert(0 <= crun && crun <= dst.Size);
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
                    Contracts.Assert(0 <= crun && crun <= src.Size);
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
            Contracts.Assert(0 <= iposMin && iposMin <= iposLim && iposLim <= rgposSrc.Length);
            Contracts.Assert(mat.Size == dst.Size * srcValues.Size);

            if (iposMin >= iposLim)
            {
                if (!add)
                    dst.ZeroItems();
                return;
            }

            Contracts.AssertNonEmpty(rgposSrc);

            if (Sse.IsSupported)
            {
                if (!tran)
                {
                    Contracts.Assert(0 <= crun && crun <= dst.Size);
                    SseIntrinsics.MatMulPA(add, mat, rgposSrc, srcValues, posMin, iposMin, iposLim, dst, crun, srcValues.Size);
                }
                else
                {
                    Contracts.Assert(0 <= crun && crun <= srcValues.Size);
                    SseIntrinsics.MatMulTranPA(add, mat, rgposSrc, srcValues, posMin, iposMin, iposLim, dst, dst.Size);
                }
            }
            else
            {
                if (!tran)
                {
                    Contracts.Assert(0 <= crun && crun <= dst.Size);
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
                    Contracts.Assert(0 <= crun && crun <= srcValues.Size);
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
            Contracts.Assert(0 < count);
            Contracts.Assert(0 < count && count <= dst.Length);

            Add(a, new Span<float>(dst, 0, count));
        }

        // dst += a
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
            Contracts.Assert(0 < count && count <= dst.Length);

            Scale(a, new Span<float>(dst, 0, count));
        }

        public static void Scale(float a, float[] dst, int offset, int count)
        {
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset < dst.Length - count);

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
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(dst);
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
            Contracts.Assert(0 < count);
            Contracts.Assert(0 < count && count <= dst.Length);

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
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count <= dst.Length);

            AddScale(a, new Span<float>(src, 0, count), new Span<float>(dst, 0, count));
        }

        public static void AddScale(float a, float[] src, float[] dst, int dstOffset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(count <= src.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 <= dstOffset && dstOffset < dst.Length);
            Contracts.Assert(0 < count && count <= dst.Length - dstOffset);

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
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count <= indices.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count < dst.Length);

            AddScale(a, new Span<float>(src), new Span<int>(indices, 0, count), new Span<float>(dst));
        }

        public static void AddScale(float a, float[] src, int[] indices, float[] dst, int dstOffset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count <= indices.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 <= dstOffset && dstOffset < dst.Length);
            Contracts.Assert(count < dst.Length - dstOffset);

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
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 < count && count <= dst.Length);
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(count <= src.Length);
            Contracts.AssertNonEmpty(res);
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
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(dst);
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
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count <= indices.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count < dst.Length);

            Add(new Span<float>(src), new Span<int>(indices, 0, count), new Span<float>(dst));
        }

        public static void Add(float[] src, int[] indices, float[] dst, int dstOffset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count <= indices.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 <= dstOffset && dstOffset < dst.Length);
            Contracts.Assert(count <= dst.Length - dstOffset);

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
            Contracts.Assert(0 < count && count <= src1.Length);
            Contracts.AssertNonEmpty(src2);
            Contracts.Assert(0 < count && count <= src2.Length);
            Contracts.AssertNonEmpty(dst);

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
            Contracts.Assert(0 < count && count <= src.Length);

            return Sum(new Span<float>(src, 0, count));
        }

        public static float Sum(float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= src.Length - count);

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
            Contracts.Assert(0 < count && count <= src.Length);

            return SumSq(new Span<float>(src, 0, count));
        }

        public static float SumSq(float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= src.Length - count);

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
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= src.Length - count);

            return SumSq(mean, new Span<float>(src, offset, count));
        }

        private static float SumSq(float mean, Span<float> src)
        {
            if (Sse.IsSupported)
            {
                if (mean == 0)
                {
                    return SseIntrinsics.SumSqU(src);
                }
                else
                {
                    return SseIntrinsics.SumSqDiffU(mean, src);
                }
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
            Contracts.Assert(0 < count && count <= src.Length);

            return SumAbs(new Span<float>(src, 0, count));
        }

        public static float SumAbs(float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= src.Length - count);

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
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= src.Length - count);

            return SumAbs(mean, new Span<float>(src, offset, count));
        }

        private static float SumAbs(float mean, Span<float> src)
        {
            if (Sse.IsSupported)
            {
                if (mean == 0)
                {
                    return SseIntrinsics.SumAbsU(src);
                }
                else
                {
                    return SseIntrinsics.SumAbsDiffU(mean, src);
                }
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
            Contracts.Assert(0 < count && count <= src.Length);

            return MaxAbs(new Span<float>(src, 0, count));
        }

        public static float MaxAbs(float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= src.Length - count);

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
            Contracts.Assert(0 < count && count <= src.Length);

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
            Contracts.Assert(0 < count);
            Contracts.Assert(a.Length >= count);
            Contracts.Assert(b.Length >= count);

            return DotProductDense(new Span<float>(a, 0, count), new Span<float>(b, 0, count));
        }

        public static float DotProductDense(float[] a, int offset, float[] b, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= a.Length - count);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(b.Length >= count);

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
            Contracts.Assert(0 < count);
            Contracts.Assert(count < a.Length);
            Contracts.Assert(count <= b.Length);
            Contracts.Assert(count <= indices.Length);

            return DotProductSparse(new Span<float>(a), new Span<float>(b),
                                    new Span<int>(indices, 0, count));
        }

        public static float DotProductSparse(float[] a, int offset, float[] b, int[] indices, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset < a.Length);
            Contracts.Assert(a.Length - offset > count);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(count <= b.Length);
            Contracts.Assert(count <= indices.Length);

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
            Contracts.Assert(0 < count && count <= a.Length);
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
            Contracts.Assert(0 < ccol && ccol <= cfltRow);

            // REVIEW NEEDED: Since the two methods below do not involve any SSE hardware intrinsics, no software fallback is needed.
            // REVIEW NEEDED: Keeping the check for SSE support so that we don't miss these two methods in case of any conditional compilation of files
            if (Sse.IsSupported)
            {
                if (ccol == cfltRow)
                {
                    SseIntrinsics.ZeroItemsU(dst, dst.Size, indices, indices.Length);
                }
                else
                {
                    SseIntrinsics.ZeroMatrixItemsCore(dst, dst.Size, ccol, cfltRow, indices, indices.Length);
                }
            }
        }

        public static void SdcaL1UpdateDense(float primalUpdate, int length, float[] src, float threshold, float[] v, float[] w)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(length <= src.Length);
            Contracts.AssertNonEmpty(v);
            Contracts.Assert(length <= v.Length);
            Contracts.AssertNonEmpty(w);
            Contracts.Assert(length <= w.Length);
            Contracts.Assert(length > 0);

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
            Contracts.Assert(count <= src.Length);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count <= indices.Length);
            Contracts.AssertNonEmpty(w);
            Contracts.Assert(length <= w.Length);
            Contracts.AssertNonEmpty(v);
            Contracts.Assert(length <= v.Length);
            Contracts.Assert(0 < count);
            Contracts.Assert(count < length);

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
