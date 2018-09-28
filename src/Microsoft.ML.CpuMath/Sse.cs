// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    /// <summary>
    /// Keep Sse.cs in sync with Avx.cs. When making changes to one, use BeyondCompare or a similar tool
    /// to view diffs and propagate appropriate changes to the other.
    /// </summary>
    public static class SseUtils
    {
        public static void MatTimesSrc(bool tran, bool add, float[] mat, float[] src, float[] dst, int crun)
        {
            Contracts.Assert(mat.Length == dst.Length * src.Length);

            unsafe
            {
                fixed (float* pmat = &mat[0])
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                {
                    if (!tran)
                    {
                        Contracts.Assert(0 <= crun && crun <= dst.Length);
                        Thunk.MatMul(add, pmat, psrc, pdst, crun, src.Length);
                    }
                    else
                    {
                        Contracts.Assert(0 <= crun && crun <= src.Length);
                        Thunk.MatMulTran(add, pmat, psrc, pdst, dst.Length, crun);
                    }
                }
            }
        }

        public static void MatTimesSrc(bool tran, bool add, float[] mat, int[] rgposSrc, float[] srcValues,
            int posMin, int iposMin, int iposLim, float[] dst, int crun)
        {
            Contracts.AssertValue(rgposSrc);
            Contracts.Assert(0 <= iposMin && iposMin <= iposLim && iposLim <= rgposSrc.Length);
            Contracts.Assert(mat.Length == dst.Length * srcValues.Length);

            if (iposMin >= iposLim)
            {
                if (!add)
                    Array.Clear(dst, 0, dst.Length);
                return;
            }
            Contracts.AssertNonEmpty(rgposSrc);
            unsafe
            {
                fixed (float* pdst = &dst[0])
                fixed (float* pmat = &mat[0])
                fixed (float* psrc = &srcValues[0])
                fixed (int* ppossrc = &rgposSrc[0])
                {
                    if (!tran)
                    {
                        Contracts.Assert(0 <= crun && crun <= dst.Length);
                        Thunk.MatMulP(add, pmat, ppossrc, psrc, posMin, iposMin, iposLim, pdst, crun, srcValues.Length);
                    }
                    else
                    {
                        Contracts.Assert(0 <= crun && crun <= srcValues.Length);
                        Thunk.MatMulTranP(add, pmat, ppossrc, psrc, posMin, iposMin, iposLim, pdst, dst.Length);
                    }
                }
            }
        }

        // dst += a
        public static void Add(float a, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 < count && count <= dst.Length);

            unsafe
            {
                fixed (float* pdst = &dst[0])
                    Thunk.AddScalarU(a, pdst, count);
            }
        }

        public static void Scale(float a, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 < count && count <= dst.Length);

            unsafe
            {
                fixed (float* pd = &dst[0])
                    Thunk.ScaleU(a, pd, count);
            }
        }

        // dst = a * src
        public static void Scale(float a, float[] src, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count <= dst.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                {
                    Thunk.ScaleSrcU(a, psrc, pdst, count);
                }
            }
        }

        // dst[i] = a * (dst[i] + b)
        public static void ScaleAdd(float a, float b, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 < count && count <= dst.Length);

            unsafe
            {
                fixed (float* pdst = &dst[0])
                    Thunk.ScaleAddU(a, b, pdst, count);
            }
        }

        public static void AddScale(float a, float[] src, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count <= dst.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                    Thunk.AddScaleU(a, psrc, pdst, count);
            }
        }

        public static void AddScale(float a, float[] src, float[] dst, int dstOffset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(count <= src.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 <= dstOffset && dstOffset < dst.Length);
            Contracts.Assert(0 < count && count <= dst.Length - dstOffset);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[dstOffset])
                    Thunk.AddScaleU(a, psrc, pdst, count);
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

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (int* pi = &indices[0])
                fixed (float* pdst = &dst[0])
                    Thunk.AddScaleSU(a, psrc, pi, pdst, count);
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

            unsafe
            {
                fixed (float* pdst = &dst[0])
                fixed (float* psrc = &src[0])
                fixed (float* pres = &res[0])
                    Thunk.AddScaleCopyU(a, psrc, pdst, pres, count);
            }
        }

        public static void AddScale(float a, float[] src, int[] indices, float[] dst,
            int dstOffset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count <= indices.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 <= dstOffset && dstOffset < dst.Length);
            Contracts.Assert(count < dst.Length - dstOffset);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (int* pi = &indices[0])
                fixed (float* pdst = &dst[dstOffset])
                    Thunk.AddScaleSU(a, psrc, pi, pdst, count);
            }
        }

        public static void Add(float[] src, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count <= dst.Length);

            unsafe
            {
                fixed (float* ps = &src[0])
                fixed (float* pd = &dst[0])
                    Thunk.AddU(ps, pd, count);
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

            unsafe
            {
                fixed (float* ps = &src[0])
                fixed (int* pi = &indices[0])
                fixed (float* pd = &dst[0])
                    Thunk.AddSU(ps, pi, pd, count);
            }
        }

        public static void MulElementWise(float[] src1, float[] src2, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src1);
            Contracts.Assert(0 < count && count <= src1.Length);
            Contracts.AssertNonEmpty(src2);
            Contracts.Assert(0 < count && count <= src2.Length);
            Contracts.AssertNonEmpty(dst);
            unsafe
            {
                fixed (float* ps1 = &src1[0])
                fixed (float* ps2 = &src2[0])
                fixed (float* pd = &dst[0])
                    Thunk.MulElementWiseU(ps1, ps2, pd, count);
            }
        }

        public static float Sum(float[] src, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                    return Thunk.SumU(psrc, count);
            }
        }

        public static float Sum(float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= src.Length - count);

            unsafe
            {
                fixed (float* psrc = &src[offset])
                    return Thunk.SumU(psrc, count);
            }
        }

        public static float SumSq(float[] src, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                    return Thunk.SumSqU(psrc, count);
            }
        }

        public static float SumSq(float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= src.Length - count);

            unsafe
            {
                fixed (float* psrc = &src[offset])
                    return Thunk.SumSqU(psrc, count);
            }
        }

        public static float SumSq(float mean, float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= src.Length - count);

            unsafe
            {
                fixed (float* psrc = &src[offset])
                    return (mean == 0 ? Thunk.SumSqU(psrc, count) : Thunk.SumSqDiffU(mean, psrc, count));
            }
        }

        public static float SumAbs(float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= src.Length - count);

            unsafe
            {
                fixed (float* psrc = &src[offset])
                    return Thunk.SumAbsU(psrc, count);
            }
        }

        public static float SumAbs(float mean, float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= src.Length - count);

            unsafe
            {
                fixed (float* psrc = &src[offset])
                    return (mean == 0 ? Thunk.SumAbsU(psrc, count) : Thunk.SumAbsDiffU(mean, psrc, count));
            }
        }

        public static float MaxAbsDiff(float mean, float[] src, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                    return Thunk.MaxAbsDiffU(mean, psrc, count);
            }
        }

        public static float MaxAbs(float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= src.Length - count);

            unsafe
            {
                fixed (float* psrc = &src[offset])
                    return Thunk.MaxAbsU(psrc, count);
            }
        }

        public static float DotProductDense(float[] a, float[] b, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(0 < count);
            Contracts.Assert(a.Length >= count);
            Contracts.Assert(b.Length >= count);

            unsafe
            {
                fixed (float* pa = &a[0])
                fixed (float* pb = &b[0])
                    return Thunk.DotU(pa, pb, count);
            }
        }

        public static float DotProductDense(float[] a, int offset, float[] b, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= a.Length - count);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(b.Length >= count);

            unsafe
            {
                fixed (float* pa = &a[offset])
                fixed (float* pb = &b[0])
                    return Thunk.DotU(pa, pb, count);
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

            unsafe
            {
                fixed (float* pa = &a[0])
                fixed (float* pb = &b[0])
                fixed (int* pi = &indices[0])
                    return Thunk.DotSU(pa, pb, pi, count);
            }
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

            unsafe
            {
                fixed (float* pa = &a[offset])
                fixed (float* pb = &b[0])
                fixed (int* pi = &indices[0])
                    return Thunk.DotSU(pa, pb, pi, count);
            }
        }

        public static float L2DistSquared(float[] a, float[] b, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(0 < count && count <= a.Length);
            Contracts.Assert(count <= b.Length);

            unsafe
            {
                fixed (float* pa = &a[0])
                fixed (float* pb = &b[0])
                    return Thunk.Dist2(pa, pb, count);
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

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (float* pd1 = &v[0])
                fixed (float* pd2 = &w[0])
                    Thunk.SdcaL1UpdateU(primalUpdate, psrc, threshold, pd1, pd2, length);
            }
        }

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

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (int* pi = &indices[0])
                fixed (float* pd1 = &v[0])
                fixed (float* pd2 = &w[0])
                    Thunk.SdcaL1UpdateSU(primalUpdate, psrc, pi, threshold, pd1, pd2, count);
            }
        }
    }
}