// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    /// <summary>
    /// Keep Sse.cs in sync with Avx.cs. When making changes to one, use BeyondCompare or a similar tool
    /// to view diffs and propagate appropriate changes to the other.
    /// </summary>
    public static class SseUtils
    {
        public const int CbAlign = 16;

        private static bool Compat(AlignedArray a)
        {
            Contracts.AssertValue(a);
            Contracts.Assert(a.Size > 0);
            return a.CbAlign == CbAlign;
        }

        private static unsafe float* Ptr(AlignedArray a, float* p)
        {
            Contracts.AssertValue(a);
            float* q = p + a.GetBase((long)p);
            Contracts.Assert(((long)q & (CbAlign - 1)) == 0);
            return q;
        }

        public static void MatTimesSrc(bool tran, AlignedArray mat, AlignedArray src, AlignedArray dst, int crun)
        {
            Contracts.Assert(Compat(mat));
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(mat.Size == dst.Size * src.Size);

            unsafe
            {
                fixed (float* pmat = &mat.Items[0])
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                {
                    if (!tran)
                    {
                        Contracts.Assert(0 <= crun && crun <= dst.Size);
                        Thunk.MatMul(Ptr(mat, pmat), Ptr(src, psrc), Ptr(dst, pdst), crun, src.Size);
                    }
                    else
                    {
                        Contracts.Assert(0 <= crun && crun <= src.Size);
                        Thunk.MatMulTran(Ptr(mat, pmat), Ptr(src, psrc), Ptr(dst, pdst), dst.Size, crun);
                    }
                }
            }
        }

        public static void MatTimesSrc(AlignedArray mat, int[] rgposSrc, AlignedArray srcValues,
            int posMin, int iposMin, int iposLim, AlignedArray dst, int crun)
        {
            Contracts.Assert(Compat(mat));
            Contracts.Assert(Compat(srcValues));
            Contracts.Assert(Compat(dst));
            Contracts.AssertValue(rgposSrc);
            Contracts.Assert(0 <= iposMin && iposMin <= iposLim && iposLim <= rgposSrc.Length);
            Contracts.Assert(mat.Size == dst.Size * srcValues.Size);

            if (iposMin >= iposLim)
            {
                dst.ZeroItems();
                return;
            }
            Contracts.AssertNonEmpty(rgposSrc);
            unsafe
            {
                fixed (float* pdst = &dst.Items[0])
                fixed (float* pmat = &mat.Items[0])
                fixed (float* psrc = &srcValues.Items[0])
                fixed (int* ppossrc = &rgposSrc[0])
                {
                    Contracts.Assert(0 <= crun && crun <= dst.Size);
                    Thunk.MatMulP(Ptr(mat, pmat), ppossrc, Ptr(srcValues, psrc), posMin, iposMin, iposLim, Ptr(dst, pdst), crun, srcValues.Size);
                }
            }
        }

        // dst += a
        public static void Add(float a, Span<float> dst)
        {
            Contracts.AssertNonEmpty(dst);

            unsafe
            {
                fixed (float* pdst = &MemoryMarshal.GetReference(dst))
                    Thunk.AddScalarU(a, pdst, dst.Length);
            }
        }

        public static void Scale(float a, Span<float> dst)
        {
            Contracts.AssertNonEmpty(dst);

            unsafe
            {
                fixed (float* pd = &MemoryMarshal.GetReference(dst))
                    Thunk.Scale(a, pd, dst.Length);
            }
        }

        // dst = a * src
        public static void Scale(float a, ReadOnlySpan<float> src, Span<float> dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count <= dst.Length);

            unsafe
            {
                fixed (float* psrc = &MemoryMarshal.GetReference(src))
                fixed (float* pdst = &MemoryMarshal.GetReference(dst))
                {
                    Thunk.ScaleSrcU(a, psrc, pdst, count);
                }
            }
        }

        // dst[i] = a * (dst[i] + b)
        public static void ScaleAdd(float a, float b, Span<float> dst)
        {
            Contracts.AssertNonEmpty(dst);

            unsafe
            {
                fixed (float* pdst = &MemoryMarshal.GetReference(dst))
                    Thunk.ScaleAddU(a, b, pdst, dst.Length);
            }
        }

        public static void AddScale(float a, ReadOnlySpan<float> src, Span<float> dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count <= dst.Length);

            unsafe
            {
                fixed (float* psrc = &MemoryMarshal.GetReference(src))
                fixed (float* pdst = &MemoryMarshal.GetReference(dst))
                    Thunk.AddScaleU(a, psrc, pdst, count);
            }
        }

        public static void AddScale(float a, ReadOnlySpan<float> src, ReadOnlySpan<int> indices, Span<float> dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count <= indices.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count < dst.Length);

            unsafe
            {
                fixed (float* psrc = &MemoryMarshal.GetReference(src))
                fixed (int* pi = &MemoryMarshal.GetReference(indices))
                fixed (float* pdst = &MemoryMarshal.GetReference(dst))
                    Thunk.AddScaleSU(a, psrc, pi, pdst, count);
            }
        }

        public static void AddScaleCopy(float a, ReadOnlySpan<float> src, ReadOnlySpan<float> dst, Span<float> res, int count)
        {
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 < count && count <= dst.Length);
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(count <= src.Length);
            Contracts.AssertNonEmpty(res);
            Contracts.Assert(count <= res.Length);

            unsafe
            {
                fixed (float* pdst = &MemoryMarshal.GetReference(dst))
                fixed (float* psrc = &MemoryMarshal.GetReference(src))
                fixed (float* pres = &MemoryMarshal.GetReference(res))
                    Thunk.AddScaleCopyU(a, psrc, pdst, pres, count);
            }
        }

        public static void Add(ReadOnlySpan<float> src, Span<float> dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count <= dst.Length);

            unsafe
            {
                fixed (float* ps = &MemoryMarshal.GetReference(src))
                fixed (float* pd = &MemoryMarshal.GetReference(dst))
                    Thunk.AddU(ps, pd, count);
            }
        }

        public static void Add(ReadOnlySpan<float> src, ReadOnlySpan<int> indices, Span<float> dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count <= indices.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count < dst.Length);

            unsafe
            {
                fixed (float* ps = &MemoryMarshal.GetReference(src))
                fixed (int* pi = &MemoryMarshal.GetReference(indices))
                fixed (float* pd = &MemoryMarshal.GetReference(dst))
                    Thunk.AddSU(ps, pi, pd, count);
            }
        }

        public static void MulElementWise(ReadOnlySpan<float> src1, ReadOnlySpan<float> src2, Span<float> dst, int count)
        {
            Contracts.AssertNonEmpty(src1);
            Contracts.Assert(0 < count && count <= src1.Length);
            Contracts.AssertNonEmpty(src2);
            Contracts.Assert(0 < count && count <= src2.Length);
            Contracts.AssertNonEmpty(dst);
            unsafe
            {
                fixed (float* ps1 = &MemoryMarshal.GetReference(src1))
                fixed (float* ps2 = &MemoryMarshal.GetReference(src2))
                fixed (float* pd = &MemoryMarshal.GetReference(dst))
                    Thunk.MulElementWiseU(ps1, ps2, pd, count);
            }
        }

        public static float Sum(ReadOnlySpan<float> src)
        {
            Contracts.AssertNonEmpty(src);

            unsafe
            {
                fixed (float* psrc = &MemoryMarshal.GetReference(src))
                    return Thunk.SumU(psrc, src.Length);
            }
        }

        public static float SumSq(ReadOnlySpan<float> src)
        {
            Contracts.AssertNonEmpty(src);

            unsafe
            {
                fixed (float* psrc = &MemoryMarshal.GetReference(src))
                    return Thunk.SumSqU(psrc, src.Length);
            }
        }

        public static float SumSq(float mean, ReadOnlySpan<float> src)
        {
            Contracts.AssertNonEmpty(src);

            unsafe
            {
                fixed (float* psrc = &MemoryMarshal.GetReference(src))
                    return (mean == 0 ? Thunk.SumSqU(psrc, src.Length) : Thunk.SumSqDiffU(mean, psrc, src.Length));
            }
        }

        public static float SumAbs(ReadOnlySpan<float> src)
        {
            Contracts.AssertNonEmpty(src);

            unsafe
            {
                fixed (float* psrc = &MemoryMarshal.GetReference(src))
                    return Thunk.SumAbsU(psrc, src.Length);
            }
        }

        public static float SumAbs(float mean, ReadOnlySpan<float> src)
        {
            Contracts.AssertNonEmpty(src);

            unsafe
            {
                fixed (float* psrc = &MemoryMarshal.GetReference(src))
                    return (mean == 0 ? Thunk.SumAbsU(psrc, src.Length) : Thunk.SumAbsDiffU(mean, psrc, src.Length));
            }
        }

        public static float MaxAbs(ReadOnlySpan<float> src)
        {
            Contracts.AssertNonEmpty(src);

            unsafe
            {
                fixed (float* psrc = &MemoryMarshal.GetReference(src))
                    return Thunk.MaxAbsU(psrc, src.Length);
            }
        }

        public static float MaxAbsDiff(float mean, ReadOnlySpan<float> src)
        {
            Contracts.AssertNonEmpty(src);

            unsafe
            {
                fixed (float* psrc = &MemoryMarshal.GetReference(src))
                    return Thunk.MaxAbsDiffU(mean, psrc, src.Length);
            }
        }

        public static float DotProductDense(ReadOnlySpan<float> a, ReadOnlySpan<float> b, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(0 < count);
            Contracts.Assert(a.Length >= count);
            Contracts.Assert(b.Length >= count);

            unsafe
            {
                fixed (float* pa = &MemoryMarshal.GetReference(a))
                fixed (float* pb = &MemoryMarshal.GetReference(b))
                    return Thunk.DotU(pa, pb, count);
            }
        }

        public static float DotProductSparse(ReadOnlySpan<float> a, ReadOnlySpan<float> b, ReadOnlySpan<int> indices, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(0 < count);
            Contracts.Assert(count < a.Length);
            Contracts.Assert(count <= b.Length);
            Contracts.Assert(count <= indices.Length);

            unsafe
            {
                fixed (float* pa = &MemoryMarshal.GetReference(a))
                fixed (float* pb = &MemoryMarshal.GetReference(b))
                fixed (int* pi = &MemoryMarshal.GetReference(indices))
                    return Thunk.DotSU(pa, pb, pi, count);
            }
        }

        public static float L2DistSquared(ReadOnlySpan<float> a, ReadOnlySpan<float> b, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(0 < count && count <= a.Length);
            Contracts.Assert(count <= b.Length);

            unsafe
            {
                fixed (float* pa = &MemoryMarshal.GetReference(a))
                fixed (float* pb = &MemoryMarshal.GetReference(b))
                    return Thunk.Dist2(pa, pb, count);
            }
        }

        public static void ZeroMatrixItems(AlignedArray dst, int ccol, int cfltRow, int[] indices)
        {
            Contracts.Assert(0 < ccol && ccol <= cfltRow);

            unsafe
            {
                fixed (float* pdst = &dst.Items[0])
                fixed (int* pi = &indices[0])
                {
                    if (ccol == cfltRow)
                        Thunk.ZeroItemsU(Ptr(dst, pdst), dst.Size, pi, indices.Length);
                    else
                        Thunk.ZeroMatrixItemsCore(Ptr(dst, pdst), dst.Size, ccol, cfltRow, pi, indices.Length);
                }
            }
        }

        public static void SdcaL1UpdateDense(float primalUpdate, int count, ReadOnlySpan<float> src, float threshold, Span<float> v, Span<float> w)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(count <= src.Length);
            Contracts.AssertNonEmpty(v);
            Contracts.Assert(count <= v.Length);
            Contracts.AssertNonEmpty(w);
            Contracts.Assert(count <= w.Length);
            Contracts.Assert(count > 0);

            unsafe
            {
                fixed (float* psrc = &MemoryMarshal.GetReference(src))
                fixed (float* pd1 = &MemoryMarshal.GetReference(v))
                fixed (float* pd2 = &MemoryMarshal.GetReference(w))
                    Thunk.SdcaL1UpdateU(primalUpdate, psrc, threshold, pd1, pd2, count);
            }
        }

        public static void SdcaL1UpdateSparse(float primalUpdate, int count, ReadOnlySpan<float> src, ReadOnlySpan<int> indices, float threshold, Span<float> v, Span<float> w)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(count <= src.Length);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count <= indices.Length);
            Contracts.AssertNonEmpty(v);
            Contracts.Assert(count <= v.Length);
            Contracts.AssertNonEmpty(w);
            Contracts.Assert(count <= w.Length);
            Contracts.Assert(count > 0);

            unsafe
            {
                fixed (float* psrc = &MemoryMarshal.GetReference(src))
                fixed (int* pi = &MemoryMarshal.GetReference(indices))
                fixed (float* pd1 = &MemoryMarshal.GetReference(v))
                fixed (float* pd2 = &MemoryMarshal.GetReference(w))
                    Thunk.SdcaL1UpdateSU(primalUpdate, psrc, pi, threshold, pd1, pd2, count);
            }
        }
    }
}