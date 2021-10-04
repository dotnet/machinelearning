// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Microsoft.ML.Internal.CpuMath.Core;

namespace Microsoft.ML.Internal.CpuMath
{
    [BestFriend]
    internal static partial class CpuMathUtils
    {
        // The count of bytes in Vector128<T>, corresponding to _cbAlign in AlignedArray
        private const int Vector128Alignment = 16;

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static int GetVectorAlignment()
            => Vector128Alignment;

        /// <summary>
        /// Check if <paramref name="a"/>'s alignment is suitable to SSE instructions. Returns <see langword="true"/>
        /// if <paramref name="a"/>'s alignment is ok and <see langword="false"/> otherwise.
        /// </summary>
        /// <param name="a">The vector being checked.</param>
        /// <returns>Whether <paramref name="a"/> is aligned well.</returns>
        private static bool Compat(AlignedArray a)
        {
            Contracts.AssertValue(a);
            Contracts.Assert(a.Size > 0);
            return a.CbAlign % Vector128Alignment == 0;
        }

        private static unsafe float* Ptr(AlignedArray a, float* p)
        {
            Contracts.AssertValue(a);
            float* q = p + a.GetBase((long)p);
            Contracts.Assert(((long)q & (Vector128Alignment - 1)) == 0);
            return q;
        }

        /// <summary>
        /// Compute the product of matrix <paramref name="mat"/> (the matrix is flattened because its type is <see cref="AlignedArray"/> instead of a matrix)
        /// and a vector <paramref name="src"/>.
        /// </summary>
        /// <param name="tran">Whether to transpose <paramref name="mat"/> before doing any computation.</param>
        /// <param name="mat">If <paramref name="tran"/> is <see langword="false"/>, <paramref name="mat"/> is a m-by-n matrix, and the value at the i-th row and the j-th column is indexed by i * n + j in <paramref name="mat"/>.
        /// If <paramref name="tran"/> is <see langword="true"/>, <paramref name="mat"/> would be viewed a n-by-m matrix, and the value at the i-th row and the j-th column in the transposed matrix is indexed by j * m + i in the
        /// original <paramref name="mat"/>.</param>
        /// <param name="src">A n-by-1 matrix, which is also a vector.</param>
        /// <param name="dst">A m-by-1 matrix, which is also a vector.</param>
        /// <param name="crun">The truncation level of <paramref name="dst"/>. For example, if <paramref name="crun"/> is 2, <paramref name="dst"/>
        /// will be considered as a 2-by-1 matrix and therefore elements after its 2nd element will be ignored. If no truncation should happen,
        /// set <paramref name="crun"/> to the length of <paramref name="dst"/>.</param>
        public static void MatrixTimesSource(bool tran, AlignedArray mat, AlignedArray src, AlignedArray dst, int crun)
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

        public static void MatrixTimesSource(AlignedArray mat, ReadOnlySpan<int> rgposSrc, AlignedArray srcValues,
            int posMin, int iposMin, int iposLim, AlignedArray dst, int crun)
        {
            Contracts.Assert(Compat(mat));
            Contracts.Assert(Compat(srcValues));
            Contracts.Assert(Compat(dst));
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
                    return Thunk.Sum(psrc, src.Length);
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

        public static void SdcaL1UpdateSparse(float primalUpdate, int count, ReadOnlySpan<float> source, ReadOnlySpan<int> indices, float threshold, Span<float> v, Span<float> w)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.Assert(count <= source.Length);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count <= indices.Length);
            Contracts.AssertNonEmpty(v);
            Contracts.Assert(count <= v.Length);
            Contracts.AssertNonEmpty(w);
            Contracts.Assert(count <= w.Length);
            Contracts.Assert(count > 0);

            unsafe
            {
                fixed (float* psrc = &MemoryMarshal.GetReference(source))
                fixed (int* pi = &MemoryMarshal.GetReference(indices))
                fixed (float* pd1 = &MemoryMarshal.GetReference(v))
                fixed (float* pd2 = &MemoryMarshal.GetReference(w))
                    Thunk.SdcaL1UpdateSU(primalUpdate, psrc, pi, threshold, pd1, pd2, count);
            }
        }
    }
}
