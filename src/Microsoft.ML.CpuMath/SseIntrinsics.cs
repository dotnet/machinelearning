// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// The exported function names need to be unique (can't be disambiguated based on signature), hence
// we introduce suffix letters to indicate the general patterns used.
// * U suffix means unaligned and unpadded.
// * S suffix means sparse (unaligned) vector.

using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    internal static class SseIntrinsics
    {
        private const int CbAlign = 16;

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

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector128<float> Load1(float* src, int* idx)
        {
            return Sse.SetScalarVector128(src[idx[0]]);
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector128<float> Load4(float* src, int* idx)
        {
            return Sse.SetVector128(src[idx[3]], src[idx[2]], src[idx[1]], src[idx[0]]);
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> Rotate(Vector128<float> x)
        {
            // The control byte shuffles the four 32-bit floats of x: ABCD -> BCDA.
            return Sse.Shuffle(x, x, 0x39);
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> RotateReverse(Vector128<float> x)
        {
            // The control byte shuffles the four 32-bit floats of x: ABCD -> DABC.
            return Sse.Shuffle(x, x, 0x93);
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static unsafe void Store4(Vector128<float> x, float* dst, int* idx)
        {
            Sse.StoreScalar(dst + idx[0], x);
            x = Rotate(x);
            Sse.StoreScalar(dst + idx[1], x);
            x = Rotate(x);
            Sse.StoreScalar(dst + idx[2], x);
            x = Rotate(x);
            Sse.StoreScalar(dst + idx[3], x);
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> VectorSum(in Vector128<float> vector)
        {
            if (Sse3.IsSupported)
            {
                Vector128<float> tmp = Sse3.HorizontalAdd(vector, vector);
                return Sse3.HorizontalAdd(tmp, tmp);
            }
            else
            {
                // SSE3 is not supported.
                Vector128<float> tmp = Sse.Add(vector, Sse.MoveHighToLow(vector, vector));
                // The control byte shuffles the four 32-bit floats of tmp: ABCD -> BADC.
                return Sse.Add(tmp, Sse.Shuffle(tmp, tmp, 0xb1));
            }
        }

        // Multiply matrix times vector into vector.
        internal static unsafe void MatMulA(bool add, AlignedArray mat, AlignedArray src, AlignedArray dst, int crow, int ccol)
        {
            Contracts.Assert(Compat(mat));
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));

            fixed (float* pSrcStart = &src.Items[0])
            fixed (float* pDstStart = &dst.Items[0])
            fixed (float* pMatStart = &mat.Items[0])
            {
                float* psrc = Ptr(src, pSrcStart);
                float* pdst = Ptr(dst, pDstStart);
                float* pmat = Ptr(mat, pMatStart);

                float* pSrcEnd = psrc + ccol;
                float* pDstEnd = pdst + crow;
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                float* pMatCurrent = pmat;

                while (pDstCurrent < pDstEnd)
                {
                    Vector128<float> res0 = Sse.SetZeroVector128();
                    Vector128<float> res1 = res0;
                    Vector128<float> res2 = res0;
                    Vector128<float> res3 = res0;

                    while (pSrcCurrent < pSrcEnd)
                    {
                        float* pMatTemp = pMatCurrent;

                        Vector128<float> x01 = Sse.LoadAlignedVector128(pMatTemp);
                        Vector128<float> x11 = Sse.LoadAlignedVector128(pMatTemp += ccol);
                        Vector128<float> x21 = Sse.LoadAlignedVector128(pMatTemp += ccol);
                        Vector128<float> x31 = Sse.LoadAlignedVector128(pMatTemp += ccol);
                        Vector128<float> x02 = Sse.LoadAlignedVector128(pSrcCurrent);

                        res0 = Sse.Add(res0, Sse.Multiply(x01, x02));
                        res1 = Sse.Add(res1, Sse.Multiply(x11, x02));
                        res2 = Sse.Add(res2, Sse.Multiply(x21, x02));
                        res3 = Sse.Add(res3, Sse.Multiply(x31, x02));

                        pSrcCurrent += 4;
                        pMatCurrent += 4;
                    }

                    // Add up the entries of each, with the 4 results in res0
                    res0 = Sse3.HorizontalAdd(res0, res1);
                    res2 = Sse3.HorizontalAdd(res2, res3);
                    res0 = Sse3.HorizontalAdd(res0, res2);

                    if (add)
                    {
                        res0 = Sse.Add(res0, Sse.LoadAlignedVector128(pDstCurrent));
                    }
                    Sse.StoreAligned(pDstCurrent, res0);

                    pDstCurrent += 4;
                    pMatCurrent += 3 * ccol;
                }
            }
        }

        // Partial sparse source vector.
        internal static unsafe void MatMulPA(bool add, AlignedArray mat, int[] rgposSrc, AlignedArray src,
                                        int posMin, int iposMin, int iposEnd, AlignedArray dst, int crow, int ccol)
        {
            Contracts.Assert(Compat(mat));
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));

            // REVIEW: For extremely sparse inputs, interchanging the loops would
            // likely be more efficient.
            fixed (float* pSrcStart = &src.Items[0])
            fixed (float* pDstStart = &dst.Items[0])
            fixed (float* pMatStart = &mat.Items[0])
            fixed (int* pposSrc = &rgposSrc[0])
            {
                float* psrc = Ptr(src, pSrcStart);
                float* pdst = Ptr(dst, pDstStart);
                float* pmat = Ptr(mat, pMatStart);

                int* pposMin = pposSrc + iposMin;
                int* pposEnd = pposSrc + iposEnd;
                float* pDstEnd = pdst + crow;
                float* pm0 = pmat - posMin;
                float* pSrcCurrent = psrc - posMin;
                float* pDstCurrent = pdst;

                while (pDstCurrent < pDstEnd)
                {
                    float* pm1 = pm0 + ccol;
                    float* pm2 = pm1 + ccol;
                    float* pm3 = pm2 + ccol;
                    Vector128<float> res = Sse.SetZeroVector128();

                    int* ppos = pposMin;

                    while (ppos < pposEnd)
                    {
                        int col = *ppos;
                        Vector128<float> x1 = Sse.SetVector128(pm3[col], pm2[col], pm1[col], pm0[col]);
                        Vector128<float> x2 = Sse.SetAllVector128(pSrcCurrent[col]);
                        x2 = Sse.Multiply(x2, x1);
                        res = Sse.Add(res, x2);

                        ppos++;
                    }

                    if (add)
                    {
                        res = Sse.Add(res, Sse.LoadAlignedVector128(pDstCurrent));
                    }
                    Sse.StoreAligned(pDstCurrent, res);

                    pDstCurrent += 4;
                    pm0 += 4 * ccol;
                }
            }
        }

        internal static unsafe void MatMulTranA(bool add, AlignedArray mat, AlignedArray src, AlignedArray dst, int crow, int ccol)
        {
            Contracts.Assert(Compat(mat));
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));

            fixed (float* pSrcStart = &src.Items[0])
            fixed (float* pDstStart = &dst.Items[0])
            fixed (float* pMatStart = &mat.Items[0])
            {
                float* psrc = Ptr(src, pSrcStart);
                float* pdst = Ptr(dst, pDstStart);
                float* pmat = Ptr(mat, pMatStart);

                float* pSrcEnd = psrc + ccol;
                float* pDstEnd = pdst + crow;
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                float* pMatCurrent = pmat;

                if (!add)
                {
                    Vector128<float> x01 = Sse.LoadAlignedVector128(pSrcCurrent);
                    // Replicate each slot of x01 into its own register.
                    Vector128<float> x11 = Sse.Shuffle(x01, x01, 0x55);
                    Vector128<float> x21 = Sse.Shuffle(x01, x01, 0xAA);
                    Vector128<float> x31 = Sse.Shuffle(x01, x01, 0xFF);
                    x01 = Sse.Shuffle(x01, x01, 0x00);

                    pSrcCurrent += 4;

                    while (pDstCurrent < pDstEnd)
                    {
                        float* pMatTemp = pMatCurrent;
                        Vector128<float> x02 = Sse.LoadAlignedVector128(pMatTemp);
                        Vector128<float> x12 = Sse.LoadAlignedVector128(pMatTemp += crow);
                        Vector128<float> x22 = Sse.LoadAlignedVector128(pMatTemp += crow);
                        Vector128<float> x32 = Sse.LoadAlignedVector128(pMatTemp += crow);

                        x02 = Sse.Multiply(x01, x02);
                        x12 = Sse.Multiply(x11, x12);
                        x22 = Sse.Multiply(x21, x22);
                        x32 = Sse.Multiply(x31, x32);

                        x02 = Sse.Add(x02, x12);
                        x22 = Sse.Add(x22, x32);
                        x02 = Sse.Add(x02, x22);

                        Sse.StoreAligned(pDstCurrent, x02);

                        pDstCurrent += 4;
                        pMatCurrent += 4;
                    }

                    pMatCurrent += 3 * crow;
                }

                while (pSrcCurrent < pSrcEnd)
                {
                    Vector128<float> x01 = Sse.LoadAlignedVector128(pSrcCurrent);
                    // Replicate each slot of x01 into its own register.
                    Vector128<float> x11 = Sse.Shuffle(x01, x01, 0x55);
                    Vector128<float> x21 = Sse.Shuffle(x01, x01, 0xAA);
                    Vector128<float> x31 = Sse.Shuffle(x01, x01, 0xFF);
                    x01 = Sse.Shuffle(x01, x01, 0x00);

                    pDstCurrent = pdst;

                    while (pDstCurrent < pDstEnd)
                    {
                        float* pMatTemp = pMatCurrent;

                        Vector128<float> x02 = Sse.LoadAlignedVector128(pMatTemp);
                        Vector128<float> x12 = Sse.LoadAlignedVector128(pMatTemp += crow);
                        Vector128<float> x22 = Sse.LoadAlignedVector128(pMatTemp += crow);
                        Vector128<float> x32 = Sse.LoadAlignedVector128(pMatTemp += crow);
                        Vector128<float> x3 = Sse.LoadAlignedVector128(pDstCurrent);

                        x02 = Sse.Multiply(x01, x02);
                        x12 = Sse.Multiply(x11, x12);
                        x22 = Sse.Multiply(x21, x22);
                        x32 = Sse.Multiply(x31, x32);

                        x02 = Sse.Add(x02, x12);
                        x22 = Sse.Add(x22, x32);
                        x02 = Sse.Add(x02, x22);
                        x3 = Sse.Add(x02, x3);

                        Sse.StoreAligned(pDstCurrent, x3);

                        pDstCurrent += 4;
                        pMatCurrent += 4;
                    }

                    pMatCurrent += 3 * crow;
                    pSrcCurrent += 4;
                }
            }
        }

        // Partial sparse source vector.
        internal static unsafe void MatMulTranPA(bool add, AlignedArray mat, int[] rgposSrc, AlignedArray src,
                                        int posMin, int iposMin, int iposEnd, AlignedArray dst, int crow)
        {
            Contracts.Assert(Compat(mat));
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));

            fixed (float* pSrcStart = &src.Items[0])
            fixed (float* pDstStart = &dst.Items[0])
            fixed (float* pMatStart = &mat.Items[0])
            fixed (int* pposSrc = &rgposSrc[0])
            {
                float* psrc = Ptr(src, pSrcStart);
                float* pdst = Ptr(dst, pDstStart);
                float* pmat = Ptr(mat, pMatStart);

                int* ppos = pposSrc + iposMin;
                int* pposEnd = pposSrc + iposEnd;
                float* pDstEnd = pdst + crow;

                if (!add)
                {
                    int col = *ppos - posMin;
                    ppos++;

                    Vector128<float> x0 = Sse.SetAllVector128(psrc[col]);
                    float* pDstCurrent = pdst;
                    float* pMatCurrent = pmat + col * crow;

                    while (pDstCurrent < pDstEnd)
                    {
                        Vector128<float> x1 = Sse.LoadAlignedVector128(pMatCurrent);
                        x1 = Sse.Multiply(x1, x0);
                        Sse.StoreAligned(pDstCurrent, x1);

                        pDstCurrent += 4;
                        pMatCurrent += 4;
                    }
                }

                // REVIEW: Should we explore unrolling the outer loop?
                while (ppos < pposEnd)
                {
                    int col = *ppos - posMin;
                    ppos++;

                    Vector128<float> x0 = Sse.SetAllVector128(psrc[col]);
                    float* pDstCurrent = pdst;
                    float* pMatCurrent = pmat + col * crow;

                    while (pDstCurrent < pDstEnd)
                    {
                        Vector128<float> x1 = Sse.LoadAlignedVector128(pMatCurrent);
                        Vector128<float> x2 = Sse.LoadAlignedVector128(pDstCurrent);
                        x1 = Sse.Multiply(x1, x0);
                        x2 = Sse.Add(x2, x1);
                        Sse.StoreAligned(pDstCurrent, x2);

                        pDstCurrent += 4;
                        pMatCurrent += 4;
                    }

                    ppos++;
                }
            }
        }

        // Sparse matrix.
        internal static unsafe void MatMulRU(bool add, int[] starts, int[] indices, float[] coefs,
                                                AlignedArray src, AlignedArray dst, int crow)
        {
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));

            fixed (int* pstarts = &starts[0])
            fixed (int* pindices = &indices[0])
            fixed (float* pcoefs = &coefs[0])
            fixed (float* pSrcStart = &src.Items[0])
            fixed (float* pDstStart = &dst.Items[0])
            {
                float* psrc = Ptr(src, pSrcStart);
                float* pdst = Ptr(dst, pDstStart);

                int* pii = pstarts + 1;
                int* pIdxCurrent = pindices;
                float* pMatCurrent = pcoefs;
                float* pDstEnd = pdst + crow;

                float* pDstCurrent = pdst;

                while (pDstCurrent < pDstEnd)
                {
                    int* pIdxEnd = pindices + *pii;
                    pii++;

                    Vector128<float> result = Sse.SetZeroVector128();

                    while (pIdxCurrent + 4 <= pIdxEnd)
                    {
                        Vector128<float> x = Sse.Multiply(Load4(psrc, pIdxCurrent), Sse.LoadVector128(pMatCurrent));
                        result = Sse.Add(result, x);

                        pIdxCurrent += 4;
                        pMatCurrent += 4;
                    }

                    while (pIdxCurrent < pIdxEnd)
                    {
                        Vector128<float> x = Sse.MultiplyScalar(Load1(psrc, pIdxCurrent), Sse.SetScalarVector128(*pMatCurrent));
                        result = Sse.AddScalar(result, x);

                        pIdxCurrent++;
                        pMatCurrent++;
                    }

                    result = VectorSum(in result);

                    if (add)
                    {
                        result = Sse.AddScalar(result, Sse.SetScalarVector128(*pDstCurrent));
                    }
                    Sse.StoreScalar(pDstCurrent, result);

                    pDstCurrent++;
                }
            }
        }

        // Unpadded convolution.
        internal static unsafe void MatMulCU(bool add, int[] mprowiv, int[] mprowcol,
            int[] runs, float[] coefs, AlignedArray src, AlignedArray dst, int crow)
        {
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));

            fixed (int* pmprowiv = &mprowiv[0])
            fixed (int* pmprowcol = &mprowcol[0])
            fixed (int* pruns = &runs[0])
            fixed (float* pcoefs = &coefs[0])
            fixed (float* pSrcStart = &src.Items[0])
            fixed (float* pDstStart = &dst.Items[0])
            {
                float* psrc = Ptr(src, pSrcStart);
                float* pdst = Ptr(dst, pDstStart);

                int size = pruns[1];
                int* psupport = pruns + 2;
                int* piv = pmprowiv;
                int* pcol = pmprowcol;
                int* pIdxEnd = psupport + size;
                float* pDstEnd = pdst + crow;

                float* pDstCurrent = pdst;

                while (pDstCurrent < pDstEnd)
                {
                    float* pMatCurrent = pcoefs + *piv;
                    piv++;
                    float* pSrcCurrent = psrc + *pcol;
                    pcol++;
                    int* pIdxCurrent = psupport;

                    Vector128<float> result = Sse.SetZeroVector128();

                    while (pIdxCurrent + 4 <= pIdxEnd)
                    {
                        Vector128<float> x = Sse.Multiply(Load4(pSrcCurrent, pIdxCurrent), Sse.LoadVector128(pMatCurrent));
                        result = Sse.Add(result, x);

                        pIdxCurrent += 4;
                        pMatCurrent += 4;
                    }

                    while (pIdxCurrent < pIdxEnd)
                    {
                        Vector128<float> x = Sse.MultiplyScalar(Load1(pSrcCurrent, pIdxCurrent), Sse.SetScalarVector128(*pMatCurrent));
                        result = Sse.AddScalar(result, x);

                        pIdxCurrent++;
                        pMatCurrent++;
                    }

                    result = VectorSum(result);

                    // Add the bias.
                    result = Sse.AddScalar(result, Sse.SetScalarVector128(*pMatCurrent));

                    if (add)
                    {
                        result = Sse.AddScalar(result, Sse.SetScalarVector128(*pDstCurrent));
                    }
                    Sse.StoreScalar(pDstCurrent, result);

                    pDstCurrent++;
                }
            }
        }

        // Padded convolution.
        internal static unsafe void MatMulDU(bool add, int[] mprowiv, int[] mprowcol, int[] mprowrun,
            int[] runs, float[] coefs, AlignedArray src, AlignedArray dst, int crow)
        {
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));

            fixed (int* pmprowiv = &mprowiv[0])
            fixed (int* pmprowcol = &mprowcol[0])
            fixed (int* pmprowrun = &mprowrun[0])
            fixed (int* pruns = &runs[0])
            fixed (float* pcoefs = &coefs[0])
            fixed (float* pSrcStart = &src.Items[0])
            fixed (float* pDstStart = &dst.Items[0])
            {
                float* psrc = Ptr(src, pSrcStart);
                float* pdst = Ptr(dst, pDstStart);

                int* piv = pmprowiv;
                int* pcol = pmprowcol;
                float* pDstEnd = pdst + crow;
                int kernelSize = pruns[1];

                int* pirun = pmprowrun;
                float* pDstCurrent = pdst;

                while (pDstCurrent < pDstEnd)
                {
                    float* pMatCurrent = pcoefs + *piv;
                    piv++;
                    float* pMatBias = pMatCurrent + kernelSize;
                    float* pSrcCurrent = psrc + *pcol;
                    pcol++;
                    int irun = *pirun;
                    pirun++;

                    int* pIdxCurrent = pruns + 2 + irun;
                    int* pIdxEnd = pIdxCurrent + pIdxCurrent[-1];

                    Vector128<float> result = Sse.SetZeroVector128();

                    if (irun == 0)
                    {
                        // No masking needed.
                        while (pIdxCurrent + 4 <= pIdxEnd)
                        {
                            Vector128<float> x = Sse.Multiply(Load4(pSrcCurrent, pIdxCurrent), Sse.LoadVector128(pMatCurrent));
                            result = Sse.Add(result, x);

                            pIdxCurrent += 4;
                            pMatCurrent += 4;
                        }

                        while (pIdxCurrent < pIdxEnd)
                        {
                            Vector128<float> x = Sse.MultiplyScalar(Load1(pSrcCurrent, pIdxCurrent), Sse.SetScalarVector128(*pMatCurrent));
                            result = Sse.AddScalar(result, x);

                            pIdxCurrent++;
                            pMatCurrent++;
                        }
                    }
                    else
                    {
                        // Need masking.
                        pMatCurrent += pIdxCurrent[-2];
                        // REVIEW NEEDED: Is it the correct translation from: "const float * pmask = reinterpret_cast<const float *>(piLim);"?
                        float* pmask = (float*)pIdxEnd;

                        while (pIdxCurrent + 4 <= pIdxEnd)
                        {
                            Vector128<float> x = Sse.Multiply(Load4(pSrcCurrent, pIdxCurrent), Sse.And(Sse.LoadVector128(pmask), Sse.LoadVector128(pMatCurrent)));
                            result = Sse.Add(result, x);

                            pIdxCurrent += 4;
                            pMatCurrent += 4;
                            pmask += 4;
                        }

                        while (pIdxCurrent < pIdxEnd)
                        {
                            Vector128<float> x = Sse.MultiplyScalar(Load1(pSrcCurrent, pIdxCurrent), Sse.And(Sse.SetScalarVector128(*pmask), Sse.SetScalarVector128(*pMatCurrent)));
                            result = Sse.AddScalar(result, x);

                            pIdxCurrent++;
                            pMatCurrent++;
                            pmask++;
                        }
                    }

                    result = VectorSum(result);

                    // Add the bias.
                    result = Sse.AddScalar(result, Sse.SetScalarVector128(*pMatBias));

                    if (add)
                    {
                        result = Sse.AddScalar(result, Sse.SetScalarVector128(*pDstCurrent));
                    }
                    Sse.StoreScalar(pDstCurrent, result);

                    pDstCurrent++;
                }
            }
        }

        internal static unsafe void ScaleU(float scale, Span<float> dst)
        {
            Vector128<float> scaleVector = Sse.SetAllVector128(scale);

            fixed (float* pdst = dst)
            {
                float* pDstCurrent = pdst;
                float* pEnd = pdst + dst.Length;

                while (pDstCurrent + 4 <= pEnd)
                {
                    Vector128<float> dstVector = Sse.LoadVector128(pDstCurrent);

                    dstVector = Sse.Multiply(scaleVector, dstVector);
                    Sse.Store(pDstCurrent, dstVector);

                    pDstCurrent += 4;
                }

                while (pDstCurrent < pEnd)
                {
                    Vector128<float> dstVector = Sse.LoadScalarVector128(pDstCurrent);

                    dstVector = Sse.MultiplyScalar(scaleVector, dstVector);
                    Sse.StoreScalar(pDstCurrent, dstVector);

                    pDstCurrent++;
                }
            }
        }

        internal static unsafe void AddScaleU(float scale, Span<float> src, Span<float> dst)
        {
            Vector128<float> scaleVector = Sse.SetAllVector128(scale);

            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                float* pEnd = pdst + dst.Length;

                while (pDstCurrent + 4 <= pEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    Vector128<float> dstVector = Sse.LoadVector128(pDstCurrent);

                    srcVector = Sse.Multiply(srcVector, scaleVector);
                    dstVector = Sse.Add(dstVector, srcVector);
                    Sse.Store(pDstCurrent, dstVector);

                    pDstCurrent += 4;
                    pSrcCurrent += 4;
                }

                while (pDstCurrent < pEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    Vector128<float> dstVector = Sse.LoadScalarVector128(pDstCurrent);

                    srcVector = Sse.MultiplyScalar(srcVector, scaleVector);
                    dstVector = Sse.AddScalar(dstVector, srcVector);
                    Sse.StoreScalar(pDstCurrent, dstVector);

                    pDstCurrent++;
                    pSrcCurrent++;
                }
            }
        }

        internal static unsafe void AddScaleSU(float scale, Span<float> src, Span<int> idx, Span<float> dst)
        {
            Vector128<float> scaleVector = Sse.SetAllVector128(scale);

            fixed (float* psrc = src)
            fixed (int* pidx = idx)
            fixed (float* pdst = dst)
            {
                float* pSrcCurrent = psrc;
                int* pIdxCurrent = pidx;
                float* pDstCurrent = pdst;
                int* pEnd = pidx + idx.Length;

                while (pIdxCurrent + 4 <= pEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    Vector128<float> dstVector = Load4(pDstCurrent, pIdxCurrent);

                    srcVector = Sse.Multiply(srcVector, scaleVector);
                    dstVector = Sse.Add(dstVector, srcVector);
                    Store4(dstVector, pDstCurrent, pIdxCurrent);

                    pIdxCurrent += 4;
                    pSrcCurrent += 4;
                }

                while (pIdxCurrent < pEnd)
                {
                    pDstCurrent[*pIdxCurrent] += scale * (*pSrcCurrent);

                    pIdxCurrent++;
                    pSrcCurrent++;
                }
            }
        }

        internal static unsafe void AddU(Span<float> src, Span<float> dst)
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                float* pEnd = psrc + src.Length;

                while (pSrcCurrent + 4 <= pEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    Vector128<float> dstVector = Sse.LoadVector128(pDstCurrent);

                    Vector128<float> result = Sse.Add(srcVector, dstVector);
                    Sse.Store(pDstCurrent, result);

                    pSrcCurrent += 4;
                    pDstCurrent += 4;
                }

                while (pSrcCurrent < pEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    Vector128<float> dstVector = Sse.LoadScalarVector128(pDstCurrent);

                    Vector128<float> result = Sse.AddScalar(srcVector, dstVector);
                    Sse.StoreScalar(pDstCurrent, result);

                    pSrcCurrent++;
                    pDstCurrent++;
                }
            }
        }

        internal static unsafe void AddSU(Span<float> src, Span<int> idx, Span<float> dst)
        {
            fixed (float* psrc = src)
            fixed (int* pidx = idx)
            fixed (float* pdst = dst)
            {
                float* pSrcCurrent = psrc;
                int* pIdxCurrent = pidx;
                float* pDstCurrent = pdst;
                int* pEnd = pidx + idx.Length;

                while (pIdxCurrent + 4 <= pEnd)
                {
                    Vector128<float> srcVector = Load4(pDstCurrent, pIdxCurrent);
                    Vector128<float> dstVector = Sse.LoadVector128(pSrcCurrent);

                    srcVector = Sse.Add(srcVector, dstVector);
                    Store4(srcVector, pDstCurrent, pIdxCurrent);

                    pIdxCurrent += 4;
                    pSrcCurrent += 4;
                }

                while (pIdxCurrent < pEnd)
                {
                    pDstCurrent[*pIdxCurrent] += *pSrcCurrent;

                    pIdxCurrent++;
                    pSrcCurrent++;
                }
            }
        }

        internal static unsafe void MulElementWiseU(Span<float> src1, Span<float> src2, Span<float> dst)
        {
            fixed (float* psrc1 = &src1[0])
            fixed (float* psrc2 = &src2[0])
            fixed (float* pdst = dst)
            {
                float* pSrc1Current = psrc1;
                float* pSrc2Current = psrc2;
                float* pDstCurrent = pdst;
                float* pEnd = pdst + dst.Length;

                while (pDstCurrent + 4 <= pEnd)
                {
                    Vector128<float> src1Vector = Sse.LoadVector128(pSrc1Current);
                    Vector128<float> src2Vector = Sse.LoadVector128(pSrc2Current);
                    src2Vector = Sse.Multiply(src1Vector, src2Vector);
                    Sse.Store(pDstCurrent, src2Vector);

                    pSrc1Current += 4;
                    pSrc2Current += 4;
                    pDstCurrent += 4;
                }

                while (pDstCurrent < pEnd)
                {
                    Vector128<float> src1Vector = Sse.LoadScalarVector128(pSrc1Current);
                    Vector128<float> src2Vector = Sse.LoadScalarVector128(pSrc2Current);
                    src2Vector = Sse.MultiplyScalar(src1Vector, src2Vector);
                    Sse.StoreScalar(pDstCurrent, src2Vector);

                    pSrc1Current++;
                    pSrc2Current++;
                    pDstCurrent++;
                }
            }
        }

        internal static unsafe float SumSqU(Span<float> src)
        {
            Vector128<float> result = Sse.SetZeroVector128();

            fixed (float* psrc = src)
            {
                float* pSrcCurrent = psrc;
                float* pEnd = psrc + src.Length;

                while (pSrcCurrent + 4 <= pEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    result = Sse.Add(result, Sse.Multiply(srcVector, srcVector));

                    pSrcCurrent += 4;
                }

                result = VectorSum(in result);

                while (pSrcCurrent < pEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    result = Sse.AddScalar(result, Sse.MultiplyScalar(srcVector, srcVector));

                    pSrcCurrent++;
                }
            }

            return Sse.ConvertToSingle(result);
        }

        internal static unsafe float SumAbsU(Span<float> src)
        {
            Vector128<float> result = Sse.SetZeroVector128();
            Vector128<float> mask;

            if (Sse2.IsSupported)
            {
                mask = Sse.StaticCast<int, float>(Sse2.SetAllVector128(0x7FFFFFFF));
            }
            else
            {
                mask = Sse.SetAllVector128(BitConverter.Int32BitsToSingle(0x7FFFFFFF));
            }

            fixed (float* psrc = src)
            {
                float* pSrcCurrent = psrc;
                float* pEnd = psrc + src.Length;

                while (pSrcCurrent + 4 <= pEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    result = Sse.Add(result, Sse.And(srcVector, mask));

                    pSrcCurrent += 4;
                }

                result = VectorSum(in result);

                while (pSrcCurrent < pEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    result = Sse.Add(result, Sse.And(srcVector, mask));

                    pSrcCurrent++;
                }
            }

            return Sse.ConvertToSingle(result);
        }

        internal static unsafe float DotU(Span<float> src, Span<float> dst)
        {
            Vector128<float> result = Sse.SetZeroVector128();

            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                float* pEnd = psrc + src.Length;

                while (pSrcCurrent + 4 <= pEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    Vector128<float> dstVector = Sse.LoadVector128(pDstCurrent);

                    result = Sse.Add(result, Sse.Multiply(srcVector, dstVector));

                    pSrcCurrent += 4;
                    pDstCurrent += 4;
                }

                result = VectorSum(in result);

                while (pSrcCurrent < pEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    Vector128<float> dstVector = Sse.LoadScalarVector128(pDstCurrent);

                    result = Sse.AddScalar(result, Sse.MultiplyScalar(srcVector, dstVector));

                    pSrcCurrent++;
                    pDstCurrent++;
                }
            }

            return Sse.ConvertToSingle(result);
        }

        internal static unsafe float DotSU(Span<float> src, Span<float> dst, Span<int> idx)
        {
            Vector128<float> result = Sse.SetZeroVector128();

            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (int* pidx = idx)
            {
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                int* pIdxCurrent = pidx;
                int* pEnd = pidx + idx.Length;

                while (pIdxCurrent + 4 <= pEnd)
                {
                    Vector128<float> srcVector = Load4(pSrcCurrent, pIdxCurrent);
                    Vector128<float> dstVector = Sse.LoadVector128(pDstCurrent);

                    result = Sse.Add(result, Sse.Multiply(srcVector, dstVector));

                    pIdxCurrent += 4;
                    pDstCurrent += 4;
                }

                result = VectorSum(in result);

                while (pIdxCurrent < pEnd)
                {
                    Vector128<float> srcVector = Load1(pSrcCurrent, pIdxCurrent);
                    Vector128<float> dstVector = Sse.LoadScalarVector128(pDstCurrent);

                    result = Sse.AddScalar(result, Sse.MultiplyScalar(srcVector, dstVector));

                    pIdxCurrent++;
                    pDstCurrent++;
                }
            }

            return Sse.ConvertToSingle(result);
        }

        internal static unsafe float Dist2(Span<float> src, Span<float> dst)
        {
            Vector128<float> sqDistanceVector = Sse.SetZeroVector128();

            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                float* pEnd = psrc + src.Length;

                while (pSrcCurrent + 4 <= pEnd)
                {
                    Vector128<float> distanceVector = Sse.Subtract(Sse.LoadVector128(pSrcCurrent),
                                                                    Sse.LoadVector128(pDstCurrent));
                    sqDistanceVector = Sse.Add(sqDistanceVector,
                                                Sse.Multiply(distanceVector, distanceVector));

                    pSrcCurrent += 4;
                    pDstCurrent += 4;
                }

                sqDistanceVector = VectorSum(in sqDistanceVector);

                float norm = Sse.ConvertToSingle(sqDistanceVector);
                while (pSrcCurrent < pEnd)
                {
                    float distance = (*pSrcCurrent) - (*pDstCurrent);
                    norm += distance * distance;

                    pSrcCurrent++;
                    pDstCurrent++;
                }

                return norm;
            }
        }

        internal static unsafe void ZeroItemsU(AlignedArray dst, int c, int[] indices, int cindices)
        {
            fixed (float* pDstStart = &dst.Items[0])
            fixed (int* pidx = &indices[0])
            {
                float* pdst = Ptr(dst, pDstStart);

                // REVIEW NEEDED: This line expands to (void)(c); but is it necessary?
                // DEBUG_ONLY(c);

                for (int i = 0; i < cindices; ++i)
                {
                    int index = pidx[i];
                    Contracts.Assert(0 <= index && index < c);
                    pdst[index] = 0;
                }
            }
        }

        internal static unsafe void ZeroMatrixItemsCore(AlignedArray dst, int c, int ccol, int cfltRow, int[] indices, int cindices)
        {
            fixed (float* pDstStart = &dst.Items[0])
            fixed (int* pidx = &indices[0])
            {
                float* pdst = Ptr(dst, pDstStart);

                // REVIEW NEEDED: This line expands to (void)(c); but is it necessary?
                // DEBUG_ONLY(c);

                int ivLogMin = 0;
                int ivLogLim = ccol;
                int ivPhyMin = 0;

                for (int i = 0; i < cindices; ++i)
                {
                    int index = pidx[i];
                    Contracts.Assert(0 <= index && index < c);

                    int col = index - ivLogMin;
                    if ((uint)col >= (uint)ccol)
                    {
                        Contracts.Assert(ivLogMin > index || index >= ivLogLim);

                        int row = index / ccol;
                        ivLogMin = row * ccol;
                        ivLogLim = ivLogMin + ccol;
                        ivPhyMin = row * cfltRow;

                        Contracts.Assert(ivLogMin <= index && index < ivLogLim);
                        col = index - ivLogMin;
                    }

                    pdst[ivPhyMin + col] = 0;
                }
            }
        }
    }
}
