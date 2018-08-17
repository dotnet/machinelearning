// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// The exported function names need to be unique (can't be disambiguated based on signature), hence
// we introduce suffix letters to indicate the general patterns used.
// * A suffix means aligned and padded for SSE operations.
// * P suffix means sparse (unaligned) partial vector - the vector is only part of a larger sparse vector.
// * Tran means the matrix is transposed.

using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    internal static class AvxIntrinsics
    {
        private const int CbAlign = 32;

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
        private static Vector256<float> ToVector256(in Vector128<float> a, in Vector128<float> b)
        {
            // REVIEW NEEDED: Is it the correct port of the following code?
            // #ifndef _WIN32
            // #define _mm256_set_m128(va, vb) _mm256_insertf128_ps(_mm256_castps128_ps256(vb), va, 1)
            // #endif
            return Avx.InsertVector128(Avx.ExtendToVector256(b), a, 1);
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static void ZeroUpper()
        {
            // Currently no-op since _mm256_zeroupper is not supported (ref: https://github.com/dotnet/coreclr/pull/16955)
            // This is a placeholder in case the intrinsic is supported later on.
            return;
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> GetLow(in Vector256<float> x)
        {
            return Avx.ExtractVector128(x, 0);
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> GetHigh(in Vector256<float> x)
        {
            return Avx.ExtractVector128(x, 1);
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> Rotate(in Vector128<float> x)
        {
            // The control byte shuffles the four 32-bit floats of x: ABCD -> BCDA.
            return Sse.Shuffle(x, x, 0x39);
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static unsafe void Store4(in Vector128<float> x, float* dst, int* idx)
        {
            Sse.StoreScalar(dst + idx[0], x);
            Vector128<float> rotated = Rotate(in x);
            Sse.StoreScalar(dst + idx[1], rotated);
            rotated = Rotate(in rotated);
            Sse.StoreScalar(dst + idx[2], rotated);
            rotated = Rotate(in rotated);
            Sse.StoreScalar(dst + idx[3], rotated);
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> VectorSum256(in Vector256<float> vector)
        {
            Vector256<float> partialSum = Avx.HorizontalAdd(vector, vector);
            return Avx.HorizontalAdd(partialSum, partialSum);
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> VectorSum128(in Vector128<float> vector)
        {
            if (Sse3.IsSupported)
            {
                Vector128<float> partialSum = Sse3.HorizontalAdd(vector, vector);
                return Sse3.HorizontalAdd(partialSum, partialSum);
            }
            else
            {
                Vector128<float> partialSum = Sse.Add(vector, Sse.MoveHighToLow(vector, vector));
                // The control byte shuffles the four 32-bit floats of partialSum: ABCD -> BADC.
                return Sse.Add(partialSum, Sse.Shuffle(partialSum, partialSum, 0xB1));
            }
        }

        // Multiply matrix times vector into vector.
        internal static unsafe void MatMulX(bool add, AlignedArray mat, AlignedArray src, AlignedArray dst, int crow, int ccol)
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
                float* pDstCurrent = pdst;
                float* pMatCurrent = pmat;

                while (pDstCurrent < pDstEnd)
                {
                    Vector256<float> res0 = Avx.SetZeroVector256<float>();
                    Vector256<float> res1 = res0;
                    Vector256<float> res2 = res0;
                    Vector256<float> res3 = res0;

                    float* pSrcCurrent = psrc;

                    while (pSrcCurrent < pSrcEnd)
                    {
                        float* pMatTemp = pMatCurrent;

                        Vector256<float> x01 = Avx.LoadAlignedVector256(pMatTemp);
                        Vector256<float> x11 = Avx.LoadAlignedVector256(pMatTemp += ccol);
                        Vector256<float> x21 = Avx.LoadAlignedVector256(pMatTemp += ccol);
                        Vector256<float> x31 = Avx.LoadAlignedVector256(pMatTemp += ccol);
                        Vector256<float> x02 = Avx.LoadAlignedVector256(pSrcCurrent);

                        res0 = Avx.Add(res0, Avx.Multiply(x01, x02));
                        res1 = Avx.Add(res1, Avx.Multiply(x11, x02));
                        res2 = Avx.Add(res2, Avx.Multiply(x21, x02));
                        res3 = Avx.Add(res3, Avx.Multiply(x31, x02));

                        pSrcCurrent += 8;
                        pMatCurrent += 8;
                    }

                    // Add up the entries of each, with the 4 results in res0
                    res0 = Avx.HorizontalAdd(res0, res1);
                    res2 = Avx.HorizontalAdd(res2, res3);
                    res0 = Avx.HorizontalAdd(res0, res2);

                    Vector128<float> sum = Sse.Add(GetLow(in res0), GetHigh(in res0));
                    if (add)
                    {
                        sum = Sse.Add(sum, Sse.LoadAlignedVector128(pDstCurrent));
                    }
                    Sse.StoreAligned(pDstCurrent, sum);

                    pDstCurrent += 4;
                    pMatCurrent += 3 * ccol;
                }

                ZeroUpper();
            }
        }

        // Partial sparse source vector.
        internal static unsafe void MatMulPX(bool add, AlignedArray mat, int[] rgposSrc, AlignedArray src,
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
                    Vector256<float> result = Avx.SetZeroVector256<float>();

                    int* ppos = pposMin;

                    while (ppos < pposEnd)
                    {
                        int col1 = *ppos;
                        int col2 = col1 + 4 * ccol;
                        Vector256<float> x1 = Avx.SetVector256(pm3[col2], pm2[col2], pm1[col2], pm0[col2],
                                                                pm3[col1], pm2[col1], pm1[col1], pm0[col1]);
                        Vector256<float> x2 = Avx.SetAllVector256(pSrcCurrent[col1]);
                        x2 = Avx.Multiply(x2, x1);
                        result = Avx.Add(result, x2);

                        ppos++;
                    }

                    if (add)
                    {
                        result = Avx.Add(result, Avx.LoadAlignedVector256(pDstCurrent));
                    }
                    Avx.StoreAligned(pDstCurrent, result);

                    pDstCurrent += 8;
                    pm0 += 8 * ccol;
                }

                ZeroUpper();
            }
        }

        internal static unsafe void MatMulTranX(bool add, AlignedArray mat, AlignedArray src, AlignedArray dst, int crow, int ccol)
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
                float* pMatCurrent = pmat;

                // We do 4-way unrolling
                if (!add)
                {
                    Vector128<float> h01 = Sse.LoadAlignedVector128(pSrcCurrent);
                    // Replicate each slot of h01 (ABCD) into its own register.
                    Vector128<float> h11 = Sse.Shuffle(h01, h01, 0x55); // B
                    Vector128<float> h21 = Sse.Shuffle(h01, h01, 0xAA); // C
                    Vector128<float> h31 = Sse.Shuffle(h01, h01, 0xFF); // D
                    h01 = Sse.Shuffle(h01, h01, 0x00); // A

                    Vector256<float> x01 = ToVector256(h01, h01);
                    Vector256<float> x11 = ToVector256(h11, h11);
                    Vector256<float> x21 = ToVector256(h21, h21);
                    Vector256<float> x31 = ToVector256(h31, h31);

                    pSrcCurrent += 4;

                    float* pDstCurrent = pdst;

                    while (pDstCurrent < pDstEnd)
                    {
                        float* pMatTemp = pMatCurrent;
                        Vector256<float> x02 = Avx.LoadAlignedVector256(pMatTemp);
                        Vector256<float> x12 = Avx.LoadAlignedVector256(pMatTemp += crow);
                        Vector256<float> x22 = Avx.LoadAlignedVector256(pMatTemp += crow);
                        Vector256<float> x32 = Avx.LoadAlignedVector256(pMatTemp += crow);

                        x02 = Avx.Multiply(x01, x02);
                        x12 = Avx.Multiply(x11, x12);
                        x22 = Avx.Multiply(x21, x22);
                        x32 = Avx.Multiply(x31, x32);

                        x02 = Avx.Add(x02, x12);
                        x22 = Avx.Add(x22, x32);
                        x02 = Avx.Add(x02, x22);

                        Avx.StoreAligned(pDstCurrent, x02);

                        pDstCurrent += 8;
                        pMatCurrent += 8;
                    }

                    pMatCurrent += 3 * crow;
                }

                while (pSrcCurrent < pSrcEnd)
                {
                    Vector128<float> h01 = Sse.LoadAlignedVector128(pSrcCurrent);
                    // Replicate each slot of h01 (ABCD) into its own register.
                    Vector128<float> h11 = Sse.Shuffle(h01, h01, 0x55); // B
                    Vector128<float> h21 = Sse.Shuffle(h01, h01, 0xAA); // C
                    Vector128<float> h31 = Sse.Shuffle(h01, h01, 0xFF); // D
                    h01 = Sse.Shuffle(h01, h01, 0x00); // A

                    Vector256<float> x01 = ToVector256(h01, h01);
                    Vector256<float> x11 = ToVector256(h11, h11);
                    Vector256<float> x21 = ToVector256(h21, h21);
                    Vector256<float> x31 = ToVector256(h31, h31);

                    float* pDstCurrent = pdst;

                    while (pDstCurrent < pDstEnd)
                    {
                        float* pMatTemp = pMatCurrent;

                        Vector256<float> x02 = Avx.LoadAlignedVector256(pMatTemp);
                        Vector256<float> x12 = Avx.LoadAlignedVector256(pMatTemp += crow);
                        Vector256<float> x22 = Avx.LoadAlignedVector256(pMatTemp += crow);
                        Vector256<float> x32 = Avx.LoadAlignedVector256(pMatTemp += crow);
                        Vector256<float> x3 = Avx.LoadAlignedVector256(pDstCurrent);

                        x02 = Avx.Multiply(x01, x02);
                        x12 = Avx.Multiply(x11, x12);
                        x22 = Avx.Multiply(x21, x22);
                        x32 = Avx.Multiply(x31, x32);

                        x02 = Avx.Add(x02, x12);
                        x22 = Avx.Add(x22, x32);
                        x02 = Avx.Add(x02, x22);
                        x3 = Avx.Add(x02, x3);

                        Avx.StoreAligned(pDstCurrent, x3);

                        pDstCurrent += 8;
                        pMatCurrent += 8;
                    }

                    pMatCurrent += 3 * crow;
                    pSrcCurrent += 4;
                }

                ZeroUpper();
            }
        }

        // Partial sparse source vector.
        internal static unsafe void MatMulTranPX(bool add, AlignedArray mat, int[] rgposSrc, AlignedArray src,
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

                    Vector256<float> x0 = Avx.SetAllVector256(psrc[col]);
                    float* pDstCurrent = pdst;
                    float* pMatCurrent = pmat + col * crow;

                    while (pDstCurrent < pDstEnd)
                    {
                        Vector256<float> x1 = Avx.LoadAlignedVector256(pMatCurrent);
                        x1 = Avx.Multiply(x1, x0);
                        Avx.StoreAligned(pDstCurrent, x1);

                        pDstCurrent += 8;
                        pMatCurrent += 8;
                    }
                }

                // REVIEW: Should we explore unrolling the outer loop?
                while (ppos < pposEnd)
                {
                    int col = *ppos - posMin;

                    Vector256<float> x0 = Avx.SetAllVector256(psrc[col]);
                    float* pDstCurrent = pdst;
                    float* pMatCurrent = pmat + col * crow;

                    while (pDstCurrent < pDstEnd)
                    {
                        Vector256<float> x1 = Avx.LoadAlignedVector256(pMatCurrent);
                        Vector256<float> x2 = Avx.LoadAlignedVector256(pDstCurrent);
                        x1 = Avx.Multiply(x1, x0);
                        x2 = Avx.Add(x2, x1);
                        Avx.StoreAligned(pDstCurrent, x2);

                        pDstCurrent += 8;
                        pMatCurrent += 8;
                    }

                    ppos++;
                }

                ZeroUpper();
            }
        }

        internal static unsafe void ScaleU(float scale, Span<float> dst)
        {
            fixed (float* pdst = dst)
            {
                float* pDstCurrent = pdst;
                float* pEnd = pdst + dst.Length;

                Vector256<float> scaleVector256 = Avx.SetAllVector256(scale);

                while (pDstCurrent + 8 <= pEnd)
                {
                    Vector256<float> dstVector = Avx.LoadVector256(pDstCurrent);

                    dstVector = Avx.Multiply(scaleVector256, dstVector);
                    Avx.Store(pDstCurrent, dstVector);

                    pDstCurrent += 8;
                }

                Vector128<float> scaleVector128 = Sse.SetAllVector128(scale);

                while (pDstCurrent + 4 <= pEnd)
                {
                    Vector128<float> dstVector = Sse.LoadVector128(pDstCurrent);

                    dstVector = Sse.Multiply(scaleVector128, dstVector);
                    Sse.Store(pDstCurrent, dstVector);

                    pDstCurrent += 4;
                }

                while (pDstCurrent < pEnd)
                {
                    Vector128<float> dstVector = Sse.LoadScalarVector128(pDstCurrent);

                    dstVector = Sse.MultiplyScalar(scaleVector128, dstVector);
                    Sse.StoreScalar(pDstCurrent, dstVector);

                    pDstCurrent++;
                }
            }

            ZeroUpper();
        }

        internal static unsafe void AddScaleU(float scale, Span<float> src, Span<float> dst)
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                float* pEnd = pdst + dst.Length;

                Vector256<float> scaleVector256 = Avx.SetAllVector256(scale);

                while (pDstCurrent + 8 <= pEnd)
                {
                    Vector256<float> srcVector = Avx.LoadVector256(pSrcCurrent);
                    Vector256<float> dstVector = Avx.LoadVector256(pDstCurrent);

                    srcVector = Avx.Multiply(srcVector, scaleVector256);
                    dstVector = Avx.Add(dstVector, srcVector);
                    Avx.Store(pDstCurrent, dstVector);

                    pSrcCurrent += 8;
                    pDstCurrent += 8;
                }

                Vector128<float> scaleVector128 = Sse.SetAllVector128(scale);

                while (pDstCurrent + 4 <= pEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    Vector128<float> dstVector = Sse.LoadVector128(pDstCurrent);

                    srcVector = Sse.Multiply(srcVector, scaleVector128);
                    dstVector = Sse.Add(dstVector, srcVector);
                    Sse.Store(pDstCurrent, dstVector);

                    pSrcCurrent += 4;
                    pDstCurrent += 4;
                }

                while (pDstCurrent < pEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    Vector128<float> dstVector = Sse.LoadScalarVector128(pDstCurrent);

                    srcVector = Sse.MultiplyScalar(srcVector, scaleVector128);
                    dstVector = Sse.AddScalar(dstVector, srcVector);
                    Sse.StoreScalar(pDstCurrent, dstVector);

                    pSrcCurrent++;
                    pDstCurrent++;
                }
            }

            ZeroUpper();
        }

        internal static unsafe void AddU(Span<float> src, Span<float> dst)
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                float* pEnd = psrc + src.Length;

                while (pSrcCurrent + 8 <= pEnd)
                {
                    Vector256<float> srcVector = Avx.LoadVector256(pSrcCurrent);
                    Vector256<float> dstVector = Avx.LoadVector256(pDstCurrent);

                    Vector256<float> result = Avx.Add(srcVector, dstVector);
                    Avx.Store(pDstCurrent, result);

                    pSrcCurrent += 8;
                    pDstCurrent += 8;
                }

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

                ZeroUpper();
            }
        }

        internal static unsafe float SumU(Span<float> src)
        {
            fixed (float* psrc = src)
            {
                float* pSrcEnd = psrc + src.Length;
                float* pSrcCurrent = psrc;

                Vector256<float> result256 = Avx.SetZeroVector256<float>();

                while (pSrcCurrent + 8 <= pSrcEnd)
                {
                    result256 = Avx.Add(result256, Avx.LoadVector256(pSrcCurrent));
                    pSrcCurrent += 8;
                }

                result256 = VectorSum256(in result256);
                Vector128<float> resultPadded = Sse.AddScalar(GetLow(result256), GetHigh(result256));

                Vector128<float> result128 = Sse.SetZeroVector128();

                while (pSrcCurrent + 4 <= pSrcEnd)
                {
                    result128 = Sse.Add(result128, Sse.LoadVector128(pSrcCurrent));
                    pSrcCurrent += 4;
                }

                result128 = VectorSum128(in result128);

                while (pSrcCurrent < pSrcEnd)
                {
                    result128 = Sse.AddScalar(result128, Sse.LoadScalarVector128(pSrcCurrent));
                    pSrcCurrent++;
                }

                float sum = Sse.ConvertToSingle(Sse.AddScalar(result128, resultPadded));
                ZeroUpper();
                return sum;
            }
        }
    }
}
