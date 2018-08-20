// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// The exported function names need to be unique (can't be disambiguated based on signature), hence
// we introduce suffix letters to indicate the general patterns used.
// * A suffix means aligned and padded for SSE operations.
// * U suffix means unaligned and unpadded.
// * P suffix means sparse (unaligned) partial vector - the vector is only part of a larger sparse vector.
// * Tran means the matrix is transposed.

using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    public static class AvxIntrinsics
    {
        private const int Vector256Alignment = 32;

        private static bool Compat(AlignedArray a)
        {
            Contracts.AssertValue(a);
            Contracts.Assert(a.Size > 0);
            return a.CbAlign == Vector256Alignment;
        }

        private static unsafe float* Ptr(AlignedArray a, float* p)
        {
            Contracts.AssertValue(a);
            float* q = p + a.GetBase((long)p);
            Contracts.Assert(((long)q & (Vector256Alignment - 1)) == 0);
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
        private static unsafe Vector256<float> Load8(float* src, int* idx)
        {
            return Avx.SetVector256(src[idx[7]], src[idx[6]], src[idx[5]], src[idx[4]], src[idx[3]], src[idx[2]], src[idx[1]], src[idx[0]]);
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
        private static unsafe void Store8(in Vector256<float> x, float* dst, int* idx)
        {
            Vector128<float> tmp = GetLow(in x);
            Sse.StoreScalar(dst + idx[0], tmp);
            tmp = Rotate(in tmp);
            Sse.StoreScalar(dst + idx[1], tmp);
            tmp = Rotate(in tmp);
            Sse.StoreScalar(dst + idx[2], tmp);
            tmp = Rotate(in tmp);
            Sse.StoreScalar(dst + idx[3], tmp);
            tmp = GetHigh(in x);
            Sse.StoreScalar(dst + idx[4], tmp);
            tmp = Rotate(in tmp);
            Sse.StoreScalar(dst + idx[5], tmp);
            tmp = Rotate(in tmp);
            Sse.StoreScalar(dst + idx[6], tmp);
            tmp = Rotate(in tmp);
            Sse.StoreScalar(dst + idx[7], tmp);
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

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> VectorSum256(in Vector256<float> vector)
        {
            Vector256<float> partialSum = Avx.HorizontalAdd(vector, vector);
            return Avx.HorizontalAdd(partialSum, partialSum);
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> VectorMax128(in Vector128<float> vector)
        {
            Vector128<float> x1 = Sse.Shuffle(vector, vector, 0xB1);
            Vector128<float> partialMax = Sse.Max(vector, x1);
            x1 = Sse.Shuffle(partialMax, partialMax, 0x02);
            return Sse.MaxScalar(partialMax, x1);
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> VectorMax256(in Vector256<float> vector)
        {
            Vector256<float> x1 = Avx.Shuffle(vector, vector, 0xB1);
            Vector256<float> partialMax = Avx.Max(vector, x1);
            x1 = Avx.Shuffle(partialMax, partialMax, 0x02);
            return Avx.Max(partialMax, x1);
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> GetAbsMask128()
        {
            return Sse2.IsSupported ?
                Sse.StaticCast<int, float>(Sse2.SetAllVector128(0x7FFFFFFF)) :
                Sse.SetAllVector128(BitConverter.Int32BitsToSingle(0x7FFFFFFF));
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> GetAbsMask256()
        {
            return Avx.StaticCast<int, float>(Avx.SetAllVector256(0x7FFFFFFF));
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> GetNewDst128(in Vector128<float> xDst1, in Vector128<float> signMask, in Vector128<float> xThreshold)
        {
            Vector128<float> xSign = Sse.And(xDst1, signMask); // result = 0x8000 0000 if xDst1 is negative or 0x0000 0000 otherwise
            Vector128<float> xDst1Abs = Sse.Xor(xDst1, xSign);
            Vector128<float> xCond = Sse.CompareGreaterThan(xDst1Abs, xThreshold); // result = 0xFFFF FFFF if true
            Vector128<float> x2 = Sse.Xor(xSign, xThreshold); // -xThreshold if xDst1 is negative and +xThreshold otherwise
            return Sse.And(Sse.Subtract(xDst1, x2), xCond);
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> GetNewDst256(in Vector256<float> xDst1, in Vector256<float> signMask, in Vector256<float> xThreshold)
        {
            Vector256<float> xSign = Avx.And(xDst1, signMask); // result = 0x8000 0000 if xDst1 is negative or 0x0000 0000 otherwise
            Vector256<float> xDst1Abs = Avx.Xor(xDst1, xSign);

            // REVIEW NEEDED: Do we want Signaling or NonSignaling?  The original functionality is NonSignaling, which does not throw an exception even when there is an NaN.
            // Signaling means that if an operand contains an NaN, an exception is raised (ref: https://stackoverflow.com/questions/16988199/how-to-choose-avx-compare-predicate-variants)
            Vector256<float> xCond = Avx.Compare(xDst1Abs, xThreshold, FloatComparisonMode.GreaterThanOrderedSignaling); // result = 0xFFFF FFFF if true
            Vector256<float> x2 = Avx.Xor(xSign, xThreshold); // -xThreshold if xDst1 is negative and +xThreshold otherwise
            return Avx.And(Avx.Subtract(xDst1, x2), xCond);
        }

        // Multiply matrix times vector into vector.
        public static unsafe void MatMulX(bool add, AlignedArray mat, AlignedArray src, AlignedArray dst, int crow, int ccol)
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
            }
        }

        // Partial sparse source vector.
        public static unsafe void MatMulPX(bool add, AlignedArray mat, int[] rgposSrc, AlignedArray src,
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
            }
        }

        public static unsafe void MatMulTranX(bool add, AlignedArray mat, AlignedArray src, AlignedArray dst, int crow, int ccol)
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
            }
        }

        // Partial sparse source vector.
        public static unsafe void MatMulTranPX(bool add, AlignedArray mat, int[] rgposSrc, AlignedArray src,
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
            }
        }

        // dst[i] += scale
        public static unsafe void AddScalarU(float scalar, Span<float> dst)
        {
            fixed (float* pdst = dst)
            {
                float* pDstEnd = pdst + dst.Length;
                float* pDstCurrent = pdst;

                Vector256<float> scalarVector256 = Avx.SetAllVector256(scalar);

                while (pDstCurrent + 8 <= pDstEnd)
                {
                    Vector256<float> dstVector = Avx.LoadVector256(pDstCurrent);
                    dstVector = Avx.Add(dstVector, scalarVector256);
                    Avx.Store(pDstCurrent, dstVector);

                    pDstCurrent += 8;
                }

                Vector128<float> scalarVector128 = Sse.SetAllVector128(scalar);

                if (pDstCurrent + 4 <= pDstEnd)
                {
                    Vector128<float> dstVector = Sse.LoadVector128(pDstCurrent);
                    dstVector = Sse.Add(dstVector, scalarVector128);
                    Sse.Store(pDstCurrent, dstVector);

                    pDstCurrent += 4;
                }

                while (pDstCurrent < pDstEnd)
                {
                    Vector128<float> dstVector = Sse.LoadScalarVector128(pDstCurrent);
                    dstVector = Sse.AddScalar(dstVector, scalarVector128);
                    Sse.StoreScalar(pDstCurrent, dstVector);

                    pDstCurrent++;
                }
            }
        }

        public static unsafe void ScaleU(float scale, Span<float> dst)
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

                if (pDstCurrent + 4 <= pEnd)
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
        }

        public static unsafe void ScaleSrcU(float scale, Span<float> src, Span<float> dst)
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                float* pDstEnd = pdst + dst.Length;
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;

                Vector256<float> scaleVector256 = Avx.SetAllVector256(scale);

                while (pDstCurrent + 8 <= pDstEnd)
                {
                    Vector256<float> srcVector = Avx.LoadVector256(pSrcCurrent);
                    srcVector = Avx.Multiply(srcVector, scaleVector256);
                    Avx.Store(pDstCurrent, srcVector);

                    pSrcCurrent += 8;
                    pDstCurrent += 8;
                }

                Vector128<float> scaleVector128 = Sse.SetAllVector128(scale);

                if (pDstCurrent + 4 <= pDstEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    srcVector = Sse.Multiply(srcVector, scaleVector128);
                    Sse.Store(pDstCurrent, srcVector);

                    pSrcCurrent += 4;
                    pDstCurrent += 4;
                }

                while (pDstCurrent < pDstEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    srcVector = Sse.MultiplyScalar(srcVector, scaleVector128);
                    Sse.StoreScalar(pDstCurrent, srcVector);

                    pSrcCurrent++;
                    pDstCurrent++;
                }
            }
        }

        // dst[i] = a * (dst[i] + b)
        public static unsafe void ScaleAddU(float a, float b, Span<float> dst)
        {
            fixed (float* pdst = dst)
            {
                float* pDstEnd = pdst + dst.Length;
                float* pDstCurrent = pdst;

                Vector256<float> a256 = Avx.SetAllVector256(a);
                Vector256<float> b256 = Avx.SetAllVector256(b);

                while (pDstCurrent + 8 <= pDstEnd)
                {
                    Vector256<float> dstVector = Avx.LoadVector256(pDstCurrent);
                    dstVector = Avx.Add(dstVector, b256);
                    dstVector = Avx.Multiply(dstVector, a256);
                    Avx.Store(pDstCurrent, dstVector);

                    pDstCurrent += 8;
                }

                Vector128<float> a128 = Sse.SetAllVector128(a);
                Vector128<float> b128 = Sse.SetAllVector128(b);

                if (pDstCurrent + 4 <= pDstEnd)
                {
                    Vector128<float> dstVector = Sse.LoadVector128(pDstCurrent);
                    dstVector = Sse.Add(dstVector, b128);
                    dstVector = Sse.Multiply(dstVector, a128);
                    Sse.Store(pDstCurrent, dstVector);

                    pDstCurrent += 4;
                }

                while (pDstCurrent < pDstEnd)
                {
                    Vector128<float> dstVector = Sse.LoadScalarVector128(pDstCurrent);
                    dstVector = Sse.AddScalar(dstVector, b128);
                    dstVector = Sse.MultiplyScalar(dstVector, a128);
                    Sse.StoreScalar(pDstCurrent, dstVector);

                    pDstCurrent++;
                }
            }
        }

        public static unsafe void AddScaleU(float scale, Span<float> src, Span<float> dst)
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

                if (pDstCurrent + 4 <= pEnd)
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
        }

        public static unsafe void AddScaleCopyU(float scale, Span<float> src, Span<float> dst, Span<float> result)
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (float* pres = result)
            {
                float* pResEnd = pres + result.Length;
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                float* pResCurrent = pres;

                Vector256<float> scaleVector256 = Avx.SetAllVector256(scale);

                while (pResCurrent + 8 <= pResEnd)
                {
                    Vector256<float> srcVector = Avx.LoadVector256(pSrcCurrent);
                    Vector256<float> dstVector = Avx.LoadVector256(pDstCurrent);
                    srcVector = Avx.Multiply(srcVector, scaleVector256);
                    dstVector = Avx.Add(dstVector, srcVector);
                    Avx.Store(pResCurrent, dstVector);

                    pSrcCurrent += 8;
                    pDstCurrent += 8;
                    pResCurrent += 8;
                }

                Vector128<float> scaleVector128 = Sse.SetAllVector128(scale);

                if (pResCurrent + 4 <= pResEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    Vector128<float> dstVector = Sse.LoadVector128(pDstCurrent);
                    srcVector = Sse.Multiply(srcVector, scaleVector128);
                    dstVector = Sse.Add(dstVector, srcVector);
                    Sse.Store(pResCurrent, dstVector);

                    pSrcCurrent += 4;
                    pDstCurrent += 4;
                    pResCurrent += 4;
                }

                while (pResCurrent < pResEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    Vector128<float> dstVector = Sse.LoadScalarVector128(pDstCurrent);
                    srcVector = Sse.MultiplyScalar(srcVector, scaleVector128);
                    dstVector = Sse.AddScalar(dstVector, srcVector);
                    Sse.StoreScalar(pResCurrent, dstVector);

                    pSrcCurrent++;
                    pDstCurrent++;
                    pResCurrent++;
                }
            }
        }

        public static unsafe void AddScaleSU(float scale, Span<float> src, Span<int> idx, Span<float> dst)
        {
            fixed (float* psrc = src)
            fixed (int* pidx = idx)
            fixed (float* pdst = dst)
            {
                float* pSrcCurrent = psrc;
                int* pIdxCurrent = pidx;
                float* pDstCurrent = pdst;
                int* pEnd = pidx + idx.Length;

                Vector256<float> scaleVector256 = Avx.SetAllVector256(scale);

                while (pIdxCurrent + 8 <= pEnd)
                {
                    Vector256<float> srcVector = Avx.LoadVector256(pSrcCurrent);
                    Vector256<float> dstVector = Load8(pDstCurrent, pIdxCurrent);

                    srcVector = Avx.Multiply(srcVector, scaleVector256);
                    dstVector = Avx.Add(dstVector, srcVector);
                    Store8(in dstVector, pDstCurrent, pIdxCurrent);

                    pIdxCurrent += 8;
                    pSrcCurrent += 8;
                }

                Vector128<float> scaleVector128 = Sse.SetAllVector128(scale);

                if (pIdxCurrent + 4 <= pEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    Vector128<float> dstVector = Load4(pDstCurrent, pIdxCurrent);

                    srcVector = Sse.Multiply(srcVector, scaleVector128);
                    dstVector = Sse.Add(dstVector, srcVector);
                    Store4(in dstVector, pDstCurrent, pIdxCurrent);

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

        public static unsafe void AddU(Span<float> src, Span<float> dst)
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

                if (pSrcCurrent + 4 <= pEnd)
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

        public static unsafe void AddSU(Span<float> src, Span<int> idx, Span<float> dst)
        {
            fixed (float* psrc = src)
            fixed (int* pidx = idx)
            fixed (float* pdst = dst)
            {
                float* pSrcCurrent = psrc;
                int* pIdxCurrent = pidx;
                float* pDstCurrent = pdst;
                int* pEnd = pidx + idx.Length;

                while (pIdxCurrent + 8 <= pEnd)
                {
                    Vector256<float> dstVector = Load8(pDstCurrent, pIdxCurrent);
                    Vector256<float> srcVector = Avx.LoadVector256(pSrcCurrent);

                    dstVector = Avx.Add(dstVector, srcVector);
                    Store8(in dstVector, pDstCurrent, pIdxCurrent);

                    pIdxCurrent += 8;
                    pSrcCurrent += 8;
                }

                if (pIdxCurrent + 4 <= pEnd)
                {
                    Vector128<float> dstVector = Load4(pDstCurrent, pIdxCurrent);
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);

                    dstVector = Sse.Add(dstVector, srcVector);
                    Store4(in dstVector, pDstCurrent, pIdxCurrent);

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

        public static unsafe void MulElementWiseU(Span<float> src1, Span<float> src2, Span<float> dst)
        {
            fixed (float* psrc1 = src1)
            fixed (float* psrc2 = src2)
            fixed (float* pdst = dst)
            {
                float* pSrc1Current = psrc1;
                float* pSrc2Current = psrc2;
                float* pDstCurrent = pdst;
                float* pEnd = pdst + dst.Length;

                while (pDstCurrent + 8 <= pEnd)
                {
                    Vector256<float> src1Vector = Avx.LoadVector256(pSrc1Current);
                    Vector256<float> src2Vector = Avx.LoadVector256(pSrc2Current);
                    src2Vector = Avx.Multiply(src1Vector, src2Vector);
                    Avx.Store(pDstCurrent, src2Vector);

                    pSrc1Current += 8;
                    pSrc2Current += 8;
                    pDstCurrent += 8;
                }

                if (pDstCurrent + 4 <= pEnd)
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

        public static unsafe float SumU(Span<float> src)
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

                if (pSrcCurrent + 4 <= pSrcEnd)
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

                return Sse.ConvertToSingle(Sse.AddScalar(result128, resultPadded));
            }
        }

        public static unsafe float SumSqU(Span<float> src)
        {
            fixed (float* psrc = src)
            {
                float* pSrcEnd = psrc + src.Length;
                float* pSrcCurrent = psrc;

                Vector256<float> result256 = Avx.SetZeroVector256<float>();

                while (pSrcCurrent + 8 <= pSrcEnd)
                {
                    Vector256<float> srcVector = Avx.LoadVector256(pSrcCurrent);
                    result256 = Avx.Add(result256, Avx.Multiply(srcVector, srcVector));

                    pSrcCurrent += 8;
                }

                result256 = VectorSum256(in result256);
                Vector128<float> resultPadded = Sse.AddScalar(GetLow(result256), GetHigh(result256));

                Vector128<float> result128 = Sse.SetZeroVector128();

                if (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    result128 = Sse.Add(result128, Sse.Multiply(srcVector, srcVector));

                    pSrcCurrent += 4;
                }

                result128 = VectorSum128(in result128);

                while (pSrcCurrent < pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    result128 = Sse.AddScalar(result128, Sse.MultiplyScalar(srcVector, srcVector));

                    pSrcCurrent++;
                }

                return Sse.ConvertToSingle(Sse.AddScalar(result128, resultPadded));
            }
        }

        public static unsafe float SumSqDiffU(float mean, Span<float> src)
        {
            fixed (float* psrc = src)
            {
                float* pSrcEnd = psrc + src.Length;
                float* pSrcCurrent = psrc;

                Vector256<float> result256 = Avx.SetZeroVector256<float>();
                Vector256<float> meanVector256 = Avx.SetAllVector256(mean);

                while (pSrcCurrent + 8 <= pSrcEnd)
                {
                    Vector256<float> srcVector = Avx.LoadVector256(pSrcCurrent);
                    srcVector = Avx.Subtract(srcVector, meanVector256);
                    result256 = Avx.Add(result256, Avx.Multiply(srcVector, srcVector));

                    pSrcCurrent += 8;
                }

                result256 = VectorSum256(in result256);
                Vector128<float> resultPadded = Sse.AddScalar(GetLow(result256), GetHigh(result256));

                Vector128<float> result128 = Sse.SetZeroVector128();
                Vector128<float> meanVector128 = Sse.SetAllVector128(mean);

                if (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    srcVector = Sse.Subtract(srcVector, meanVector128);
                    result128 = Sse.Add(result128, Sse.Multiply(srcVector, srcVector));

                    pSrcCurrent += 4;
                }

                result128 = VectorSum128(in result128);

                while (pSrcCurrent < pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    srcVector = Sse.SubtractScalar(srcVector, meanVector128);
                    result128 = Sse.AddScalar(result128, Sse.MultiplyScalar(srcVector, srcVector));

                    pSrcCurrent++;
                }

                return Sse.ConvertToSingle(Sse.AddScalar(result128, resultPadded));
            }
        }

        public static unsafe float SumAbsU(Span<float> src)
        {
            fixed (float* psrc = src)
            {
                float* pSrcEnd = psrc + src.Length;
                float* pSrcCurrent = psrc;

                Vector256<float> result256 = Avx.SetZeroVector256<float>();
                Vector256<float> mask256 = GetAbsMask256();

                while (pSrcCurrent + 8 <= pSrcEnd)
                {
                    Vector256<float> srcVector = Avx.LoadVector256(pSrcCurrent);
                    result256 = Avx.Add(result256, Avx.And(srcVector, mask256));

                    pSrcCurrent += 8;
                }

                result256 = VectorSum256(in result256);
                Vector128<float> resultPadded = Sse.AddScalar(GetLow(result256), GetHigh(result256));

                Vector128<float> result128 = Sse.SetZeroVector128();
                Vector128<float> mask128 = GetAbsMask128();

                if (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    result128 = Sse.Add(result128, Sse.And(srcVector, mask128));

                    pSrcCurrent += 4;
                }

                result128 = VectorSum128(in result128);

                while (pSrcCurrent < pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    result128 = Sse.AddScalar(result128, Sse.And(srcVector, mask128));

                    pSrcCurrent++;
                }

                return Sse.ConvertToSingle(Sse.AddScalar(result128, resultPadded));
            }
        }

        public static unsafe float SumAbsDiffU(float mean, Span<float> src)
        {
            fixed (float* psrc = src)
            {
                float* pSrcEnd = psrc + src.Length;
                float* pSrcCurrent = psrc;

                Vector256<float> result256 = Avx.SetZeroVector256<float>();
                Vector256<float> meanVector256 = Avx.SetAllVector256(mean);
                Vector256<float> mask256 = GetAbsMask256();

                while (pSrcCurrent + 8 <= pSrcEnd)
                {
                    Vector256<float> srcVector = Avx.LoadVector256(pSrcCurrent);
                    srcVector = Avx.Subtract(srcVector, meanVector256);
                    result256 = Avx.Add(result256, Avx.And(srcVector, mask256));

                    pSrcCurrent += 8;
                }

                result256 = VectorSum256(in result256);
                Vector128<float> resultPadded = Sse.AddScalar(GetLow(result256), GetHigh(result256));

                Vector128<float> result128 = Sse.SetZeroVector128();
                Vector128<float> meanVector128 = Sse.SetAllVector128(mean);
                Vector128<float> mask128 = GetAbsMask128();

                if (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    srcVector = Sse.Subtract(srcVector, meanVector128);
                    result128 = Sse.Add(result128, Sse.And(srcVector, mask128));

                    pSrcCurrent += 4;
                }

                result128 = VectorSum128(in result128);

                while (pSrcCurrent < pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    srcVector = Sse.SubtractScalar(srcVector, meanVector128);
                    result128 = Sse.AddScalar(result128, Sse.And(srcVector, mask128));

                    pSrcCurrent++;
                }

                return Sse.ConvertToSingle(Sse.AddScalar(result128, resultPadded));
            }
        }

        public static unsafe float MaxAbsU(Span<float> src)
        {
            fixed (float* psrc = src)
            {
                float* pSrcEnd = psrc + src.Length;
                float* pSrcCurrent = psrc;

                Vector256<float> result256 = Avx.SetZeroVector256<float>();
                Vector256<float> mask256 = GetAbsMask256();

                while (pSrcCurrent + 8 <= pSrcEnd)
                {
                    Vector256<float> srcVector = Avx.LoadVector256(pSrcCurrent);
                    result256 = Avx.Max(result256, Avx.And(srcVector, mask256));

                    pSrcCurrent += 8;
                }

                result256 = VectorMax256(in result256);
                Vector128<float> resultPadded = Sse.MaxScalar(GetLow(result256), GetHigh(result256));

                Vector128<float> result128 = Sse.SetZeroVector128();
                Vector128<float> mask128 = GetAbsMask128();

                if (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    result128 = Sse.Max(result128, Sse.And(srcVector, mask128));

                    pSrcCurrent += 4;
                }

                result128 = VectorMax128(in result128);

                while (pSrcCurrent < pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    result128 = Sse.MaxScalar(result128, Sse.And(srcVector, mask128));

                    pSrcCurrent++;
                }

                return Sse.ConvertToSingle(Sse.MaxScalar(result128, resultPadded));
            }
        }

        public static unsafe float MaxAbsDiffU(float mean, Span<float> src)
        {
            fixed (float* psrc = src)
            {
                float* pSrcEnd = psrc + src.Length;
                float* pSrcCurrent = psrc;

                Vector256<float> result256 = Avx.SetZeroVector256<float>();
                Vector256<float> meanVector256 = Avx.SetAllVector256(mean);
                Vector256<float> mask256 = GetAbsMask256();

                while (pSrcCurrent + 8 <= pSrcEnd)
                {
                    Vector256<float> srcVector = Avx.LoadVector256(pSrcCurrent);
                    srcVector = Avx.Subtract(srcVector, meanVector256);
                    result256 = Avx.Max(result256, Avx.And(srcVector, mask256));

                    pSrcCurrent += 8;
                }

                result256 = VectorMax256(in result256);
                Vector128<float> resultPadded = Sse.MaxScalar(GetLow(result256), GetHigh(result256));

                Vector128<float> result128 = Sse.SetZeroVector128();
                Vector128<float> meanVector128 = Sse.SetAllVector128(mean);
                Vector128<float> mask128 = GetAbsMask128();

                if (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    srcVector = Sse.Subtract(srcVector, meanVector128);
                    result128 = Sse.Max(result128, Sse.And(srcVector, mask128));

                    pSrcCurrent += 4;
                }

                result128 = VectorMax128(in result128);

                while (pSrcCurrent < pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    srcVector = Sse.SubtractScalar(srcVector, meanVector128);
                    result128 = Sse.MaxScalar(result128, Sse.And(srcVector, mask128));

                    pSrcCurrent++;
                }

                return Sse.ConvertToSingle(Sse.MaxScalar(result128, resultPadded));
            }
        }

        public static unsafe float DotU(Span<float> src, Span<float> dst)
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                float* pSrcEnd = psrc + src.Length;

                Vector256<float> result256 = Avx.SetZeroVector256<float>();

                while (pSrcCurrent + 8 <= pSrcEnd)
                {
                    Vector256<float> srcVector = Avx.LoadVector256(pSrcCurrent);
                    Vector256<float> dstVector = Avx.LoadVector256(pDstCurrent);

                    result256 = Avx.Add(result256, Avx.Multiply(srcVector, dstVector));

                    pSrcCurrent += 8;
                    pDstCurrent += 8;
                }

                result256 = VectorSum256(in result256);
                Vector128<float> resultPadded = Sse.AddScalar(GetLow(result256), GetHigh(result256));

                Vector128<float> result128 = Sse.SetZeroVector128();

                if (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    Vector128<float> dstVector = Sse.LoadVector128(pDstCurrent);

                    result128 = Sse.Add(result128, Sse.Multiply(srcVector, dstVector));

                    pSrcCurrent += 4;
                    pDstCurrent += 4;
                }

                result128 = VectorSum128(in result128);

                while (pSrcCurrent < pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    Vector128<float> dstVector = Sse.LoadScalarVector128(pDstCurrent);

                    result128 = Sse.AddScalar(result128, Sse.MultiplyScalar(srcVector, dstVector));

                    pSrcCurrent++;
                    pDstCurrent++;
                }

                return Sse.ConvertToSingle(Sse.AddScalar(result128, resultPadded));
            }
        }

        public static unsafe float DotSU(Span<float> src, Span<float> dst, Span<int> idx)
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            fixed (int* pidx = idx)
            {
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                int* pIdxCurrent = pidx;
                int* pIdxEnd = pidx + idx.Length;

                Vector256<float> result256 = Avx.SetZeroVector256<float>();

                while (pIdxCurrent + 8 <= pIdxEnd)
                {
                    Vector256<float> srcVector = Load8(pSrcCurrent, pIdxCurrent);
                    Vector256<float> dstVector = Avx.LoadVector256(pDstCurrent);

                    result256 = Avx.Add(result256, Avx.Multiply(srcVector, dstVector));

                    pIdxCurrent += 8;
                    pDstCurrent += 8;
                }

                result256 = VectorSum256(in result256);
                Vector128<float> resultPadded = Sse.AddScalar(GetLow(result256), GetHigh(result256));

                Vector128<float> result128 = Sse.SetZeroVector128();

                if (pIdxCurrent + 4 <= pIdxEnd)
                {
                    Vector128<float> srcVector = Load4(pSrcCurrent, pIdxCurrent);
                    Vector128<float> dstVector = Sse.LoadVector128(pDstCurrent);

                    result128 = Sse.Add(result128, Sse.Multiply(srcVector, dstVector));

                    pIdxCurrent += 4;
                    pDstCurrent += 4;
                }

                result128 = VectorSum128(in result128);

                while (pIdxCurrent < pIdxEnd)
                {
                    Vector128<float> srcVector = Load1(pSrcCurrent, pIdxCurrent);
                    Vector128<float> dstVector = Sse.LoadScalarVector128(pDstCurrent);

                    result128 = Sse.AddScalar(result128, Sse.MultiplyScalar(srcVector, dstVector));

                    pIdxCurrent++;
                    pDstCurrent++;
                }

                return Sse.ConvertToSingle(Sse.AddScalar(result128, resultPadded));
            }
        }

        public static unsafe float Dist2(Span<float> src, Span<float> dst)
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                float* pSrcEnd = psrc + src.Length;

                Vector256<float> sqDistanceVector256 = Avx.SetZeroVector256<float>();

                while (pSrcCurrent + 8 <= pSrcEnd)
                {
                    Vector256<float> distanceVector = Avx.Subtract(Avx.LoadVector256(pSrcCurrent),
                                                                    Avx.LoadVector256(pDstCurrent));
                    sqDistanceVector256 = Avx.Add(sqDistanceVector256,
                                                Avx.Multiply(distanceVector, distanceVector));

                    pSrcCurrent += 8;
                    pDstCurrent += 8;
                }

                sqDistanceVector256 = VectorSum256(in sqDistanceVector256);
                Vector128<float> sqDistanceVectorPadded = Sse.AddScalar(GetLow(sqDistanceVector256), GetHigh(sqDistanceVector256));

                Vector128<float> sqDistanceVector128 = Sse.SetZeroVector128();

                if (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> distanceVector = Sse.Subtract(Sse.LoadVector128(pSrcCurrent),
                                                                    Sse.LoadVector128(pDstCurrent));
                    sqDistanceVector128 = Sse.Add(sqDistanceVector128,
                                                Sse.Multiply(distanceVector, distanceVector));

                    pSrcCurrent += 4;
                    pDstCurrent += 4;
                }

                sqDistanceVector128 = VectorSum128(in sqDistanceVector128);

                float norm = Sse.ConvertToSingle(Sse.AddScalar(sqDistanceVector128, sqDistanceVectorPadded));
                while (pSrcCurrent < pSrcEnd)
                {
                    float distance = (*pSrcCurrent) - (*pDstCurrent);
                    norm += distance * distance;

                    pSrcCurrent++;
                    pDstCurrent++;
                }

                return norm;
            }
        }

        public static unsafe void SdcaL1UpdateU(float primalUpdate, Span<float> src, float threshold, Span<float> v, Span<float> w)
        {
            fixed (float* psrc = src)
            fixed (float* pdst1 = v)
            fixed (float* pdst2 = w)
            {
                float* pSrcEnd = psrc + src.Length;
                float* pSrcCurrent = psrc;
                float* pDst1Current = pdst1;
                float* pDst2Current = pdst2;

                Vector256<float> xPrimal256 = Avx.SetAllVector256(primalUpdate);

                Vector256<float> signMask256 = Avx.SetAllVector256(-0.0f); // 0x8000 0000
                Vector256<float> xThreshold256 = Avx.SetAllVector256(threshold);

                while (pSrcCurrent + 8 <= pSrcEnd)
                {
                    Vector256<float> xSrc = Avx.LoadVector256(pSrcCurrent);

                    Vector256<float> xDst1 = Avx.LoadVector256(pDst1Current);
                    xDst1 = Avx.Add(xDst1, Avx.Multiply(xSrc, xPrimal256));
                    Vector256<float> xDst2 = GetNewDst256(xDst1, signMask256, xThreshold256);

                    Avx.Store(pDst1Current, xDst1);
                    Avx.Store(pDst2Current, xDst2);

                    pSrcCurrent += 8;
                    pDst1Current += 8;
                    pDst2Current += 8;
                }

                Vector128<float> xPrimal128 = Sse.SetAllVector128(primalUpdate);

                Vector128<float> signMask128 = Sse.SetAllVector128(-0.0f); // 0x8000 0000
                Vector128<float> xThreshold128 = Sse.SetAllVector128(threshold);

                if (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> xSrc = Sse.LoadVector128(pSrcCurrent);

                    Vector128<float> xDst1 = Sse.LoadVector128(pDst1Current);
                    xDst1 = Sse.Add(xDst1, Sse.Multiply(xSrc, xPrimal128));
                    Vector128<float> xDst2 = GetNewDst128(xDst1, signMask128, xThreshold128);

                    Sse.Store(pDst1Current, xDst1);
                    Sse.Store(pDst2Current, xDst2);

                    pSrcCurrent += 4;
                    pDst1Current += 4;
                    pDst2Current += 4;
                }

                while (pSrcCurrent < pSrcEnd)
                {
                    *pDst1Current += (*pSrcCurrent) * primalUpdate;
                    float dst1 = *pDst1Current;
                    *pDst2Current = Math.Abs(dst1) > threshold ? (dst1 > 0 ? dst1 - threshold : dst1 + threshold) : 0;

                    pSrcCurrent++;
                    pDst1Current++;
                    pDst2Current++;
                }
            }
        }

        public static unsafe void SdcaL1UpdateSU(float primalUpdate, Span<float> src, Span<int> indices, float threshold, Span<float> v, Span<float> w)
        {
            fixed (float* psrc = src)
            fixed (int* pidx = indices)
            fixed (float* pdst1 = v)
            fixed (float* pdst2 = w)
            {
                int* pIdxEnd = pidx + indices.Length;
                float* pSrcCurrent = psrc;
                int* pIdxCurrent = pidx;

                Vector256<float> xPrimal256 = Avx.SetAllVector256(primalUpdate);

                Vector256<float> signMask = Avx.SetAllVector256(-0.0f); // 0x8000 0000
                Vector256<float> xThreshold = Avx.SetAllVector256(threshold);

                while (pIdxCurrent + 8 <= pIdxEnd)
                {
                    Vector256<float> xSrc = Avx.LoadVector256(pSrcCurrent);

                    Vector256<float> xDst1 = Load8(pdst1, pIdxCurrent);
                    xDst1 = Avx.Add(xDst1, Avx.Multiply(xSrc, xPrimal256));
                    Vector256<float> xDst2 = GetNewDst256(xDst1, signMask, xThreshold);

                    Store8(in xDst1, pdst1, pIdxCurrent);
                    Store8(in xDst2, pdst2, pIdxCurrent);

                    pIdxCurrent += 8;
                    pSrcCurrent += 8;
                }

                Vector128<float> xPrimal128 = Sse.SetAllVector128(primalUpdate);

                Vector128<float> signMask128 = Sse.SetAllVector128(-0.0f); // 0x8000 0000
                Vector128<float> xThreshold128 = Sse.SetAllVector128(threshold);

                if (pIdxCurrent + 4 <= pIdxEnd)
                {
                    Vector128<float> xSrc = Sse.LoadVector128(pSrcCurrent);

                    Vector128<float> xDst1 = Load4(pdst1, pIdxCurrent);
                    xDst1 = Sse.Add(xDst1, Sse.Multiply(xSrc, xPrimal128));
                    Vector128<float> xDst2 = GetNewDst128(xDst1, signMask128, xThreshold128);

                    Store4(in xDst1, pdst1, pIdxCurrent);
                    Store4(in xDst2, pdst2, pIdxCurrent);

                    pIdxCurrent += 4;
                    pSrcCurrent += 4;
                }

                while (pIdxCurrent < pIdxEnd)
                {
                    int index = *pIdxCurrent;
                    pdst1[index] += (*pSrcCurrent) * primalUpdate;
                    float dst1 = pdst1[index];
                    pdst2[index] = Math.Abs(dst1) > threshold ? (dst1 > 0 ? dst1 - threshold : dst1 + threshold) : 0;

                    pIdxCurrent++;
                    pSrcCurrent++;
                }
            }
        }
    }
}
