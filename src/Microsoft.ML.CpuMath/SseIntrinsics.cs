// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// The exported function names need to be unique (can't be disambiguated based on signature), hence
// we introduce suffix letters to indicate the general patterns used.
// * A suffix means aligned and padded for SSE operations.
// * U suffix means unaligned and unpadded.
// * S suffix means sparse (unaligned) vector.
// * P suffix means sparse (unaligned) partial vector - the vector is only part of a larger sparse vector.
// * R suffix means sparse matrix.
// * C suffix means convolution matrix.
// * D suffix means convolution matrix, with implicit source padding.
// * Tran means the matrix is transposed.

using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    internal static class SseIntrinsics
    {
        internal static readonly Vector128<float> AbsMask128 = Sse2.IsSupported ?
            Sse.StaticCast<int, float>(Sse2.SetAllVector128(0x7FFFFFFF)) :
            Sse.SetAllVector128(BitConverter.Int32BitsToSingle(0x7FFFFFFF));

        // The count of bytes in Vector128<T>, corresponding to _cbAlign in AlignedArray
        private const int Vector128Alignment = 16;

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static bool HasCompatibleAlignment(AlignedArray alignedArray)
        {
            Contracts.AssertValue(alignedArray);
            Contracts.Assert(alignedArray.Size > 0);
            return (alignedArray.CbAlign % Vector128Alignment) == 0;
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static unsafe float* GetAlignedBase(AlignedArray alignedArray, float* unalignedBase)
        {
            Contracts.AssertValue(alignedArray);
            float* alignedBase = unalignedBase + alignedArray.GetBase((long)unalignedBase);
            Contracts.Assert(((long)alignedBase & (Vector128Alignment - 1)) == 0);
            return alignedBase;
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        internal static unsafe Vector128<float> Load1(float* src, int* idx)
             => Sse.SetScalarVector128(src[idx[0]]);

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        internal static unsafe Vector128<float> Load4(float* src, int* idx)
            => Sse.SetVector128(src[idx[3]], src[idx[2]], src[idx[1]], src[idx[0]]);

        // The control byte shuffles the four 32-bit floats of x: ABCD -> BCDA.
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        internal static Vector128<float> Rotate(in Vector128<float> x)
            => Sse.Shuffle(x, x, 0x39);

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void Store4(in Vector128<float> x, float* dst, int* idx)
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
       internal static Vector128<float> VectorSum128(in Vector128<float> vector)
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
        internal static Vector128<float> VectorMax128(in Vector128<float> vector)
        {
            // The control byte shuffles the four 32-bit floats of partialMax: ABCD -> BADC.
            Vector128<float> x1 = Sse.Shuffle(vector, vector, 0xB1);

            // Performs element-wise maximum operation: The 1st and 3rd 32-bit slots become
            // max(A, B) and max(C, D).
            Vector128<float> partialMax = Sse.Max(vector, x1);

            // The control byte shuffles the four 32-bit floats of partialMax: ABCD -> CAAA.
            x1 = Sse.Shuffle(partialMax, partialMax, 0x02);

            // Performs element-wise maximum operation: The 1st 32-bit slot becomes
            // max(A, B, C, D).
            return Sse.MaxScalar(partialMax, x1);
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        internal static Vector128<float> GetNewDst128(in Vector128<float> xDst1, in Vector128<float> xThreshold)
        {
            Vector128<float> signMask = Sse.SetAllVector128(-0.0f); // 0x8000 0000
            Vector128<float> xSign = Sse.And(xDst1, signMask); // result = 0x8000 0000 if xDst1 is negative or 0x0000 0000 otherwise
            Vector128<float> xDst1Abs = Sse.Xor(xDst1, xSign);
            Vector128<float> xCond = Sse.CompareGreaterThan(xDst1Abs, xThreshold); // result = 0xFFFF FFFF if true
            Vector128<float> x2 = Sse.Xor(xSign, xThreshold); // -xThreshold if xDst1 is negative and +xThreshold otherwise
            return Sse.And(Sse.Subtract(xDst1, x2), xCond);
        }

        // Multiply matrix times vector into vector.
        public static unsafe void MatMulA(bool add, AlignedArray mat, AlignedArray src, AlignedArray dst, int crow, int ccol)
        {
            Contracts.Assert(HasCompatibleAlignment(mat));
            Contracts.Assert(HasCompatibleAlignment(src));
            Contracts.Assert(HasCompatibleAlignment(dst));

            fixed (float* pSrcStart = &src.Items[0])
            fixed (float* pDstStart = &dst.Items[0])
            fixed (float* pMatStart = &mat.Items[0])
            {
                float* psrc = GetAlignedBase(src, pSrcStart);
                float* pdst = GetAlignedBase(dst, pDstStart);
                float* pmat = GetAlignedBase(mat, pMatStart);

                float* pSrcEnd = psrc + ccol;
                float* pDstEnd = pdst + crow;
                float* pDstCurrent = pdst;
                float* pMatCurrent = pmat;

                while (pDstCurrent < pDstEnd)
                {
                    Vector128<float> res0 = Sse.SetZeroVector128();
                    Vector128<float> res1 = res0;
                    Vector128<float> res2 = res0;
                    Vector128<float> res3 = res0;

                    float* pSrcCurrent = psrc;

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
        public static unsafe void MatMulPA(bool add, AlignedArray mat, int[] rgposSrc, AlignedArray src,
                                        int posMin, int iposMin, int iposEnd, AlignedArray dst, int crow, int ccol)
        {
            Contracts.Assert(HasCompatibleAlignment(mat));
            Contracts.Assert(HasCompatibleAlignment(src));
            Contracts.Assert(HasCompatibleAlignment(dst));

            // REVIEW: For extremely sparse inputs, interchanging the loops would
            // likely be more efficient.
            fixed (float* pSrcStart = &src.Items[0])
            fixed (float* pDstStart = &dst.Items[0])
            fixed (float* pMatStart = &mat.Items[0])
            fixed (int* pposSrc = &rgposSrc[0])
            {
                float* psrc = GetAlignedBase(src, pSrcStart);
                float* pdst = GetAlignedBase(dst, pDstStart);
                float* pmat = GetAlignedBase(mat, pMatStart);

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
                    Vector128<float> result = Sse.SetZeroVector128();

                    int* ppos = pposMin;

                    while (ppos < pposEnd)
                    {
                        int col = *ppos;
                        Vector128<float> x1 = Sse.SetVector128(pm3[col], pm2[col], pm1[col], pm0[col]);
                        Vector128<float> x2 = Sse.SetAllVector128(pSrcCurrent[col]);
                        x2 = Sse.Multiply(x2, x1);
                        result = Sse.Add(result, x2);

                        ppos++;
                    }

                    if (add)
                    {
                        result = Sse.Add(result, Sse.LoadAlignedVector128(pDstCurrent));
                    }
                    Sse.StoreAligned(pDstCurrent, result);

                    pDstCurrent += 4;
                    pm0 += 4 * ccol;
                }
            }
        }

        public static unsafe void MatMulTranA(bool add, AlignedArray mat, AlignedArray src, AlignedArray dst, int crow, int ccol)
        {
            Contracts.Assert(HasCompatibleAlignment(mat));
            Contracts.Assert(HasCompatibleAlignment(src));
            Contracts.Assert(HasCompatibleAlignment(dst));

            fixed (float* pSrcStart = &src.Items[0])
            fixed (float* pDstStart = &dst.Items[0])
            fixed (float* pMatStart = &mat.Items[0])
            {
                float* psrc = GetAlignedBase(src, pSrcStart);
                float* pdst = GetAlignedBase(dst, pDstStart);
                float* pmat = GetAlignedBase(mat, pMatStart);

                float* pSrcEnd = psrc + ccol;
                float* pDstEnd = pdst + crow;
                float* pSrcCurrent = psrc;
                float* pMatCurrent = pmat;

                if (!add)
                {
                    Vector128<float> x01 = Sse.LoadAlignedVector128(pSrcCurrent);
                    // Replicate each 32-bit slot of x01 (ABCD) into its own register.
                    Vector128<float> x11 = Sse.Shuffle(x01, x01, 0x55); // B
                    Vector128<float> x21 = Sse.Shuffle(x01, x01, 0xAA); // C
                    Vector128<float> x31 = Sse.Shuffle(x01, x01, 0xFF); // D
                    x01 = Sse.Shuffle(x01, x01, 0x00); // A

                    pSrcCurrent += 4;

                    float* pDstCurrent = pdst;

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
                    // Replicate each 32-bit slot of x01 (ABCD) into its own register.
                    Vector128<float> x11 = Sse.Shuffle(x01, x01, 0x55); // B
                    Vector128<float> x21 = Sse.Shuffle(x01, x01, 0xAA); // C
                    Vector128<float> x31 = Sse.Shuffle(x01, x01, 0xFF); // D
                    x01 = Sse.Shuffle(x01, x01, 0x00); // A

                    float* pDstCurrent = pdst;

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
        public static unsafe void MatMulTranPA(bool add, AlignedArray mat, int[] rgposSrc, AlignedArray src,
                                        int posMin, int iposMin, int iposEnd, AlignedArray dst, int crow)
        {
            Contracts.Assert(HasCompatibleAlignment(mat));
            Contracts.Assert(HasCompatibleAlignment(src));
            Contracts.Assert(HasCompatibleAlignment(dst));

            fixed (float* pSrcStart = &src.Items[0])
            fixed (float* pDstStart = &dst.Items[0])
            fixed (float* pMatStart = &mat.Items[0])
            fixed (int* pposSrc = &rgposSrc[0])
            {
                float* psrc = GetAlignedBase(src, pSrcStart);
                float* pdst = GetAlignedBase(dst, pDstStart);
                float* pmat = GetAlignedBase(mat, pMatStart);

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

        // dst[i] += scale
        public static unsafe void AddScalarU(float scalar, Span<float> dst)
        {
            fixed (float* pdst = dst)
            {
                Vector128<float> scalarVector = Sse.SetAllVector128(scalar);
                int count = Math.DivRem(dst.Length, 4, out int remainder);
                float* pDstCurrent = pdst;

                for (int i = 0; i < count; i++)
                {
                    Vector128<float> dstVector = Sse.LoadVector128(pDstCurrent);
                    dstVector = Sse.Add(dstVector, scalarVector);
                    Sse.Store(pDstCurrent, dstVector);

                    pDstCurrent += 4;
                }

                for (int i = 0; i < remainder; i++)
                {
                    pDstCurrent[i] += scalar;
                }
            }
        }

        public static unsafe void ScaleU(float scale, Span<float> dst)
        {
            fixed (float* pdst = dst)
            {
                float* pDstCurrent = pdst;
                float* pEnd = pdst + dst.Length;

                Vector128<float> scaleVector = Sse.SetAllVector128(scale);

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

        public static unsafe void ScaleSrcU(float scale, Span<float> src, Span<float> dst)
        {
            fixed (float* psrc = src)
            fixed (float* pdst = dst)
            {
                float* pDstEnd = pdst + dst.Length;
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;

                Vector128<float> scaleVector = Sse.SetAllVector128(scale);

                while (pDstCurrent + 4 <= pDstEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    srcVector = Sse.Multiply(srcVector, scaleVector);
                    Sse.Store(pDstCurrent, srcVector);

                    pSrcCurrent += 4;
                    pDstCurrent += 4;
                }

                while (pDstCurrent < pDstEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    srcVector = Sse.MultiplyScalar(srcVector, scaleVector);
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

                Vector128<float> aVector = Sse.SetAllVector128(a);
                Vector128<float> bVector = Sse.SetAllVector128(b);

                while (pDstCurrent + 4 <= pDstEnd)
                {
                    Vector128<float> dstVector = Sse.LoadVector128(pDstCurrent);
                    dstVector = Sse.Add(dstVector, bVector);
                    dstVector = Sse.Multiply(dstVector, aVector);
                    Sse.Store(pDstCurrent, dstVector);

                    pDstCurrent += 4;
                }

                while (pDstCurrent < pDstEnd)
                {
                    Vector128<float> dstVector = Sse.LoadScalarVector128(pDstCurrent);
                    dstVector = Sse.AddScalar(dstVector, bVector);
                    dstVector = Sse.MultiplyScalar(dstVector, aVector);
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

                Vector128<float> scaleVector = Sse.SetAllVector128(scale);

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

                Vector128<float> scaleVector = Sse.SetAllVector128(scale);

                while (pResCurrent + 4 <= pResEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    Vector128<float> dstVector = Sse.LoadVector128(pDstCurrent);
                    srcVector = Sse.Multiply(srcVector, scaleVector);
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
                    srcVector = Sse.MultiplyScalar(srcVector, scaleVector);
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

                Vector128<float> scaleVector = Sse.SetAllVector128(scale);

                while (pIdxCurrent + 4 <= pEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    Vector128<float> dstVector = Load4(pDstCurrent, pIdxCurrent);

                    srcVector = Sse.Multiply(srcVector, scaleVector);
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

                while (pIdxCurrent + 4 <= pEnd)
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

        public static unsafe float SumU(Span<float> src)
        {
            fixed (float* psrc = src)
            {
                float* pSrcEnd = psrc + src.Length;
                float* pSrcCurrent = psrc;

                Vector128<float> result = Sse.SetZeroVector128();

                while (pSrcCurrent + 4 < pSrcEnd)
                {
                    result = Sse.Add(result, Sse.LoadVector128(pSrcCurrent));
                    pSrcCurrent += 4;
                }

                result = VectorSum128(in result);

                while (pSrcCurrent < pSrcEnd)
                {
                    result = Sse.AddScalar(result, Sse.LoadScalarVector128(pSrcCurrent));
                    pSrcCurrent++;
                }

                return Sse.ConvertToSingle(result);
            }
        }

        public static unsafe float SumSqU(Span<float> src)
        {
            fixed (float* psrc = src)
            {
                float* pSrcEnd = psrc + src.Length;
                float* pSrcCurrent = psrc;

                Vector128<float> result = Sse.SetZeroVector128();

                while (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    result = Sse.Add(result, Sse.Multiply(srcVector, srcVector));

                    pSrcCurrent += 4;
                }

                result = VectorSum128(in result);

                while (pSrcCurrent < pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    result = Sse.AddScalar(result, Sse.MultiplyScalar(srcVector, srcVector));

                    pSrcCurrent++;
                }

                return Sse.ConvertToSingle(result);
            }
        }

        public static unsafe float SumSqDiffU(float mean, Span<float> src)
        {
            fixed (float* psrc = src)
            {
                float* pSrcEnd = psrc + src.Length;
                float* pSrcCurrent = psrc;

                Vector128<float> result = Sse.SetZeroVector128();
                Vector128<float> meanVector = Sse.SetAllVector128(mean);

                while (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    srcVector = Sse.Subtract(srcVector, meanVector);
                    result = Sse.Add(result, Sse.Multiply(srcVector, srcVector));

                    pSrcCurrent += 4;
                }

                result = VectorSum128(in result);

                while (pSrcCurrent < pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    srcVector = Sse.SubtractScalar(srcVector, meanVector);
                    result = Sse.AddScalar(result, Sse.MultiplyScalar(srcVector, srcVector));

                    pSrcCurrent++;
                }

                return Sse.ConvertToSingle(result);
            }
        }

        public static unsafe float SumAbsU(Span<float> src)
        {
            fixed (float* psrc = src)
            {
                float* pSrcEnd = psrc + src.Length;
                float* pSrcCurrent = psrc;

                Vector128<float> result = Sse.SetZeroVector128();

                while (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    result = Sse.Add(result, Sse.And(srcVector, AbsMask128));

                    pSrcCurrent += 4;
                }

                result = VectorSum128(in result);

                while (pSrcCurrent < pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    result = Sse.AddScalar(result, Sse.And(srcVector, AbsMask128));

                    pSrcCurrent++;
                }

                return Sse.ConvertToSingle(result);
            }
        }

        public static unsafe float SumAbsDiffU(float mean, Span<float> src)
        {
            fixed (float* psrc = src)
            {
                float* pSrcEnd = psrc + src.Length;
                float* pSrcCurrent = psrc;

                Vector128<float> result = Sse.SetZeroVector128();
                Vector128<float> meanVector = Sse.SetAllVector128(mean);

                while (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    srcVector = Sse.Subtract(srcVector, meanVector);
                    result = Sse.Add(result, Sse.And(srcVector, AbsMask128));

                    pSrcCurrent += 4;
                }

                result = VectorSum128(in result);

                while (pSrcCurrent < pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    srcVector = Sse.SubtractScalar(srcVector, meanVector);
                    result = Sse.AddScalar(result, Sse.And(srcVector, AbsMask128));

                    pSrcCurrent++;
                }

                return Sse.ConvertToSingle(result);
            }
        }

        public static unsafe float MaxAbsU(Span<float> src)
        {
            fixed (float* psrc = src)
            {
                float* pSrcEnd = psrc + src.Length;
                float* pSrcCurrent = psrc;

                Vector128<float> result = Sse.SetZeroVector128();

                while (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    result = Sse.Max(result, Sse.And(srcVector, AbsMask128));

                    pSrcCurrent += 4;
                }

                result = VectorMax128(in result);

                while (pSrcCurrent < pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    result = Sse.MaxScalar(result, Sse.And(srcVector, AbsMask128));

                    pSrcCurrent++;
                }

                return Sse.ConvertToSingle(result);
            }
        }

        public static unsafe float MaxAbsDiffU(float mean, Span<float> src)
        {
            fixed (float* psrc = src)
            {
                float* pSrcEnd = psrc + src.Length;
                float* pSrcCurrent = psrc;

                Vector128<float> result = Sse.SetZeroVector128();
                Vector128<float> meanVector = Sse.SetAllVector128(mean);

                while (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    srcVector = Sse.Subtract(srcVector, meanVector);
                    result = Sse.Max(result, Sse.And(srcVector, AbsMask128));

                    pSrcCurrent += 4;
                }

                result = VectorMax128(in result);

                while (pSrcCurrent < pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    srcVector = Sse.SubtractScalar(srcVector, meanVector);
                    result = Sse.MaxScalar(result, Sse.And(srcVector, AbsMask128));

                    pSrcCurrent++;
                }

                return Sse.ConvertToSingle(result);
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

                Vector128<float> result = Sse.SetZeroVector128();

                while (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    Vector128<float> dstVector = Sse.LoadVector128(pDstCurrent);

                    result = Sse.Add(result, Sse.Multiply(srcVector, dstVector));

                    pSrcCurrent += 4;
                    pDstCurrent += 4;
                }

                result = VectorSum128(in result);

                while (pSrcCurrent < pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    Vector128<float> dstVector = Sse.LoadScalarVector128(pDstCurrent);

                    result = Sse.AddScalar(result, Sse.MultiplyScalar(srcVector, dstVector));

                    pSrcCurrent++;
                    pDstCurrent++;
                }

                return Sse.ConvertToSingle(result);
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

                Vector128<float> result = Sse.SetZeroVector128();

                while (pIdxCurrent + 4 <= pIdxEnd)
                {
                    Vector128<float> srcVector = Load4(pSrcCurrent, pIdxCurrent);
                    Vector128<float> dstVector = Sse.LoadVector128(pDstCurrent);

                    result = Sse.Add(result, Sse.Multiply(srcVector, dstVector));

                    pIdxCurrent += 4;
                    pDstCurrent += 4;
                }

                result = VectorSum128(in result);

                while (pIdxCurrent < pIdxEnd)
                {
                    Vector128<float> srcVector = Load1(pSrcCurrent, pIdxCurrent);
                    Vector128<float> dstVector = Sse.LoadScalarVector128(pDstCurrent);

                    result = Sse.AddScalar(result, Sse.MultiplyScalar(srcVector, dstVector));

                    pIdxCurrent++;
                    pDstCurrent++;
                }

                return Sse.ConvertToSingle(result);
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

                Vector128<float> sqDistanceVector = Sse.SetZeroVector128();

                while (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> distanceVector = Sse.Subtract(Sse.LoadVector128(pSrcCurrent),
                                                                    Sse.LoadVector128(pDstCurrent));
                    sqDistanceVector = Sse.Add(sqDistanceVector,
                                                Sse.Multiply(distanceVector, distanceVector));

                    pSrcCurrent += 4;
                    pDstCurrent += 4;
                }

                sqDistanceVector = VectorSum128(in sqDistanceVector);

                float norm = Sse.ConvertToSingle(sqDistanceVector);
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

                Vector128<float> xPrimal = Sse.SetAllVector128(primalUpdate);

                Vector128<float> signMask = Sse.SetAllVector128(-0.0f); // 0x8000 0000
                Vector128<float> xThreshold = Sse.SetAllVector128(threshold);

                while (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> xSrc = Sse.LoadVector128(pSrcCurrent);

                    Vector128<float> xDst1 = Sse.LoadVector128(pDst1Current);
                    xDst1 = Sse.Add(xDst1, Sse.Multiply(xSrc, xPrimal));
                    Vector128<float> xDst2 = GetNewDst128(xDst1, xThreshold);

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

                Vector128<float> xPrimal = Sse.SetAllVector128(primalUpdate);

                Vector128<float> signMask = Sse.SetAllVector128(-0.0f); // 0x8000 0000
                Vector128<float> xThreshold = Sse.SetAllVector128(threshold);

                while (pIdxCurrent + 4 <= pIdxEnd)
                {
                    Vector128<float> xSrc = Sse.LoadVector128(pSrcCurrent);

                    Vector128<float> xDst1 = Load4(pdst1, pIdxCurrent);
                    xDst1 = Sse.Add(xDst1, Sse.Multiply(xSrc, xPrimal));
                    Vector128<float> xDst2 = GetNewDst128(xDst1, xThreshold);

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
