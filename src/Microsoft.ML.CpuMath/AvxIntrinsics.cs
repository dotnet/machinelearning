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
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using nuint = System.UInt64;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    internal static class AvxIntrinsics
    {
        public static readonly uint[] LeadingAlignmentMask = new uint[64]
        {
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000,
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000,
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000,
        };

        public static readonly uint[] TrailingAlignmentMask = new uint[64]
        {
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
            0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
            0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
            0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
        };

        private static readonly Vector256<float> _absMask256 = Avx.StaticCast<int, float>(Avx.SetAllVector256(0x7FFFFFFF));

        private const int Vector256Alignment = 32;

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static bool HasCompatibleAlignment(AlignedArray alignedArray)
        {
            Contracts.AssertValue(alignedArray);
            Contracts.Assert(alignedArray.Size > 0);
            return (alignedArray.CbAlign % Vector256Alignment) == 0;
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static unsafe float* GetAlignedBase(AlignedArray alignedArray, float* unalignedBase)
        {
            Contracts.AssertValue(alignedArray);
            float* alignedBase = unalignedBase + alignedArray.GetBase((long)unalignedBase);
            Contracts.Assert(((long)alignedBase % Vector256Alignment) == 0);
            return alignedBase;
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> GetHigh(in Vector256<float> x)
            => Avx.ExtractVector128(x, 1);

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<float> Load8(float* src, int* idx)
        {
            if (Avx2.IsSupported)
            {
                Vector256<int> idx256 = Avx.LoadVector256(idx);
                return Avx2.GatherVector256(src, idx256, 4);
            }
            else
            {
                return Avx.SetVector256(src[idx[7]], src[idx[6]], src[idx[5]], src[idx[4]], src[idx[3]], src[idx[2]], src[idx[1]], src[idx[0]]);
            }
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static unsafe void Store8(in Vector256<float> x, float* dst, int* idx)
        {
            Vector128<float> tmp = Avx.GetLowerHalf(x);
            Sse.StoreScalar(dst + idx[0], tmp);
            tmp = SseIntrinsics.Rotate(in tmp);
            Sse.StoreScalar(dst + idx[1], tmp);
            tmp = SseIntrinsics.Rotate(in tmp);
            Sse.StoreScalar(dst + idx[2], tmp);
            tmp = SseIntrinsics.Rotate(in tmp);
            Sse.StoreScalar(dst + idx[3], tmp);
            tmp = GetHigh(in x);
            Sse.StoreScalar(dst + idx[4], tmp);
            tmp = SseIntrinsics.Rotate(in tmp);
            Sse.StoreScalar(dst + idx[5], tmp);
            tmp = SseIntrinsics.Rotate(in tmp);
            Sse.StoreScalar(dst + idx[6], tmp);
            tmp = SseIntrinsics.Rotate(in tmp);
            Sse.StoreScalar(dst + idx[7], tmp);
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> VectorSum256(in Vector256<float> vector)
        {
            Vector256<float> partialSum = Avx.HorizontalAdd(vector, vector);
            return Avx.HorizontalAdd(partialSum, partialSum);
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> VectorMax256(in Vector256<float> vector)
        {
            // The control byte shuffles the eight 32-bit floats of partialMax: ABCD|EFGH -> BADC|FEHG.
            Vector256<float> x1 = Avx.Shuffle(vector, vector, 0xB1);

            // Performs element-wise maximum operation: The 1st, 3rd, 5th, and 7th 32-bit slots become
            // max(A, B), max(C, D), max(E, F), and max(G, H).
            Vector256<float> partialMax = Avx.Max(vector, x1);

            // The control byte shuffles the eight 32-bit floats of partialMax: ABCD|EFGH -> CAAA|GEEE.
            x1 = Avx.Shuffle(partialMax, partialMax, 0x02);

            // Performs element-wise maximum operation: The 1st and 5th 32-bit slots become
            // max(max(A, B), max(C, D)) = max(A, B, C, D) and
            // max(max(E, F), max(G, H)) = max(E, F, G, H).
            return Avx.Max(partialMax, x1);
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> GetNewDst256(in Vector256<float> xDst1, in Vector256<float> xThreshold)
        {
            Vector256<float> signMask = Avx.SetAllVector256(-0.0f); // 0x8000 0000
            Vector256<float> xSign = Avx.And(xDst1, signMask); // result = 0x8000 0000 if xDst1 is negative or 0x0000 0000 otherwise
            Vector256<float> xDst1Abs = Avx.Xor(xDst1, xSign);
            Vector256<float> xCond = Avx.Compare(xDst1Abs, xThreshold, FloatComparisonMode.GreaterThanOrderedNonSignaling); // result = 0xFFFF FFFF if true
            Vector256<float> x2 = Avx.Xor(xSign, xThreshold); // -xThreshold if xDst1 is negative and +xThreshold otherwise
            return Avx.And(Avx.Subtract(xDst1, x2), xCond);
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<float> MultiplyAdd(float* psrc1, Vector256<float> src2, Vector256<float> src3)
        {
            if (Fma.IsSupported)
            {
                return Fma.MultiplyAdd(Avx.LoadVector256(psrc1), src2, src3);
            }
            else
            {
                return Avx.Add(Avx.Multiply(Avx.LoadVector256(psrc1), src2), src3);
            }
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> MultiplyAdd(Vector256<float> src1, Vector256<float> src2, Vector256<float> src3)
        {
            if (Fma.IsSupported)
            {
                return Fma.MultiplyAdd(src1, src2, src3);
            }
            else
            {
                return Avx.Add(Avx.Multiply(src1, src2), src3);
            }
        }

        // Multiply matrix times vector into vector.
        public static unsafe void MatMulX(AlignedArray mat, AlignedArray src, AlignedArray dst, int crow, int ccol)
        {
            Contracts.Assert(crow % 4 == 0);
            Contracts.Assert(ccol % 4 == 0);

            MatMulX(mat.Items, src.Items, dst.Items, crow, ccol);
        }

        public static unsafe void MatMulX(float[] mat, float[] src, float[] dst, int crow, int ccol)
        {
            fixed (float* psrc = &src[0])
            fixed (float* pdst = &dst[0])
            fixed (float* pmat = &mat[0])
            fixed (uint* pLeadingAlignmentMask = &LeadingAlignmentMask[0])
            fixed (uint* pTrailingAlignmentMask = &TrailingAlignmentMask[0])
            {
                float* pSrcEnd = psrc + ccol;
                float* pDstEnd = pdst + crow;
                float* pDstCurrent = pdst;
                float* pMatCurrent = pmat;

                while (pDstCurrent < pDstEnd)
                {
                    Vector256<float> res0 = Avx.SetZeroVector256<float>();
                    Vector256<float> res1 = Avx.SetZeroVector256<float>();
                    Vector256<float> res2 = Avx.SetZeroVector256<float>();
                    Vector256<float> res3 = Avx.SetZeroVector256<float>();

                    int length = ccol;
                    float* pSrcCurrent = psrc;

                    nuint address = (nuint)(pMatCurrent);
                    int misalignment = (int)(address % 32);

                    int remainder = 0;
                    if ((misalignment & 3) != 0)
                    {
                        // Handles cases where the data is not 32-bit aligned and we can't ever use aligned operations
                        while (pSrcCurrent < pSrcEnd)
                        {
                            Vector256<float> vector = Avx.LoadVector256(pSrcCurrent);

                            float* pMatTemp = pMatCurrent;
                            res0 = MultiplyAdd(pMatTemp, vector, res0);
                            res1 = MultiplyAdd(pMatTemp += ccol, vector, res1);
                            res2 = MultiplyAdd(pMatTemp += ccol, vector, res2);
                            res3 = MultiplyAdd(pMatTemp += ccol, vector, res3);

                            pSrcCurrent += 8;
                            pMatCurrent += 8;
                        }
                    }
                    else
                    {
                        if (misalignment != 0)
                        {
                            // Handle cases where the data is not 256-bit aligned by doing an unaligned read and then
                            // masking any elements that will be included in the first aligned read
                            misalignment >>= 2;
                            misalignment = 8 - misalignment;

                            Vector256<float> mask = Avx.LoadVector256(((float*)(pLeadingAlignmentMask)) + (misalignment * 8));

                            // We only align pMat since it has significantly more reads.
                            float* pMatTemp = pMatCurrent;
                            Vector256<float> x01 = Avx.And(mask, Avx.LoadVector256(pMatTemp));
                            Vector256<float> x11 = Avx.And(mask, Avx.LoadVector256(pMatTemp += ccol));
                            Vector256<float> x21 = Avx.And(mask, Avx.LoadVector256(pMatTemp += ccol));
                            Vector256<float> x31 = Avx.And(mask, Avx.LoadVector256(pMatTemp += ccol));
                            Vector256<float> vector = Avx.And(mask, Avx.LoadVector256(pSrcCurrent));

                            res0 = Avx.Multiply(x01, vector);
                            res1 = Avx.Multiply(x11, vector);
                            res2 = Avx.Multiply(x21, vector);
                            res3 = Avx.Multiply(x31, vector);

                            pMatCurrent += misalignment;
                            pSrcCurrent += misalignment;
                            length -= misalignment;
                        }

                        if (length > 7)
                        {
                            remainder = length % 8;
                            while (pSrcCurrent + 8 <= pSrcEnd)
                            {
                                Vector256<float> vector = Avx.LoadVector256(pSrcCurrent);

                                float* pMatTemp = pMatCurrent;
                                res0 = MultiplyAdd(pMatTemp, vector, res0);
                                res1 = MultiplyAdd(pMatTemp += ccol, vector, res1);
                                res2 = MultiplyAdd(pMatTemp += ccol, vector, res2);
                                res3 = MultiplyAdd(pMatTemp += ccol, vector, res3);

                                pSrcCurrent += 8;
                                pMatCurrent += 8;
                            }
                        }
                        else
                        {
                            remainder = length;
                        }

                        if (remainder != 0)
                        {
                            pMatCurrent -= (8 - remainder);
                            pSrcCurrent -= (8 - remainder);

                            Vector256<float> mask = Avx.LoadVector256(((float*)(pTrailingAlignmentMask)) + (remainder * 8));

                            float* pMatTemp = pMatCurrent;
                            Vector256<float> x01 = Avx.And(mask, Avx.LoadVector256(pMatTemp));
                            Vector256<float> x11 = Avx.And(mask, Avx.LoadVector256(pMatTemp += ccol));
                            Vector256<float> x21 = Avx.And(mask, Avx.LoadVector256(pMatTemp += ccol));
                            Vector256<float> x31 = Avx.And(mask, Avx.LoadVector256(pMatTemp += ccol));
                            Vector256<float> vector = Avx.And(mask, Avx.LoadVector256(pSrcCurrent));

                            res0 = MultiplyAdd(x01, vector, res0);
                            res1 = MultiplyAdd(x11, vector, res1);
                            res2 = MultiplyAdd(x21, vector, res2);
                            res3 = MultiplyAdd(x31, vector, res3);

                            pMatCurrent += 8;
                            pSrcCurrent += 8;
                        }
                    }

                    // Add up the entries of each, with the 4 results in res0
                    res0 = Avx.HorizontalAdd(res0, res1);
                    res2 = Avx.HorizontalAdd(res2, res3);
                    res0 = Avx.HorizontalAdd(res0, res2);

                    Vector128<float> sum = Sse.Add(Avx.GetLowerHalf(res0), GetHigh(in res0));
                    Sse.Store(pDstCurrent, sum);

                    pDstCurrent += 4;
                    pMatCurrent += 3 * ccol;
                }
            }
        }

        // Partial sparse source vector.
        public static unsafe void MatMulPX(AlignedArray mat, int[] rgposSrc, AlignedArray src,
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
                    Vector256<float> result = Avx.SetZeroVector256<float>();

                    int* ppos = pposMin;

                    while (ppos < pposEnd)
                    {
                        int col1 = *ppos;
                        int col2 = col1 + 4 * ccol;
                        Vector256<float> x1 = Avx.SetVector256(pm3[col2], pm2[col2], pm1[col2], pm0[col2],
                                                                pm3[col1], pm2[col1], pm1[col1], pm0[col1]);
                        Vector256<float> x2 = Avx.SetAllVector256(pSrcCurrent[col1]);
                        result = MultiplyAdd(x2, x1, result);

                        ppos++;
                    }

                    Avx.StoreAligned(pDstCurrent, result);
                    pDstCurrent += 8;
                    pm0 += 8 * ccol;
                }
            }
        }

        public static unsafe void MatMulTranX(AlignedArray mat, AlignedArray src, AlignedArray dst, int crow, int ccol)
        {
            Contracts.Assert(crow % 4 == 0);
            Contracts.Assert(ccol % 4 == 0);

            MatMulTranX(mat.Items, src.Items, dst.Items, crow, ccol);
        }

        public static unsafe void MatMulTranX(float[] mat, float[] src, float[] dst, int crow, int ccol)
        {
            fixed (float* psrc = &src[0])
            fixed (float* pdst = &dst[0])
            fixed (float* pmat = &mat[0])
            fixed (uint* pLeadingAlignmentMask = &LeadingAlignmentMask[0])
            fixed (uint* pTrailingAlignmentMask = &TrailingAlignmentMask[0])
            {
                float* pSrcEnd = psrc + ccol;
                float* pDstEnd = pdst + crow;
                float* pSrcCurrent = psrc;
                float* pMatCurrent = pmat;

                // The reason behind adding the if condtion instead of boolean flag
                // is to avoid branching in codegen.
                if (pSrcCurrent < pSrcEnd)
                {
                    Vector128<float> h01 = Sse.LoadVector128(pSrcCurrent);
                    // Replicate each slot of h01 (ABCD) into its own register.
                    Vector128<float> h11 = Avx.Permute(h01, 0x55); // B
                    Vector128<float> h21 = Avx.Permute(h01, 0xAA); // C
                    Vector128<float> h31 = Avx.Permute(h01, 0xFF); // D
                    h01 = Avx.Permute(h01, 0x00); // A

                    Vector256<float> x01 = Avx.SetHighLow(h01, h01);
                    Vector256<float> x11 = Avx.SetHighLow(h11, h11);
                    Vector256<float> x21 = Avx.SetHighLow(h21, h21);
                    Vector256<float> x31 = Avx.SetHighLow(h31, h31);

                    int length = crow;
                    float* pDstCurrent = pdst;

                    nuint address = (nuint)(pMatCurrent);
                    int misalignment = (int)(address % 32);

                    if ((misalignment & 3) != 0)
                    {
                        while (pDstCurrent < pDstEnd)
                        {
                            float* pMatTemp = pMatCurrent;
                            Vector256<float> x02 = Avx.Multiply(x01, Avx.LoadVector256(pMatTemp));
                            Vector256<float> x12 = Avx.Multiply(x11, Avx.LoadVector256(pMatTemp += crow));
                            Vector256<float> x22 = Avx.Multiply(x21, Avx.LoadVector256(pMatTemp += crow));
                            Vector256<float> x32 = Avx.Multiply(x31, Avx.LoadVector256(pMatTemp += crow));

                            x02 = Avx.Add(x02, x12);
                            x22 = Avx.Add(x22, x32);
                            x02 = Avx.Add(x02, x22);

                            Avx.Store(pDstCurrent, x02);
                            pDstCurrent += 8;
                            pMatCurrent += 8;
                        }
                    }
                    else
                    {
                        int remainder = 0;
                        if (misalignment != 0)
                        {
                            // Handle cases where the data is not 256-bit aligned by doing an unaligned read and then
                            // masking any elements that will be included in the first aligned read
                            misalignment >>= 2;
                            misalignment = 8 - misalignment;

                            Vector256<float> leadingMask = Avx.LoadVector256(((float*)(pLeadingAlignmentMask)) + (misalignment * 8));

                            // We only align pMat since it has significantly more reads.
                            float* pMatTemp = pMatCurrent;
                            Vector256<float> x02 = Avx.And(leadingMask, Avx.LoadVector256(pMatTemp));
                            Vector256<float> x12 = Avx.And(leadingMask, Avx.LoadVector256(pMatTemp += crow));
                            Vector256<float> x22 = Avx.And(leadingMask, Avx.LoadVector256(pMatTemp += crow));
                            Vector256<float> x32 = Avx.And(leadingMask, Avx.LoadVector256(pMatTemp += crow));

                            x02 = Avx.Multiply(x01, x02);
                            x12 = Avx.Multiply(x11, x12);
                            x22 = Avx.Multiply(x21, x22);
                            x32 = Avx.Multiply(x31, x32);

                            x02 = Avx.Add(x02, x12);
                            x22 = Avx.Add(x22, x32);
                            x02 = Avx.Add(x02, x22);

                            Vector256<float> trailingMask = Avx.LoadVector256(((float*)(pTrailingAlignmentMask)) + ((8 - misalignment) * 8));
                            Vector256<float> x3 = Avx.LoadVector256(pDstCurrent);
                            x02 = Avx.Or(x02, Avx.And(x3, trailingMask));

                            Avx.Store(pDstCurrent, x02);
                            pMatCurrent += misalignment;
                            pDstCurrent += misalignment;
                            length -= misalignment;
                        }
                        if (length > 7)
                        {
                            remainder = length % 8;
                            while (pDstCurrent + 8 <= pDstEnd)
                            {
                                float* pMatTemp = pMatCurrent;

                                Vector256<float> x02 = Avx.Multiply(x01, Avx.LoadVector256(pMatTemp));
                                Vector256<float> x12 = Avx.Multiply(x11, Avx.LoadVector256(pMatTemp += crow));
                                Vector256<float> x22 = Avx.Multiply(x21, Avx.LoadVector256(pMatTemp += crow));
                                Vector256<float> x32 = Avx.Multiply(x31, Avx.LoadVector256(pMatTemp += crow));

                                x02 = Avx.Add(x02, x12);
                                x22 = Avx.Add(x22, x32);
                                x02 = Avx.Add(x02, x22);

                                Avx.Store(pDstCurrent, x02);
                                pDstCurrent += 8;
                                pMatCurrent += 8;
                            }
                        }
                        else
                        {
                            remainder = length;
                        }

                        if (remainder != 0)
                        {
                            pMatCurrent -= (8 - remainder);
                            pDstCurrent -= (8 - remainder);
                            Vector256<float> trailingMask = Avx.LoadVector256(((float*)(pTrailingAlignmentMask)) + (remainder * 8));

                            float* pMatTemp = pMatCurrent;
                            Vector256<float> x02 = Avx.And(trailingMask, Avx.LoadVector256(pMatTemp));
                            Vector256<float> x12 = Avx.And(trailingMask, Avx.LoadVector256(pMatTemp += crow));
                            Vector256<float> x22 = Avx.And(trailingMask, Avx.LoadVector256(pMatTemp += crow));
                            Vector256<float> x32 = Avx.And(trailingMask, Avx.LoadVector256(pMatTemp += crow));

                            x02 = Avx.Multiply(x01, x02);
                            x12 = Avx.Multiply(x11, x12);
                            x22 = Avx.Multiply(x21, x22);
                            x32 = Avx.Multiply(x31, x32);

                            x02 = Avx.Add(x02, x12);
                            x22 = Avx.Add(x22, x32);
                            x02 = Avx.Add(x02, x22);

                            Vector256<float> leadingMask = Avx.LoadVector256(((float*)(pLeadingAlignmentMask)) + ((8 - remainder) * 8));
                            Vector256<float> x3 = Avx.LoadVector256(pDstCurrent);
                            x02 = Avx.Or(x02, Avx.And(x3, leadingMask));

                            Avx.Store(pDstCurrent, x02);
                            pDstCurrent += 8;
                            pMatCurrent += 8;
                        }
                    }

                    pMatCurrent += 3 * crow;
                    pSrcCurrent += 4;
                }

                // We do 4-way unrolling
                while (pSrcCurrent < pSrcEnd)
                {
                    Vector128<float> h01 = Sse.LoadVector128(pSrcCurrent);
                    // Replicate each slot of h01 (ABCD) into its own register.
                    Vector128<float> h11 = Avx.Permute(h01, 0x55); // B
                    Vector128<float> h21 = Avx.Permute(h01, 0xAA); // C
                    Vector128<float> h31 = Avx.Permute(h01, 0xFF); // D
                    h01 = Avx.Permute(h01, 0x00); // A

                    Vector256<float> x01 = Avx.SetHighLow(h01, h01);
                    Vector256<float> x11 = Avx.SetHighLow(h11, h11);
                    Vector256<float> x21 = Avx.SetHighLow(h21, h21);
                    Vector256<float> x31 = Avx.SetHighLow(h31, h31);

                    int length = crow;
                    float* pDstCurrent = pdst;

                    nuint address = (nuint)(pMatCurrent);
                    int misalignment = (int)(address % 32);

                    if ((misalignment & 3) != 0)
                    {
                        while (pDstCurrent < pDstEnd)
                        {
                            float* pMatTemp = pMatCurrent;
                            Vector256<float> x02 = Avx.Multiply(x01, Avx.LoadVector256(pMatTemp));
                            Vector256<float> x12 = Avx.Multiply(x11, Avx.LoadVector256(pMatTemp += crow));
                            Vector256<float> x22 = Avx.Multiply(x21, Avx.LoadVector256(pMatTemp += crow));
                            Vector256<float> x32 = Avx.Multiply(x31, Avx.LoadVector256(pMatTemp += crow));

                            x02 = Avx.Add(x02, x12);
                            x22 = Avx.Add(x22, x32);
                            x02 = Avx.Add(x02, x22);

                            x02 = Avx.Add(x02, Avx.LoadVector256(pDstCurrent));

                            Avx.Store(pDstCurrent, x02);
                            pDstCurrent += 8;
                            pMatCurrent += 8;
                        }
                    }
                    else
                    {
                        int remainder = 0;
                        if (misalignment != 0)
                        {
                            // Handle cases where the data is not 256-bit aligned by doing an unaligned read and then
                            // masking any elements that will be included in the first aligned read
                            misalignment >>= 2;
                            misalignment = 8 - misalignment;

                            Vector256<float> leadingMask = Avx.LoadVector256(((float*)(pLeadingAlignmentMask)) + (misalignment * 8));

                            // We only align pMat since it has significantly more reads.
                            float* pMatTemp = pMatCurrent;
                            Vector256<float> x02 = Avx.And(leadingMask, Avx.LoadVector256(pMatTemp));
                            Vector256<float> x12 = Avx.And(leadingMask, Avx.LoadVector256(pMatTemp += crow));
                            Vector256<float> x22 = Avx.And(leadingMask, Avx.LoadVector256(pMatTemp += crow));
                            Vector256<float> x32 = Avx.And(leadingMask, Avx.LoadVector256(pMatTemp += crow));

                            x02 = Avx.Multiply(x01, x02);
                            x12 = Avx.Multiply(x11, x12);
                            x22 = Avx.Multiply(x21, x22);
                            x32 = Avx.Multiply(x31, x32);

                            x02 = Avx.Add(x02, x12);
                            x22 = Avx.Add(x22, x32);
                            x02 = Avx.Add(x02, x22);

                            Vector256<float> trailingMask = Avx.LoadVector256(((float*)(pTrailingAlignmentMask)) + ((8 - misalignment) * 8));
                            Vector256<float> x3 = Avx.LoadVector256(pDstCurrent);
                            x02 = Avx.Or(x02, Avx.And(x3, trailingMask));

                            x02 = Avx.Add(x02, Avx.And(x3, leadingMask));

                            Avx.Store(pDstCurrent, x02);
                            pMatCurrent += misalignment;
                            pDstCurrent += misalignment;
                            length -= misalignment;
                        }
                        if (length > 7)
                        {
                            remainder = length % 8;
                            while (pDstCurrent + 8 <= pDstEnd)
                            {
                                float* pMatTemp = pMatCurrent;

                                Vector256<float> x02 = Avx.Multiply(x01, Avx.LoadVector256(pMatTemp));
                                Vector256<float> x12 = Avx.Multiply(x11, Avx.LoadVector256(pMatTemp += crow));
                                Vector256<float> x22 = Avx.Multiply(x21, Avx.LoadVector256(pMatTemp += crow));
                                Vector256<float> x32 = Avx.Multiply(x31, Avx.LoadVector256(pMatTemp += crow));

                                x02 = Avx.Add(x02, x12);
                                x22 = Avx.Add(x22, x32);
                                x02 = Avx.Add(x02, x22);

                                x02 = Avx.Add(x02, Avx.LoadVector256(pDstCurrent));

                                Avx.Store(pDstCurrent, x02);
                                pDstCurrent += 8;
                                pMatCurrent += 8;
                            }
                        }
                        else
                        {
                            remainder = length;
                        }

                        if (remainder != 0)
                        {
                            pMatCurrent -= (8 - remainder);
                            pDstCurrent -= (8 - remainder);
                            Vector256<float> trailingMask = Avx.LoadVector256(((float*)(pTrailingAlignmentMask)) + (remainder * 8));

                            float* pMatTemp = pMatCurrent;
                            Vector256<float> x02 = Avx.And(trailingMask, Avx.LoadVector256(pMatTemp));
                            Vector256<float> x12 = Avx.And(trailingMask, Avx.LoadVector256(pMatTemp += crow));
                            Vector256<float> x22 = Avx.And(trailingMask, Avx.LoadVector256(pMatTemp += crow));
                            Vector256<float> x32 = Avx.And(trailingMask, Avx.LoadVector256(pMatTemp += crow));

                            x02 = Avx.Multiply(x01, x02);
                            x12 = Avx.Multiply(x11, x12);
                            x22 = Avx.Multiply(x21, x22);
                            x32 = Avx.Multiply(x31, x32);

                            x02 = Avx.Add(x02, x12);
                            x22 = Avx.Add(x22, x32);
                            x02 = Avx.Add(x02, x22);

                            Vector256<float> leadingMask = Avx.LoadVector256(((float*)(pLeadingAlignmentMask)) + ((8 - remainder) * 8));
                            Vector256<float> x3 = Avx.LoadVector256(pDstCurrent);
                            x02 = Avx.Or(x02, Avx.And(x3, leadingMask));

                            x02 = Avx.Add(x02, Avx.And(x3, trailingMask));

                            Avx.Store(pDstCurrent, x02);
                            pDstCurrent += 8;
                            pMatCurrent += 8;
                        }
                    }

                    pMatCurrent += 3 * crow;
                    pSrcCurrent += 4;
                }
            }
        }

        // dst[i] += scale
        public static unsafe void AddScalarU(float scalar, Span<float> dst)
        {
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
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

        public static unsafe void Scale(float scale, Span<float> dst)
        {
            fixed (uint* pLeadingAlignmentMask = &LeadingAlignmentMask[0])
            fixed (uint* pTrailingAlignmentMask = &TrailingAlignmentMask[0])
            fixed (float* pd = &MemoryMarshal.GetReference(dst))
            {
                float* pDstCurrent = pd;
                int length = dst.Length;
                Vector256<float> scaleVector256 = Avx.SetAllVector256(scale);

                if (length < 8)
                {
                    switch (length)
                    {
                        case 7: dst[6] *= scale; goto case 6;
                        case 6: dst[5] *= scale; goto case 5;
                        case 5: dst[4] *= scale; goto case 4;
                        case 4: dst[3] *= scale; goto case 3;
                        case 3: dst[2] *= scale; goto case 2;
                        case 2: dst[1] *= scale; goto case 1;
                        case 1: dst[0] *= scale; break;
                    }
                    return;
                }

                nuint address = (nuint)(pd);
                int misalignment = (int)(address % 32);
                int remainder = 0;

                if ((misalignment & 3) != 0)
                {
                    // Handles cases where the data is not 32-bit aligned and we can't ever use aligned operations
                    remainder = length % 8;

                    for (float* pEnd = pd + (length - remainder); pDstCurrent < pEnd; pDstCurrent += 8)
                    {
                        Vector256<float> temp = Avx.LoadVector256(pDstCurrent);
                        temp = Avx.Multiply(scaleVector256, temp);
                        Avx.Store(pDstCurrent, temp);
                    }
                }
                else
                {
                    if (misalignment != 0)
                    {
                        // Handle cases where the data is not 256-bit aligned by doing an unaligned read and then
                        // masking any elements that will be included in the first aligned read

                        misalignment >>= 2;
                        misalignment = 8 - misalignment;

                        Vector256<float> result = Avx.LoadVector256(pDstCurrent);

                        Vector256<float> leadingMask = Avx.LoadVector256(((float*)(pLeadingAlignmentMask)) + (misalignment * 8));
                        Vector256<float> trailingMask = Avx.LoadVector256(((float*)(pTrailingAlignmentMask)) + ((8 - misalignment) * 8));

                        Vector256<float> temp = Avx.And(result, leadingMask);
                        result = Avx.And(result, trailingMask);

                        temp = Avx.Multiply(scaleVector256, temp);
                        result = Avx.Or(temp, result);

                        Avx.Store(pDstCurrent, result);

                        pDstCurrent += misalignment;
                        length -= misalignment;
                    }

                    if (length > 7)
                    {
                        // Handle all the 256-bit blocks that we can now that we have offset to an aligned address

                        remainder = length % 8;

                        for (float* pEnd = pDstCurrent + (length - remainder); pDstCurrent < pEnd; pDstCurrent += 8)
                        {
                            // The JIT will only fold away unaligned loads due to the semantics behind
                            // the VEX-encoding of the memory operand for `ins xmm, xmm, [mem]`. Since
                            // modern hardware has unaligned loads that are as fast as aligned loads,
                            // when it doesn't cross a cache-line/page boundary, we will just assert
                            // that the alignment is correct and allow for the more-efficient codegen.

                            Contracts.Assert(((nuint)(pDstCurrent) % 32) == 0);
                            Vector256<float> temp = Avx.LoadVector256(pDstCurrent);
                            temp = Avx.Multiply(scaleVector256, temp);
                            Avx.Store(pDstCurrent, temp);
                        }
                    }
                    else
                    {
                        // Handle the "worst-case" scenario, which is when we have 8-16 elements and the input is not
                        // 256-bit aligned. This means we can't do any aligned loads and will just end up doing two
                        // unaligned loads where we mask the input each time.
                        remainder = length;
                    }
                }

                if (remainder != 0)
                {
                    // Handle any trailing elements that don't fit into a 128-bit block by moving back so that the next
                    // unaligned load will read to the end of the array and then mask out any elements already processed

                    pDstCurrent -= (8 - remainder);

                    Vector256<float> result = Avx.LoadVector256(pDstCurrent);

                    Vector256<float> trailingMask = Avx.LoadVector256(((float*)(pTrailingAlignmentMask)) + (remainder * 8));
                    Vector256<float> leadingMask = Avx.LoadVector256(((float*)(pLeadingAlignmentMask)) + ((8 - remainder) * 8));

                    Vector256<float> temp = Avx.And(result, trailingMask);
                    result = Avx.And(result, leadingMask);

                    temp = Avx.Multiply(scaleVector256, temp);
                    temp = Avx.Or(temp, result);

                    Avx.Store(pDstCurrent, temp);
                }
            }
        }

        public static unsafe void ScaleSrcU(float scale, ReadOnlySpan<float> src, Span<float> dst, int count)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
            {
                float* pDstEnd = pdst + count;
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
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
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

        public static unsafe void AddScaleU(float scale, ReadOnlySpan<float> src, Span<float> dst, int count)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
            {
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                float* pEnd = pdst + count;

                Vector256<float> scaleVector256 = Avx.SetAllVector256(scale);

                while (pDstCurrent + 8 <= pEnd)
                {
                    Vector256<float> dstVector = Avx.LoadVector256(pDstCurrent);

                    dstVector = MultiplyAdd(pSrcCurrent, scaleVector256, dstVector);
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

        public static unsafe void AddScaleCopyU(float scale, ReadOnlySpan<float> src, ReadOnlySpan<float> dst, Span<float> result, int count)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
            fixed (float* pres = &MemoryMarshal.GetReference(result))
            {
                float* pResEnd = pres + count;
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                float* pResCurrent = pres;

                Vector256<float> scaleVector256 = Avx.SetAllVector256(scale);

                while (pResCurrent + 8 <= pResEnd)
                {
                    Vector256<float> dstVector = Avx.LoadVector256(pDstCurrent);
                    dstVector = MultiplyAdd(pSrcCurrent, scaleVector256, dstVector);
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

        public static unsafe void AddScaleSU(float scale, ReadOnlySpan<float> src, ReadOnlySpan<int> idx, Span<float> dst, int count)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            fixed (int* pidx = &MemoryMarshal.GetReference(idx))
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
            {
                float* pSrcCurrent = psrc;
                int* pIdxCurrent = pidx;
                float* pDstCurrent = pdst;
                int* pEnd = pidx + count;

                Vector256<float> scaleVector256 = Avx.SetAllVector256(scale);

                while (pIdxCurrent + 8 <= pEnd)
                {
                    Vector256<float> dstVector = Load8(pDstCurrent, pIdxCurrent);
                    dstVector = MultiplyAdd(pSrcCurrent, scaleVector256, dstVector);
                    Store8(in dstVector, pDstCurrent, pIdxCurrent);

                    pIdxCurrent += 8;
                    pSrcCurrent += 8;
                }

                Vector128<float> scaleVector128 = Sse.SetAllVector128(scale);

                if (pIdxCurrent + 4 <= pEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    Vector128<float> dstVector = SseIntrinsics.Load4(pDstCurrent, pIdxCurrent);

                    srcVector = Sse.Multiply(srcVector, scaleVector128);
                    dstVector = Sse.Add(dstVector, srcVector);
                    SseIntrinsics.Store4(in dstVector, pDstCurrent, pIdxCurrent);

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

        public static unsafe void AddU(ReadOnlySpan<float> src, Span<float> dst, int count)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
            {
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                float* pEnd = psrc + count;

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

        public static unsafe void AddSU(ReadOnlySpan<float> src, ReadOnlySpan<int> idx, Span<float> dst, int count)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            fixed (int* pidx = &MemoryMarshal.GetReference(idx))
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
            {
                float* pSrcCurrent = psrc;
                int* pIdxCurrent = pidx;
                float* pDstCurrent = pdst;
                int* pEnd = pidx + count;

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
                    Vector128<float> dstVector = SseIntrinsics.Load4(pDstCurrent, pIdxCurrent);
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);

                    dstVector = Sse.Add(dstVector, srcVector);
                    SseIntrinsics.Store4(in dstVector, pDstCurrent, pIdxCurrent);

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

        public static unsafe void MulElementWiseU(ReadOnlySpan<float> src1, ReadOnlySpan<float> src2, Span<float> dst, int count)
        {
            fixed (float* psrc1 = &MemoryMarshal.GetReference(src1))
            fixed (float* psrc2 = &MemoryMarshal.GetReference(src2))
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
            {
                float* pSrc1Current = psrc1;
                float* pSrc2Current = psrc2;
                float* pDstCurrent = pdst;
                float* pEnd = pdst + count;

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

        public static unsafe float SumU(ReadOnlySpan<float> src)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
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
                Vector128<float> resultPadded = Sse.AddScalar(Avx.GetLowerHalf(result256), GetHigh(result256));

                Vector128<float> result128 = Sse.SetZeroVector128();

                if (pSrcCurrent + 4 <= pSrcEnd)
                {
                    result128 = Sse.Add(result128, Sse.LoadVector128(pSrcCurrent));
                    pSrcCurrent += 4;
                }

                result128 = SseIntrinsics.VectorSum128(in result128);

                while (pSrcCurrent < pSrcEnd)
                {
                    result128 = Sse.AddScalar(result128, Sse.LoadScalarVector128(pSrcCurrent));
                    pSrcCurrent++;
                }

                return Sse.ConvertToSingle(Sse.AddScalar(result128, resultPadded));
            }
        }

        public static unsafe float SumSqU(ReadOnlySpan<float> src)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            {
                float* pSrcEnd = psrc + src.Length;
                float* pSrcCurrent = psrc;

                Vector256<float> result256 = Avx.SetZeroVector256<float>();

                while (pSrcCurrent + 8 <= pSrcEnd)
                {
                    Vector256<float> srcVector = Avx.LoadVector256(pSrcCurrent);
                    result256 = MultiplyAdd(srcVector, srcVector, result256);

                    pSrcCurrent += 8;
                }

                result256 = VectorSum256(in result256);
                Vector128<float> resultPadded = Sse.AddScalar(Avx.GetLowerHalf(result256), GetHigh(result256));

                Vector128<float> result128 = Sse.SetZeroVector128();

                if (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    result128 = Sse.Add(result128, Sse.Multiply(srcVector, srcVector));

                    pSrcCurrent += 4;
                }

                result128 = SseIntrinsics.VectorSum128(in result128);

                while (pSrcCurrent < pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    result128 = Sse.AddScalar(result128, Sse.MultiplyScalar(srcVector, srcVector));

                    pSrcCurrent++;
                }

                return Sse.ConvertToSingle(Sse.AddScalar(result128, resultPadded));
            }
        }

        public static unsafe float SumSqDiffU(float mean, ReadOnlySpan<float> src)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            {
                float* pSrcEnd = psrc + src.Length;
                float* pSrcCurrent = psrc;

                Vector256<float> result256 = Avx.SetZeroVector256<float>();
                Vector256<float> meanVector256 = Avx.SetAllVector256(mean);

                while (pSrcCurrent + 8 <= pSrcEnd)
                {
                    Vector256<float> srcVector = Avx.LoadVector256(pSrcCurrent);
                    srcVector = Avx.Subtract(srcVector, meanVector256);
                    result256 = MultiplyAdd(srcVector, srcVector, result256);
                    pSrcCurrent += 8;
                }

                result256 = VectorSum256(in result256);
                Vector128<float> resultPadded = Sse.AddScalar(Avx.GetLowerHalf(result256), GetHigh(result256));

                Vector128<float> result128 = Sse.SetZeroVector128();
                Vector128<float> meanVector128 = Sse.SetAllVector128(mean);

                if (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    srcVector = Sse.Subtract(srcVector, meanVector128);
                    result128 = Sse.Add(result128, Sse.Multiply(srcVector, srcVector));

                    pSrcCurrent += 4;
                }

                result128 = SseIntrinsics.VectorSum128(in result128);

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

        public static unsafe float SumAbsU(ReadOnlySpan<float> src)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            {
                float* pSrcEnd = psrc + src.Length;
                float* pSrcCurrent = psrc;

                Vector256<float> result256 = Avx.SetZeroVector256<float>();

                while (pSrcCurrent + 8 <= pSrcEnd)
                {
                    Vector256<float> srcVector = Avx.LoadVector256(pSrcCurrent);
                    result256 = Avx.Add(result256, Avx.And(srcVector, _absMask256));

                    pSrcCurrent += 8;
                }

                result256 = VectorSum256(in result256);
                Vector128<float> resultPadded = Sse.AddScalar(Avx.GetLowerHalf(result256), GetHigh(result256));

                Vector128<float> result128 = Sse.SetZeroVector128();

                if (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    result128 = Sse.Add(result128, Sse.And(srcVector, SseIntrinsics.AbsMask128));

                    pSrcCurrent += 4;
                }

                result128 = SseIntrinsics.VectorSum128(in result128);

                while (pSrcCurrent < pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    result128 = Sse.AddScalar(result128, Sse.And(srcVector, SseIntrinsics.AbsMask128));

                    pSrcCurrent++;
                }

                return Sse.ConvertToSingle(Sse.AddScalar(result128, resultPadded));
            }
        }

        public static unsafe float SumAbsDiffU(float mean, ReadOnlySpan<float> src)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            {
                float* pSrcEnd = psrc + src.Length;
                float* pSrcCurrent = psrc;

                Vector256<float> result256 = Avx.SetZeroVector256<float>();
                Vector256<float> meanVector256 = Avx.SetAllVector256(mean);

                while (pSrcCurrent + 8 <= pSrcEnd)
                {
                    Vector256<float> srcVector = Avx.LoadVector256(pSrcCurrent);
                    srcVector = Avx.Subtract(srcVector, meanVector256);
                    result256 = Avx.Add(result256, Avx.And(srcVector, _absMask256));

                    pSrcCurrent += 8;
                }

                result256 = VectorSum256(in result256);
                Vector128<float> resultPadded = Sse.AddScalar(Avx.GetLowerHalf(result256), GetHigh(result256));

                Vector128<float> result128 = Sse.SetZeroVector128();
                Vector128<float> meanVector128 = Sse.SetAllVector128(mean);

                if (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    srcVector = Sse.Subtract(srcVector, meanVector128);
                    result128 = Sse.Add(result128, Sse.And(srcVector, SseIntrinsics.AbsMask128));

                    pSrcCurrent += 4;
                }

                result128 = SseIntrinsics.VectorSum128(in result128);

                while (pSrcCurrent < pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    srcVector = Sse.SubtractScalar(srcVector, meanVector128);
                    result128 = Sse.AddScalar(result128, Sse.And(srcVector, SseIntrinsics.AbsMask128));

                    pSrcCurrent++;
                }

                return Sse.ConvertToSingle(Sse.AddScalar(result128, resultPadded));
            }
        }

        public static unsafe float MaxAbsU(ReadOnlySpan<float> src)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            {
                float* pSrcEnd = psrc + src.Length;
                float* pSrcCurrent = psrc;

                Vector256<float> result256 = Avx.SetZeroVector256<float>();

                while (pSrcCurrent + 8 <= pSrcEnd)
                {
                    Vector256<float> srcVector = Avx.LoadVector256(pSrcCurrent);
                    result256 = Avx.Max(result256, Avx.And(srcVector, _absMask256));

                    pSrcCurrent += 8;
                }

                result256 = VectorMax256(in result256);
                Vector128<float> resultPadded = Sse.MaxScalar(Avx.GetLowerHalf(result256), GetHigh(result256));

                Vector128<float> result128 = Sse.SetZeroVector128();

                if (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    result128 = Sse.Max(result128, Sse.And(srcVector, SseIntrinsics.AbsMask128));

                    pSrcCurrent += 4;
                }

                result128 = SseIntrinsics.VectorMax128(in result128);

                while (pSrcCurrent < pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    result128 = Sse.MaxScalar(result128, Sse.And(srcVector, SseIntrinsics.AbsMask128));

                    pSrcCurrent++;
                }

                return Sse.ConvertToSingle(Sse.MaxScalar(result128, resultPadded));
            }
        }

        public static unsafe float MaxAbsDiffU(float mean, ReadOnlySpan<float> src)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            {
                float* pSrcEnd = psrc + src.Length;
                float* pSrcCurrent = psrc;

                Vector256<float> result256 = Avx.SetZeroVector256<float>();
                Vector256<float> meanVector256 = Avx.SetAllVector256(mean);

                while (pSrcCurrent + 8 <= pSrcEnd)
                {
                    Vector256<float> srcVector = Avx.LoadVector256(pSrcCurrent);
                    srcVector = Avx.Subtract(srcVector, meanVector256);
                    result256 = Avx.Max(result256, Avx.And(srcVector, _absMask256));

                    pSrcCurrent += 8;
                }

                result256 = VectorMax256(in result256);
                Vector128<float> resultPadded = Sse.MaxScalar(Avx.GetLowerHalf(result256), GetHigh(result256));

                Vector128<float> result128 = Sse.SetZeroVector128();
                Vector128<float> meanVector128 = Sse.SetAllVector128(mean);

                if (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    srcVector = Sse.Subtract(srcVector, meanVector128);
                    result128 = Sse.Max(result128, Sse.And(srcVector, SseIntrinsics.AbsMask128));

                    pSrcCurrent += 4;
                }

                result128 = SseIntrinsics.VectorMax128(in result128);

                while (pSrcCurrent < pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                    srcVector = Sse.SubtractScalar(srcVector, meanVector128);
                    result128 = Sse.MaxScalar(result128, Sse.And(srcVector, SseIntrinsics.AbsMask128));

                    pSrcCurrent++;
                }

                return Sse.ConvertToSingle(Sse.MaxScalar(result128, resultPadded));
            }
        }

        public static unsafe float DotU(ReadOnlySpan<float> src, ReadOnlySpan<float> dst, int count)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
            {
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                float* pSrcEnd = psrc + count;

                Vector256<float> result256 = Avx.SetZeroVector256<float>();

                while (pSrcCurrent + 8 <= pSrcEnd)
                {
                    Vector256<float> dstVector = Avx.LoadVector256(pDstCurrent);
                    result256 = MultiplyAdd(pSrcCurrent, dstVector, result256);
                    pSrcCurrent += 8;
                    pDstCurrent += 8;
                }

                result256 = VectorSum256(in result256);
                Vector128<float> resultPadded = Sse.AddScalar(Avx.GetLowerHalf(result256), GetHigh(result256));

                Vector128<float> result128 = Sse.SetZeroVector128();

                if (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                    Vector128<float> dstVector = Sse.LoadVector128(pDstCurrent);

                    result128 = Sse.Add(result128, Sse.Multiply(srcVector, dstVector));

                    pSrcCurrent += 4;
                    pDstCurrent += 4;
                }

                result128 = SseIntrinsics.VectorSum128(in result128);

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

        public static unsafe float DotSU(ReadOnlySpan<float> src, ReadOnlySpan<float> dst, ReadOnlySpan<int> idx, int count)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
            fixed (int* pidx = &MemoryMarshal.GetReference(idx))
            {
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                int* pIdxCurrent = pidx;
                int* pIdxEnd = pidx + count;

                Vector256<float> result256 = Avx.SetZeroVector256<float>();

                while (pIdxCurrent + 8 <= pIdxEnd)
                {
                    Vector256<float> srcVector = Load8(pSrcCurrent, pIdxCurrent);
                    result256 = MultiplyAdd(pDstCurrent, srcVector, result256);
                    pIdxCurrent += 8;
                    pDstCurrent += 8;
                }

                result256 = VectorSum256(in result256);
                Vector128<float> resultPadded = Sse.AddScalar(Avx.GetLowerHalf(result256), GetHigh(result256));

                Vector128<float> result128 = Sse.SetZeroVector128();

                if (pIdxCurrent + 4 <= pIdxEnd)
                {
                    Vector128<float> srcVector = SseIntrinsics.Load4(pSrcCurrent, pIdxCurrent);
                    Vector128<float> dstVector = Sse.LoadVector128(pDstCurrent);

                    result128 = Sse.Add(result128, Sse.Multiply(srcVector, dstVector));

                    pIdxCurrent += 4;
                    pDstCurrent += 4;
                }

                result128 = SseIntrinsics.VectorSum128(in result128);

                while (pIdxCurrent < pIdxEnd)
                {
                    Vector128<float> srcVector = SseIntrinsics.Load1(pSrcCurrent, pIdxCurrent);
                    Vector128<float> dstVector = Sse.LoadScalarVector128(pDstCurrent);

                    result128 = Sse.AddScalar(result128, Sse.MultiplyScalar(srcVector, dstVector));

                    pIdxCurrent++;
                    pDstCurrent++;
                }

                return Sse.ConvertToSingle(Sse.AddScalar(result128, resultPadded));
            }
        }

        public static unsafe float Dist2(ReadOnlySpan<float> src, ReadOnlySpan<float> dst, int count)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
            {
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                float* pSrcEnd = psrc + count;

                Vector256<float> sqDistanceVector256 = Avx.SetZeroVector256<float>();

                while (pSrcCurrent + 8 <= pSrcEnd)
                {
                    Vector256<float> distanceVector = Avx.Subtract(Avx.LoadVector256(pSrcCurrent),
                                                                    Avx.LoadVector256(pDstCurrent));
                    sqDistanceVector256 = MultiplyAdd(distanceVector, distanceVector, sqDistanceVector256);
                    pSrcCurrent += 8;
                    pDstCurrent += 8;
                }

                sqDistanceVector256 = VectorSum256(in sqDistanceVector256);
                Vector128<float> sqDistanceVectorPadded = Sse.AddScalar(Avx.GetLowerHalf(sqDistanceVector256), GetHigh(sqDistanceVector256));

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

                sqDistanceVector128 = SseIntrinsics.VectorSum128(in sqDistanceVector128);

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

        public static unsafe void SdcaL1UpdateU(float primalUpdate, int count, ReadOnlySpan<float> src, float threshold, Span<float> v, Span<float> w)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            fixed (float* pdst1 = &MemoryMarshal.GetReference(v))
            fixed (float* pdst2 = &MemoryMarshal.GetReference(w))
            {
                float* pSrcEnd = psrc + count;
                float* pSrcCurrent = psrc;
                float* pDst1Current = pdst1;
                float* pDst2Current = pdst2;

                Vector256<float> xPrimal256 = Avx.SetAllVector256(primalUpdate);
                Vector256<float> xThreshold256 = Avx.SetAllVector256(threshold);

                while (pSrcCurrent + 8 <= pSrcEnd)
                {
                    Vector256<float> xDst1 = Avx.LoadVector256(pDst1Current);
                    xDst1 = MultiplyAdd(pSrcCurrent, xPrimal256, xDst1);
                    Vector256<float> xDst2 = GetNewDst256(xDst1, xThreshold256);

                    Avx.Store(pDst1Current, xDst1);
                    Avx.Store(pDst2Current, xDst2);

                    pSrcCurrent += 8;
                    pDst1Current += 8;
                    pDst2Current += 8;
                }

                Vector128<float> xPrimal128 = Sse.SetAllVector128(primalUpdate);
                Vector128<float> xThreshold128 = Sse.SetAllVector128(threshold);

                if (pSrcCurrent + 4 <= pSrcEnd)
                {
                    Vector128<float> xSrc = Sse.LoadVector128(pSrcCurrent);

                    Vector128<float> xDst1 = Sse.LoadVector128(pDst1Current);
                    xDst1 = Sse.Add(xDst1, Sse.Multiply(xSrc, xPrimal128));
                    Vector128<float> xDst2 = SseIntrinsics.GetNewDst128(xDst1, xThreshold128);

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

        public static unsafe void SdcaL1UpdateSU(float primalUpdate, int count, ReadOnlySpan<float> src, ReadOnlySpan<int> indices, float threshold, Span<float> v, Span<float> w)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            fixed (int* pidx = &MemoryMarshal.GetReference(indices))
            fixed (float* pdst1 = &MemoryMarshal.GetReference(v))
            fixed (float* pdst2 = &MemoryMarshal.GetReference(w))
            {
                int* pIdxEnd = pidx + count;
                float* pSrcCurrent = psrc;
                int* pIdxCurrent = pidx;

                Vector256<float> xPrimal256 = Avx.SetAllVector256(primalUpdate);
                Vector256<float> xThreshold = Avx.SetAllVector256(threshold);

                while (pIdxCurrent + 8 <= pIdxEnd)
                {
                    Vector256<float> xDst1 = Load8(pdst1, pIdxCurrent);
                    xDst1 = MultiplyAdd(pSrcCurrent, xPrimal256, xDst1);
                    Vector256<float> xDst2 = GetNewDst256(xDst1, xThreshold);

                    Store8(in xDst1, pdst1, pIdxCurrent);
                    Store8(in xDst2, pdst2, pIdxCurrent);

                    pIdxCurrent += 8;
                    pSrcCurrent += 8;
                }

                Vector128<float> xPrimal128 = Sse.SetAllVector128(primalUpdate);
                Vector128<float> xThreshold128 = Sse.SetAllVector128(threshold);

                if (pIdxCurrent + 4 <= pIdxEnd)
                {
                    Vector128<float> xSrc = Sse.LoadVector128(pSrcCurrent);

                    Vector128<float> xDst1 = SseIntrinsics.Load4(pdst1, pIdxCurrent);
                    xDst1 = Sse.Add(xDst1, Sse.Multiply(xSrc, xPrimal128));
                    Vector128<float> xDst2 = SseIntrinsics.GetNewDst128(xDst1, xThreshold128);

                    SseIntrinsics.Store4(in xDst1, pdst1, pIdxCurrent);
                    SseIntrinsics.Store4(in xDst2, pdst2, pIdxCurrent);

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
