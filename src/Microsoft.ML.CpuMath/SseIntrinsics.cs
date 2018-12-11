﻿// Licensed to the .NET Foundation under one or more agreements.
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

using Microsoft.ML.Runtime.Internal.CpuMath.Core;
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using nuint = System.UInt64;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    internal static class SseIntrinsics
    {
        public static readonly uint[] LeadingAlignmentMask = new uint[16]
        {
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000,
            0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000,
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000,
        };

        public static readonly uint[] TrailingAlignmentMask = new uint[16]
        {
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF,
            0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF,
            0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
        };

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

        internal static readonly Vector128<float> AbsMask128 = Sse2.IsSupported ?
            Sse.StaticCast<int, float>(Sse2.SetAllVector128(0x7FFFFFFF)) :
            Sse.SetAllVector128(BitConverter.Int32BitsToSingle(0x7FFFFFFF));

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
        public static unsafe void MatMul(AlignedArray mat, AlignedArray src, AlignedArray dst, int crow, int ccol)
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

                    Sse.StoreAligned(pDstCurrent, res0);

                    pDstCurrent += 4;
                    pMatCurrent += 3 * ccol;
                }
            }
        }

        // Partial sparse source vector.
        public static unsafe void MatMulP(AlignedArray mat, ReadOnlySpan<int> rgposSrc, AlignedArray src,
                                int posMin, int iposMin, int iposEnd, AlignedArray dst, int crow, int ccol)
        {
            // REVIEW: For extremely sparse inputs, interchanging the loops would
            // likely be more efficient.
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

                    Sse.StoreAligned(pDstCurrent, result);
                    pDstCurrent += 4;
                    pm0 += 4 * ccol;
                }
            }
        }

        public static unsafe void MatMulTran(AlignedArray mat, AlignedArray src, AlignedArray dst, int crow, int ccol)
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

                while (pSrcCurrent < pSrcEnd)
                {
                    x01 = Sse.LoadAlignedVector128(pSrcCurrent);
                    // Replicate each 32-bit slot of x01 (ABCD) into its own register.
                    x11 = Sse.Shuffle(x01, x01, 0x55); // B
                    x21 = Sse.Shuffle(x01, x01, 0xAA); // C
                    x31 = Sse.Shuffle(x01, x01, 0xFF); // D
                    x01 = Sse.Shuffle(x01, x01, 0x00); // A

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

        // dst[i] += scale
        public static unsafe void AddScalarU(float scalar, Span<float> dst)
        {
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
            {
                float* pDstEnd = pdst + dst.Length;
                float* pDstCurrent = pdst;
                float* pVectorizationEnd = pDstEnd - 4;

                Vector128<float> scalarVector = Sse.SetAllVector128(scalar);

                while (pDstCurrent <= pVectorizationEnd)
                {
                    Vector128<float> dstVector = Sse.LoadVector128(pDstCurrent);
                    dstVector = Sse.Add(dstVector, scalarVector);
                    Sse.Store(pDstCurrent, dstVector);

                    pDstCurrent += 4;
                }

                while (pDstCurrent < pDstEnd)
                {
                    Vector128<float> dstVector = Sse.LoadScalarVector128(pDstCurrent);
                    dstVector = Sse.AddScalar(dstVector, scalarVector);
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
                Vector128<float> scaleVector128 = Sse.SetAllVector128(scale);

                if (length < 4)
                {
                    // Handle cases where we have less than 128-bits total and can't ever use SIMD acceleration.
                    switch (length)
                    {
                        case 3: dst[2] *= scale; goto case 2;
                        case 2: dst[1] *= scale; goto case 1;
                        case 1: dst[0] *= scale; break;
                    }
                    return;
                }

                nuint address = (nuint)(pd);
                int misalignment = (int)(address % 16);
                int remainder = 0;

                if ((misalignment & 3) != 0)
                {
                    // Handles cases where the data is not 32-bit aligned and we can't ever use aligned operations
                    remainder = length % 4;

                    for (float* pEnd = pd + (length - remainder); pDstCurrent < pEnd; pDstCurrent += 4)
                    {
                        Vector128<float> temp = Sse.LoadVector128(pDstCurrent);
                        temp = Sse.Multiply(scaleVector128, temp);
                        Sse.Store(pDstCurrent, temp);
                    }
                }
                else
                {
                    if (misalignment != 0)
                    {
                        // Handle cases where the data is not 128-bit aligned by doing an unaligned read and then
                        // masking any elements that will be included in the first aligned read

                        misalignment >>= 2;
                        misalignment = 4 - misalignment;

                        Vector128<float> result = Sse.LoadVector128(pDstCurrent);

                        Vector128<float> leadingMask = Sse.LoadVector128(((float*)(pLeadingAlignmentMask)) + (misalignment * 4));
                        Vector128<float> trailingMask = Sse.LoadVector128(((float*)(pTrailingAlignmentMask)) + ((4 - misalignment) * 4));

                        Vector128<float> temp = Sse.And(result, trailingMask);
                        result = Sse.Multiply(scaleVector128, result);

                        // Masking operation is done at the end to avoid doing an Or operation with negative Zero.
                        result = Sse.And(result, leadingMask);
                        result = Sse.Or(result, temp);

                        Sse.Store(pDstCurrent, result);

                        pDstCurrent += misalignment;
                        length -= misalignment;
                    }

                    if (length > 3)
                    {
                        // Handle all the 128-bit blocks that we can now that we have offset to an aligned address
                        remainder = length % 4;

                        for (float* pEnd = pDstCurrent + (length - remainder); pDstCurrent < pEnd; pDstCurrent += 4)
                        {
                            // If we aren't using the VEX-encoding, the JIT will only fold away aligned loads
                            // (due to semantics of the legacy encoding).
                            // We don't need an assert, since the instruction will throw for unaligned inputs.
                            Vector128<float> temp = Sse.LoadAlignedVector128(pDstCurrent);
                            temp = Sse.Multiply(scaleVector128, temp);
                            Sse.Store(pDstCurrent, temp);
                        }
                    }
                    else
                    {
                        // Handle the "worst-case" scenario, which is when we have 4-8 elements and the input is not
                        // 128-bit aligned. This means we can't do any aligned loads and will just end up doing two
                        // unaligned loads where we mask the input each time.
                        remainder = length;
                    }
                }

                if (remainder != 0)
                {
                    // Handle any trailing elements that don't fit into a 128-bit block by moving back so that the next
                    // unaligned load will read to the end of the array and then mask out any elements already processed

                    pDstCurrent -= (4 - remainder);

                    Vector128<float> result = Sse.LoadVector128(pDstCurrent);

                    Vector128<float> trailingMask = Sse.LoadVector128(((float*)(pTrailingAlignmentMask)) + (remainder * 4));
                    Vector128<float> leadingMask = Sse.LoadVector128(((float*)(pLeadingAlignmentMask)) + ((4 - remainder) * 4));

                    Vector128<float> temp = Sse.And(result, leadingMask);
                    result = Sse.Multiply(scaleVector128, result);

                    // Masking operation is done at the end to avoid doing an Or operation with negative Zero.
                    result = Sse.And(result, trailingMask);
                    result = Sse.Or(result, temp);

                    Sse.Store(pDstCurrent, result);
                }
            }
        }

        public static unsafe void ScaleSrcU(float scale, ReadOnlySpan<float> src, Span<float> dst, int count)
        {
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= dst.Length);
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
            {
                float* pDstEnd = pdst + count;
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                float* pVectorizationEnd = pDstEnd - 4;

                Vector128<float> scaleVector = Sse.SetAllVector128(scale);

                while (pDstCurrent <= pVectorizationEnd)
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
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
            {
                float* pDstEnd = pdst + dst.Length;
                float* pDstCurrent = pdst;
                float* pVectorizationEnd = pDstEnd - 4;

                Vector128<float> aVector = Sse.SetAllVector128(a);
                Vector128<float> bVector = Sse.SetAllVector128(b);

                while (pDstCurrent <= pVectorizationEnd)
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

        public static unsafe void AddScaleU(float scale, ReadOnlySpan<float> src, Span<float> dst, int count)
        {
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= dst.Length);
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
            {
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                float* pEnd = pdst + count;

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

        public static unsafe void AddScaleCopyU(float scale, ReadOnlySpan<float> src, ReadOnlySpan<float> dst, Span<float> result, int count)
        {
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= dst.Length);
            Contracts.Assert(count <= result.Length);
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
            fixed (float* pres = &MemoryMarshal.GetReference(result))
            {
                float* pResEnd = pres + count;
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

        public static unsafe void AddScaleSU(float scale, ReadOnlySpan<float> src, ReadOnlySpan<int> idx, Span<float> dst, int count)
        {
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= dst.Length);
            Contracts.Assert(count <= idx.Length);
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            fixed (int* pidx = &MemoryMarshal.GetReference(idx))
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
            {
                float* pSrcCurrent = psrc;
                int* pIdxCurrent = pidx;
                float* pDstCurrent = pdst;
                int* pEnd = pidx + count;

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

        public static unsafe void AddU(ReadOnlySpan<float> src, Span<float> dst, int count)
        {
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= dst.Length);
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
            {
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                float* pEnd = psrc + count;

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

        public static unsafe void AddSU(ReadOnlySpan<float> src, ReadOnlySpan<int> idx, Span<float> dst, int count)
        {
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= dst.Length);
            Contracts.Assert(count <= idx.Length);
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            fixed (int* pidx = &MemoryMarshal.GetReference(idx))
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
            {
                float* pSrcCurrent = psrc;
                int* pIdxCurrent = pidx;
                float* pDstCurrent = pdst;
                int* pEnd = pidx + count;

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

        public static unsafe void MulElementWiseU(ReadOnlySpan<float> src1, ReadOnlySpan<float> src2, Span<float> dst, int count)
        {
            Contracts.Assert(count <= src1.Length);
            Contracts.Assert(count <= src2.Length);
            Contracts.Assert(count <= dst.Length);
            fixed (float* psrc1 = &MemoryMarshal.GetReference(src1))
            fixed (float* psrc2 = &MemoryMarshal.GetReference(src2))
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
            {
                float* pSrc1Current = psrc1;
                float* pSrc2Current = psrc2;
                float* pDstCurrent = pdst;
                float* pEnd = pdst + count;

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

        public static unsafe float Sum(ReadOnlySpan<float> src)
        {
            fixed (float* pSrc = &MemoryMarshal.GetReference(src))
            fixed (uint* pLeadingAlignmentMask = &LeadingAlignmentMask[0])
            fixed (uint* pTrailingAlignmentMask = &TrailingAlignmentMask[0])
            {
                float* pValues = pSrc;
                int length = src.Length;

                if (length < 4)
                {
                    // Handle cases where we have less than 128-bits total and can't ever use SIMD acceleration.

                    float res = 0;

                    switch (length)
                    {
                        case 3: res += pValues[2]; goto case 2;
                        case 2: res += pValues[1]; goto case 1;
                        case 1: res += pValues[0]; break;
                    }

                    return res;
                }

                Vector128<float> result = Sse.SetZeroVector128();

                nuint address = (nuint)(pValues);
                int misalignment = (int)(address % 16);
                int remainder = 0;

                if ((misalignment & 3) != 0)
                {
                    // Handles cases where the data is not 32-bit aligned and we can't ever use aligned operations

                    remainder = length % 4;

                    for (float* pEnd = pValues + (length - remainder); pValues < pEnd; pValues += 4)
                    {
                        result = Sse.Add(result, Sse.LoadVector128(pValues));
                    }
                }
                else
                {
                    if (misalignment != 0)
                    {
                        // Handle cases where the data is not 128-bit aligned by doing an unaligned read and then
                        // masking any elements that will be included in the first aligned read

                        misalignment >>= 2;
                        misalignment = 4 - misalignment;

                        Vector128<float> mask = Sse.LoadVector128(((float*)(pLeadingAlignmentMask)) + (misalignment * 4));
                        Vector128<float> temp = Sse.And(mask, Sse.LoadVector128(pValues));
                        result = Sse.Add(result, temp);

                        pValues += misalignment;
                        length -= misalignment;
                    }

                    if (length > 3)
                    {
                        // Handle all the 128-bit blocks that we can now that we have offset to an aligned address

                        remainder = length % 4;

                        for (float* pEnd = pValues + (length - remainder); pValues < pEnd; pValues += 4)
                        {
                            // If we aren't using the VEX-encoding, the JIT will only fold away aligned loads
                            // (due to semantics of the legacy encoding).
                            // We don't need an assert, since the instruction will throw for unaligned inputs.

                            result = Sse.Add(result, Sse.LoadAlignedVector128(pValues));
                        }
                    }
                    else
                    {
                        // Handle the "worst-case" scenario, which is when we have 4-8 elements and the input is not
                        // 128-bit aligned. This means we can't do any aligned loads and will just end up doing two
                        // unaligned loads where we mask the input each time.
                        remainder = length;
                    }
                }

                if (remainder != 0)
                {
                    // Handle any trailing elements that don't fit into a 128-bit block by moving back so that the next
                    // unaligned load will read to the end of the array and then mask out any elements already processed

                    pValues -= (4 - remainder);

                    Vector128<float> mask = Sse.LoadVector128(((float*)(pTrailingAlignmentMask)) + (remainder * 4));
                    Vector128<float> temp = Sse.And(mask, Sse.LoadVector128(pValues));
                    result = Sse.Add(result, temp);
                }

                // Sum all the elements together and return the result
                result = VectorSum128(in result);
                return Sse.ConvertToSingle(result);
            }
        }

        public static unsafe float SumSqU(ReadOnlySpan<float> src)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
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

        public static unsafe float SumSqDiffU(float mean, ReadOnlySpan<float> src)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
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

        public static unsafe float SumAbsU(ReadOnlySpan<float> src)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
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

        public static unsafe float SumAbsDiffU(float mean, ReadOnlySpan<float> src)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
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

        public static unsafe float MaxAbsU(ReadOnlySpan<float> src)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
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

        public static unsafe float MaxAbsDiffU(float mean, ReadOnlySpan<float> src)
        {
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
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

        public static unsafe float DotU(ReadOnlySpan<float> src, ReadOnlySpan<float> dst, int count)
        {
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= dst.Length);
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
            {
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                float* pSrcEnd = psrc + count;

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

        public static unsafe float DotSU(ReadOnlySpan<float> src, ReadOnlySpan<float> dst, ReadOnlySpan<int> idx, int count)
        {
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= dst.Length);
            Contracts.Assert(count <= idx.Length);
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
            fixed (int* pidx = &MemoryMarshal.GetReference(idx))
            {
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                int* pIdxCurrent = pidx;
                int* pIdxEnd = pidx + count;

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

        public static unsafe float Dist2(ReadOnlySpan<float> src, ReadOnlySpan<float> dst, int count)
        {
            Contracts.Assert(count <= src.Length);
            Contracts.Assert(count <= dst.Length);
            fixed (float* psrc = &MemoryMarshal.GetReference(src))
            fixed (float* pdst = &MemoryMarshal.GetReference(dst))
            {
                float* pSrcCurrent = psrc;
                float* pDstCurrent = pdst;
                float* pSrcEnd = psrc + count;

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
