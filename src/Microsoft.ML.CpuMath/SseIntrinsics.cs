// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// The exported function names need to be unique (can't be disambiguated based on signature), hence
// we introduce suffix letters to indicate the general patterns used.
// * U suffix means unaligned and unpadded.
// * S suffix means sparse (unaligned) vector.

using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    internal static class SseIntrinsics
    {
        private static unsafe Vector128<float> Load1(float* src, int* idx)
        {
            return Sse.SetScalarVector128(src[idx[0]]);
        }

        private static unsafe Vector128<float> Load4(float* src, int* idx)
        {
            return Sse.SetVector128(src[idx[3]], src[idx[2]], src[idx[1]], src[idx[0]]);
        }

        private static Vector128<float> Rotate(Vector128<float> x)
        {
            // The control byte shuffles the four 32-bit floats of x: ABCD -> BCDA.
            return Sse.Shuffle(x, x, 0x39);
        }

        private static Vector128<float> RotateReverse(Vector128<float> x)
        {
            // The control byte shuffles the four 32-bit floats of x: ABCD -> DABC.
            return Sse.Shuffle(x, x, 0x93);
        }

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

        private static unsafe Vector128<float> VectorSum(Vector128<float> vector)
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

                result = VectorSum(result);

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

                result = VectorSum(result);

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

                result = VectorSum(result);

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

                result = VectorSum(result);

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

                sqDistanceVector = VectorSum(sqDistanceVector);

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

    }
}
