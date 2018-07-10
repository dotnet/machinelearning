using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System;

namespace Intrinsics
{    public class IntrinsicsUtils
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
            return Sse.Shuffle(x, x, 0x39);
        }

        private static Vector128<float> RotateReverse(Vector128<float> x)
        {
            return Sse.Shuffle(x, x, 0x93);
        }

        // Warning: this operation changes the value of x => do not reuse x
        private static unsafe void Store4(Vector128<float> x, float* dst, int* idx)
        {
            Sse.StoreScalar(dst + idx[0], x);
            for (int i = 1; i <= 3; i++)
            {
                x = Rotate(x);
                Sse.StoreScalar(dst + idx[i], x);
            }
        }

        //// Multiply matrix times vector into vector.
        //private static float MatMulASse(bool add, float[] matrix, float[] src, float[] dst)
        //{
        //    unsafe
        //    {
        //        fixed (float* psrc = src)
        //        fixed (float* pdst = dst)
        //        fixed (float* pmat = matrix)
        //        {
        //            float* pSrcCurrent = psrc;
        //            float* pDstCurrent = pdst;
        //            float* pSrcEnd = psrc + src.Length;
        //            float* pDstEnd = pdst + dst.Length;
        //            float* pMatCurrent = pmat;

        //            while (pDstCurrent < pDstEnd)
        //            {
        //                float* pMatTemp = pMatCurrent;
        //                Vector128<float> matVector1 = Sse.LoadAlignedVector128(pMatTemp);
        //                Vector128<float> matVector2 = Sse.LoadAlignedVector128(pMatTemp += src.Length);
        //                Vector128<float> matVector3 = Sse.LoadAlignedVector128(pMatTemp += src.Length);
        //                Vector128<float> matVector4 = Sse.LoadAlignedVector128(pMatTemp += src.Length);

        //                Vector128<float> srcVector = Sse.LoadAlignedVector128(pSrcCurrent);


        //                result = Sse.Add(result, Sse.Multiply(srcVector, dstVector));

        //                pDstCurrent += 4;
        //                pMatCurrent += 3 * src.Length;
        //            }

        //            if (Sse3.IsSupported)
        //            {
        //                // SSE3 is supported.
        //                result = Sse3.HorizontalAdd(result, result);
        //                result = Sse3.HorizontalAdd(result, result);
        //            }
        //            else
        //            {
        //                // SSE3 is not supported.
        //                result = Sse.Add(result, Sse.MoveHighToLow(result, result));
        //                result = Sse.Add(result, Sse.MoveHighToLow(result, Sse.UnpackLow(result, result)));
        //            }


        //            while (pSrcCurrent < pEnd)
        //            {
        //                Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
        //                Vector128<float> dstVector = Sse.LoadScalarVector128(pDstCurrent);

        //                result = Sse.AddScalar(result, Sse.MultiplyScalar(srcVector, dstVector));

        //                pSrcCurrent++;
        //                pDstCurrent++;
        //            }
        //        }
        //    }

        //    return Sse.ConvertToSingle(result);
        //}


        //public static float MatMulA((bool add, float[] matrix, float[] src, float[] dst)
        //{

        //    if (Sse.IsSupported)
        //    {
        //        return MatMulASse(add, matrix, src, dst);
        //    }
        //    else
        //    {
        //        // Software fallback.
        //        float result = 0;
        //        for (int i = 0; i < src.Length; i++)
        //        {
        //            result += src[i] * dst[i];
        //        }
        //        return result;
        //    }
        //}

        private static float DotUSse(float[] src, float[] dst, int count)
        {
            Vector128<float> result = Sse.SetZeroVector128();

            unsafe
            {

                fixed (float* psrc = src)
                fixed (float* pdst = dst)
                {
                    float* pSrcCurrent = psrc;
                    float* pDstCurrent = pdst;
                    float* pEnd = psrc + count;

                    while (pSrcCurrent + 4 <= pEnd)
                    {
                        Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                        Vector128<float> dstVector = Sse.LoadVector128(pDstCurrent);

                        result = Sse.Add(result, Sse.Multiply(srcVector, dstVector));

                        pSrcCurrent += 4;
                        pDstCurrent += 4;
                    }

                    if (Sse3.IsSupported)
                    {
                        // SSE3 is supported.
                        result = Sse3.HorizontalAdd(result, result);
                        result = Sse3.HorizontalAdd(result, result);
                    }
                    else
                    {
                        // SSE3 is not supported.
                        result = Sse.Add(result, Sse.MoveHighToLow(result, result));
                        result = Sse.Add(result, Sse.MoveHighToLow(result, Sse.UnpackLow(result, result)));
                    }


                    while (pSrcCurrent < pEnd)
                    {
                        Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                        Vector128<float> dstVector = Sse.LoadScalarVector128(pDstCurrent);

                        result = Sse.AddScalar(result, Sse.MultiplyScalar(srcVector, dstVector));

                        pSrcCurrent++;
                        pDstCurrent++;
                    }
                }
            }

            return Sse.ConvertToSingle(result);
        }


        public static float DotU(float[] src, float[] dst, int count)
        {
            if (Sse.IsSupported)
            {
                return DotUSse(src, dst, count);
            }
            else
            {
                // Software fallback.
                float result = 0;
                for (int i = 0; i < count; i++)
                {
                    result += src[i] * dst[i];
                }
                return result;
            }
        }

        private static float DotSUSse(float[] src, float[] dst, int[] idx, int count)
        {
            Vector128<float> result = Sse.SetZeroVector128();

            unsafe
            {

                fixed (float* psrc = src)
                fixed (float* pdst = dst)
                fixed (int* pidx = idx)
                {
                    float* pSrcCurrent = psrc;
                    float* pDstCurrent = pdst;
                    int* pIdxCurrent = pidx;
                    int* pEnd = pidx + count;

                    while (pIdxCurrent + 4 <= pEnd)
                    {
                        Vector128<float> srcVector = Load4(pSrcCurrent, pIdxCurrent);
                        Vector128<float> dstVector = Sse.LoadVector128(pDstCurrent);

                        result = Sse.Add(result, Sse.Multiply(srcVector, dstVector));

                        pIdxCurrent += 4;
                        pDstCurrent += 4;
                    }

                    if (Sse3.IsSupported)
                    {
                        // SSE3 is supported.
                        result = Sse3.HorizontalAdd(result, result);
                        result = Sse3.HorizontalAdd(result, result);
                    }
                    else
                    {
                        // SSE3 is not supported.
                        result = Sse.Add(result, Sse.MoveHighToLow(result, result));
                        result = Sse.Add(result, Sse.MoveHighToLow(result, Sse.UnpackLow(result, result)));
                    }


                    while (pIdxCurrent < pEnd)
                    {
                        Vector128<float> srcVector = Load1(pSrcCurrent, pIdxCurrent);
                        Vector128<float> dstVector = Sse.LoadScalarVector128(pDstCurrent);

                        result = Sse.AddScalar(result, Sse.MultiplyScalar(srcVector, dstVector));

                        pIdxCurrent++;
                        pDstCurrent++;
                    }
                }
            }

            return Sse.ConvertToSingle(result);
        }

        public static float DotSU(float[] src, float[] dst, int[] idx, int count)
        {
            if (Sse.IsSupported)
            {
                return DotSUSse(src, dst, idx, count);
            }
            else
            {
                // Software fallback.
                float result = 0;
                for (int i = 0; i < count; i++)
                {
                    result += src[idx[i]] * dst[i];
                }
                return result;
            }
        }

        private static float SumSqUSse(float[] src, int count)
        {
            Vector128<float> result = Sse.SetZeroVector128();

            unsafe
            {

                fixed (float* psrc = src)
                {
                    float* pSrcCurrent = psrc;
                    float* pEnd = psrc + count;

                    while (pSrcCurrent + 4 <= pEnd)
                    {
                        Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                        result = Sse.Add(result, Sse.Multiply(srcVector, srcVector));

                        pSrcCurrent += 4;
                    }

                    if (Sse3.IsSupported)
                    {
                        // SSE3 is supported.
                        result = Sse3.HorizontalAdd(result, result);
                        result = Sse3.HorizontalAdd(result, result);
                    }
                    else
                    {
                        // SSE3 is not supported.
                        result = Sse.Add(result, Sse.MoveHighToLow(result, result));
                        result = Sse.Add(result, Sse.MoveHighToLow(result, Sse.UnpackLow(result, result)));
                    }


                    while (pSrcCurrent < pEnd)
                    {
                        Vector128<float> srcVector = Sse.LoadScalarVector128(pSrcCurrent);
                        result = Sse.AddScalar(result, Sse.MultiplyScalar(srcVector, srcVector));

                        pSrcCurrent++;
                    }
                }
            }

            return Sse.ConvertToSingle(result);
        }

        public static float SumSqU(float[] src, int count)
        {
            if (Sse.IsSupported)
            {
                return SumSqUSse(src, count);
            }
            else
            {
                // Software fallback.
                float result = 0;
                for (int i = 0; i < count; i++)
                {
                    result += src[i] * src[i];
                }
                return result;
            }
        }

        private static void AddUSse(float[] src, float[] dst, int count)
        {
            unsafe
            {
                fixed (float* psrc = src)
                fixed (float* pdst = dst)
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
        }

        public static void AddU(float[] src, float[] dst, int count)
        {
            if (Sse.IsSupported)
            {
                AddUSse(src, dst, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    dst[i] += src[i];
                }
            }
        }

        private static void AddSUSse(float[] src, int[] idx, float[] dst, int count)
        {
            unsafe
            {
                fixed (float* psrc = src)
                fixed (int* pidx = idx)
                fixed (float* pdst = dst)
                {
                    float* pSrcCurrent = psrc;
                    int* pIdxCurrent = pidx;
                    float* pDstCurrent = pdst;
                    int* pEnd = pidx + count;

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
        }

        public static void AddSU(float[] src, int[] idx, float[] dst, int count)
        {
            if (Sse.IsSupported)
            {
                AddSUSse(src, idx, dst, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    dst[idx[i]] += src[i];
                }
            }
        }

        private static void AddScaleUSse(float scale, float[] src, float[] dst, int count)
        {
            Vector128<float> scaleVector = Sse.SetAllVector128(scale);

            unsafe
            {
                fixed (float* psrc = src)
                fixed (float* pdst = dst)
                {
                    float* pSrcCurrent = psrc;
                    float* pDstCurrent = pdst;
                    float* pEnd = pdst + count;

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
        }

        public static void AddScaleU(float scale, float[] src, float[] dst, int count)
        {
            if (Sse.IsSupported)
            {
                AddScaleUSse(scale, src, dst,count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    dst[i] += scale * src[i];
                }
            }
        }

        private static void AddScaleSUSse(float scale, float[] src, int[] idx, float[] dst, int count)
        {
            Vector128<float> scaleVector = Sse.SetAllVector128(scale);

            unsafe
            {
                fixed (float* psrc = src)
                fixed (int* pidx = idx)
                fixed (float* pdst = dst)
                {
                    float* pSrcCurrent = psrc;
                    int* pIdxCurrent = pidx;
                    float* pDstCurrent = pdst;
                    int* pEnd = pidx + count;

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
                        pDstCurrent[*pIdxCurrent] += scale * *pSrcCurrent;

                        pIdxCurrent++;
                        pSrcCurrent++;
                    }
                }
            }
        }

        public static void AddScaleSU(float scale, float[] src, int[] idx, float[] dst, int count)
        {
            if (Sse.IsSupported)
            {
                AddScaleSUSse(scale, src, idx, dst, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    dst[idx[i]] += scale * src[i];
                }
            }
        }

        private static void ScaleUSse(float scale, float[] dst, int count)
        {
            Vector128<float> scaleVector = Sse.SetAllVector128(scale);

            unsafe
            {
                fixed (float* pdst = dst)
                {
                    float* pDstCurrent = pdst;
                    float* pEnd = pdst + count;

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
        }

        public static void ScaleU(float scale, float[] dst, int count)
        {
            if (Sse.IsSupported)
            {
                ScaleUSse(scale, dst, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    dst[i] *= scale;
                }
            }
        }

        private static float Dist2Sse(float[] src, float[] dst, int count)
        {
            Vector128<float> SqDistanceVector = Sse.SetZeroVector128();

            unsafe
            {
                fixed (float* psrc = src)
                fixed (float* pdst = dst)
                {
                    float* pSrcCurrent = psrc;
                    float* pDstCurrent = pdst;
                    float* pEnd = psrc + count;

                    while (pSrcCurrent + 4 <= pEnd)
                    {
                        Vector128<float> distanceVector = Sse.Subtract(Sse.LoadVector128(pSrcCurrent), Sse.LoadVector128(pDstCurrent));
                        SqDistanceVector = Sse.Add(SqDistanceVector, Sse.Multiply(distanceVector, distanceVector));

                        pSrcCurrent += 4;
                        pDstCurrent += 4;
                    }

                    // TODO: Check code length per line
                    if (Sse3.IsSupported)
                    {
                        // SSE3 is supported.
                        SqDistanceVector = Sse3.HorizontalAdd(SqDistanceVector, SqDistanceVector);
                        SqDistanceVector = Sse3.HorizontalAdd(SqDistanceVector, SqDistanceVector);
                    }
                    else
                    {
                        // SSE3 is not supported.
                        SqDistanceVector = Sse.Add(SqDistanceVector, Sse.MoveHighToLow(SqDistanceVector, SqDistanceVector));
                        SqDistanceVector = Sse.Add(SqDistanceVector, Sse.MoveHighToLow(SqDistanceVector, Sse.UnpackLow(SqDistanceVector, SqDistanceVector)));
                    }

                    float norm = Sse.ConvertToSingle(SqDistanceVector);
                    while (pSrcCurrent < pEnd)
                    {
                        float distance = *pSrcCurrent - *pDstCurrent;
                        norm += distance * distance;

                        pSrcCurrent++;
                        pDstCurrent++;
                    }

                    return norm;
                }
            }
        }

        public static float Dist2(float[] src, float[] dst, int count)
        {
            if (Sse.IsSupported)
            {
                return Dist2Sse(src, dst, count);
            }
            else
            {
                float norm = 0;
                for (int i = 0; i < count; i++)
                {
                    float distance = src[i] - dst[i];
                    norm += distance * distance;
                }
                return norm;
            }
        }

        private static float SumAbsUSse(float[] src, int count)
        {
            Vector128<float> result = Sse.SetZeroVector128();
            Vector128<float> mask;

            if (Sse2.IsSupported)
            {
                mask = Sse.StaticCast<int, float>(Sse2.SetAllVector128(0x7FFFFFFF));
            }
            else
            {
                mask = Sse.SetAllVector128(0x7FFFFFFF);
            }

            unsafe
            {
                fixed (float* psrc = src)
                {
                    float* pSrcCurrent = psrc;
                    float* pEnd = psrc + count;

                    while (pSrcCurrent + 4 <= pEnd)
                    {
                        Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                        result = Sse.Add(result, Sse.And(srcVector, mask));

                        pSrcCurrent += 4;
                    }

                    // TODO: Check code length per line
                    if (Sse3.IsSupported)
                    {
                        // SSE3 is supported.
                        result = Sse3.HorizontalAdd(result, result);
                        result = Sse3.HorizontalAdd(result, result);
                    }
                    else
                    {
                        // SSE3 is not supported.
                        result = Sse.Add(result, Sse.MoveHighToLow(result, result));
                        result = Sse.Add(result, Sse.MoveHighToLow(result, Sse.UnpackLow(result, result)));
                    }

                    while (pSrcCurrent < pEnd)
                    {
                        Vector128<float> srcVector = Sse.LoadVector128(pSrcCurrent);
                        result = Sse.Add(result, Sse.And(srcVector, mask));

                        pSrcCurrent++;
                    }
                }
            }

            return Sse.ConvertToSingle(result);
        }

        public static float SumAbsU(float[] src, int count)
        {
            if (Sse.IsSupported)
            {
                return SumAbsUSse(src, count);
            }
            else
            {
                float sum = 0;
                for (int i = 0; i < count; i++)
                {
                    sum += Math.Abs(src[i]);
                }
                return sum;
            }
        }

        private static void MulElementWiseUSse(float[] src1, float[] src2, float[] dst, int count)
        {
            unsafe
            {
                fixed (float* psrc1 = &src1[0])
                fixed (float* psrc2 = &src2[0])
                fixed (float* pdst = dst)
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
        }

        public static void MulElementWiseU(float[] src1, float[] src2, float[] dst, int count)
        {
            if (Sse.IsSupported)
            {
                MulElementWiseUSse(src1, src2, dst, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    dst[i] = src1[i] * src2[i];
                }
            }
        }
    }
}
