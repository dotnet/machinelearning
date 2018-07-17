// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    public static partial class CpuMathUtils
    {
        public const int CbAlign = 16;

        private static bool Compat(AlignedArray a)
        {
            Contracts.AssertValue(a);
            Contracts.Assert(a.Size > 0);
            return a.CbAlign == CbAlign;
        }

        private unsafe static float* Ptr(AlignedArray a, float* p)
        {
            Contracts.AssertValue(a);
            float* q = p + a.GetBase((long)p);
            Contracts.Assert(((long)q & (CbAlign - 1)) == 0);
            return q;
        }

        public static void MatTimesSrc(bool tran, bool add, AlignedArray mat, AlignedArray src, AlignedArray dst, int crun)
        {
            Contracts.Assert(Compat(mat));
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(mat.Size == dst.Size * src.Size);

            if (Sse.IsSupported)
            {
                if (!tran)
                {
                    Contracts.Assert(0 <= crun && crun <= dst.Size);
                    SseIntrinsics.MatMulASse(add, mat, src, dst, crun, src.Size);
                }
                else
                {
                    Contracts.Assert(0 <= crun && crun <= src.Size);
                    SseIntrinsics.MatMulTranA(add, mat, src, dst, dst.Size, crun);
                }
            }
            else
            {
                // TODO: Software fallback.
            }
        }

        public static void Scale(float a, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 < count && count <= dst.Length);

            if (Sse.IsSupported)
            {
                SseIntrinsics.ScaleU(a, dst, count);
            }
            else
            {
                for (int i = 0; i < dst.Length; i++)
                {
                    dst[i] *= a;
                }
            }
        }

        public static void Scale(float a, float[] dst, int offset, int count)
        {
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset < dst.Length - count);

            ArraySegment<float> dstSeg = new ArraySegment<float>(dst, offset, count);
            Scale(a, dstSeg.Array, count);
        }

        public static void AddScale(float a, float[] src, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count <= dst.Length);

            if (Sse.IsSupported)
            {
                SseIntrinsics.AddScaleU(a, src, dst, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    dst[i] += a * src[i];
                }
            }
        }

        public static void AddScale(float a, float[] src, float[] dst, int dstOffset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(count <= src.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 <= dstOffset && dstOffset < dst.Length);
            Contracts.Assert(0 < count && count <= dst.Length - dstOffset);

            ArraySegment<float> dstSeg = new ArraySegment<float>(dst, dstOffset, count);
            AddScale(a, src, dstSeg.Array, count);
        }

        public static void AddScale(float a, float[] src, int[] indices, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count <= indices.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count < dst.Length);

            if (Sse.IsSupported)
            {
                SseIntrinsics.AddScaleSU(a, src, indices, dst, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    dst[indices[i]] += a * src[i];
                }
            }
        } 

        public static void AddScale(float a, float[] src, int[] indices, float[] dst, int dstOffset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count <= indices.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 <= dstOffset && dstOffset < dst.Length);
            Contracts.Assert(count < dst.Length - dstOffset);

            ArraySegment<float> dstSeg = new ArraySegment<float>(dst, dstOffset, count);
            AddScale(a, src, indices, dstSeg.Array, count);
        }

        public static void Add(float[] src, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count <= dst.Length);

            if (Sse.IsSupported)
            {
                SseIntrinsics.AddU(src, dst, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    dst[i] += src[i];
                }
            }
        }

        public static void Add(float[] src, int[] indices, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count <= indices.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count < dst.Length);

            if (Sse.IsSupported)
            {
                SseIntrinsics.AddSU(src, indices, dst, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    dst[indices[i]] += src[i];
                }
            }
        }

        public static void Add(float[] src, int[] indices, float[] dst, int dstOffset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count <= indices.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 <= dstOffset && dstOffset < dst.Length);
            Contracts.Assert(count <= dst.Length - dstOffset);

            ArraySegment<float> dstSeg = new ArraySegment<float>(dst, dstOffset, count);
            Add(src, indices, dstSeg.Array, count);
        }

        public static void MulElementWise(float[] src1, float[] src2, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src1);
            Contracts.Assert(0 < count && count <= src1.Length);
            Contracts.AssertNonEmpty(src2);
            Contracts.Assert(0 < count && count <= src2.Length);
            Contracts.AssertNonEmpty(dst);

            if (Sse.IsSupported)
            {
                SseIntrinsics.MulElementWiseU(src1, src2, dst, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    dst[i] = src1[i] * src2[i];
                }
            }
        }

        public static float SumSq(float[] src, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);

            if (Sse.IsSupported)
            {
                return SseIntrinsics.SumSqU(src, count);
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

        public static float SumSq(float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= src.Length - count);

            ArraySegment<float> srcSeg = new ArraySegment<float>(src, offset, count);
            return SumSq(srcSeg.Array, count);
        }

        public static float SumAbs(float[] src, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);

            if (Sse.IsSupported)
            {
                return SseIntrinsics.SumAbsU(src, count);
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

        public static float SumAbs(float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= src.Length - count);

            ArraySegment<float> srcSeg = new ArraySegment<float>(src, offset, count);
            return SumAbs(srcSeg.Array, count);
        }

        public static float DotProductDense(float[] a, float[] b, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(0 < count);
            Contracts.Assert(a.Length >= count);
            Contracts.Assert(b.Length >= count);

            if (Sse.IsSupported)
            {
                return SseIntrinsics.DotU(a, b, count);
            }
            else
            {
                // Software fallback.
                float result = 0;
                for (int i = 0; i < count; i++)
                {
                    result += a[i] * b[i];
                }
                return result;
            }
        }

        public static float DotProductDense(float[] a, int offset, float[] b, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= a.Length - count);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(b.Length >= count);

            ArraySegment<float> aSeg = new ArraySegment<float>(a, offset, count);
            return DotProductDense(aSeg.Array, b, count);
        }

        public static float DotProductSparse(float[] a, float[] b, int[] indices, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(0 < count);
            Contracts.Assert(count < a.Length);
            Contracts.Assert(count <= b.Length);
            Contracts.Assert(count <= indices.Length);

            if (Sse.IsSupported)
            {
                return SseIntrinsics.DotSU(a, b, indices, count);
            }
            else
            {
                // Software fallback.
                float result = 0;
                for (int i = 0; i < count; i++)
                {
                    result += a[indices[i]] * b[i];
                }
                return result;
            }
        }

        public static float DotProductSparse(float[] a, int offset, float[] b, int[] indices, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset < a.Length);
            Contracts.Assert(a.Length - offset > count);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(count <= b.Length);
            Contracts.Assert(count <= indices.Length);

            ArraySegment<float> aSeg = new ArraySegment<float>(a, offset, count);
            return DotProductSparse(aSeg.Array, b, indices, count);
        }

        public static float L2DistSquared(float[] a, float[] b, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(0 < count && count <= a.Length);
            Contracts.Assert(count <= b.Length);

            if (Sse.IsSupported)
            {
                return SseIntrinsics.Dist2(a, b, count);
            }
            else
            {
                float norm = 0;
                for (int i = 0; i < count; i++)
                {
                    float distance = a[i] - b[i];
                    norm += distance * distance;
                }
                return norm;
            }
        }

    }
}
