// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.Intrinsics.X86;
using System;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    public static partial class CpuMathUtils
    {
        public static void Scale(float a, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 < count && count <= dst.Length);

            Scale(a, new Span<float>(dst, 0, count));
        }

        public static void Scale(float a, float[] dst, int offset, int count)
        {
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset < dst.Length - count);

            Scale(a, new Span<float>(dst, offset, count));
        }

        private static void Scale(float a, Span<float> dst)
        {
            if (Sse.IsSupported)
            {
                SseIntrinsics.ScaleU(a, dst);
            }
            else
            {
                for (int i = 0; i < dst.Length; i++)
                {
                    dst[i] *= a;
                }
            }
        }

        public static void AddScale(float a, float[] src, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count <= dst.Length);

            AddScale(a, new Span<float>(src, 0, count), new Span<float>(dst, 0, count));
        }

        public static void AddScale(float a, float[] src, float[] dst, int dstOffset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(count <= src.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 <= dstOffset && dstOffset < dst.Length);
            Contracts.Assert(0 < count && count <= dst.Length - dstOffset);

            AddScale(a, new Span<float>(src, 0, count), new Span<float>(dst, dstOffset, count));
        }

        private static void AddScale(float a, Span<float> src, Span<float> dst)
        {
            if (Sse.IsSupported)
            {
                SseIntrinsics.AddScaleU(a, src, dst);
            }
            else
            {
                for (int i = 0; i < dst.Length; i++)
                {
                    dst[i] += a * src[i];
                }
            }
        }

        public static void AddScale(float a, float[] src, int[] indices, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count <= indices.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count < dst.Length);

            AddScale(a, new Span<float>(src), new Span<int>(indices, 0, count), new Span<float>(dst));
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

            AddScale(a, new Span<float>(src), new Span<int>(indices, 0, count), 
                    new Span<float>(dst, dstOffset, dst.Length - dstOffset));
        }

        private static void AddScale(float a, Span<float> src, Span<int> indices, Span<float> dst)
        {
            if (Sse.IsSupported)
            {
                SseIntrinsics.AddScaleSU(a, src, indices, dst);
            }
            else
            {
                for (int i = 0; i < indices.Length; i++)
                {
                    int index = indices[i];
                    dst[index] += a * src[i];
                }
            }
        }

        public static void Add(float[] src, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count <= dst.Length);

            Add(new Span<float>(src, 0, count), new Span<float>(dst, 0, count));
        }

        private static void Add(Span<float> src, Span<float> dst)
        {
            if (Sse.IsSupported)
            {
                SseIntrinsics.AddU(src, dst);
            }
            else
            {
                for (int i = 0; i < dst.Length; i++)
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

            Add(new Span<float>(src), new Span<int>(indices, 0, count), new Span<float>(dst));
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

            Add(new Span<float>(src), new Span<int>(indices, 0, count), 
                new Span<float>(dst, dstOffset, dst.Length - dstOffset));
        }

        private static void Add(Span<float> src, Span<int> indices, Span<float> dst)
        {
            if (Sse.IsSupported)
            {
                SseIntrinsics.AddSU(src, indices, dst);
            }
            else
            {
                for (int i = 0; i < indices.Length; i++)
                {
                    int index = indices[i];
                    dst[index] += src[i];
                }
            }
        }

        public static void MulElementWise(float[] src1, float[] src2, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src1);
            Contracts.Assert(0 < count && count <= src1.Length);
            Contracts.AssertNonEmpty(src2);
            Contracts.Assert(0 < count && count <= src2.Length);
            Contracts.AssertNonEmpty(dst);

            MulElementWise(new Span<float>(src1, 0, count), new Span<float>(src2, 0, count), 
                            new Span<float>(dst, 0, count));
        }

        private static void MulElementWise(Span<float> src1, Span<float> src2, Span<float> dst)
        {
            if (Sse.IsSupported)
            {
                SseIntrinsics.MulElementWiseU(src1, src2, dst);
            }
            else
            {
                for (int i = 0; i < dst.Length; i++)
                {
                    dst[i] = src1[i] * src2[i];
                }
            }
        }

        public static float SumSq(float[] src, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);

            return SumSq(new Span<float>(src, 0, count));
        }

        public static float SumSq(float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= src.Length - count);

            return SumSq(new Span<float>(src, offset, count));
        }

        private static float SumSq(Span<float> src)
        {
            if (Sse.IsSupported)
            {
                return SseIntrinsics.SumSqU(src);
            }
            else
            {
                float result = 0;
                for (int i = 0; i < src.Length; i++)
                {
                    result += src[i] * src[i];
                }
                return result;
            }
        }

        public static float SumAbs(float[] src, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);

            return SumAbs(new Span<float>(src, 0, count));
        }

        public static float SumAbs(float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= src.Length - count);

            return SumAbs(new Span<float>(src, offset, count));
        }

        private static float SumAbs(Span<float> src)
        {
            if (Sse.IsSupported)
            {
                return SseIntrinsics.SumAbsU(src);
            }
            else
            {
                float sum = 0;
                for (int i = 0; i < src.Length; i++)
                {
                    sum += Math.Abs(src[i]);
                }
                return sum;
            }
        }

        public static float DotProductDense(float[] a, float[] b, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(0 < count);
            Contracts.Assert(a.Length >= count);
            Contracts.Assert(b.Length >= count);

            return DotProductDense(new Span<float>(a, 0, count), new Span<float>(b, 0, count));
        }

        public static float DotProductDense(float[] a, int offset, float[] b, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= a.Length - count);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(b.Length >= count);

            return DotProductDense(new Span<float>(a, offset, count), new Span<float>(b, 0, count));
        }

        private static float DotProductDense(Span<float> a, Span<float> b)
        {
            if (Sse.IsSupported)
            {
                return SseIntrinsics.DotU(a, b);
            }
            else
            {
                float result = 0;
                for (int i = 0; i < b.Length; i++)
                {
                    result += a[i] * b[i];
                }
                return result;
            }
        }

        public static float DotProductSparse(float[] a, float[] b, int[] indices, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(0 < count);
            Contracts.Assert(count < a.Length);
            Contracts.Assert(count <= b.Length);
            Contracts.Assert(count <= indices.Length);

            return DotProductSparse(new Span<float>(a), new Span<float>(b), 
                                    new Span<int>(indices, 0, count));
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

            return DotProductSparse(new Span<float>(a, offset, a.Length - offset), 
                                    new Span<float>(b), new Span<int>(indices, 0, count));
        }

        private static float DotProductSparse(Span<float> a, Span<float> b, Span<int> indices)
        {
            if (Sse.IsSupported)
            {
                return SseIntrinsics.DotSU(a, b, indices);
            }
            else
            {
                float result = 0;
                for (int i = 0; i < indices.Length; i++)
                {
                    int index = indices[i];
                    result += a[index] * b[i];
                }
                return result;
            }
        }

        public static float L2DistSquared(float[] a, float[] b, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(0 < count && count <= a.Length);
            Contracts.Assert(count <= b.Length);

            return L2DistSquared(new Span<float>(a, 0, count), new Span<float>(b, 0, count));
        }

        private static float L2DistSquared(Span<float> a, Span<float> b)
        {
            if (Sse.IsSupported)
            {
                return SseIntrinsics.Dist2(a, b);
            }
            else
            {
                float norm = 0;
                for (int i = 0; i < b.Length; i++)
                {
                    float distance = a[i] - b[i];
                    norm += distance * distance;
                }
                return norm;
            }
        }

    }
}
