// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    public static partial class CpuMathUtils
    {
        // REVIEW NEEDED: AVX support cannot be checked in .NET Standard 2.0, so we assume Vector128 alignment for SSE.  Is it okay?

        // The count of bytes in Vector128<T>, corresponding to _cbAlign in AlignedArray
        private const int Vector128Alignment = 16;

        public static int GetVectorAlignment()
        {
            // Assumes SSE support on machines that run ML.NET.
            return Vector128Alignment;
        }

        public static void MatTimesSrc(bool tran, bool add, AlignedArray mat, AlignedArray src, AlignedArray dst, int crun) => SseUtils.MatTimesSrc(tran, add, mat, src, dst, crun);

        public static void MatTimesSrc(bool tran, bool add, AlignedArray mat, int[] rgposSrc, AlignedArray srcValues,
            int posMin, int iposMin, int iposLim, AlignedArray dst, int crun) => SseUtils.MatTimesSrc(tran, add, mat, rgposSrc, srcValues, posMin, iposMin, iposLim, dst, crun);

        public static void Add(float a, float[] dst, int count) => SseUtils.Add(a, dst, count);

        public static void Scale(float a, float[] dst, int count) => SseUtils.Scale(a, dst, count);

        public static void Scale(float a, float[] dst, int offset, int count) => SseUtils.Scale(a, dst, offset, count);

        public static void Scale(float a, float[] src, float[] dst, int count) => SseUtils.Scale(a, src, dst, count);

        public static void ScaleAdd(float a, float b, float[] dst, int count) => SseUtils.ScaleAdd(a, b, dst, count);

        public static void AddScale(float a, float[] src, float[] dst, int count) => SseUtils.AddScale(a, src, dst, count);

        public static void AddScale(float a, float[] src, float[] dst, int dstOffset, int count) => SseUtils.AddScale(a, src, dst, dstOffset, count);

        public static void AddScale(float a, float[] src, int[] indices, float[] dst, int count) => SseUtils.AddScale(a, src, indices, dst, count);

        public static void AddScale(float a, float[] src, int[] indices, float[] dst, int dstOffset, int count) => SseUtils.AddScale(a, src, indices, dst, dstOffset, count);

        public static void AddScaleCopy(float a, float[] src, float[] dst, float[] res, int count) => SseUtils.AddScaleCopy(a, src, dst, res, count);

        public static void Add(float[] src, float[] dst, int count) => SseUtils.Add(src, dst, count);

        public static void Add(float[] src, int[] indices, float[] dst, int count) => SseUtils.Add(src, indices, dst, count);

        public static void Add(float[] src, int[] indices, float[] dst, int dstOffset, int count) => SseUtils.Add(src, indices, dst, dstOffset, count);

        public static void MulElementWise(float[] src1, float[] src2, float[] dst, int count) => SseUtils.MulElementWise(src1, src2, dst, count);

        public static float Sum(float[] src, int count) => SseUtils.Sum(src, count);

        public static float Sum(float[] src, int offset, int count) => SseUtils.Sum(src, offset, count);

        public static float SumSq(float[] src, int count) => SseUtils.SumSq(src, count);

        public static float SumSq(float[] src, int offset, int count) => SseUtils.SumSq(src, offset, count);

        public static float SumSq(float mean, float[] src, int offset, int count) => SseUtils.SumSq(mean, src, offset, count);

        public static float SumAbs(float[] src, int count) => SseUtils.SumAbs(src, count);

        public static float SumAbs(float[] src, int offset, int count) => SseUtils.SumAbs(src, offset, count);

        public static float SumAbs(float mean, float[] src, int offset, int count) => SseUtils.SumAbs(mean, src, offset, count);

        public static float MaxAbs(float[] src, int count) => SseUtils.MaxAbs(src, count);

        public static float MaxAbs(float[] src, int offset, int count) => SseUtils.MaxAbs(src, offset, count);

        public static float MaxAbsDiff(float mean, float[] src, int count) => SseUtils.MaxAbsDiff(mean, src, count);

        public static float DotProductDense(float[] a, float[] b, int count) => SseUtils.DotProductDense(a, b, count);

        public static float DotProductDense(float[] a, int offset, float[] b, int count) => SseUtils.DotProductDense(a, offset, b, count);

        public static float DotProductSparse(float[] a, float[] b, int[] indices, int count) => SseUtils.DotProductSparse(a, b, indices, count);

        public static float DotProductSparse(float[] a, int offset, float[] b, int[] indices, int count) => SseUtils.DotProductSparse(a, offset, b, indices, count);

        public static float L2DistSquared(float[] a, float[] b, int count) => SseUtils.L2DistSquared(a, b, count);

        public static void ZeroMatrixItems(AlignedArray dst, int ccol, int cfltRow, int[] indices) => SseUtils.ZeroMatrixItems(dst, ccol, cfltRow, indices);

        public static void SdcaL1UpdateDense(float primalUpdate, int length, float[] src, float threshold, float[] v, float[] w)
            => SseUtils.SdcaL1UpdateDense(primalUpdate, length, src, threshold, v, w);

        public static void SdcaL1UpdateSparse(float primalUpdate, int length, float[] src, int[] indices, int count, float threshold, float[] v, float[] w)
            => SseUtils.SdcaL1UpdateSparse(primalUpdate, length, src, indices, count, threshold, v, w);
    }
}
