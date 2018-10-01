// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.CompilerServices;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    public static partial class CpuMathUtils
    {
        // The count of bytes in Vector128<T>, corresponding to _cbAlign in AlignedArray
        private const int Vector128Alignment = 16;

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static int GetVectorAlignment()
            => Vector128Alignment;

        public static void MatrixTimesSource(bool transpose, bool add, AlignedArray matrix, AlignedArray source, AlignedArray destination, int stride) => SseUtils.MatTimesSrc(transpose, add, matrix, source, destination, stride);

        public static void MatrixTimesSource(bool transpose, bool add, AlignedArray matrix, int[] rgposSrc, AlignedArray srcValues,
            int posMin, int iposMin, int iposLim, AlignedArray destination, int stride) => SseUtils.MatTimesSrc(transpose, add, matrix, rgposSrc, srcValues, posMin, iposMin, iposLim, destination, stride);

        public static void Add(float value, float[] destination, int count) => SseUtils.Add(value, destination, count);

        public static void Scale(float value, float[] destination, int count) => SseUtils.Scale(value, destination, count);

        public static void Scale(float value, float[] destination, int offset, int count) => SseUtils.Scale(value, destination, offset, count);

        public static void Scale(float value, float[] source, float[] destination, int count) => SseUtils.Scale(value, source, destination, count);

        public static void ScaleAdd(float value, float addend, float[] destination, int count) => SseUtils.ScaleAdd(value, addend, destination, count);

        public static void AddScale(float value, float[] source, float[] destination, int count) => SseUtils.AddScale(value, source, destination, count);

        public static void AddScale(float value, float[] source, float[] destination, int dstOffset, int count) => SseUtils.AddScale(value, source, destination, dstOffset, count);

        public static void AddScale(float value, float[] source, int[] indices, float[] destination, int count) => SseUtils.AddScale(value, source, indices, destination, count);

        public static void AddScale(float value, float[] source, int[] indices, float[] destination, int dstOffset, int count) => SseUtils.AddScale(value, source, indices, destination, dstOffset, count);

        public static void AddScaleCopy(float value, float[] source, float[] destination, float[] res, int count) => SseUtils.AddScaleCopy(value, source, destination, res, count);

        public static void Add(float[] source, float[] destination, int count) => SseUtils.Add(source, destination, count);

        public static void Add(float[] source, int[] indices, float[] destination, int count) => SseUtils.Add(source, indices, destination, count);

        public static void Add(float[] source, int[] indices, float[] destination, int dstOffset, int count) => SseUtils.Add(source, indices, destination, dstOffset, count);

        public static void MulElementWise(float[] left, float[] right, float[] destination, int count) => SseUtils.MulElementWise(left, right, destination, count);

        public static float Sum(float[] source, int count) => SseUtils.Sum(source, count);

        public static float Sum(float[] source, int offset, int count) => SseUtils.Sum(source, offset, count);

        public static float SumSq(float[] source, int count) => SseUtils.SumSq(source, count);

        public static float SumSq(float[] source, int offset, int count) => SseUtils.SumSq(source, offset, count);

        public static float SumSq(float mean, float[] source, int offset, int count) => SseUtils.SumSq(mean, source, offset, count);

        public static float SumAbs(float[] source, int count) => SseUtils.SumAbs(source, count);

        public static float SumAbs(float[] source, int offset, int count) => SseUtils.SumAbs(source, offset, count);

        public static float SumAbs(float mean, float[] source, int offset, int count) => SseUtils.SumAbs(mean, source, offset, count);

        public static float MaxAbs(float[] source, int count) => SseUtils.MaxAbs(source, count);

        public static float MaxAbs(float[] source, int offset, int count) => SseUtils.MaxAbs(source, offset, count);

        public static float MaxAbsDiff(float mean, float[] source, int count) => SseUtils.MaxAbsDiff(mean, source, count);

        public static float DotProductDense(float[] value, float[] b, int count) => SseUtils.DotProductDense(value, b, count);

        public static float DotProductDense(float[] value, int offset, float[] b, int count) => SseUtils.DotProductDense(value, offset, b, count);

        public static float DotProductSparse(float[] value, float[] b, int[] indices, int count) => SseUtils.DotProductSparse(value, b, indices, count);

        public static float DotProductSparse(float[] value, int offset, float[] b, int[] indices, int count) => SseUtils.DotProductSparse(value, offset, b, indices, count);

        public static float L2DistSquared(float[] value, float[] b, int count) => SseUtils.L2DistSquared(value, b, count);

        public static void ZeroMatrixItems(AlignedArray destination, int ccol, int cfltRow, int[] indices) => SseUtils.ZeroMatrixItems(destination, ccol, cfltRow, indices);

        public static void SdcaL1UpdateDense(float primalUpdate, int length, float[] source, float threshold, float[] v, float[] w)
            => SseUtils.SdcaL1UpdateDense(primalUpdate, length, source, threshold, v, w);

        public static void SdcaL1UpdateSparse(float primalUpdate, int length, float[] source, int[] indices, int count, float threshold, float[] v, float[] w)
            => SseUtils.SdcaL1UpdateSparse(primalUpdate, length, source, indices, count, threshold, v, w);
    }
}
