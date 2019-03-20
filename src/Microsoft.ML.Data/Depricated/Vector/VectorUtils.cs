// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Numeric
{
    /// <summary>
    /// A series of vector utility functions, generally operating over arrays or <see cref="VBuffer{T}"/>
    /// structures. The convention is that if a array or buffer is not modified, that is, it is treated
    /// as a constant, it might have the name <c>a</c> or <c>b</c> or <c>src</c>, but in a situation
    /// where the vector structure might be changed the parameter might have the name <c>dst</c>.
    /// </summary>
    [BestFriend]
    internal static partial class VectorUtils
    {
        public static float DotProduct(float[] a, float[] b)
        {
            Contracts.Check(Utils.Size(a) == Utils.Size(b), "Arrays must have the same length");
            Contracts.Check(Utils.Size(a) > 0);
            return CpuMathUtils.DotProductDense(a, b, a.Length);
        }

        public static float DotProduct(float[] a, in VBuffer<float> b)
        {
            Contracts.Check(Utils.Size(a) == b.Length, "Vectors must have the same dimensionality.");
            var bValues = b.GetValues();
            if (bValues.Length == 0)
                return 0;
            if (b.IsDense)
                return CpuMathUtils.DotProductDense(a, bValues, b.Length);
            return CpuMathUtils.DotProductSparse(a, bValues, b.GetIndices(), bValues.Length);
        }

        public static float DotProduct(in VBuffer<float> a, in VBuffer<float> b)
        {
            Contracts.Check(a.Length == b.Length, "Vectors must have the same dimensionality.");

            var aValues = a.GetValues();
            var bValues = b.GetValues();
            if (aValues.Length == 0 || bValues.Length == 0)
                return 0;

            if (a.IsDense)
            {
                if (b.IsDense)
                    return CpuMathUtils.DotProductDense(aValues, bValues, a.Length);
                return CpuMathUtils.DotProductSparse(aValues, bValues, b.GetIndices(), bValues.Length);
            }

            if (b.IsDense)
                return CpuMathUtils.DotProductSparse(bValues, aValues, a.GetIndices(), aValues.Length);
            return DotProductSparse(aValues, a.GetIndices(), 0, aValues.Length, bValues, b.GetIndices(), 0, bValues.Length);
        }

        /// <summary>
        ///  Sparsify vector A (keep at most <paramref name="top"/>+<paramref name="bottom"/> values)
        /// and optionally rescale values to the [-1, 1] range.
        /// <param name="a">Vector to be sparsified and normalized.</param>
        /// <param name="top">How many top (positive) elements to preserve after sparsification.</param>
        /// <param name="bottom">How many bottom (negative) elements to preserve after sparsification.</param>
        /// <param name="normalize">Whether to normalize results to [-1,1] range.</param>
        /// </summary>
        public static void SparsifyNormalize(ref VBuffer<float> a, int top, int bottom, bool normalize)
        {
            Contracts.CheckParam(top >= 0, nameof(top), "Top count needs to be non-negative");
            Contracts.CheckParam(bottom >= 0, nameof(bottom), "Bottom count needs to be non-negative");
            float absMax = 0;

            // In the top heap, we pop the smallest values, so that the 'top' largest remain.
            var topHeap = new Heap<KeyValuePair<int, float>>((left, right) => right.Value < left.Value, top + 1);
            var bottomHeap = new Heap<KeyValuePair<int, float>>((left, right) => right.Value > left.Value, bottom + 1);
            bool isDense = a.IsDense;

            var aValues = a.GetValues();
            var aIndices = a.GetIndices();
            for (int i = 0; i < aValues.Length; i++)
            {
                int idx = isDense ? i : aIndices[i];
                var value = aValues[i];

                if (value < 0 && bottom > 0)
                {
                    if (bottomHeap.Count == bottom && value > bottomHeap.Top.Value)
                        continue;

                    bottomHeap.Add(new KeyValuePair<int, float>(idx, value));
                    if (bottomHeap.Count > bottom)
                    {
                        bottomHeap.Pop();
                        Contracts.Assert(bottomHeap.Count == bottom);
                    }
                }

                if (value > 0 && top > 0)
                {
                    if (topHeap.Count == top && value < topHeap.Top.Value)
                        continue;

                    topHeap.Add(new KeyValuePair<int, float>(idx, value));
                    if (topHeap.Count > top)
                    {
                        topHeap.Pop();
                        Contracts.Assert(topHeap.Count == top);
                    }
                }
            }

            var newCount = topHeap.Count + bottomHeap.Count;
            var aEditor = VBufferEditor.Create(ref a, a.Length, newCount, requireIndicesOnDense: true);
            int count = 0;
            while (topHeap.Count > 0)
            {
                var pair = topHeap.Pop();
                aEditor.Indices[count] = pair.Key;
                aEditor.Values[count++] = pair.Value;
            }

            while (bottomHeap.Count > 0)
            {
                var pair = bottomHeap.Pop();
                aEditor.Indices[count] = pair.Key;
                aEditor.Values[count++] = pair.Value;
            }

            Contracts.Assert(count == newCount);

            if (normalize)
            {
                for (var i = 0; i < newCount; i++)
                {
                    var value = aEditor.Values[i];
                    var absValue = Math.Abs(value);
                    if (absValue > absMax)
                        absMax = absValue;
                }

                if (absMax != 0)
                {
                    var ratio = 1 / absMax;
                    for (var i = 0; i < newCount; i++)
                        aEditor.Values[i] = ratio * aEditor.Values[i];
                }
            }

            if (!aEditor.Indices.IsEmpty)
                GenericSpanSortHelper<int>.Sort(aEditor.Indices, aEditor.Values, 0, newCount);
            a = aEditor.Commit();
        }

        /// <summary>
        /// Multiplies arrays Dst *= A element by element and returns the result in <paramref name="dst"/> (Hadamard product).
        /// </summary>
        public static void MulElementWise(in VBuffer<float> a, ref VBuffer<float> dst)
        {
            Contracts.Check(a.Length == dst.Length, "Vectors must have the same dimensionality.");

            if (a.IsDense && dst.IsDense)
            {
                var editor = VBufferEditor.CreateFromBuffer(ref dst);
                CpuMathUtils.MulElementWise(a.GetValues(), dst.GetValues(), editor.Values, a.Length);
            }
            else
                VBufferUtils.ApplyWithEitherDefined(in a, ref dst, (int ind, float v1, ref float v2) => { v2 *= v1; });
        }

        private static float L2DistSquaredSparse(ReadOnlySpan<float> valuesA, ReadOnlySpan<int> indicesA, ReadOnlySpan<float> valuesB, ReadOnlySpan<int> indicesB)
        {
            Contracts.Assert(valuesA.Length == indicesA.Length);
            Contracts.Assert(valuesB.Length == indicesB.Length);

            float res = 0;

            int ia = 0;
            int ib = 0;
            while (ia < indicesA.Length && ib < indicesB.Length)
            {
                int diff = indicesA[ia] - indicesB[ib];
                float d;
                if (diff == 0)
                {
                    d = valuesA[ia] - valuesB[ib];
                    ia++;
                    ib++;
                }
                else if (diff < 0)
                {
                    d = valuesA[ia];
                    ia++;
                }
                else
                {
                    d = valuesB[ib];
                    ib++;
                }
                res += d * d;
            }

            while (ia < indicesA.Length)
            {
                var d = valuesA[ia];
                res += d * d;
                ia++;
            }

            while (ib < indicesB.Length)
            {
                var d = valuesB[ib];
                res += d * d;
                ib++;
            }

            return res;
        }

        private static float L2DistSquaredHalfSparse(ReadOnlySpan<float> valuesA, ReadOnlySpan<float> valuesB, ReadOnlySpan<int> indicesB)
        {
            var normA = CpuMathUtils.SumSq(valuesA);
            if (valuesB.Length == 0)
                return normA;
            var normB = CpuMathUtils.SumSq(valuesB);
            var dotP = CpuMathUtils.DotProductSparse(valuesA, valuesB, indicesB, valuesB.Length);
            var res = normA + normB - 2 * dotP;
            return res < 0 ? 0 : res;
        }

        private static float L2DiffSquaredDense(ReadOnlySpan<float> valuesA, ReadOnlySpan<float> valuesB, int length)
        {
            Contracts.Assert(0 <= length && length <= valuesA.Length);
            Contracts.Assert(0 <= length && length <= valuesB.Length);

            if (length == 0)
                return 0;
            return CpuMathUtils.L2DistSquared(valuesA, valuesB, length);
        }

        /// <summary>
        /// Computes the dot product of two arrays
        /// Where "offset" is considered to be a's zero index
        /// </summary>
        /// <param name="a">one array</param>
        /// <param name="b">the second array (given as a VBuffer)</param>
        /// <param name="offset">offset in 'a'</param>
        /// <returns>the dot product</returns>
        public static float DotProductWithOffset(in VBuffer<float> a, int offset, in VBuffer<float> b)
        {
            Contracts.Check(0 <= offset && offset <= a.Length);
            Contracts.Check(b.Length <= a.Length - offset, "VBuffer b must be no longer than a.Length - offset.");

            var aValues = a.GetValues();
            var bValues = b.GetValues();
            if (aValues.Length == 0 || bValues.Length == 0)
                return 0;
            if (a.IsDense)
            {
                if (b.IsDense)
                    return CpuMathUtils.DotProductDense(aValues.Slice(offset), bValues, b.Length);
                return CpuMathUtils.DotProductSparse(aValues.Slice(offset), bValues, b.GetIndices(), bValues.Length);
            }
            else
            {
                float result = 0;
                var aIndices = a.GetIndices();
                int aMin = Utils.FindIndexSorted(aIndices, 0, aIndices.Length, offset);
                int aLim = Utils.FindIndexSorted(aIndices, 0, aIndices.Length, offset + b.Length);
                if (b.IsDense)
                {
                    for (int iA = aMin; iA < aLim; ++iA)
                        result += aValues[iA] * bValues[aIndices[iA] - offset];
                    return result;
                }
                var bIndices = b.GetIndices();
                for (int iA = aMin, iB = 0; iA < aLim && iB < bIndices.Length; )
                {
                    int aIndex = aIndices[iA];
                    int bIndex = bIndices[iB];
                    int comp = (aIndex - offset) - bIndex;
                    if (comp == 0)
                        result += aValues[iA++] * bValues[iB++];
                    else if (comp < 0)
                        iA++;
                    else
                        iB++;
                }
                return result;
            }
        }

        /// <summary>
        /// Computes the dot product of two arrays
        /// Where "offset" is considered to be a's zero index
        /// </summary>
        /// <param name="a">one array</param>
        /// <param name="b">the second array (given as a VBuffer)</param>
        /// <param name="offset">offset in 'a'</param>
        /// <returns>the dot product</returns>
        public static float DotProductWithOffset(float[] a, int offset, in VBuffer<float> b)
        {
            Contracts.Check(0 <= offset && offset <= a.Length);
            Contracts.Check(b.Length <= a.Length - offset, "VBuffer b must be no longer than a.Length - offset.");

            var bValues = b.GetValues();
            if (bValues.Length == 0)
                return 0;

            if (b.IsDense)
                return CpuMathUtils.DotProductDense(a.AsSpan(offset), bValues, b.Length);
            return CpuMathUtils.DotProductSparse(a.AsSpan(offset), bValues, b.GetIndices(), bValues.Length);
        }

        private static float DotProductSparse(ReadOnlySpan<float> aValues, ReadOnlySpan<int> aIndices, int ia, int iaLim, ReadOnlySpan<float> bValues, ReadOnlySpan<int> bIndices, int ib, int ibLim)
        {
            Contracts.AssertNonEmpty(aValues);
            Contracts.AssertNonEmpty(aIndices);
            Contracts.AssertNonEmpty(bValues);
            Contracts.AssertNonEmpty(bIndices);
            Contracts.Assert(0 <= ia && ia < iaLim && iaLim <= aIndices.Length);
            Contracts.Assert(0 <= ib && ib < ibLim && ibLim <= bIndices.Length);

            float res = 0;

            // Do binary searches when the indices mismatch by more than this.
            const int thresh = 20;

            for (; ; )
            {
                int d = aIndices[ia] - bIndices[ib];
                if (d == 0)
                {
                    res += aValues[ia] * bValues[ib];
                    if (++ia >= iaLim)
                        break;
                    if (++ib >= ibLim)
                        break;
                }
                else if (d < 0)
                {
                    ia++;
                    if (d < -thresh)
                        ia = Utils.FindIndexSorted(aIndices, ia, iaLim, bIndices[ib]);
                    if (ia >= iaLim)
                        break;
                }
                else
                {
                    ib++;
                    if (d > thresh)
                        ib = Utils.FindIndexSorted(bIndices, ib, ibLim, aIndices[ia]);
                    if (ib >= ibLim)
                        break;
                }
            }

            return res;
        }

        /// <summary>
        /// Computes the L1 distance between two VBuffers
        /// </summary>
        /// <param name="a">one VBuffer</param>
        /// <param name="b">another VBuffer</param>
        /// <returns>L1 Distance from a to b</returns>
        public static float L1Distance(in VBuffer<float> a, in VBuffer<float> b)
        {
            float res = 0;
            VBufferUtils.ForEachEitherDefined(in a, in b,
                (slot, val1, val2) => res += Math.Abs(val1 - val2));
            return res;
        }

        /// <summary>
        /// Computes the Euclidean distance between two VBuffers
        /// </summary>
        /// <param name="a">one VBuffer</param>
        /// <param name="b">another VBuffer</param>
        /// <returns>Distance from a to b</returns>
        public static float Distance(in VBuffer<float> a, in VBuffer<float> b)
        {
            return MathUtils.Sqrt(L2DistSquared(in a, in b));
        }

        /// <summary>
        /// Computes the Euclidean distance squared between two VBuffers
        /// </summary>
        /// <param name="a">one VBuffer</param>
        /// <param name="b">another VBuffer</param>
        /// <returns>Distance from a to b</returns>
        public static float L2DistSquared(in VBuffer<float> a, in VBuffer<float> b)
        {
            Contracts.Check(a.Length == b.Length, "Vectors must have the same dimensionality.");
            if (a.IsDense)
            {
                if (b.IsDense)
                    return L2DiffSquaredDense(a.GetValues(), b.GetValues(), b.Length);
                return L2DistSquaredHalfSparse(a.GetValues(), b.GetValues(), b.GetIndices());
            }
            if (b.IsDense)
                return L2DistSquaredHalfSparse(b.GetValues(), a.GetValues(), a.GetIndices());
            return L2DistSquaredSparse(a.GetValues(), a.GetIndices(), b.GetValues(), b.GetIndices());
        }

        /// <summary>
        /// Given two vectors a and b, calculate their L2 distance squared (|a-b|^2).
        /// </summary>
        /// <param name="a">The first vector, given as an array</param>
        /// <param name="b">The second vector, given as a VBuffer{float}</param>
        /// <returns>The squared L2 distance between a and b</returns>
        public static float L2DistSquared(float[] a, in VBuffer<float> b)
        {
            Contracts.CheckValue(a, nameof(a));
            Contracts.Check(Utils.Size(a) == b.Length, "Vectors must have the same dimensionality.");
            if (b.IsDense)
                return L2DiffSquaredDense(a, b.GetValues(), b.Length);
            return L2DistSquaredHalfSparse(a.AsSpan(0, a.Length), b.GetValues(), b.GetIndices());
        }

        /// <summary>
        /// Perform in-place vector addition <c><paramref name="dst"/> += <paramref name="src"/></c>.
        /// </summary>
        public static void Add(float[] src, float[] dst)
        {
            Contracts.CheckValue(src, nameof(src));
            Contracts.CheckValue(dst, nameof(dst));
            Contracts.CheckParam(src.Length == dst.Length, nameof(dst), "Arrays must have the same dimensionality.");
            if (src.Length == 0)
                return;
            CpuMathUtils.Add(src, dst, src.Length);
        }

        /// <summary>
        /// Adds a multiple of a <see cref="VBuffer{T}"/> to a <see cref="float"/> array.
        /// </summary>
        /// <param name="src">Buffer to add</param>
        /// <param name="dst">Span to add to</param>
        /// <param name="c">Coefficient</param>
        public static void AddMult(in VBuffer<float> src, Span<float> dst, float c)
        {
            Contracts.CheckParam(src.Length == dst.Length, nameof(dst), "Arrays must have the same dimensionality.");

            var srcValues = src.GetValues();
            if (srcValues.Length == 0 || c == 0)
                return;

            if (src.IsDense)
                CpuMathUtils.AddScale(c, srcValues, dst, srcValues.Length);
            else
            {
                var srcIndices = src.GetIndices();
                for (int i = 0; i < srcValues.Length; i++)
                    dst[srcIndices[i]] += c * srcValues[i];
            }
        }

        /// <summary>
        /// Adds a multiple of a <see cref="VBuffer{T}"/> to a <see cref="float"/> array, with an offset into the destination.
        /// </summary>
        /// <param name="src">Buffer to add</param>
        /// <param name="dst">Array to add to</param>
        /// <param name="offset">The offset into <paramref name="dst"/> at which to add</param>
        /// <param name="c">Coefficient</param>

        public static void AddMultWithOffset(in VBuffer<float> src, float[] dst, int offset, float c)
        {
            Contracts.CheckValue(dst, nameof(dst));
            Contracts.Check(0 <= offset && offset <= dst.Length);
            Contracts.Check(src.Length <= dst.Length - offset, "Vector src must be no longer than dst.Length - offset.");

            var srcValues = src.GetValues();
            if (srcValues.Length == 0 || c == 0)
                return;

            if (src.IsDense)
            {
                for (int i = 0; i < src.Length; i++)
                    dst[i + offset] += c * srcValues[i];
            }
            else
            {
                var srcIndices = src.GetIndices();
                for (int i = 0; i < srcValues.Length; i++)
                    dst[srcIndices[i] + offset] += c * srcValues[i];
            }
        }

        /// <summary>
        /// Adds a multiple of an array to a second array.
        /// </summary>
        /// <param name="src">Array to add</param>
        /// <param name="dst">Array to add to</param>
        /// <param name="c">Multiple</param>
        public static void AddMult(float[] src, float[] dst, float c)
        {
            Contracts.Check(src.Length == dst.Length, "Arrays must have the same dimensionality.");

            if (c == 0)
                return;

            CpuMathUtils.AddScale(c, src, dst, src.Length);
        }

        /// <summary>
        /// Returns the L2 norm of the vector (sum of squares of the components).
        /// </summary>
        public static float Norm(float[] a)
        {
            return MathUtils.Sqrt(CpuMathUtils.SumSq(a));
        }

        /// <summary>
        /// Returns sum of elements in array
        /// </summary>
        public static float Sum(float[] a)
        {
            if (a == null || a.Length == 0)
                return 0;
            return CpuMathUtils.Sum(a);
        }

        /// <summary>
        /// Multiples the array by a real value
        /// </summary>
        /// <param name="dst">The array</param>
        /// <param name="c">Value to multiply vector with</param>
        public static void ScaleBy(float[] dst, float c)
        {
            if (c == 1)
                return;

            if (c != 0)
                CpuMathUtils.Scale(c, dst);
            else
                Array.Clear(dst, 0, dst.Length);
        }

        public static float Distance(float[] a, float[] b)
        {
            Contracts.Check(a.Length == b.Length, "Arrays must have the same dimensionality.");

            float res = 0;
            for (int i = 0; i < a.Length; i++)
            {
                var diff = a[i] - b[i];
                res += diff * diff;
            }

            return res;
        }
    }
}
