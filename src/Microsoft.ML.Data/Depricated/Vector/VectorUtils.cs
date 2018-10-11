// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.Internal.Utilities;
using Float = System.Single;

namespace Microsoft.ML.Runtime.Numeric
{
    /// <summary>
    /// A series of vector utility functions, generally operating over arrays or <see cref="VBuffer{T}"/>
    /// structures. The convention is that if a array or buffer is not modified, that is, it is treated
    /// as a constant, it might have the name <c>a</c> or <c>b</c> or <c>src</c>, but in a situation
    /// where the vector structure might be changed the parameter might have the name <c>dst</c>.
    /// </summary>
    public static partial class VectorUtils
    {
        public static Float DotProduct(Float[] a, Float[] b)
        {
            Contracts.Check(Utils.Size(a) == Utils.Size(b), "Arrays must have the same length");
            Contracts.Check(Utils.Size(a) > 0);
            return CpuMathUtils.DotProductDense(a, b, a.Length);
        }

        public static Float DotProduct(Float[] a, ref VBuffer<Float> b)
        {
            Contracts.Check(Utils.Size(a) == b.Length, "Vectors must have the same dimensionality.");
            if (b.Count == 0)
                return 0;
            if (b.IsDense)
                return CpuMathUtils.DotProductDense(a, b.Values, b.Length);
            return CpuMathUtils.DotProductSparse(a, b.Values, b.Indices, b.Count);
        }

        public static Float DotProduct(in ReadOnlyVBuffer<Float> a, in ReadOnlyVBuffer<Float> b)
        {
            Contracts.Check(a.Length == b.Length, "Vectors must have the same dimensionality.");

            if (a.Count == 0 || b.Count == 0)
                return 0;

            if (a.IsDense)
            {
                if (b.IsDense)
                    return CpuMathUtils.DotProductDense(a.Values, b.Values, a.Length);
                return CpuMathUtils.DotProductSparse(a.Values, b.Values, b.Indices, b.Count);
            }

            if (b.IsDense)
                return CpuMathUtils.DotProductSparse(b.Values, a.Values, a.Indices, a.Count);
            return DotProductSparse(a.Values, a.Indices, 0, a.Count, b.Values, b.Indices, 0, b.Count);
        }

        /// <summary>
        ///  Sparsify vector A (keep at most <paramref name="top"/>+<paramref name="bottom"/> values)
        /// and optionally rescale values to the [-1, 1] range.
        /// <param name="a">Vector to be sparsified and normalized.</param>
        /// <param name="top">How many top (positive) elements to preserve after sparsification.</param>
        /// <param name="bottom">How many bottom (negative) elements to preserve after sparsification.</param>
        /// <param name="normalize">Whether to normalize results to [-1,1] range.</param>
        /// </summary>
        public static void SparsifyNormalize(ref VBuffer<Float> a, int top, int bottom, bool normalize)
        {
            Contracts.CheckParam(top >= 0, nameof(top), "Top count needs to be non-negative");
            Contracts.CheckParam(bottom >= 0, nameof(bottom), "Bottom count needs to be non-negative");
            Float absMax = 0;

            // In the top heap, we pop the smallest values, so that the 'top' largest remain.
            var topHeap = new Heap<KeyValuePair<int, Float>>((left, right) => right.Value < left.Value, top + 1);
            var bottomHeap = new Heap<KeyValuePair<int, Float>>((left, right) => right.Value > left.Value, bottom + 1);
            bool isDense = a.IsDense;

            for (int i = 0; i < a.Count; i++)
            {
                int idx = isDense ? i : a.Indices[i];
                var value = a.Values[i];

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
            var indices = a.Indices;
            Utils.EnsureSize(ref indices, newCount);
            Contracts.Assert(Utils.Size(a.Values) >= newCount);
            int count = 0;
            while (topHeap.Count > 0)
            {
                var pair = topHeap.Pop();
                indices[count] = pair.Key;
                a.Values[count++] = pair.Value;
            }

            while (bottomHeap.Count > 0)
            {
                var pair = bottomHeap.Pop();
                indices[count] = pair.Key;
                a.Values[count++] = pair.Value;
            }

            Contracts.Assert(count == newCount);

            if (normalize)
            {
                for (var i = 0; i < newCount; i++)
                {
                    var value = a.Values[i];
                    var absValue = Math.Abs(value);
                    if (absValue > absMax)
                        absMax = absValue;
                }

                if (absMax != 0)
                {
                    var ratio = 1 / absMax;
                    for (var i = 0; i < newCount; i++)
                        a.Values[i] = ratio * a.Values[i];
                }
            }

            if (indices != null)
                Array.Sort(indices, a.Values, 0, newCount);
            a = new VBuffer<float>(a.Length, newCount, a.Values, indices);
        }

        /// <summary>
        /// Multiplies arrays Dst *= A element by element and returns the result in <paramref name="dst"/> (Hadamard product).
        /// </summary>
        public static void MulElementWise(in ReadOnlyVBuffer<Float> a, ref VBuffer<Float> dst)
        {
            Contracts.Check(a.Length == dst.Length, "Vectors must have the same dimensionality.");

            if (a.IsDense && dst.IsDense)
                CpuMathUtils.MulElementWise(a.Values, dst.Values, dst.Values, a.Length);
            else
                VBufferUtils.ApplyWithEitherDefined(in a, ref dst, (int ind, Float v1, ref Float v2) => { v2 *= v1; });
        }

        private static Float L2DistSquaredSparse(Float[] valuesA, int[] indicesA, int countA, Float[] valuesB, int[] indicesB, int countB, int length)
        {
            Contracts.AssertValueOrNull(valuesA);
            Contracts.AssertValueOrNull(indicesA);
            Contracts.AssertValueOrNull(valuesB);
            Contracts.AssertValueOrNull(indicesB);
            Contracts.Assert(0 <= countA && countA <= Utils.Size(indicesA));
            Contracts.Assert(0 <= countB && countB <= Utils.Size(indicesB));
            Contracts.Assert(countA <= Utils.Size(valuesA));
            Contracts.Assert(countB <= Utils.Size(valuesB));

            Float res = 0;

            int ia = 0;
            int ib = 0;
            while (ia < countA && ib < countB)
            {
                int diff = indicesA[ia] - indicesB[ib];
                Float d;
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

            while (ia < countA)
            {
                var d = valuesA[ia];
                res += d * d;
                ia++;
            }

            while (ib < countB)
            {
                var d = valuesB[ib];
                res += d * d;
                ib++;
            }

            return res;
        }

        private static Float L2DistSquaredHalfSparse(Float[] valuesA, int lengthA, Float[] valuesB, int[] indicesB, int countB)
        {
            Contracts.AssertValueOrNull(valuesA);
            Contracts.AssertValueOrNull(valuesB);
            Contracts.AssertValueOrNull(indicesB);
            Contracts.Assert(0 <= lengthA && lengthA <= Utils.Size(valuesA));
            Contracts.Assert(0 <= countB && countB <= Utils.Size(indicesB));
            Contracts.Assert(countB <= Utils.Size(valuesB));

            var normA = CpuMathUtils.SumSq(valuesA.AsSpan(0, lengthA));
            if (countB == 0)
                return normA;
            var normB = CpuMathUtils.SumSq(valuesB.AsSpan(0, countB));
            var dotP = CpuMathUtils.DotProductSparse(valuesA, valuesB, indicesB, countB);
            var res = normA + normB - 2 * dotP;
            return res < 0 ? 0 : res;
        }

        private static Float L2DiffSquaredDense(Float[] valuesA, Float[] valuesB, int length)
        {
            Contracts.AssertValueOrNull(valuesA);
            Contracts.AssertValueOrNull(valuesB);
            Contracts.Assert(0 <= length && length <= Utils.Size(valuesA));
            Contracts.Assert(0 <= length && length <= Utils.Size(valuesB));

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
        public static Float DotProductWithOffset(in ReadOnlyVBuffer<Float> a, int offset, in ReadOnlyVBuffer<Float> b)
        {
            Contracts.Check(0 <= offset && offset <= a.Length);
            Contracts.Check(b.Length <= a.Length - offset, "VBuffer b must be no longer than a.Length - offset.");

            if (a.Count == 0 || b.Count == 0)
                return 0;
            if (a.IsDense)
            {
                if (b.IsDense)
                    return CpuMathUtils.DotProductDense(a.Values.Slice(offset), b.Values, b.Length);
                return CpuMathUtils.DotProductSparse(a.Values.Slice(offset), b.Values, b.Indices, b.Count);
            }
            else
            {
                Float result = 0;
                int aMin = Utils.FindIndexSorted(a.Indices, 0, a.Count, offset);
                int aLim = Utils.FindIndexSorted(a.Indices, 0, a.Count, offset + b.Length);
                if (b.IsDense)
                {
                    for (int iA = aMin; iA < aLim; ++iA)
                        result += a.Values[iA] * b.Values[a.Indices[iA] - offset];
                    return result;
                }
                for (int iA = aMin, iB = 0; iA < aLim && iB < b.Count; )
                {
                    int aIndex = a.Indices[iA];
                    int bIndex = b.Indices[iB];
                    int comp = (aIndex - offset) - bIndex;
                    if (comp == 0)
                        result += a.Values[iA++] * b.Values[iB++];
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
        public static Float DotProductWithOffset(Float[] a, int offset, ref VBuffer<Float> b)
        {
            Contracts.Check(0 <= offset && offset <= a.Length);
            Contracts.Check(b.Length <= a.Length - offset, "VBuffer b must be no longer than a.Length - offset.");

            if (b.Count == 0)
                return 0;

            if (b.IsDense)
                return CpuMathUtils.DotProductDense(a.AsSpan(offset), b.Values, b.Length);
            return CpuMathUtils.DotProductSparse(a.AsSpan(offset), b.Values, b.Indices, b.Count);
        }

        private static Float DotProductSparse(ReadOnlySpan<Float> aValues, ReadOnlySpan<int> aIndices, int ia, int iaLim, ReadOnlySpan<Float> bValues, ReadOnlySpan<int> bIndices, int ib, int ibLim)
        {
            Contracts.AssertNonEmpty(aValues);
            Contracts.AssertNonEmpty(aIndices);
            Contracts.AssertNonEmpty(bValues);
            Contracts.AssertNonEmpty(bIndices);
            Contracts.Assert(0 <= ia && ia < iaLim && iaLim <= aIndices.Length);
            Contracts.Assert(0 <= ib && ib < ibLim && ibLim <= bIndices.Length);

            Float res = 0;

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
        public static Float L1Distance(ref VBuffer<Float> a, ref VBuffer<Float> b)
        {
            Float res = 0;
            VBufferUtils.ForEachEitherDefined(ref a, ref b,
                (slot, val1, val2) => res += Math.Abs(val1 - val2));
            return res;
        }

        /// <summary>
        /// Computes the Euclidean distance between two VBuffers
        /// </summary>
        /// <param name="a">one VBuffer</param>
        /// <param name="b">another VBuffer</param>
        /// <returns>Distance from a to b</returns>
        public static Float Distance(ref VBuffer<Float> a, ref VBuffer<Float> b)
        {
            return MathUtils.Sqrt(L2DistSquared(ref a, ref b));
        }

        /// <summary>
        /// Computes the Euclidean distance squared between two VBuffers
        /// </summary>
        /// <param name="a">one VBuffer</param>
        /// <param name="b">another VBuffer</param>
        /// <returns>Distance from a to b</returns>
        public static Float L2DistSquared(ref VBuffer<Float> a, ref VBuffer<Float> b)
        {
            Contracts.Check(a.Length == b.Length, "Vectors must have the same dimensionality.");
            if (a.IsDense)
            {
                if (b.IsDense)
                    return L2DiffSquaredDense(a.Values, b.Values, b.Length);
                return L2DistSquaredHalfSparse(a.Values, a.Length, b.Values, b.Indices, b.Count);
            }
            if (b.IsDense)
                return L2DistSquaredHalfSparse(b.Values, b.Length, a.Values, a.Indices, a.Count);
            return L2DistSquaredSparse(a.Values, a.Indices, a.Count, b.Values, b.Indices, b.Count, a.Length);
        }

        /// <summary>
        /// Given two vectors a and b, calculate their L2 distance squared (|a-b|^2).
        /// </summary>
        /// <param name="a">The first vector, given as an array</param>
        /// <param name="b">The second vector, given as a VBuffer{Float}</param>
        /// <returns>The squared L2 distance between a and b</returns>
        public static Float L2DistSquared(Float[] a, ref VBuffer<Float> b)
        {
            Contracts.CheckValue(a, nameof(a));
            Contracts.Check(Utils.Size(a) == b.Length, "Vectors must have the same dimensionality.");
            if (b.IsDense)
                return L2DiffSquaredDense(a, b.Values, b.Length);
            return L2DistSquaredHalfSparse(a, a.Length, b.Values, b.Indices, b.Count);
        }

        /// <summary>
        /// Perform in-place vector addition <c><paramref name="dst"/> += <paramref name="src"/></c>.
        /// </summary>
        public static void Add(Float[] src, Float[] dst)
        {
            Contracts.CheckValue(src, nameof(src));
            Contracts.CheckValue(dst, nameof(dst));
            Contracts.CheckParam(src.Length == dst.Length, nameof(dst), "Arrays must have the same dimensionality.");
            if (src.Length == 0)
                return;
            CpuMathUtils.Add(src, dst, src.Length);
        }

        /// <summary>
        /// Adds a multiple of a <see cref="VBuffer{T}"/> to a <see cref="Float"/> array.
        /// </summary>
        /// <param name="src">Buffer to add</param>
        /// <param name="dst">Array to add to</param>
        /// <param name="c">Coefficient</param>
        public static void AddMult(in ReadOnlyVBuffer<Float> src, Float[] dst, Float c)
        {
            Contracts.CheckValue(dst, nameof(dst));
            Contracts.CheckParam(src.Length == dst.Length, nameof(dst), "Arrays must have the same dimensionality.");

            if (src.Count == 0 || c == 0)
                return;

            if (src.IsDense)
                CpuMathUtils.AddScale(c, src.Values, dst, src.Count);
            else
            {
                for (int i = 0; i < src.Count; i++)
                    dst[src.Indices[i]] += c * src.Values[i];
            }
        }

        /// <summary>
        /// Adds a multiple of a <see cref="VBuffer{T}"/> to a <see cref="Float"/> array, with an offset into the destination.
        /// </summary>
        /// <param name="src">Buffer to add</param>
        /// <param name="dst">Array to add to</param>
        /// <param name="offset">The offset into <paramref name="dst"/> at which to add</param>
        /// <param name="c">Coefficient</param>

        public static void AddMultWithOffset(ref VBuffer<Float> src, Float[] dst, int offset, Float c)
        {
            Contracts.CheckValue(dst, nameof(dst));
            Contracts.Check(0 <= offset && offset <= dst.Length);
            Contracts.Check(src.Length <= dst.Length - offset, "Vector src must be no longer than dst.Length - offset.");

            if (src.Count == 0 || c == 0)
                return;

            if (src.IsDense)
            {
                for (int i = 0; i < src.Length; i++)
                    dst[i + offset] += c * src.Values[i];
            }
            else
            {
                for (int i = 0; i < src.Count; i++)
                    dst[src.Indices[i] + offset] += c * src.Values[i];
            }
        }

        /// <summary>
        /// Adds a multiple of an array to a second array.
        /// </summary>
        /// <param name="src">Array to add</param>
        /// <param name="dst">Array to add to</param>
        /// <param name="c">Multiple</param>
        public static void AddMult(Float[] src, Float[] dst, Float c)
        {
            Contracts.Check(src.Length == dst.Length, "Arrays must have the same dimensionality.");

            if (c == 0)
                return;

            CpuMathUtils.AddScale(c, src, dst, src.Length);
        }

        /// <summary>
        /// Returns the L2 norm of the vector (sum of squares of the components).
        /// </summary>
        public static Float Norm(Float[] a)
        {
            return MathUtils.Sqrt(CpuMathUtils.SumSq(a));
        }

        /// <summary>
        /// Returns sum of elements in array
        /// </summary>
        public static Float Sum(Float[] a)
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
        public static void ScaleBy(Float[] dst, Float c)
        {
            if (c == 1)
                return;

            if (c != 0)
                CpuMathUtils.Scale(c, dst);
            else
                Array.Clear(dst, 0, dst.Length);
        }

        public static Float Distance(Float[] a, Float[] b)
        {
            Contracts.Check(a.Length == b.Length, "Arrays must have the same dimensionality.");

            Float res = 0;
            for (int i = 0; i < a.Length; i++)
            {
                var diff = a[i] - b[i];
                res += diff * diff;
            }

            return res;
        }
    }
}
