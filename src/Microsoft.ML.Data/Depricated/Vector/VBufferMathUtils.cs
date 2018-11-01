// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Numeric
{
    // REVIEW: Once we do the conversions from Vector/WritableVector, review names of methods,
    //   parameters, parameter order, etc.
    using Float = System.Single;

    public static partial class VectorUtils
    {
        /// <summary>
        /// Returns the L2 norm squared of the vector (sum of squares of the components).
        /// </summary>
        public static Float NormSquared(in VBuffer<Float> a)
        {
            if (a.Count == 0)
                return 0;
            return CpuMathUtils.SumSq(a.Values.AsSpan(0, a.Count));
        }

        /// <summary>
        /// Returns the L2 norm squared of the vector (sum of squares of the components).
        /// </summary>
        public static Float NormSquared(Float[] a, int offset, int count)
        {
            return CpuMathUtils.SumSq(a.AsSpan(offset, count));
        }

        /// <summary>
        /// Returns the L2 norm of the vector.
        /// </summary>
        /// <returns>L2 norm of the vector</returns>
        public static Float Norm(in VBuffer<Float> a)
        {
            return MathUtils.Sqrt(NormSquared(in a));
        }

        /// <summary>
        /// Returns the L1 norm of the vector.
        /// </summary>
        /// <returns>L1 norm of the vector</returns>
        public static Float L1Norm(in VBuffer<Float> a)
        {
            if (a.Count == 0)
                return 0;
            return CpuMathUtils.SumAbs(a.Values.AsSpan(0, a.Count));
        }

        /// <summary>
        /// Returns the L-infinity norm of the vector (i.e., the maximum absolute value).
        /// </summary>
        /// <returns>L-infinity norm of the vector</returns>
        public static Float MaxNorm(in VBuffer<Float> a)
        {
            if (a.Count == 0)
                return 0;
            return CpuMathUtils.MaxAbs(a.Values.AsSpan(0, a.Count));
        }

        /// <summary>
        /// Returns the sum of elements in the vector.
        /// </summary>
        public static Float Sum(in VBuffer<Float> a)
        {
            if (a.Count == 0)
                return 0;
            return CpuMathUtils.Sum(a.Values.AsSpan(0, a.Count));
        }

        /// <summary>
        /// Scales the vector by a real value.
        /// </summary>
        /// <param name="dst">Incoming vector</param>
        /// <param name="c">Value to multiply vector with</param>
        public static void ScaleBy(ref VBuffer<Float> dst, Float c)
        {
            if (c == 1 || dst.Count == 0)
                return;
            if (c != 0)
                CpuMathUtils.Scale(c, dst.Values.AsSpan(0, dst.Count));
            else // Maintain density of dst.
                Array.Clear(dst.Values, 0, dst.Count);
            // REVIEW: Any benefit in sparsifying?
        }

        /// <summary>
        /// Scales the vector by a real value.
        /// <c><paramref name="dst"/> = <paramref name="c"/> * <paramref name="src"/></c>
        /// </summary>
        public static void ScaleBy(in VBuffer<Float> src, ref VBuffer<Float> dst, Float c)
        {
            int length = src.Length;
            int count = src.Count;

            if (count == 0)
            {
                // dst is a zero vector.
                dst = new VBuffer<Float>(length, 0, dst.Values, dst.Indices);
                return;
            }

            var dstValues = Utils.Size(dst.Values) >= count ? dst.Values : new Float[count];
            if (src.IsDense)
            {
                // Maintain the density of src to dst in order to avoid slow down of L-BFGS.
                Contracts.Assert(length == count);
                if (c == 0)
                    Array.Clear(dstValues, 0, length);
                else
                    CpuMathUtils.Scale(c, src.Values, dstValues, length);
                dst = new VBuffer<Float>(length, dstValues, dst.Indices);
            }
            else
            {
                var dstIndices = Utils.Size(dst.Indices) >= count ? dst.Indices : new int[count];
                Array.Copy(src.Indices, dstIndices, count);
                if (c == 0)
                    Array.Clear(dstValues, 0, count);
                else
                    CpuMathUtils.Scale(c, src.Values, dstValues, count);
                dst = new VBuffer<Float>(length, count, dstValues, dstIndices);
            }
        }

        /// <summary>
        /// Perform in-place vector addition <c><paramref name="dst"/> += <paramref name="src"/></c>.
        /// </summary>
        public static void Add(in VBuffer<Float> src, ref VBuffer<Float> dst)
        {
            Contracts.Check(src.Length == dst.Length, "Vectors must have the same dimensionality.");

            if (src.Count == 0)
                return;

            if (dst.IsDense)
            {
                if (src.IsDense)
                    CpuMathUtils.Add(src.Values, dst.Values, src.Length);
                else
                    CpuMathUtils.Add(src.Values, src.Indices, dst.Values, src.Count);
                return;
            }
            // REVIEW: Should we use SSE for any of these possibilities?
            VBufferUtils.ApplyWith(in src, ref dst, (int i, Float v1, ref Float v2) => v2 += v1);
        }

        // REVIEW: Rename all instances of AddMult to AddScale, as soon as convesion concerns are no more.
        /// <summary>
        /// Perform in-place scaled vector addition
        /// <c><paramref name="dst"/> += <paramref name="c"/> * <paramref name="src"/></c>.
        /// If either vector is dense, <paramref name="dst"/> will be dense, unless
        /// <paramref name="c"/> is 0 in which case this method does nothing.
        /// </summary>
        public static void AddMult(in VBuffer<Float> src, Float c, ref VBuffer<Float> dst)
        {
            Contracts.Check(src.Length == dst.Length, "Vectors must have the same dimensionality.");

            if (src.Count == 0 || c == 0)
                return;

            if (dst.IsDense)
            {
                if (src.IsDense)
                    CpuMathUtils.AddScale(c, src.Values, dst.Values, src.Length);
                else
                    CpuMathUtils.AddScale(c, src.Values, src.Indices, dst.Values, src.Count);
                return;
            }
            // REVIEW: Should we use SSE for any of these possibilities?
            VBufferUtils.ApplyWith(in src, ref dst, (int i, Float v1, ref Float v2) => v2 += c * v1);
        }

        /// <summary>
        /// Perform scalar vector addition
        /// <c><paramref name="res"/> = <paramref name="c"/> * <paramref name="src"/> + <paramref name="dst"/></c>
        /// </summary>
        public static void AddMult(in VBuffer<Float> src, Float c, ref VBuffer<Float> dst, ref VBuffer<Float> res)
        {
            Contracts.Check(src.Length == dst.Length, "Vectors must have the same dimensionality.");
            int length = src.Length;

            if (src.Count == 0 || c == 0)
            {
                // src is zero vector, res = dst
                dst.CopyTo(ref res);
                return;
            }

            Contracts.Assert(length > 0);
            if (dst.IsDense && src.IsDense)
            {
                Float[] resValues = Utils.Size(res.Values) >= length ? res.Values : new Float[length];
                CpuMathUtils.AddScaleCopy(c, src.Values, dst.Values, resValues, length);
                res = new VBuffer<Float>(length, resValues, res.Indices);
                return;
            }

            VBufferUtils.ApplyWithCopy(in src, ref dst, ref res, (int i, Float v1, Float v2, ref Float v3) => v3 = v2 + c * v1);
        }

        /// <summary>
        /// Calculate
        /// <c><paramref name="a"/> + <paramref name="c"/> * <paramref name="b"/></c>
        /// and store the result in <paramref name="dst"/>.
        /// </summary>
        public static void AddMultInto(in VBuffer<Float> a, Float c, in VBuffer<Float> b, ref VBuffer<Float> dst)
        {
            Contracts.Check(a.Length == b.Length, "Vectors must have the same dimensionality.");

            if (c == 0 || b.Count == 0)
                a.CopyTo(ref dst);
            else if (a.Count == 0)
                ScaleInto(in b, c, ref dst);
            else
                VBufferUtils.ApplyInto(in a, in b, ref dst, (ind, v1, v2) => v1 + c * v2);
        }

        /// <summary>
        /// Perform in-place scaled vector addition
        /// <c><paramref name="dst"/> += <paramref name="c"/> * <paramref name="src"/></c>,
        /// except that this takes place in the section of <paramref name="dst"/> starting
        /// at slot <paramref name="offset"/>.
        /// </summary>
        public static void AddMultWithOffset(in VBuffer<Float> src, Float c, ref VBuffer<Float> dst, int offset)
        {
            Contracts.CheckParam(0 <= offset && offset <= dst.Length, nameof(offset));
            Contracts.CheckParam(src.Length <= dst.Length - offset, nameof(offset));

            if (src.Count == 0 || c == 0)
                return;
            if (dst.IsDense)
            {
                // This is by far the most common case.
                if (src.IsDense)
                    CpuMathUtils.AddScale(c, src.Values, dst.Values.AsSpan(offset), src.Count);
                else
                    CpuMathUtils.AddScale(c, src.Values, src.Indices, dst.Values.AsSpan(offset), src.Count);
                return;
            }
            // REVIEW: Perhaps implementing an ApplyInto with an offset would be more
            // appropriate, as well as more general, considering that this case is less important.

            // dst is sparse. I expect this will see limited practical use, since accumulants
            // are often better off going into a dense vector in all applications of interest to us.
            // Correspondingly, this implementation will be functional, but not optimized.
            int dMin = dst.Count == 0 ? 0 : Utils.FindIndexSorted(dst.Indices, 0, dst.Count, offset);
            int dLim = dst.Count == 0 ? 0 : Utils.FindIndexSorted(dst.Indices, dMin, dst.Count, offset + src.Length);
            Contracts.Assert(dMin - dLim <= src.Length);
            // First get the number of extra values that we will need to accomodate.
            int gapCount;
            if (src.IsDense)
                gapCount = src.Length - (dLim - dMin);
            else
            {
                gapCount = src.Count;
                for (int iS = 0, iD = dMin; iS < src.Count && iD < dLim; )
                {
                    var comp = src.Indices[iS] - dst.Indices[iD] + offset;
                    if (comp < 0) // dst index is larger.
                        iS++;
                    else if (comp > 0) // src index is larger.
                        iD++;
                    else
                    {
                        iS++;
                        iD++;
                        gapCount--;
                    }
                }
            }
            // Extend dst so that it has room for this additional stuff. Shift things over as well.
            var indices = dst.Indices;
            var values = dst.Values;
            if (gapCount > 0)
            {
                Utils.EnsureSize(ref indices, dst.Count + gapCount, dst.Length);
                Utils.EnsureSize(ref values, dst.Count + gapCount, dst.Length);
                // Shift things over, unless there's nothing to shift over, or no new elements are being introduced anyway.
                if (dst.Count != dLim)
                {
                    Contracts.Assert(dLim < dst.Count);
                    Array.Copy(indices, dLim, indices, dLim + gapCount, dst.Count - dLim);
                    Array.Copy(values, dLim, values, dLim + gapCount, dst.Count - dLim);
                }
            }
            // Now, fill in the stuff in this "gap." Both of these implementations work
            // backwards from the end, since they can potentially be working in place if
            // the EnsureSize calls did not actually result in a new array.
            if (src.IsDense)
            {
                // dst is sparse, src is dense.
                int iD = dLim - 1;
                int iS = src.Length - 1;
                for (int iDD = dLim + gapCount; --iDD >= dMin; --iS)
                {
                    Contracts.Assert(iDD == iS + dMin);
                    // iDD and iD are the points in where we are writing and reading from.
                    Contracts.Assert(iDD >= iD);
                    if (iD >= 0 && offset + iS == dst.Indices[iD]) // Collision.
                        values[iDD] = dst.Values[iD--] + c * src.Values[iS];
                    else // Miss.
                        values[iDD] = c * src.Values[iS];
                    indices[iDD] = offset + iS;
                }
            }
            else
            {
                // Both dst and src are sparse.
                int iD = dLim - 1;
                int iS = src.Count - 1;
                int sIndex = iS < 0 ? -1 : src.Indices[iS];
                int dIndex = iD < 0 ? -1 : dst.Indices[iD] - offset;

                for (int iDD = dLim + gapCount; --iDD >= dMin; )
                {
                    Contracts.Assert(iDD >= iD);
                    int comp = sIndex - dIndex;
                    if (comp == 0) // Collision on both.
                    {
                        indices[iDD] = dst.Indices[iD];
                        values[iDD] = dst.Values[iD--] + c * src.Values[iS--];
                        sIndex = iS < 0 ? -1 : src.Indices[iS];
                        dIndex = iD < 0 ? -1 : dst.Indices[iD] - offset;
                    }
                    else if (comp < 0) // Collision on dst.
                    {
                        indices[iDD] = dst.Indices[iD];
                        values[iDD] = dst.Values[iD--];
                        dIndex = iD < 0 ? -1 : dst.Indices[iD] - offset;
                    }
                    else // Collision on src.
                    {
                        indices[iDD] = sIndex + offset;
                        values[iDD] = c * src.Values[iS--];
                        sIndex = iS < 0 ? -1 : src.Indices[iS];
                    }
                }
            }
            dst = new VBuffer<Float>(dst.Length, dst.Count + gapCount, values, indices);
        }

        /// <summary>
        /// Perform in-place scaling of a vector into another vector as
        /// <c><paramref name="dst"/> = <paramref name="src"/> * <paramref name="c"/></c>.
        /// This is more or less equivalent to performing the same operation with
        /// <see cref="VBufferUtils.ApplyInto{TSrc1,TSrc2,TDst}"/> except perhaps more efficiently,
        /// with one exception: if <paramref name="c"/> is 0 and <paramref name="src"/>
        /// is sparse, <paramref name="dst"/> will have a count of zero, instead of the
        /// same count as <paramref name="src"/>.
        /// </summary>
        public static void ScaleInto(in VBuffer<Float> src, Float c, ref VBuffer<Float> dst)
        {
            // REVIEW: The analogous WritableVector method insisted on
            // equal lengths, but I assume I don't care here.
            if (c == 1)
                src.CopyTo(ref dst);
            else if (src.Count == 0 || c == 0)
            {
                if (src.Length > 0 && src.IsDense)
                {
                    var values = dst.Values;
                    // Due to sparsity preservation from src, dst must be dense, in the same way.
                    Utils.EnsureSize(ref values, src.Length, src.Length, keepOld: false);
                    if (values == dst.Values) // We need to clear it.
                        Array.Clear(values, 0, src.Length);
                    dst = new VBuffer<Float>(src.Length, values, dst.Indices);
                }
                else
                    dst = new VBuffer<Float>(src.Length, 0, dst.Values, dst.Indices);
            }
            else if (c == -1)
                VBufferUtils.ApplyIntoEitherDefined(in src, ref dst, (i, v) => -v);
            else
                VBufferUtils.ApplyIntoEitherDefined(in src, ref dst, (i, v) => c * v);
        }

        public static int ArgMax(in VBuffer<Float> src)
        {
            if (src.Length == 0)
                return -1;
            if (src.Count == 0)
                return 0;

            int ind = MathUtils.ArgMax(src.Values, src.Count);
            // ind < 0 iff all explicit values are NaN.
            Contracts.Assert(-1 <= ind && ind < src.Count);

            if (src.IsDense)
                return ind;

            if (ind >= 0)
            {
                Contracts.Assert(src.Indices[ind] >= ind);
                if (src.Values[ind] > 0)
                    return src.Indices[ind];
                // This covers the case where there is an explicit zero, and zero is the max,
                // and the first explicit zero is before any implicit entries.
                if (src.Values[ind] == 0 && src.Indices[ind] == ind)
                    return ind;
            }

            // All explicit values are non-positive or NaN, so return the first index not in src.Indices.
            ind = 0;
            while (ind < src.Count && src.Indices[ind] == ind)
                ind++;
            Contracts.Assert(ind <= src.Count);
            Contracts.Assert(ind == src.Count || ind < src.Indices[ind]);
            return ind;
        }

        public static int ArgMin(in VBuffer<Float> src)
        {
            if (src.Length == 0)
                return -1;
            if (src.Count == 0)
                return 0;

            int ind = MathUtils.ArgMin(src.Values, src.Count);
            // ind < 0 iff all explicit values are NaN.
            Contracts.Assert(-1 <= ind && ind < src.Count);

            if (src.IsDense)
                return ind;

            if (ind >= 0)
            {
                Contracts.Assert(src.Indices[ind] >= ind);
                if (src.Values[ind] < 0)
                    return src.Indices[ind];
                // This covers the case where there is an explicit zero, and zero is the min,
                // and the first explicit zero is before any implicit entries.
                if (src.Values[ind] == 0 && src.Indices[ind] == ind)
                    return ind;
            }

            // All explicit values are non-negative or NaN, so return the first index not in src.Indices.
            ind = 0;
            while (ind < src.Count && src.Indices[ind] == ind)
                ind++;
            Contracts.Assert(ind <= src.Count);
            Contracts.Assert(ind == src.Count || ind < src.Indices[ind]);
            return ind;
        }
    }
}
