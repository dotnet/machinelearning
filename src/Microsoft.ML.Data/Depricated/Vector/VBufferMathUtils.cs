// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Numeric
{

    internal static partial class VectorUtils
    {
        /// <summary>
        /// Returns the L2 norm squared of the vector (sum of squares of the components).
        /// </summary>
        public static float NormSquared(in VBuffer<float> a)
        {
            var aValues = a.GetValues();
            if (aValues.Length == 0)
                return 0;
            return CpuMathUtils.SumSq(aValues);
        }

        /// <summary>
        /// Returns the L2 norm squared of the vector (sum of squares of the components).
        /// </summary>
        public static float NormSquared(ReadOnlySpan<float> a)
        {
            return CpuMathUtils.SumSq(a);
        }

        /// <summary>
        /// Returns the L2 norm of the vector.
        /// </summary>
        /// <returns>L2 norm of the vector</returns>
        public static float Norm(in VBuffer<float> a)
        {
            return MathUtils.Sqrt(NormSquared(in a));
        }

        /// <summary>
        /// Returns the L1 norm of the vector.
        /// </summary>
        /// <returns>L1 norm of the vector</returns>
        public static float L1Norm(in VBuffer<float> a)
        {
            var aValues = a.GetValues();
            if (aValues.Length == 0)
                return 0;
            return CpuMathUtils.SumAbs(aValues);
        }

        /// <summary>
        /// Returns the L-infinity norm of the vector (i.e., the maximum absolute value).
        /// </summary>
        /// <returns>L-infinity norm of the vector</returns>
        public static float MaxNorm(in VBuffer<float> a)
        {
            var aValues = a.GetValues();
            if (aValues.Length == 0)
                return 0;
            return CpuMathUtils.MaxAbs(aValues);
        }

        /// <summary>
        /// Returns the sum of elements in the vector.
        /// </summary>
        public static float Sum(in VBuffer<float> a)
        {
            var aValues = a.GetValues();
            if (aValues.Length == 0)
                return 0;
            return CpuMathUtils.Sum(aValues);
        }

        /// <summary>
        /// Scales the vector by a real value.
        /// </summary>
        /// <param name="dst">Incoming vector</param>
        /// <param name="c">Value to multiply vector with</param>
        public static void ScaleBy(ref VBuffer<float> dst, float c)
        {
            if (c == 1 || dst.GetValues().Length == 0)
                return;
            var editor = VBufferEditor.CreateFromBuffer(ref dst);
            if (c != 0)
                CpuMathUtils.Scale(c, editor.Values);
            else // Maintain density of dst.
                editor.Values.Clear();
            // REVIEW: Any benefit in sparsifying?
        }

        /// <summary>
        /// Scales the vector by a real value.
        /// <c><paramref name="dst"/> = <paramref name="c"/> * <paramref name="src"/></c>
        /// </summary>
        public static void ScaleBy(in VBuffer<float> src, ref VBuffer<float> dst, float c)
        {
            int length = src.Length;
            var srcValues = src.GetValues();
            int count = srcValues.Length;

            if (count == 0)
            {
                // dst is a zero vector.
                VBufferUtils.Resize(ref dst, length, 0);
                return;
            }

            if (src.IsDense)
            {
                // Maintain the density of src to dst in order to avoid slow down of L-BFGS.
                var editor = VBufferEditor.Create(ref dst, length);
                Contracts.Assert(length == count);
                if (c == 0)
                    editor.Values.Clear();
                else
                    CpuMathUtils.Scale(c, srcValues, editor.Values, length);
                dst = editor.Commit();
            }
            else
            {
                var editor = VBufferEditor.Create(ref dst, length, count);
                src.GetIndices().CopyTo(editor.Indices);
                if (c == 0)
                    editor.Values.Clear();
                else
                    CpuMathUtils.Scale(c, srcValues, editor.Values, count);
                dst = editor.Commit();
            }
        }

        /// <summary>
        /// Perform in-place vector addition <c><paramref name="dst"/> += <paramref name="src"/></c>.
        /// </summary>
        public static void Add(in VBuffer<float> src, ref VBuffer<float> dst)
        {
            Contracts.Check(src.Length == dst.Length, "Vectors must have the same dimensionality.");

            var srcValues = src.GetValues();
            if (srcValues.Length == 0)
                return;

            if (dst.IsDense)
            {
                var editor = VBufferEditor.Create(ref dst, dst.Length);
                if (src.IsDense)
                    CpuMathUtils.Add(srcValues, editor.Values, src.Length);
                else
                    CpuMathUtils.Add(srcValues, src.GetIndices(), editor.Values, srcValues.Length);
                return;
            }
            // REVIEW: Should we use SSE for any of these possibilities?
            VBufferUtils.ApplyWith(in src, ref dst, (int i, float v1, ref float v2) => v2 += v1);
        }

        // REVIEW: Rename all instances of AddMult to AddScale, as soon as convesion concerns are no more.
        /// <summary>
        /// Perform in-place scaled vector addition
        /// <c><paramref name="dst"/> += <paramref name="c"/> * <paramref name="src"/></c>.
        /// If either vector is dense, <paramref name="dst"/> will be dense, unless
        /// <paramref name="c"/> is 0 in which case this method does nothing.
        /// </summary>
        public static void AddMult(in VBuffer<float> src, float c, ref VBuffer<float> dst)
        {
            Contracts.Check(src.Length == dst.Length, "Vectors must have the same dimensionality.");

            var srcValues = src.GetValues();
            if (srcValues.Length == 0 || c == 0)
                return;

            if (dst.IsDense)
            {
                var editor = VBufferEditor.Create(ref dst, dst.Length);
                if (src.IsDense)
                    CpuMathUtils.AddScale(c, srcValues, editor.Values, src.Length);
                else
                    CpuMathUtils.AddScale(c, srcValues, src.GetIndices(), editor.Values, srcValues.Length);
                return;
            }
            // REVIEW: Should we use SSE for any of these possibilities?
            VBufferUtils.ApplyWith(in src, ref dst, (int i, float v1, ref float v2) => v2 += c * v1);
        }

        /// <summary>
        /// Perform scalar vector addition
        /// <c><paramref name="res"/> = <paramref name="c"/> * <paramref name="src"/> + <paramref name="dst"/></c>
        /// </summary>
        public static void AddMult(in VBuffer<float> src, float c, ref VBuffer<float> dst, ref VBuffer<float> res)
        {
            Contracts.Check(src.Length == dst.Length, "Vectors must have the same dimensionality.");
            int length = src.Length;

            var srcValues = src.GetValues();
            if (srcValues.Length == 0 || c == 0)
            {
                // src is zero vector, res = dst
                dst.CopyTo(ref res);
                return;
            }

            Contracts.Assert(length > 0);
            if (dst.IsDense && src.IsDense)
            {
                var editor = VBufferEditor.Create(ref res, length);
                CpuMathUtils.AddScaleCopy(c, srcValues, dst.GetValues(), editor.Values, length);
                res = editor.Commit();
                return;
            }

            VBufferUtils.ApplyWithCopy(in src, ref dst, ref res, (int i, float v1, float v2, ref float v3) => v3 = v2 + c * v1);
        }

        /// <summary>
        /// Calculate
        /// <c><paramref name="a"/> + <paramref name="c"/> * <paramref name="b"/></c>
        /// and store the result in <paramref name="dst"/>.
        /// </summary>
        public static void AddMultInto(in VBuffer<float> a, float c, in VBuffer<float> b, ref VBuffer<float> dst)
        {
            Contracts.Check(a.Length == b.Length, "Vectors must have the same dimensionality.");

            if (c == 0 || b.GetValues().Length == 0)
                a.CopyTo(ref dst);
            else if (a.GetValues().Length == 0)
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
        public static void AddMultWithOffset(in VBuffer<float> src, float c, ref VBuffer<float> dst, int offset)
        {
            Contracts.CheckParam(0 <= offset && offset <= dst.Length, nameof(offset));
            Contracts.CheckParam(src.Length <= dst.Length - offset, nameof(offset));

            var srcValues = src.GetValues();
            if (srcValues.Length == 0 || c == 0)
                return;
            VBufferEditor<float> editor;
            Span<float> values;
            if (dst.IsDense)
            {
                // This is by far the most common case.
                editor = VBufferEditor.Create(ref dst, dst.Length);
                values = editor.Values.Slice(offset);
                if (src.IsDense)
                    CpuMathUtils.AddScale(c, srcValues, values, srcValues.Length);
                else
                    CpuMathUtils.AddScale(c, srcValues, src.GetIndices(), values, srcValues.Length);
                return;
            }
            // REVIEW: Perhaps implementing an ApplyInto with an offset would be more
            // appropriate, as well as more general, considering that this case is less important.

            // dst is sparse. I expect this will see limited practical use, since accumulants
            // are often better off going into a dense vector in all applications of interest to us.
            // Correspondingly, this implementation will be functional, but not optimized.
            var dstIndices = dst.GetIndices();
            int dMin = dstIndices.Length == 0 ? 0 : dstIndices.FindIndexSorted(0, dstIndices.Length, offset);
            int dLim = dstIndices.Length == 0 ? 0 : dstIndices.FindIndexSorted(dMin, dstIndices.Length, offset + src.Length);
            Contracts.Assert(dMin - dLim <= src.Length);
            // First get the number of extra values that we will need to accomodate.
            int gapCount;
            if (src.IsDense)
                gapCount = src.Length - (dLim - dMin);
            else
            {
                gapCount = srcValues.Length;
                var srcIndices = src.GetIndices();
                for (int iS = 0, iD = dMin; iS < srcIndices.Length && iD < dLim; )
                {
                    var comp = srcIndices[iS] - dstIndices[iD] + offset;
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
            var dstValues = dst.GetValues();
            editor = VBufferEditor.Create(ref dst,
                dst.Length,
                dstValues.Length + gapCount,
                keepOldOnResize: true,
                requireIndicesOnDense: true);
            var indices = editor.Indices;
            values = editor.Values;
            if (gapCount > 0)
            {
                // Shift things over, unless there's nothing to shift over, or no new elements are being introduced anyway.
                if (dstValues.Length != dLim)
                {
                    Contracts.Assert(dLim < dstValues.Length);
                    indices.Slice(dLim, dstValues.Length - dLim)
                        .CopyTo(indices.Slice(dLim + gapCount));
                    values.Slice(dLim, dstValues.Length - dLim)
                        .CopyTo(values.Slice(dLim + gapCount));
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
                    if (iD >= 0 && offset + iS == dstIndices[iD]) // Collision.
                        values[iDD] = dstValues[iD--] + c * srcValues[iS];
                    else // Miss.
                        values[iDD] = c * srcValues[iS];
                    indices[iDD] = offset + iS;
                }
            }
            else
            {
                // Both dst and src are sparse.
                int iD = dLim - 1;
                var srcIndices = src.GetIndices();
                int iS = srcIndices.Length - 1;
                int sIndex = iS < 0 ? -1 : srcIndices[iS];
                int dIndex = iD < 0 ? -1 : dstIndices[iD] - offset;

                for (int iDD = dLim + gapCount; --iDD >= dMin; )
                {
                    Contracts.Assert(iDD >= iD);
                    int comp = sIndex - dIndex;
                    if (comp == 0) // Collision on both.
                    {
                        indices[iDD] = dstIndices[iD];
                        values[iDD] = dstValues[iD--] + c * srcValues[iS--];
                        sIndex = iS < 0 ? -1 : srcIndices[iS];
                        dIndex = iD < 0 ? -1 : dstIndices[iD] - offset;
                    }
                    else if (comp < 0) // Collision on dst.
                    {
                        indices[iDD] = dstIndices[iD];
                        values[iDD] = dstValues[iD--];
                        dIndex = iD < 0 ? -1 : dstIndices[iD] - offset;
                    }
                    else // Collision on src.
                    {
                        indices[iDD] = sIndex + offset;
                        values[iDD] = c * srcValues[iS--];
                        sIndex = iS < 0 ? -1 : srcIndices[iS];
                    }
                }
            }
            dst = editor.Commit();
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
        public static void ScaleInto(in VBuffer<float> src, float c, ref VBuffer<float> dst)
        {
            // REVIEW: The analogous WritableVector method insisted on
            // equal lengths, but I assume I don't care here.
            if (c == 1)
                src.CopyTo(ref dst);
            else if (src.GetValues().Length == 0 || c == 0)
            {
                if (src.Length > 0 && src.IsDense)
                {
                    // Due to sparsity preservation from src, dst must be dense, in the same way.
                    var editor = VBufferEditor.Create(ref dst, src.Length);
                    if (!editor.CreatedNewValues) // We need to clear it.
                        editor.Values.Clear();
                    dst = editor.Commit();
                }
                else
                {
                    VBufferUtils.Resize(ref dst, src.Length, 0);
                }
            }
            else if (c == -1)
                VBufferUtils.ApplyIntoEitherDefined(in src, ref dst, (i, v) => -v);
            else
                VBufferUtils.ApplyIntoEitherDefined(in src, ref dst, (i, v) => c * v);
        }

        public static int ArgMax(in VBuffer<float> src)
        {
            if (src.Length == 0)
                return -1;
            var srcValues = src.GetValues();
            if (srcValues.Length == 0)
                return 0;

            int ind = MathUtils.ArgMax(srcValues);
            // ind < 0 iff all explicit values are NaN.
            Contracts.Assert(-1 <= ind && ind < srcValues.Length);

            if (src.IsDense)
                return ind;

            var srcIndices = src.GetIndices();
            if (ind >= 0)
            {
                Contracts.Assert(srcIndices[ind] >= ind);
                if (srcValues[ind] > 0)
                    return srcIndices[ind];
                // This covers the case where there is an explicit zero, and zero is the max,
                // and the first explicit zero is before any implicit entries.
                if (srcValues[ind] == 0 && srcIndices[ind] == ind)
                    return ind;
            }

            // All explicit values are non-positive or NaN, so return the first index not in src.Indices.
            ind = 0;
            while (ind < srcIndices.Length && srcIndices[ind] == ind)
                ind++;
            Contracts.Assert(ind <= srcIndices.Length);
            Contracts.Assert(ind == srcIndices.Length || ind < srcIndices[ind]);
            return ind;
        }

        public static int ArgMin(in VBuffer<float> src)
        {
            if (src.Length == 0)
                return -1;
            var srcValues = src.GetValues();
            if (srcValues.Length == 0)
                return 0;

            int ind = MathUtils.ArgMin(srcValues);
            // ind < 0 iff all explicit values are NaN.
            Contracts.Assert(-1 <= ind && ind < srcValues.Length);

            if (src.IsDense)
                return ind;

            var srcIndices = src.GetIndices();
            if (ind >= 0)
            {
                Contracts.Assert(srcIndices[ind] >= ind);
                if (srcValues[ind] < 0)
                    return srcIndices[ind];
                // This covers the case where there is an explicit zero, and zero is the min,
                // and the first explicit zero is before any implicit entries.
                if (srcValues[ind] == 0 && srcIndices[ind] == ind)
                    return ind;
            }

            // All explicit values are non-negative or NaN, so return the first index not in srcIndices.
            ind = 0;
            while (ind < srcIndices.Length && srcIndices[ind] == ind)
                ind++;
            Contracts.Assert(ind <= srcIndices.Length);
            Contracts.Assert(ind == srcIndices.Length || ind < srcIndices[ind]);
            return ind;
        }
    }
}
