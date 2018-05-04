// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    // REVIEW: Consider automatic densification in some of the operations, where appropriate.
    // REVIEW: Once we do the conversions from Vector/WritableVector, review names of methods,
    //   parameters, parameter order, etc.
    /// <summary>
    /// Convenience utilities for vector operations on <see cref="VBuffer{T}"/>.
    /// </summary>
    public static class VBufferUtils
    {
        private const float SparsityThreshold = 0.25f;

        /// <summary>
        /// A helper method that gives us an iterable over the items given the fields from a <see cref="VBuffer{T}"/>.
        /// Note that we have this in a separate utility class, rather than in its more natural location of
        /// <see cref="VBuffer{T}"/> itself, due to a bug in the C++/CLI compiler. (DevDiv 1097919:
        /// [C++/CLI] Nested generic types are not correctly imported from metadata). So, if we want to use
        /// <see cref="VBuffer{T}"/> in C++/CLI projects, we cannot have a generic struct with a nested class
        /// that has the outer struct type as a field.
        /// </summary>
        internal static IEnumerable<KeyValuePair<int, T>> Items<T>(T[] values, int[] indices, int length, int count, bool all)
        {
            Contracts.AssertValueOrNull(values);
            Contracts.Assert(0 <= count && count <= Utils.Size(values));
            Contracts.Assert(count <= length);
            Contracts.Assert(count == length || count <= Utils.Size(indices));

            if (count == length)
            {
                for (int i = 0; i < count; i++)
                    yield return new KeyValuePair<int, T>(i, values[i]);
            }
            else if (!all)
            {
                for (int i = 0; i < count; i++)
                    yield return new KeyValuePair<int, T>(indices[i], values[i]);
            }
            else
            {
                int slotCur = -1;
                for (int i = 0; i < count; i++)
                {
                    int slot = indices[i];
                    Contracts.Assert(slotCur < slot && slot < length);
                    while (++slotCur < slot)
                        yield return new KeyValuePair<int, T>(slotCur, default(T));
                    Contracts.Assert(slotCur == slot);
                    yield return new KeyValuePair<int, T>(slotCur, values[i]);
                }
                Contracts.Assert(slotCur < length);
                while (++slotCur < length)
                    yield return new KeyValuePair<int, T>(slotCur, default(T));
            }
        }

        internal static IEnumerable<T> DenseValues<T>(T[] values, int[] indices, int length, int count)
        {
            Contracts.AssertValueOrNull(values);
            Contracts.Assert(0 <= count && count <= Utils.Size(values));
            Contracts.Assert(count <= length);
            Contracts.Assert(count == length || count <= Utils.Size(indices));

            if (count == length)
            {
                for (int i = 0; i < length; i++)
                    yield return values[i];
            }
            else
            {
                int slotCur = -1;
                for (int i = 0; i < count; i++)
                {
                    int slot = indices[i];
                    Contracts.Assert(slotCur < slot && slot < length);
                    while (++slotCur < slot)
                        yield return default(T);
                    Contracts.Assert(slotCur == slot);
                    yield return values[i];
                }
                Contracts.Assert(slotCur < length);
                while (++slotCur < length)
                    yield return default(T);
            }
        }

        public static bool HasNaNs(ref VBuffer<Single> buffer)
        {
            for (int i = 0; i < buffer.Count; i++)
            {
                if (Single.IsNaN(buffer.Values[i]))
                    return true;
            }
            return false;
        }

        public static bool HasNaNs(ref VBuffer<Double> buffer)
        {
            for (int i = 0; i < buffer.Count; i++)
            {
                if (Double.IsNaN(buffer.Values[i]))
                    return true;
            }
            return false;
        }

        public static bool HasNonFinite(ref VBuffer<Single> buffer)
        {
            for (int i = 0; i < buffer.Count; i++)
            {
                if (!FloatUtils.IsFinite(buffer.Values[i]))
                    return true;
            }
            return false;
        }

        public static bool HasNonFinite(ref VBuffer<Double> buffer)
        {
            for (int i = 0; i < buffer.Count; i++)
            {
                if (!FloatUtils.IsFinite(buffer.Values[i]))
                    return true;
            }
            return false;
        }

        public static VBuffer<T> CreateEmpty<T>(int length)
        {
            Contracts.CheckParam(length >= 0, nameof(length));
            return new VBuffer<T>(length, 0, null, null);
        }

        public static VBuffer<T> CreateDense<T>(int length)
        {
            Contracts.CheckParam(length >= 0, nameof(length));
            return new VBuffer<T>(length, new T[length]);
        }

        /// <summary>
        /// Applies <paramref name="visitor"/> to every explicitly defined element of the vector,
        /// in order of index.
        /// </summary>
        public static void ForEachDefined<T>(ref VBuffer<T> a, Action<int, T> visitor)
        {
            Contracts.CheckValue(visitor, nameof(visitor));

            // REVIEW: This is analogous to an old Vector method, but is there
            // any real reason to have it given that we have the Items extension method?
            if (a.IsDense)
            {
                for (int i = 0; i < a.Length; i++)
                    visitor(i, a.Values[i]);
            }
            else
            {
                for (int i = 0; i < a.Count; i++)
                    visitor(a.Indices[i], a.Values[i]);
            }
        }

        /// <summary>
        /// Applies the <paramref name="visitor "/>to each corresponding pair of elements
        /// where the item is emplicitly defined in the vector. By explicitly defined,
        /// we mean that for a given index <c>i</c>, both vectors have an entry in
        /// <see cref="VBuffer{T}.Values"/> corresponding to that index.
        /// </summary>
        /// <param name="a">The first vector</param>
        /// <param name="b">The second vector</param>
        /// <param name="visitor">Delegate to apply to each pair of non-zero values.
        /// This is passed the index, and two values</param>
        public static void ForEachBothDefined<T>(ref VBuffer<T> a, ref VBuffer<T> b, Action<int, T, T> visitor)
        {
            Contracts.Check(a.Length == b.Length, "Vectors must have the same dimensionality.");
            Contracts.CheckValue(visitor, nameof(visitor));

            if (a.IsDense && b.IsDense)
            {
                for (int i = 0; i < a.Length; i++)
                    visitor(i, a.Values[i], b.Values[i]);
            }
            else if (b.IsDense)
            {
                for (int i = 0; i < a.Count; i++)
                    visitor(a.Indices[i], a.Values[i], b.Values[a.Indices[i]]);
            }
            else if (a.IsDense)
            {
                for (int i = 0; i < b.Count; i++)
                    visitor(b.Indices[i], a.Values[b.Indices[i]], b.Values[i]);
            }
            else
            {
                // Both sparse.
                int aI = 0;
                int bI = 0;
                while (aI < a.Count && bI < b.Count)
                {
                    int i = a.Indices[aI];
                    int j = b.Indices[bI];
                    if (i == j)
                        visitor(i, a.Values[aI++], b.Values[bI++]);
                    else if (i < j)
                        aI++;
                    else
                        bI++;
                }
            }
        }

        /// <summary>
        /// Applies the ParallelVisitor to each corresponding pair of elements where at least one is non-zero, in order of index.
        /// </summary>
        /// <param name="a">a vector</param>
        /// <param name="b">another vector</param>
        /// <param name="visitor">Function to apply to each pair of non-zero values - passed the index, and two values</param>
        public static void ForEachEitherDefined<T>(ref VBuffer<T> a, ref VBuffer<T> b, Action<int, T, T> visitor)
        {
            Contracts.Check(a.Length == b.Length, "Vectors must have the same dimensionality.");
            Contracts.CheckValue(visitor, nameof(visitor));

            if (a.IsDense && b.IsDense)
            {
                for (int i = 0; i < a.Length; ++i)
                    visitor(i, a.Values[i], b.Values[i]);
            }
            else if (b.IsDense)
            {
                int aI = 0;
                for (int i = 0; i < b.Length; i++)
                {
                    T aVal = (aI < a.Count && i == a.Indices[aI]) ? a.Values[aI++] : default(T);
                    visitor(i, aVal, b.Values[i]);
                }
            }
            else if (a.IsDense)
            {
                int bI = 0;
                for (int i = 0; i < a.Length; i++)
                {
                    T bVal = (bI < b.Count && i == b.Indices[bI]) ? b.Values[bI++] : default(T);
                    visitor(i, a.Values[i], bVal);
                }
            }
            else
            {
                // Both sparse
                int aI = 0;
                int bI = 0;
                while (aI < a.Count && bI < b.Count)
                {
                    int diff = a.Indices[aI] - b.Indices[bI];
                    if (diff == 0)
                    {
                        visitor(b.Indices[bI], a.Values[aI], b.Values[bI]);
                        aI++;
                        bI++;
                    }
                    else if (diff < 0)
                    {
                        visitor(a.Indices[aI], a.Values[aI], default(T));
                        aI++;
                    }
                    else
                    {
                        visitor(b.Indices[bI], default(T), b.Values[bI]);
                        bI++;
                    }
                }

                while (aI < a.Count)
                {
                    visitor(a.Indices[aI], a.Values[aI], default(T));
                    aI++;
                }

                while (bI < b.Count)
                {
                    visitor(b.Indices[bI], default(T), b.Values[bI]);
                    bI++;
                }
            }
        }

        /// <summary>
        /// Sets all values in the vector to the default value for the type, without changing the
        /// density or index structure of the input array. That is to say, the count of the input
        /// vector will be the same afterwards as it was before.
        /// </summary>
        public static void Clear<T>(ref VBuffer<T> dst)
        {
            if (dst.Count == 0)
                return;
            Array.Clear(dst.Values, 0, dst.Count);
        }

        // REVIEW: Look into removing slot in this and other manipulators, so that we
        // could potentially have something around, say, skipping default entries.

        /// <summary>
        /// A delegate for functions that can change a value.
        /// </summary>
        /// <param name="slot">Index of entry</param>
        /// <param name="value">Value to change</param>
        public delegate void SlotValueManipulator<T>(int slot, ref T value);

        /// <summary>
        /// A predicate on some sort of value.
        /// </summary>
        /// <param name="src">The value to test</param>
        /// <returns>The result of some sort of test from that value</returns>
        public delegate bool ValuePredicate<T>(ref T src);

        /// <summary>
        /// Applies the <paramref name="manip"/> to every explicitly defined
        /// element of the vector.
        /// </summary>
        public static void Apply<T>(ref VBuffer<T> dst, SlotValueManipulator<T> manip)
        {
            Contracts.CheckValue(manip, nameof(manip));

            if (dst.IsDense)
            {
                for (int i = 0; i < dst.Length; i++)
                    manip(i, ref dst.Values[i]);
            }
            else
            {
                for (int i = 0; i < dst.Count; i++)
                    manip(dst.Indices[i], ref dst.Values[i]);
            }
        }

        /// <summary>
        /// Applies some function on a value at a particular slot value, changing that slot value.
        /// This function will, wherever possible, not change the structure of <paramref name="dst"/>.
        /// If the vector is sparse, and the corresponding slot is not explicitly represented,
        /// then this can involve memory copying and possibly memory reallocation on <paramref name="dst"/>.
        /// However, if the item is explicitly represented, even if the item is set to the default
        /// value of <typeparamref name="T"/> it will not change the structure of <paramref name="dst"/>,
        /// in terms of sparsifying a dense array, or dropping indices.
        /// </summary>
        /// <param name="dst">The vector to modify</param>
        /// <param name="slot">The slot of the vector to modify</param>
        /// <param name="manip">The manipulation function</param>
        /// <param name="pred">A predicate that returns true if we should skip insertion of a value into 
        /// sparse vector if it was default. If the predicate is null, we insert any non-default.</param>
        public static void ApplyAt<T>(ref VBuffer<T> dst, int slot, SlotValueManipulator<T> manip, ValuePredicate<T> pred = null)
        {
            Contracts.CheckParam(0 <= slot && slot < dst.Length, nameof(slot));
            Contracts.CheckValue(manip, nameof(manip));
            Contracts.CheckValueOrNull(pred);

            if (dst.IsDense)
            {
                // The vector is dense, so we can just do a direct access.
                manip(slot, ref dst.Values[slot]);
                return;
            }
            int idx = 0;
            if (dst.Count > 0 && Utils.TryFindIndexSorted(dst.Indices, 0, dst.Count, slot, out idx))
            {
                // Vector is sparse, but the item exists so we can access it.
                manip(slot, ref dst.Values[idx]);
                return;
            }
            // The vector is sparse and there is no correpsonding item, yet.
            T value = default(T);
            manip(slot, ref value);
            // If this item is not defined and it's default, no need to proceed of course.
            pred = pred ?? ((ref T val) => Comparer<T>.Default.Compare(val, default(T)) == 0);
            if (pred(ref value))
                return;
            // We have to insert this value, somehow.
            int[] indices = dst.Indices;
            T[] values = dst.Values;
            // There is a modest special case where there is exactly one free slot
            // we are modifying in the sparse vector, in which case the vector becomes
            // dense. Then there is no need to do anything with indices.
            bool needIndices = dst.Count + 1 < dst.Length;
            if (needIndices)
                Utils.EnsureSize(ref indices, dst.Count + 1, dst.Length - 1);
            Utils.EnsureSize(ref values, dst.Count + 1, dst.Length);
            if (idx != dst.Count)
            {
                // We have to do some sort of shift copy.
                if (needIndices)
                    Array.Copy(indices, idx, indices, idx + 1, dst.Count - idx);
                Array.Copy(values, idx, values, idx + 1, dst.Count - idx);
            }
            if (needIndices)
                indices[idx] = slot;
            values[idx] = value;
            dst = new VBuffer<T>(dst.Length, dst.Count + 1, values, indices);
        }

        /// <summary>
        /// Given a vector, turns it into an equivalent dense representation.
        /// </summary>
        public static void Densify<T>(ref VBuffer<T> dst)
        {
            if (dst.IsDense)
                return;
            var indices = dst.Indices;
            var values = dst.Values;
            if (Utils.Size(values) >= dst.Length)
            {
                // Densify in place.
                for (int i = dst.Count; --i >= 0; )
                {
                    Contracts.Assert(i <= indices[i]);
                    values[indices[i]] = values[i];
                }
                if (dst.Count == 0)
                    Array.Clear(values, 0, dst.Length);
                else
                {
                    int min = 0;
                    for (int ii = 0; ii < dst.Count; ++ii)
                    {
                        Array.Clear(values, min, indices[ii] - min);
                        min = indices[ii] + 1;
                    }
                    Array.Clear(values, min, dst.Length - min);
                }
            }
            else
            {
                T[] newValues = new T[dst.Length];
                for (int i = 0; i < dst.Count; ++i)
                    newValues[indices[i]] = values[i];
                values = newValues;
            }
            dst = new VBuffer<T>(dst.Length, values, indices);
        }

        /// <summary>
        /// Given a vector, ensure that the first <paramref name="denseCount"/> slots are explicitly
        /// represented.
        /// </summary>
        public static void DensifyFirst<T>(ref VBuffer<T> dst, int denseCount)
        {
            Contracts.Check(0 <= denseCount && denseCount <= dst.Length);
            if (dst.IsDense || denseCount == 0 || (dst.Count >= denseCount && dst.Indices[denseCount - 1] == denseCount - 1))
                return;
            if (denseCount == dst.Length)
            {
                Densify(ref dst);
                return;
            }

            // Densify the first BiasCount entries.
            int[] indices = dst.Indices;
            T[] values = dst.Values;
            if (indices == null)
            {
                Contracts.Assert(dst.Count == 0);
                indices = Utils.GetIdentityPermutation(denseCount);
                Utils.EnsureSize(ref values, denseCount, dst.Length, keepOld: false);
                Array.Clear(values, 0, denseCount);
                dst = new VBuffer<T>(dst.Length, denseCount, values, indices);
                return;
            }
            int lim = Utils.FindIndexSorted(indices, 0, dst.Count, denseCount);
            Contracts.Assert(lim < denseCount);
            int newLen = dst.Count + denseCount - lim;
            if (newLen == dst.Length)
            {
                Densify(ref dst);
                return;
            }
            Utils.EnsureSize(ref values, newLen, dst.Length);
            Utils.EnsureSize(ref indices, newLen, dst.Length);
            Array.Copy(values, lim, values, denseCount, dst.Count - lim);
            Array.Copy(indices, lim, indices, denseCount, dst.Count - lim);
            int i = lim - 1;
            for (int ii = denseCount; --ii >= 0; )
            {
                values[ii] = i >= 0 && indices[i] == ii ? values[i--] : default(T);
                indices[ii] = ii;
            }
            dst = new VBuffer<T>(dst.Length, newLen, values, indices);
        }

        /// <summary>
        /// Creates a maybe sparse copy of a VBuffer. 
        /// Whether the created copy is sparse or not is determined by the proportion of non-default entries compared to the sparsity parameter.
        /// </summary>
        public static void CreateMaybeSparseCopy<T>(ref VBuffer<T> src, ref VBuffer<T> dst, RefPredicate<T> isDefaultPredicate, float sparsityThreshold = SparsityThreshold)
        {
            Contracts.CheckParam(0 < sparsityThreshold && sparsityThreshold < 1, nameof(sparsityThreshold));
            if (!src.IsDense || src.Length < 20)
            {
                src.CopyTo(ref dst);
                return;
            }

            int sparseCount = 0;
            var sparseCountThreshold = (int)(src.Length * sparsityThreshold);
            for (int i = 0; i < src.Length; i++)
            {
                if (!isDefaultPredicate(ref src.Values[i]))
                    sparseCount++;

                if (sparseCount > sparseCountThreshold)
                {
                    src.CopyTo(ref dst);
                    return;
                }
            }

            var indices = dst.Indices;
            var values = dst.Values;

            if (sparseCount > 0)
            {
                if (Utils.Size(values) < sparseCount)
                    values = new T[sparseCount];
                if (Utils.Size(indices) < sparseCount)
                    indices = new int[sparseCount];
                int j = 0;
                for (int i = 0; i < src.Length; i++)
                {
                    if (!isDefaultPredicate(ref src.Values[i]))
                    {
                        Contracts.Assert(j < sparseCount);
                        indices[j] = i;
                        values[j] = src.Values[i];
                        j++;
                    }
                }

                Contracts.Assert(j == sparseCount);
            }

            dst = new VBuffer<T>(src.Length, sparseCount, values, indices);
        }

        /// <summary>
        /// A delegate for functions that access an index and two corresponding
        /// values, possibly changing one of them.
        /// </summary>
        /// <param name="slot">Slot index of the entry.</param>
        /// <param name="src">Value from first vector.</param>
        /// <param name="dst">Value from second vector, which may be manipulated.</param>
        public delegate void PairManipulator<TSrc, TDst>(int slot, TSrc src, ref TDst dst);

        /// <summary>
        /// A delegate for functions that access an index and two corresponding
        /// values, stores the result in another vector.
        /// </summary>
        /// <param name="slot">Slot index of the entry.</param>
        /// <param name="src">Value from first vector.</param>
        /// <param name="dst">Value from second vector.</param>
        /// <param name="res">The value to store the result.</param>
        public delegate void PairManipulatorCopy<TSrc, TDst>(int slot, TSrc src, TDst dst, ref TDst res);

        /// <summary>
        /// Applies the <see cref="PairManipulator{TSrc,TDst}"/> to each pair of elements
        /// where <paramref name="src"/> is defined, in order of index. If there is
        /// some value at an index in <paramref name="dst"/> that is not defined in
        /// <paramref name="src"/>, that item remains without any further modification.
        /// If either of the vectors are dense, the resulting <paramref name="dst"/>
        /// will be dense. Otherwise, if both are sparse, the output will be sparse iff
        /// there is any slot that is not explicitly represented in either vector.
        /// </summary>
        /// <param name="src">Argument vector, whose elements are only read</param>
        /// <param name="dst">Argument vector, that could change</param>
        /// <param name="manip">Function to apply to each pair of elements</param>
        public static void ApplyWith<TSrc, TDst>(ref VBuffer<TSrc> src, ref VBuffer<TDst> dst, PairManipulator<TSrc, TDst> manip)
        {
            ApplyWithCore(ref src, ref dst, manip, outer: false);
        }

        /// <summary>
        /// Applies the <see cref="PairManipulator{TSrc,TDst}"/> to each pair of elements
        /// where <paramref name="src"/> is defined, in order of index. It stores the result 
        /// in another vector. If there is some value at an index in <paramref name="dst"/> 
        /// that is not defined in <paramref name="src"/>, that slot value is copied to the 
        /// corresponding slot in the result vector without any further modification.
        /// If either of the vectors are dense, the resulting <paramref name="res"/>
        /// will be dense. Otherwise, if both are sparse, the output will be sparse iff
        /// there is any slot that is not explicitly represented in either vector.
        /// </summary>
        /// <param name="src">Argument vector, whose elements are only read</param>
        /// <param name="dst">Argument vector, whose elements are only read</param>
        /// <param name="res">Result vector</param>
        /// <param name="manip">Function to apply to each pair of elements</param>
        public static void ApplyWithCopy<TSrc, TDst>(ref VBuffer<TSrc> src, ref VBuffer<TDst> dst, ref VBuffer<TDst> res, PairManipulatorCopy<TSrc, TDst> manip)
        {
            ApplyWithCoreCopy(ref src, ref dst, ref res, manip, outer: false);
        }

        /// <summary>
        /// Applies the <see cref="PairManipulator{TSrc,TDst}"/> to each pair of elements
        /// where either <paramref name="src"/> or <paramref name="dst"/>, has an element
        /// defined at that index. If either of the vectors are dense, the resulting
        /// <paramref name="dst"/> will be dense. Otherwise, if both are sparse, the output
        /// will be sparse iff there is any slot that is not explicitly represented in
        /// either vector.
        /// </summary>
        /// <param name="src">Argument vector, whose elements are only read</param>
        /// <param name="dst">Argument vector, that could change</param>
        /// <param name="manip">Function to apply to each pair of elements</param>
        public static void ApplyWithEitherDefined<TSrc, TDst>(ref VBuffer<TSrc> src, ref VBuffer<TDst> dst, PairManipulator<TSrc, TDst> manip)
        {
            ApplyWithCore(ref src, ref dst, manip, outer: true);
        }

        /// <summary>
        /// Applies the <see cref="PairManipulator{TSrc,TDst}"/> to each pair of elements
        /// where either <paramref name="src"/> or <paramref name="dst"/>, has an element
        /// defined at that index. It stores the result in another vector <paramref name="res"/>. 
        /// If either of the vectors are dense, the resulting <paramref name="res"/>
        /// will be dense. Otherwise, if both are sparse, the output will be sparse iff
        /// there is any slot that is not explicitly represented in either vector.
        /// </summary>
        /// <param name="src">Argument vector, whose elements are only read</param>
        /// <param name="dst">Argument vector, whose elements are only read</param>
        /// <param name="res">Result vector</param>
        /// <param name="manip">Function to apply to each pair of elements</param>
        public static void ApplyWithEitherDefinedCopy<TSrc, TDst>(ref VBuffer<TSrc> src, ref VBuffer<TDst> dst, ref VBuffer<TDst> res, PairManipulatorCopy<TSrc, TDst> manip)
        {
            ApplyWithCoreCopy(ref src, ref dst, ref res, manip, outer: true);
        }

        /// <summary>
        /// The actual implementation of <see cref="ApplyWith"/> and
        /// <see cref="ApplyWithEitherDefined{TSrc,TDst}"/>, that has internal branches on the implementation
        /// where necessary depending on whether this is an inner or outer join of the
        /// indices of <paramref name="src"/> on <paramref name="dst"/>.
        /// </summary>
        private static void ApplyWithCore<TSrc, TDst>(ref VBuffer<TSrc> src, ref VBuffer<TDst> dst, PairManipulator<TSrc, TDst> manip, bool outer)
        {
            Contracts.Check(src.Length == dst.Length, "Vectors must have the same dimensionality.");
            Contracts.CheckValue(manip, nameof(manip));

            // We handle all of the permutations of the density/sparsity of src/dst through
            // special casing below. Each subcase in turn handles appropriately the treatment
            // of the "outer" parameter. There are nine, top level cases. Each case is
            // considered in this order.

            // 1. src.Count == 0.
            // 2. src.Dense.
            // 3. dst.Dense.
            // 4. dst.Count == 0.

            // Beyond this point the cases can assume both src/dst are sparse non-empty vectors.
            // We then calculate the size of the resulting output array, then use that to fall
            // through to more special cases.

            // 5. The union will result in dst becoming dense. So just densify it, then recurse.
            // 6. Neither src nor dst's indices is a subset of the other.
            // 7. The two sets of indices are identical.
            // 8. src's indices are a subset of dst's.
            // 9. dst's indices are a subset of src's.

            // Each one of these subcases also separately handles the "outer" parameter, if
            // necessary. It is unnecessary if src's indices form a superset (proper or improper)
            // of dst's indices. So, for example, cases 2, 4, 7, 9 do not require special handling.
            // Case 5 does not require special handling, because it falls through to other cases
            // that do the special handling for them.

            if (src.Count == 0)
            {
                // Major case 1, with src.Count == 0.
                if (!outer)
                    return;
                if (dst.IsDense)
                {
                    for (int i = 0; i < dst.Length; i++)
                        manip(i, default(TSrc), ref dst.Values[i]);
                }
                else
                {
                    for (int i = 0; i < dst.Count; i++)
                        manip(dst.Indices[i], default(TSrc), ref dst.Values[i]);
                }
                return;
            }

            if (src.IsDense)
            {
                // Major case 2, with src.Dense.
                if (!dst.IsDense)
                    Densify(ref dst);
                // Both are now dense. Both cases of outer are covered.
                for (int i = 0; i < src.Length; i++)
                    manip(i, src.Values[i], ref dst.Values[i]);
                return;
            }

            if (dst.IsDense)
            {
                // Major case 3, with dst.Dense. Note that !a.Dense.
                if (outer)
                {
                    int sI = 0;
                    int sIndex = src.Indices[sI];
                    for (int i = 0; i < dst.Length; ++i)
                    {
                        if (i == sIndex)
                        {
                            manip(i, src.Values[sI], ref dst.Values[i]);
                            sIndex = ++sI == src.Count ? src.Length : src.Indices[sI];
                        }
                        else
                            manip(i, default(TSrc), ref dst.Values[i]);
                    }
                }
                else
                {
                    for (int i = 0; i < src.Count; i++)
                        manip(src.Indices[i], src.Values[i], ref dst.Values[src.Indices[i]]);
                }
                return;
            }

            if (dst.Count == 0)
            {
                // Major case 4, with dst empty. Note that !src.Dense.
                // Neither is dense, and dst is empty. Both cases of outer are covered.
                var values = dst.Values;
                var indices = dst.Indices;
                Utils.EnsureSize(ref values, src.Count, src.Length);
                Array.Clear(values, 0, src.Count);
                Utils.EnsureSize(ref indices, src.Count, src.Length);
                for (int i = 0; i < src.Count; i++)
                    manip(indices[i] = src.Indices[i], src.Values[i], ref values[i]);
                dst = new VBuffer<TDst>(src.Length, src.Count, values, indices);
                return;
            }

            // Beyond this point, we can assume both a and b are sparse with positive count.
            int dI = 0;
            int newCount = dst.Count;
            // Try to find each src index in dst indices, counting how many more we'll add.
            for (int sI = 0; sI < src.Count; sI++)
            {
                int sIndex = src.Indices[sI];
                while (dI < dst.Count && dst.Indices[dI] < sIndex)
                    dI++;
                if (dI == dst.Count)
                {
                    newCount += src.Count - sI;
                    break;
                }
                if (dst.Indices[dI] == sIndex)
                    dI++;
                else
                    newCount++;
            }
            Contracts.Assert(newCount > 0);
            Contracts.Assert(0 < src.Count && src.Count <= newCount);
            Contracts.Assert(0 < dst.Count && dst.Count <= newCount);

            // REVIEW: Densify above a certain threshold, not just if
            // the output will necessarily become dense? But then we get into
            // the dubious business of trying to pick the "right" densification
            // threshold.
            if (newCount == dst.Length)
            {
                // Major case 5, dst will become dense through the application of
                // this. Just recurse one level so one of the initial conditions
                // can catch it, specifically, the major case 3.

                // This is unnecessary -- falling through to the sparse code will
                // actually handle this case just fine -- but it is more efficient.
                Densify(ref dst);
                ApplyWithCore(ref src, ref dst, manip, outer);
                return;
            }

            if (newCount != src.Count && newCount != dst.Count)
            {
                // Major case 6, neither set of indices is a subset of the other.
                // This subcase used to fall through to another subcase, but this
                // proved to be inefficient so we go to the little bit of extra work
                // to handle it here.

                var indices = dst.Indices;
                var values = dst.Values;
                Utils.EnsureSize(ref indices, newCount, dst.Length, keepOld: false);
                Utils.EnsureSize(ref values, newCount, dst.Length, keepOld: false);
                int sI = src.Count - 1;
                dI = dst.Count - 1;
                int sIndex = src.Indices[sI];
                int dIndex = dst.Indices[dI];

                // Go from the end, so that even if we're writing over dst's vectors in
                // place, we do not corrupt the data as we are reorganizing it.
                for (int i = newCount; --i >= 0; )
                {
                    if (sIndex < dIndex)
                    {
                        indices[i] = dIndex;
                        values[i] = dst.Values[dI];
                        if (outer)
                            manip(dIndex, default(TSrc), ref values[i]);
                        dIndex = --dI >= 0 ? dst.Indices[dI] : -1;
                    }
                    else if (sIndex > dIndex)
                    {
                        indices[i] = sIndex;
                        values[i] = default(TDst);
                        manip(sIndex, src.Values[sI], ref values[i]);
                        sIndex = --sI >= 0 ? src.Indices[sI] : -1;
                    }
                    else
                    {
                        // We should not have run past the beginning, due to invariants.
                        Contracts.Assert(sIndex >= 0);
                        Contracts.Assert(sIndex == dIndex);
                        indices[i] = dIndex;
                        values[i] = dst.Values[dI];
                        manip(sIndex, src.Values[sI], ref values[i]);
                        sIndex = --sI >= 0 ? src.Indices[sI] : -1;
                        dIndex = --dI >= 0 ? dst.Indices[dI] : -1;
                    }
                }
                dst = new VBuffer<TDst>(dst.Length, newCount, values, indices);
                return;
            }

            if (newCount == dst.Count)
            {
                if (newCount == src.Count)
                {
                    // Major case 7, the set of indices is the same for src and dst.
                    Contracts.Assert(src.Count == dst.Count);
                    for (int i = 0; i < src.Count; i++)
                    {
                        Contracts.Assert(src.Indices[i] == dst.Indices[i]);
                        manip(src.Indices[i], src.Values[i], ref dst.Values[i]);
                    }
                    return;
                }
                // Major case 8, the indices of src must be a subset of dst's indices.
                Contracts.Assert(newCount > src.Count);
                dI = 0;
                if (outer)
                {
                    int sI = 0;
                    int sIndex = src.Indices[sI];
                    for (int i = 0; i < dst.Count; ++i)
                    {
                        if (dst.Indices[i] == sIndex)
                        {
                            manip(sIndex, src.Values[sI], ref dst.Values[i]);
                            sIndex = ++sI == src.Count ? src.Length : src.Indices[sI];
                        }
                        else
                            manip(dst.Indices[i], default(TSrc), ref dst.Values[i]);
                    }
                }
                else
                {
                    for (int sI = 0; sI < src.Count; sI++)
                    {
                        int sIndex = src.Indices[sI];
                        while (dst.Indices[dI] < sIndex)
                            dI++;
                        Contracts.Assert(dst.Indices[dI] == sIndex);
                        manip(sIndex, src.Values[sI], ref dst.Values[dI++]);
                    }
                }
                return;
            }

            if (newCount == src.Count)
            {
                // Major case 9, the indices of dst must be a subset of src's indices. Both cases of outer are covered.

                // First do a "quasi" densification of dst, by making the indices
                // of dst correspond to those in src.
                int sI = 0;
                for (dI = 0; dI < dst.Count; ++dI)
                {
                    int bIndex = dst.Indices[dI];
                    while (src.Indices[sI] < bIndex)
                        sI++;
                    Contracts.Assert(src.Indices[sI] == bIndex);
                    dst.Indices[dI] = sI++;
                }
                dst = new VBuffer<TDst>(newCount, dst.Count, dst.Values, dst.Indices);
                Densify(ref dst);
                int[] indices = dst.Indices;
                Utils.EnsureSize(ref indices, src.Count, src.Length, keepOld: false);
                Array.Copy(src.Indices, indices, newCount);
                dst = new VBuffer<TDst>(src.Length, newCount, dst.Values, indices);
                for (sI = 0; sI < src.Count; sI++)
                    manip(src.Indices[sI], src.Values[sI], ref dst.Values[sI]);
                return;
            }

            Contracts.Assert(false);
        }

        /// <summary>
        /// The actual implementation of <see cref="ApplyWithCopy{TSrc,TDst}"/> and
        /// <see cref="ApplyWithEitherDefinedCopy{TSrc,TDst}"/>, that has internal branches on the implementation
        /// where necessary depending on whether this is an inner or outer join of the
        /// indices of <paramref name="src"/> on <paramref name="dst"/>.
        /// </summary>
        private static void ApplyWithCoreCopy<TSrc, TDst>(ref VBuffer<TSrc> src, ref VBuffer<TDst> dst, ref VBuffer<TDst> res, PairManipulatorCopy<TSrc, TDst> manip, bool outer)
        {
            Contracts.Check(src.Length == dst.Length, "Vectors must have the same dimensionality.");
            Contracts.CheckValue(manip, nameof(manip));
            Contracts.Assert(Utils.Size(src.Values) >= src.Count);
            Contracts.Assert(Utils.Size(dst.Values) >= dst.Count);
            int length = src.Length;

            if (dst.Count == 0)
            {
                if (src.Count == 0)
                    res = new VBuffer<TDst>(length, 0, res.Values, res.Indices);
                else if (src.IsDense)
                {
                    Contracts.Assert(src.Count == src.Length);
                    TDst[] resValues = Utils.Size(res.Values) >= length ? res.Values : new TDst[length];
                    for (int i = 0; i < length; i++)
                        manip(i, src.Values[i], default(TDst), ref resValues[i]);
                    res = new VBuffer<TDst>(length, resValues, res.Indices);
                }
                else
                {
                    // src is non-empty sparse.
                    int count = src.Count;
                    Contracts.Assert(0 < count && count < length);
                    int[] resIndices = Utils.Size(res.Indices) >= count ? res.Indices : new int[count];
                    TDst[] resValues = Utils.Size(res.Values) >= count ? res.Values : new TDst[count];
                    Array.Copy(src.Indices, resIndices, count);
                    for (int ii = 0; ii < count; ii++)
                    {
                        int i = src.Indices[ii];
                        resIndices[ii] = i;
                        manip(i, src.Values[ii], default(TDst), ref resValues[ii]);
                    }
                    res = new VBuffer<TDst>(length, count, resValues, resIndices);
                }
            }
            else if (dst.IsDense)
            {
                TDst[] resValues = Utils.Size(res.Values) >= length ? res.Values : new TDst[length];
                if (src.Count == 0)
                {
                    if (outer)
                    {
                        // Apply manip to all slots, as all slots of dst are defined.
                        for (int j = 0; j < length; j++)
                            manip(j, default(TSrc), dst.Values[j], ref resValues[j]);
                    }
                    else
                    {
                        // Copy only. No slot of src is defined.
                        for (int j = 0; j < length; j++)
                            resValues[j] = dst.Values[j];
                    }
                    res = new VBuffer<TDst>(length, resValues, res.Indices);
                }
                else if (src.IsDense)
                {
                    Contracts.Assert(src.Count == src.Length);
                    for (int i = 0; i < length; i++)
                        manip(i, src.Values[i], dst.Values[i], ref resValues[i]);
                    res = new VBuffer<TDst>(length, resValues, res.Indices);
                }
                else
                {
                    // src is sparse and non-empty.
                    int count = src.Count;
                    Contracts.Assert(0 < count && count < length);

                    int ii = 0;
                    int i = src.Indices[ii];
                    if (outer)
                    {
                        // All slots of dst are defined. Always apply manip.
                        for (int j = 0; j < length; j++)
                        {
                            if (j == i)
                            {
                                manip(j, src.Values[ii], dst.Values[j], ref resValues[j]);
                                i = ++ii == count ? length : src.Indices[ii];
                            }
                            else
                                manip(j, default(TSrc), dst.Values[j], ref resValues[j]);
                        }
                    }
                    else
                    {
                        // Only apply manip for those slots where src is defined. Otherwise just copy.
                        for (int j = 0; j < length; j++)
                        {
                            if (j == i)
                            {
                                manip(j, src.Values[ii], dst.Values[j], ref resValues[j]);
                                i = ++ii == count ? length : src.Indices[ii];
                            }
                            else
                                resValues[j] = dst.Values[j];
                        }
                    }
                    res = new VBuffer<TDst>(length, resValues, res.Indices);
                }
            }
            else
            {
                // dst is non-empty sparse
                int dstCount = dst.Count;
                Contracts.Assert(dstCount > 0);
                if (src.Count == 0)
                {
                    int[] resIndices = Utils.Size(res.Indices) >= dstCount ? res.Indices : new int[dstCount];
                    TDst[] resValues = Utils.Size(res.Values) >= dstCount ? res.Values : new TDst[dstCount];
                    if (outer)
                    {
                        for (int jj = 0; jj < dstCount; jj++)
                        {
                            int j = dst.Indices[jj];
                            resIndices[jj] = j;
                            manip(j, default(TSrc), dst.Values[jj], ref resValues[jj]);
                        }
                    }
                    else
                    {
                        for (int jj = 0; jj < dstCount; jj++)
                        {
                            resIndices[jj] = dst.Indices[jj];
                            resValues[jj] = dst.Values[jj];
                        }
                    }
                    res = new VBuffer<TDst>(length, dstCount, resValues, resIndices);
                }
                else if (src.IsDense)
                {
                    // res will be dense.
                    TDst[] resValues = Utils.Size(res.Values) >= length ? res.Values : new TDst[length];
                    int jj = 0;
                    int j = dst.Indices[jj];
                    for (int i = 0; i < length; i++)
                    {
                        if (i == j)
                        {
                            manip(i, src.Values[i], dst.Values[jj], ref resValues[i]);
                            j = ++jj == dstCount ? length : dst.Indices[jj];
                        }
                        else
                            manip(i, src.Values[i], default(TDst), ref resValues[i]);
                    }
                    res = new VBuffer<TDst>(length, resValues, res.Indices);
                }
                else
                {
                    // Both src and dst are non-empty sparse.
                    Contracts.Assert(src.Count > 0);

                    // Find the count of result, which is the size of the union of the indices set of src and dst.
                    int resCount = dstCount;
                    for (int ii = 0, jj = 0; ii < src.Count; ii++)
                    {
                        int i = src.Indices[ii];
                        while (jj < dst.Count && dst.Indices[jj] < i)
                            jj++;
                        if (jj == dst.Count)
                        {
                            resCount += src.Count - ii;
                            break;
                        }
                        if (dst.Indices[jj] == i)
                            jj++;
                        else
                            resCount++;
                    }

                    Contracts.Assert(0 < resCount && resCount <= length);
                    Contracts.Assert(resCount <= src.Count + dstCount);
                    Contracts.Assert(src.Count <= resCount);
                    Contracts.Assert(dstCount <= resCount);

                    if (resCount == length)
                    {
                        // result will become dense.
                        // This is unnecessary -- falling through to the sparse code will
                        // actually handle this case just fine -- but it is more efficient.
                        Densify(ref dst);
                        ApplyWithCoreCopy(ref src, ref dst, ref res, manip, outer);
                    }
                    else
                    {
                        int[] resIndices = Utils.Size(res.Indices) >= resCount ? res.Indices : new int[resCount];
                        TDst[] resValues = Utils.Size(res.Values) >= resCount ? res.Values : new TDst[resCount];

                        int ii = 0;
                        int i = src.Indices[ii];
                        int jj = 0;
                        int j = dst.Indices[jj];

                        for (int kk = 0; kk < resCount; kk++)
                        {
                            Contracts.Assert(i < length || j < length);
                            if (i == j)
                            {
                                // Slot (i == j) both defined in src and dst. Apply manip.
                                resIndices[kk] = i;
                                manip(i, src.Values[ii], dst.Values[jj], ref resValues[kk]);
                                i = ++ii == src.Count ? length : src.Indices[ii];
                                j = ++jj == dstCount ? length : dst.Indices[jj];
                            }
                            else if (i < j)
                            {
                                // Slot i defined only in src, but not in dst. Apply manip.
                                resIndices[kk] = i;
                                manip(i, src.Values[ii], default(TDst), ref resValues[kk]);
                                i = ++ii == src.Count ? length : src.Indices[ii];
                            }
                            else
                            {
                                // Slot j defined only in dst, but not in src. Apply manip if outer.
                                // Otherwise just copy.
                                resIndices[kk] = j;
                                // REVIEW: Should we move checking of outer outside the loop?
                                if (outer)
                                    manip(j, default(TSrc), dst.Values[jj], ref resValues[kk]);
                                else
                                    resValues[kk] = dst.Values[jj];
                                j = ++jj == dstCount ? length : dst.Indices[jj];
                            }
                        }

                        Contracts.Assert(ii == src.Count && jj == dstCount);
                        Contracts.Assert(i == length && j == length);
                        res = new VBuffer<TDst>(length, resCount, resValues, resIndices);
                    }
                }
            }
        }

        /// <summary>
        /// Applies a function to explicitly defined elements in a vector <paramref name="src"/>,
        /// storing the result in <paramref name="dst"/>, overwriting any of its existing contents.
        /// The contents of <paramref name="dst"/> do not affect calculation. If you instead wish
        /// to calculate a function that reads and writes <paramref name="dst"/>, see
        /// <see cref="ApplyWith"/> and <see cref="ApplyWithEitherDefined"/>. Post-operation,
        /// <paramref name="dst"/> will be dense iff <paramref name="src"/> is dense.
        /// </summary>
        /// <seealso cref="ApplyWith"/>
        /// <seealso cref="ApplyWithEitherDefined"/>
        public static void ApplyIntoEitherDefined<TSrc, TDst>(ref VBuffer<TSrc> src, ref VBuffer<TDst> dst, Func<int, TSrc, TDst> func)
        {
            Contracts.CheckValue(func, nameof(func));

            // REVIEW: The analogous WritableVector method insisted on
            // equal lengths, but I don't care here.
            if (src.Count == 0)
            {
                dst = new VBuffer<TDst>(src.Length, src.Count, dst.Values, dst.Indices);
                return;
            }
            int[] indices = dst.Indices;
            TDst[] values = dst.Values;
            Utils.EnsureSize(ref values, src.Count, src.Length, keepOld: false);
            if (src.IsDense)
            {
                for (int i = 0; i < src.Length; ++i)
                    values[i] = func(i, src.Values[i]);
            }
            else
            {
                Utils.EnsureSize(ref indices, src.Count, src.Length, keepOld: false);
                Array.Copy(src.Indices, indices, src.Count);
                for (int i = 0; i < src.Count; ++i)
                    values[i] = func(src.Indices[i], src.Values[i]);
            }
            dst = new VBuffer<TDst>(src.Length, src.Count, values, indices);
        }

        /// <summary>
        /// Applies a function <paramref name="func"/> to two vectors, storing the result in
        /// <paramref name="dst"/>, whose existing contents are discarded and overwritten. The
        /// function is called for every index value that appears in either <paramref name="a"/>
        /// or <paramref name="b"/>. If either of the two inputs is dense, the output will
        /// necessarily be dense. Otherwise, if both are sparse, the output will be sparse iff
        /// there is any slot that is not explicitly represented in either vector.
        /// </summary>
        public static void ApplyInto<TSrc1, TSrc2, TDst>(ref VBuffer<TSrc1> a, ref VBuffer<TSrc2> b, ref VBuffer<TDst> dst, Func<int, TSrc1, TSrc2, TDst> func)
        {
            Contracts.Check(a.Length == b.Length, "Vectors must have the same dimensionality.");
            Contracts.CheckValue(func, nameof(func));

            // We handle the following cases:
            // 1. When a and b are both empty, we set the result to empty.
            // 2. When either a or b are dense, then the result will be dense, and we have some
            //    special casing for the sparsity of either a or b.
            // Then we have the case where both are sparse. We calculate the size of the output,
            // then fall through to the various cases.
            // 3. a and b have the same indices.
            // 4. a's indices are a subset of b's.
            // 5. b's indices are a subset of a's.
            // 6. Neither a nor b's indices are a subset of the other.

            if (a.Count == 0 && b.Count == 0)
            {
                // Case 1. Output will be empty.
                dst = new VBuffer<TDst>(a.Length, 0, dst.Values, dst.Indices);
                return;
            }

            int aI = 0;
            int bI = 0;
            TDst[] values = dst.Values;
            if (a.IsDense || b.IsDense)
            {
                // Case 2. One of the two inputs is dense. The output will be dense.
                Utils.EnsureSize(ref values, a.Length, a.Length, keepOld: false);

                if (!a.IsDense)
                {
                    // a is sparse, b is dense
                    for (int i = 0; i < b.Length; i++)
                    {
                        TSrc1 aVal = (aI < a.Count && i == a.Indices[aI]) ? a.Values[aI++] : default(TSrc1);
                        values[i] = func(i, aVal, b.Values[i]);
                    }
                }
                else if (!b.IsDense)
                {
                    // b is sparse, a is dense
                    for (int i = 0; i < a.Length; i++)
                    {
                        TSrc2 bVal = (bI < b.Count && i == b.Indices[bI]) ? b.Values[bI++] : default(TSrc2);
                        values[i] = func(i, a.Values[i], bVal);
                    }
                }
                else
                {
                    // both dense
                    for (int i = 0; i < a.Length; i++)
                        values[i] = func(i, a.Values[i], b.Values[i]);
                }
                dst = new VBuffer<TDst>(a.Length, values, dst.Indices);
                return;
            }

            // a, b both sparse.
            int newCount = 0;
            while (aI < a.Count && bI < b.Count)
            {
                int aCompB = a.Indices[aI] - b.Indices[bI];
                if (aCompB <= 0) // a is no larger than b.
                    aI++;
                if (aCompB >= 0) // b is no larger than a.
                    bI++;
                newCount++;
            }

            if (aI < a.Count)
                newCount += a.Count - aI;
            if (bI < b.Count)
                newCount += b.Count - bI;

            // REVIEW: Worth optimizing the newCount == a.Length case?
            // Probably not...

            int[] indices = dst.Indices;
            Utils.EnsureSize(ref indices, newCount, a.Length, keepOld: false);
            Utils.EnsureSize(ref values, newCount, a.Length, keepOld: false);

            if (newCount == b.Count)
            {
                if (newCount == a.Count)
                {
                    // Case 3, a and b actually have the same indices!
                    Array.Copy(a.Indices, indices, a.Count);
                    for (aI = 0; aI < a.Count; aI++)
                    {
                        Contracts.Assert(a.Indices[aI] == b.Indices[aI]);
                        values[aI] = func(a.Indices[aI], a.Values[aI], b.Values[aI]);
                    }
                }
                else
                {
                    // Case 4, a's indices are a subset of b's.
                    Array.Copy(b.Indices, indices, b.Count);
                    aI = 0;
                    for (bI = 0; aI < a.Count && bI < b.Count; bI++)
                    {
                        Contracts.Assert(a.Indices[aI] >= b.Indices[bI]);
                        TSrc1 aVal = a.Indices[aI] == b.Indices[bI] ? a.Values[aI++] : default(TSrc1);
                        values[bI] = func(b.Indices[bI], aVal, b.Values[bI]);
                    }
                    for (; bI < b.Count; bI++)
                        values[bI] = func(b.Indices[bI], default(TSrc1), b.Values[bI]);
                }
            }
            else if (newCount == a.Count)
            {
                // Case 5, b's indices are a subset of a's.
                Array.Copy(a.Indices, indices, a.Count);
                bI = 0;
                for (aI = 0; bI < b.Count && aI < a.Count; aI++)
                {
                    Contracts.Assert(b.Indices[bI] >= a.Indices[aI]);
                    TSrc2 bVal = a.Indices[aI] == b.Indices[bI] ? b.Values[bI++] : default(TSrc2);
                    values[aI] = func(a.Indices[aI], a.Values[aI], bVal);
                }
                for (; aI < a.Count; aI++)
                    values[aI] = func(a.Indices[aI], a.Values[aI], default(TSrc2));
            }
            else
            {
                // Case 6, neither a nor b's indices are a subset of the other.
                int newI = aI = bI = 0;
                TSrc1 aVal = default(TSrc1);
                TSrc2 bVal = default(TSrc2);
                while (aI < a.Count && bI < b.Count)
                {
                    int aCompB = a.Indices[aI] - b.Indices[bI];
                    int index = 0;

                    if (aCompB < 0)
                    {
                        index = a.Indices[aI];
                        aVal = a.Values[aI++];
                        bVal = default(TSrc2);
                    }
                    else if (aCompB > 0)
                    {
                        index = b.Indices[bI];
                        aVal = default(TSrc1);
                        bVal = b.Values[bI++];
                    }
                    else
                    {
                        index = a.Indices[aI];
                        Contracts.Assert(index == b.Indices[bI]);
                        aVal = a.Values[aI++];
                        bVal = b.Values[bI++];
                    }
                    values[newI] = func(index, aVal, bVal);
                    indices[newI++] = index;
                }

                for (; aI < a.Count; aI++)
                {
                    int index = a.Indices[aI];
                    values[newI] = func(index, a.Values[aI], default(TSrc2));
                    indices[newI++] = index;
                }

                for (; bI < b.Count; bI++)
                {
                    int index = b.Indices[bI];
                    values[newI] = func(index, default(TSrc1), b.Values[bI]);
                    indices[newI++] = index;
                }
            }
            dst = new VBuffer<TDst>(a.Length, newCount, values, indices);
        }

        /// <summary>
        /// Copy from a source list to the given VBuffer destination.
        /// </summary>
        public static void Copy<T>(List<T> src, ref VBuffer<T> dst, int length)
        {
            Contracts.CheckParam(0 <= length && length <= Utils.Size(src), nameof(length));
            var values = dst.Values;
            if (length > 0)
            {
                if (Utils.Size(values) < length)
                    values = new T[length];
                src.CopyTo(values);
            }
            dst = new VBuffer<T>(length, values, dst.Indices);
        }
    }
}
