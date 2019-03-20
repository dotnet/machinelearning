// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Internal.Utilities
{
    // REVIEW: Consider automatic densification in some of the operations, where appropriate.
    // REVIEW: Once we do the conversions from Vector/WritableVector, review names of methods,
    //   parameters, parameter order, etc.
    /// <summary>
    /// Convenience utilities for vector operations on <see cref="VBuffer{T}"/>.
    /// </summary>
    [BestFriend]
    internal static class VBufferUtils
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

        public static bool HasNaNs(in VBuffer<Single> buffer)
        {
            var values = buffer.GetValues();
            for (int i = 0; i < values.Length; i++)
            {
                if (Single.IsNaN(values[i]))
                    return true;
            }
            return false;
        }

        public static bool HasNaNs(in VBuffer<Double> buffer)
        {
            var values = buffer.GetValues();
            for (int i = 0; i < values.Length; i++)
            {
                if (Double.IsNaN(values[i]))
                    return true;
            }
            return false;
        }

        public static bool HasNonFinite(in VBuffer<Single> buffer)
        {
            var values = buffer.GetValues();
            for (int i = 0; i < values.Length; i++)
            {
                if (!FloatUtils.IsFinite(values[i]))
                    return true;
            }
            return false;
        }

        public static bool HasNonFinite(in VBuffer<Double> buffer)
        {
            var values = buffer.GetValues();
            for (int i = 0; i < values.Length; i++)
            {
                if (!FloatUtils.IsFinite(values[i]))
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
        public static void ForEachDefined<T>(in VBuffer<T> a, Action<int, T> visitor)
        {
            Contracts.CheckValue(visitor, nameof(visitor));

            // REVIEW: This is analogous to an old Vector method, but is there
            // any real reason to have it given that we have the Items extension method?
            var aValues = a.GetValues();
            if (a.IsDense)
            {
                for (int i = 0; i < aValues.Length; i++)
                    visitor(i, aValues[i]);
            }
            else
            {
                var aIndices = a.GetIndices();
                for (int i = 0; i < aValues.Length; i++)
                    visitor(aIndices[i], aValues[i]);
            }
        }

        /// <summary>
        /// Applies the <paramref name="visitor "/>to each corresponding pair of elements
        /// where the item is emplicitly defined in the vector. By explicitly defined,
        /// we mean that for a given index <c>i</c>, both vectors have an entry in
        /// <see cref="VBuffer{T}.GetValues"/> corresponding to that index.
        /// </summary>
        /// <param name="a">The first vector</param>
        /// <param name="b">The second vector</param>
        /// <param name="visitor">Delegate to apply to each pair of non-zero values.
        /// This is passed the index, and two values</param>
        public static void ForEachBothDefined<T>(in VBuffer<T> a, in VBuffer<T> b, Action<int, T, T> visitor)
        {
            Contracts.Check(a.Length == b.Length, "Vectors must have the same dimensionality.");
            Contracts.CheckValue(visitor, nameof(visitor));

            var aValues = a.GetValues();
            var bValues = b.GetValues();
            if (a.IsDense && b.IsDense)
            {
                for (int i = 0; i < a.Length; i++)
                    visitor(i, aValues[i], bValues[i]);
            }
            else if (b.IsDense)
            {
                var aIndices = a.GetIndices();
                for (int i = 0; i < aValues.Length; i++)
                    visitor(aIndices[i], aValues[i], bValues[aIndices[i]]);
            }
            else if (a.IsDense)
            {
                var bIndices = b.GetIndices();
                for (int i = 0; i < bValues.Length; i++)
                    visitor(bIndices[i], aValues[bIndices[i]], bValues[i]);
            }
            else
            {
                // Both sparse.
                int aI = 0;
                int bI = 0;
                var aIndices = a.GetIndices();
                var bIndices = b.GetIndices();
                while (aI < aValues.Length && bI < bValues.Length)
                {
                    int i = aIndices[aI];
                    int j = bIndices[bI];
                    if (i == j)
                        visitor(i, aValues[aI++], bValues[bI++]);
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
        public static void ForEachEitherDefined<T>(in VBuffer<T> a, in VBuffer<T> b, Action<int, T, T> visitor)
        {
            Contracts.Check(a.Length == b.Length, "Vectors must have the same dimensionality.");
            Contracts.CheckValue(visitor, nameof(visitor));

            var aValues = a.GetValues();
            var bValues = b.GetValues();
            if (a.IsDense && b.IsDense)
            {
                for (int i = 0; i < a.Length; ++i)
                    visitor(i, aValues[i], bValues[i]);
            }
            else if (b.IsDense)
            {
                int aI = 0;
                var aIndices = a.GetIndices();
                for (int i = 0; i < b.Length; i++)
                {
                    T aVal = (aI < aValues.Length && i == aIndices[aI]) ? aValues[aI++] : default(T);
                    visitor(i, aVal, bValues[i]);
                }
            }
            else if (a.IsDense)
            {
                int bI = 0;
                var bIndices = b.GetIndices();
                for (int i = 0; i < a.Length; i++)
                {
                    T bVal = (bI < bValues.Length && i == bIndices[bI]) ? bValues[bI++] : default(T);
                    visitor(i, aValues[i], bVal);
                }
            }
            else
            {
                // Both sparse
                int aI = 0;
                int bI = 0;
                var aIndices = a.GetIndices();
                var bIndices = b.GetIndices();
                while (aI < aValues.Length && bI < bValues.Length)
                {
                    int diff = aIndices[aI] - bIndices[bI];
                    if (diff == 0)
                    {
                        visitor(bIndices[bI], aValues[aI], bValues[bI]);
                        aI++;
                        bI++;
                    }
                    else if (diff < 0)
                    {
                        visitor(aIndices[aI], aValues[aI], default(T));
                        aI++;
                    }
                    else
                    {
                        visitor(bIndices[bI], default(T), bValues[bI]);
                        bI++;
                    }
                }

                while (aI < aValues.Length)
                {
                    visitor(aIndices[aI], aValues[aI], default(T));
                    aI++;
                }

                while (bI < bValues.Length)
                {
                    visitor(bIndices[bI], default(T), bValues[bI]);
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
            var editor = VBufferEditor.CreateFromBuffer(ref dst);
            editor.Values.Clear();
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

            var editor = VBufferEditor.CreateFromBuffer(ref dst);
            if (dst.IsDense)
            {
                for (int i = 0; i < editor.Values.Length; i++)
                    manip(i, ref editor.Values[i]);
            }
            else
            {
                var dstIndices = dst.GetIndices();
                for (int i = 0; i < editor.Values.Length; i++)
                    manip(dstIndices[i], ref editor.Values[i]);
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

            var editor = VBufferEditor.CreateFromBuffer(ref dst);
            int dstValuesCount = editor.Values.Length;
            if (dst.IsDense)
            {
                // The vector is dense, so we can just do a direct access.
                manip(slot, ref editor.Values[slot]);
                return;
            }
            int idx = 0;
            if (dstValuesCount > 0 && Utils.TryFindIndexSorted(editor.Indices, 0, dstValuesCount, slot, out idx))
            {
                // Vector is sparse, but the item exists so we can access it.
                manip(slot, ref editor.Values[idx]);
                return;
            }
            // The vector is sparse and there is no corresponding item, yet.
            T value = default(T);
            manip(slot, ref value);
            // If this item is not defined and it's default, no need to proceed of course.
            pred = pred ?? ((ref T val) => Comparer<T>.Default.Compare(val, default(T)) == 0);
            if (pred(ref value))
                return;
            // We have to insert this value, somehow.

            // There is a modest special case where there is exactly one free slot
            // we are modifying in the sparse vector, in which case the vector becomes
            // dense. Then there is no need to do anything with indices.
            bool needIndices = dstValuesCount + 1 < dst.Length;
            editor = VBufferEditor.Create(ref dst, dst.Length, dstValuesCount + 1, keepOldOnResize: true);
            if (idx != dstValuesCount)
            {
                // We have to do some sort of shift copy.
                int sliceLength = dstValuesCount - idx;
                if (needIndices)
                    editor.Indices.Slice(idx, sliceLength).CopyTo(editor.Indices.Slice(idx + 1));
                editor.Values.Slice(idx, sliceLength).CopyTo(editor.Values.Slice(idx + 1));
            }
            if (needIndices)
                editor.Indices[idx] = slot;
            editor.Values[idx] = value;
            dst = editor.Commit();
        }

        /// <summary>
        /// Given a vector, turns it into an equivalent dense representation.
        /// </summary>
        public static void Densify<T>(ref VBuffer<T> dst)
        {
            if (dst.IsDense)
                return;

            var indices = dst.GetIndices();
            var values = dst.GetValues();
            var editor = VBufferEditor.Create(
                ref dst,
                dst.Length);

            if (!editor.CreatedNewValues)
            {
                // Densify in place.
                for (int i = values.Length; --i >= 0;)
                {
                    Contracts.Assert(i <= indices[i]);
                    editor.Values[indices[i]] = values[i];
                }
                if (values.Length == 0)
                    editor.Values.Clear();
                else
                {
                    int min = 0;
                    for (int ii = 0; ii < values.Length; ++ii)
                    {
                        editor.Values.Slice(min, indices[ii] - min).Clear();
                        min = indices[ii] + 1;
                    }
                    editor.Values.Slice(min, dst.Length - min).Clear();
                }
            }
            else
            {
                // createdNewValues is true, keepOldOnResize is false, so Values is already cleared
                for (int i = 0; i < values.Length; ++i)
                    editor.Values[indices[i]] = values[i];
            }
            dst = editor.Commit();
        }

        /// <summary>
        /// Given a vector, ensure that the first <paramref name="denseCount"/> slots are explicitly
        /// represented.
        /// </summary>
        public static void DensifyFirst<T>(ref VBuffer<T> dst, int denseCount)
        {
            Contracts.Check(0 <= denseCount && denseCount <= dst.Length);
            var dstValues = dst.GetValues();
            var dstIndices = dst.GetIndices();
            if (dst.IsDense || denseCount == 0 || (dstValues.Length >= denseCount && dstIndices[denseCount - 1] == denseCount - 1))
                return;
            if (denseCount == dst.Length)
            {
                Densify(ref dst);
                return;
            }

            // Densify the first denseCount entries.
            if (dstIndices.IsEmpty)
            {
                // no previous values
                var newIndicesEditor = VBufferEditor.Create(ref dst, dst.Length, denseCount);
                Utils.FillIdentity(newIndicesEditor.Indices, denseCount);
                newIndicesEditor.Values.Clear();
                dst = newIndicesEditor.Commit();
                return;
            }
            int lim = Utils.FindIndexSorted(dstIndices, 0, dstValues.Length, denseCount);
            Contracts.Assert(lim < denseCount);
            int newLen = dstValues.Length + denseCount - lim;
            if (newLen == dst.Length)
            {
                Densify(ref dst);
                return;
            }

            var editor = VBufferEditor.Create(ref dst, dst.Length, newLen, keepOldOnResize: true);
            int sliceLength = dstValues.Length - lim;
            editor.Values.Slice(lim, sliceLength).CopyTo(editor.Values.Slice(denseCount));
            editor.Indices.Slice(lim, sliceLength).CopyTo(editor.Indices.Slice(denseCount));
            int i = lim - 1;
            for (int ii = denseCount; --ii >= 0;)
            {
                editor.Values[ii] = i >= 0 && dstIndices[i] == ii ? dstValues[i--] : default(T);
                editor.Indices[ii] = ii;
            }
            dst = editor.Commit();
        }

        /// <summary>
        /// Creates a maybe sparse copy of a VBuffer.
        /// Whether the created copy is sparse or not is determined by the proportion of non-default entries compared to the sparsity parameter.
        /// </summary>
        public static void CreateMaybeSparseCopy<T>(in VBuffer<T> src, ref VBuffer<T> dst, InPredicate<T> isDefaultPredicate, float sparsityThreshold = SparsityThreshold)
        {
            Contracts.CheckParam(0 < sparsityThreshold && sparsityThreshold < 1, nameof(sparsityThreshold));
            if (!src.IsDense || src.Length < 20)
            {
                src.CopyTo(ref dst);
                return;
            }

            int sparseCount = 0;
            var sparseCountThreshold = (int)(src.Length * sparsityThreshold);
            var srcValues = src.GetValues();
            for (int i = 0; i < src.Length; i++)
            {
                if (!isDefaultPredicate(in srcValues[i]))
                    sparseCount++;

                if (sparseCount > sparseCountThreshold)
                {
                    src.CopyTo(ref dst);
                    return;
                }
            }

            var editor = VBufferEditor.Create(ref dst, src.Length, sparseCount);
            if (sparseCount > 0)
            {
                int j = 0;
                for (int i = 0; i < src.Length; i++)
                {
                    if (!isDefaultPredicate(in srcValues[i]))
                    {
                        Contracts.Assert(j < sparseCount);
                        editor.Indices[j] = i;
                        editor.Values[j] = srcValues[i];
                        j++;
                    }
                }

                Contracts.Assert(j == sparseCount);
            }

            dst = editor.Commit();
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
        public static void ApplyWith<TSrc, TDst>(in VBuffer<TSrc> src, ref VBuffer<TDst> dst, PairManipulator<TSrc, TDst> manip)
        {
            ApplyWithCore(in src, ref dst, manip, outer: false);
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
        /// <param name="dst">Argument vector, whose elements are read in most cases. But in some
        /// cases <paramref name="dst"/> may be densified.</param>
        /// <param name="res">Result vector</param>
        /// <param name="manip">Function to apply to each pair of elements</param>
        public static void ApplyWithCopy<TSrc, TDst>(in VBuffer<TSrc> src, ref VBuffer<TDst> dst, ref VBuffer<TDst> res, PairManipulatorCopy<TSrc, TDst> manip)
        {
            ApplyWithCoreCopy(in src, ref dst, ref res, manip, outer: false);
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
        public static void ApplyWithEitherDefined<TSrc, TDst>(in VBuffer<TSrc> src, ref VBuffer<TDst> dst, PairManipulator<TSrc, TDst> manip)
        {
            ApplyWithCore(in src, ref dst, manip, outer: true);
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
        /// <param name="dst">Argument vector, whose elements are read in most cases. But in some
        /// cases <paramref name="dst"/> may be densified.</param>
        /// <param name="res">Result vector</param>
        /// <param name="manip">Function to apply to each pair of elements</param>
        public static void ApplyWithEitherDefinedCopy<TSrc, TDst>(in VBuffer<TSrc> src, ref VBuffer<TDst> dst, ref VBuffer<TDst> res, PairManipulatorCopy<TSrc, TDst> manip)
        {
            ApplyWithCoreCopy(in src, ref dst, ref res, manip, outer: true);
        }

        /// <summary>
        /// The actual implementation of <see cref="ApplyWith"/> and
        /// <see cref="ApplyWithEitherDefined{TSrc,TDst}"/>, that has internal branches on the implementation
        /// where necessary depending on whether this is an inner or outer join of the
        /// indices of <paramref name="src"/> on <paramref name="dst"/>.
        /// </summary>
        private static void ApplyWithCore<TSrc, TDst>(in VBuffer<TSrc> src, ref VBuffer<TDst> dst, PairManipulator<TSrc, TDst> manip, bool outer)
        {
            Contracts.Check(src.Length == dst.Length, "Vectors must have the same dimensionality.");
            Contracts.CheckValue(manip, nameof(manip));

            // We handle all of the permutations of the density/sparsity of src/dst through
            // special casing below. Each subcase in turn handles appropriately the treatment
            // of the "outer" parameter. There are nine, top level cases. Each case is
            // considered in this order.

            // 1. srcValues.Length == 0.
            // 2. src.Dense.
            // 3. dst.Dense.
            // 4. dstValues.Length == 0.

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

            var srcValues = src.GetValues();
            var dstValues = dst.GetValues();
            var dstIndices = dst.GetIndices();
            var editor = VBufferEditor.CreateFromBuffer(ref dst);
            if (srcValues.Length == 0)
            {
                // Major case 1, with srcValues.Length == 0.
                if (!outer)
                    return;
                if (dst.IsDense)
                {
                    for (int i = 0; i < dst.Length; i++)
                        manip(i, default(TSrc), ref editor.Values[i]);
                }
                else
                {
                    for (int i = 0; i < dstValues.Length; i++)
                        manip(dstIndices[i], default(TSrc), ref editor.Values[i]);
                }
                return;
            }

            if (src.IsDense)
            {
                // Major case 2, with src.Dense.
                if (!dst.IsDense)
                {
                    Densify(ref dst);
                    editor = VBufferEditor.CreateFromBuffer(ref dst);
                }

                // Both are now dense. Both cases of outer are covered.
                for (int i = 0; i < srcValues.Length; i++)
                    manip(i, srcValues[i], ref editor.Values[i]);
                return;
            }

            var srcIndices = src.GetIndices();
            if (dst.IsDense)
            {
                // Major case 3, with dst.Dense. Note that !src.Dense.
                if (outer)
                {
                    int sI = 0;
                    int sIndex = srcIndices[sI];
                    for (int i = 0; i < dst.Length; ++i)
                    {
                        if (i == sIndex)
                        {
                            manip(i, srcValues[sI], ref editor.Values[i]);
                            sIndex = ++sI == srcValues.Length ? src.Length : srcIndices[sI];
                        }
                        else
                            manip(i, default(TSrc), ref editor.Values[i]);
                    }
                }
                else
                {
                    for (int i = 0; i < srcValues.Length; i++)
                        manip(srcIndices[i], srcValues[i], ref editor.Values[srcIndices[i]]);
                }
                return;
            }

            if (dstValues.Length == 0)
            {
                // Major case 4, with dst empty. Note that !src.Dense.
                // Neither is dense, and dst is empty. Both cases of outer are covered.
                editor = VBufferEditor.Create(ref dst,
                    src.Length,
                    srcValues.Length,
                    maxValuesCapacity: src.Length);
                editor.Values.Clear();
                for (int i = 0; i < srcValues.Length; i++)
                    manip(editor.Indices[i] = srcIndices[i], srcValues[i], ref editor.Values[i]);
                dst = editor.Commit();
                return;
            }

            // Beyond this point, we can assume both a and b are sparse with positive count.
            int dI = 0;
            int newCount = dstValues.Length;
            // Try to find each src index in dst indices, counting how many more we'll add.
            for (int sI = 0; sI < srcValues.Length; sI++)
            {
                int sIndex = srcIndices[sI];
                while (dI < dstValues.Length && dstIndices[dI] < sIndex)
                    dI++;
                if (dI == dstValues.Length)
                {
                    newCount += srcValues.Length - sI;
                    break;
                }
                if (dstIndices[dI] == sIndex)
                    dI++;
                else
                    newCount++;
            }
            Contracts.Assert(newCount > 0);
            Contracts.Assert(0 < srcValues.Length && srcValues.Length <= newCount);
            Contracts.Assert(0 < dstValues.Length && dstValues.Length <= newCount);

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
                ApplyWithCore(in src, ref dst, manip, outer);
                return;
            }

            if (newCount != srcValues.Length && newCount != dstValues.Length)
            {
                // Major case 6, neither set of indices is a subset of the other.
                // This subcase used to fall through to another subcase, but this
                // proved to be inefficient so we go to the little bit of extra work
                // to handle it here.

                editor = VBufferEditor.Create(ref dst,
                    src.Length,
                    newCount,
                    maxValuesCapacity: dst.Length);
                var indices = editor.Indices;
                var values = editor.Values;
                int sI = srcValues.Length - 1;
                dI = dstValues.Length - 1;
                int sIndex = srcIndices[sI];
                int dIndex = dstIndices[dI];

                // Go from the end, so that even if we're writing over dst's vectors in
                // place, we do not corrupt the data as we are reorganizing it.
                for (int i = newCount; --i >= 0;)
                {
                    if (sIndex < dIndex)
                    {
                        indices[i] = dIndex;
                        values[i] = dstValues[dI];
                        if (outer)
                            manip(dIndex, default(TSrc), ref values[i]);
                        dIndex = --dI >= 0 ? dstIndices[dI] : -1;
                    }
                    else if (sIndex > dIndex)
                    {
                        indices[i] = sIndex;
                        values[i] = default(TDst);
                        manip(sIndex, srcValues[sI], ref values[i]);
                        sIndex = --sI >= 0 ? srcIndices[sI] : -1;
                    }
                    else
                    {
                        // We should not have run past the beginning, due to invariants.
                        Contracts.Assert(sIndex >= 0);
                        Contracts.Assert(sIndex == dIndex);
                        indices[i] = dIndex;
                        values[i] = dstValues[dI];
                        manip(sIndex, srcValues[sI], ref values[i]);
                        sIndex = --sI >= 0 ? srcIndices[sI] : -1;
                        dIndex = --dI >= 0 ? dstIndices[dI] : -1;
                    }
                }
                dst = editor.Commit();
                return;
            }

            if (newCount == dstValues.Length)
            {
                if (newCount == srcValues.Length)
                {
                    // Major case 7, the set of indices is the same for src and dst.
                    Contracts.Assert(srcValues.Length == dstValues.Length);
                    for (int i = 0; i < srcValues.Length; i++)
                    {
                        Contracts.Assert(srcIndices[i] == dstIndices[i]);
                        manip(srcIndices[i], srcValues[i], ref editor.Values[i]);
                    }
                    return;
                }
                // Major case 8, the indices of src must be a subset of dst's indices.
                Contracts.Assert(newCount > srcValues.Length);
                dI = 0;
                if (outer)
                {
                    int sI = 0;
                    int sIndex = srcIndices[sI];
                    for (int i = 0; i < dstValues.Length; ++i)
                    {
                        if (dstIndices[i] == sIndex)
                        {
                            manip(sIndex, srcValues[sI], ref editor.Values[i]);
                            sIndex = ++sI == srcValues.Length ? src.Length : srcIndices[sI];
                        }
                        else
                            manip(dstIndices[i], default(TSrc), ref editor.Values[i]);
                    }
                }
                else
                {
                    for (int sI = 0; sI < srcValues.Length; sI++)
                    {
                        int sIndex = srcIndices[sI];
                        while (dstIndices[dI] < sIndex)
                            dI++;
                        Contracts.Assert(dstIndices[dI] == sIndex);
                        manip(sIndex, srcValues[sI], ref editor.Values[dI++]);
                    }
                }
                return;
            }

            if (newCount == srcValues.Length)
            {
                // Major case 9, the indices of dst must be a subset of src's indices. Both cases of outer are covered.

                // First do a "quasi" densification of dst, by making the indices
                // of dst correspond to those in src.
                editor = VBufferEditor.Create(ref dst, newCount, dstValues.Length);
                int sI = 0;
                for (dI = 0; dI < dstValues.Length; ++dI)
                {
                    int bIndex = dstIndices[dI];
                    while (srcIndices[sI] < bIndex)
                        sI++;
                    Contracts.Assert(srcIndices[sI] == bIndex);
                    editor.Indices[dI] = sI++;
                }
                dst = editor.Commit();
                Densify(ref dst);

                editor = VBufferEditor.Create(ref dst,
                    src.Length,
                    newCount,
                    maxValuesCapacity: src.Length);
                srcIndices.CopyTo(editor.Indices);
                for (sI = 0; sI < srcValues.Length; sI++)
                    manip(srcIndices[sI], srcValues[sI], ref editor.Values[sI]);
                dst = editor.Commit();
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
        private static void ApplyWithCoreCopy<TSrc, TDst>(in VBuffer<TSrc> src, ref VBuffer<TDst> dst, ref VBuffer<TDst> res, PairManipulatorCopy<TSrc, TDst> manip, bool outer)
        {
            Contracts.Check(src.Length == dst.Length, "Vectors must have the same dimensionality.");
            Contracts.CheckValue(manip, nameof(manip));

            int length = src.Length;

            var srcValues = src.GetValues();
            var dstValues = dst.GetValues();

            if (dstValues.Length == 0)
            {
                if (srcValues.Length == 0)
                {
                    Resize(ref res, length, 0);
                }
                else if (src.IsDense)
                {
                    Contracts.Assert(srcValues.Length == src.Length);
                    var editor = VBufferEditor.Create(ref res, length);
                    for (int i = 0; i < length; i++)
                        manip(i, srcValues[i], default(TDst), ref editor.Values[i]);
                    res = editor.Commit();
                }
                else
                {
                    // src is non-empty sparse.
                    int count = srcValues.Length;
                    Contracts.Assert(0 < count && count < length);
                    var editor = VBufferEditor.Create(ref res, length, count);
                    var srcIndices = src.GetIndices();
                    srcIndices.CopyTo(editor.Indices);
                    for (int ii = 0; ii < count; ii++)
                    {
                        int i = srcIndices[ii];
                        editor.Indices[ii] = i;
                        manip(i, srcValues[ii], default(TDst), ref editor.Values[ii]);
                    }
                    res = editor.Commit();
                }
            }
            else if (dst.IsDense)
            {
                var editor = VBufferEditor.Create(ref res, length);
                if (srcValues.Length == 0)
                {
                    if (outer)
                    {
                        // Apply manip to all slots, as all slots of dst are defined.
                        for (int j = 0; j < length; j++)
                            manip(j, default(TSrc), dstValues[j], ref editor.Values[j]);
                    }
                    else
                    {
                        // Copy only. No slot of src is defined.
                        for (int j = 0; j < length; j++)
                            editor.Values[j] = dstValues[j];
                    }
                    res = editor.Commit();
                }
                else if (src.IsDense)
                {
                    Contracts.Assert(srcValues.Length == src.Length);
                    for (int i = 0; i < length; i++)
                        manip(i, srcValues[i], dstValues[i], ref editor.Values[i]);
                    res = editor.Commit();
                }
                else
                {
                    // src is sparse and non-empty.
                    int count = srcValues.Length;
                    Contracts.Assert(0 < count && count < length);

                    int ii = 0;
                    var srcIndices = src.GetIndices();
                    int i = srcIndices[ii];
                    if (outer)
                    {
                        // All slots of dst are defined. Always apply manip.
                        for (int j = 0; j < length; j++)
                        {
                            if (j == i)
                            {
                                manip(j, srcValues[ii], dstValues[j], ref editor.Values[j]);
                                i = ++ii == count ? length : srcIndices[ii];
                            }
                            else
                                manip(j, default(TSrc), dstValues[j], ref editor.Values[j]);
                        }
                    }
                    else
                    {
                        // Only apply manip for those slots where src is defined. Otherwise just copy.
                        for (int j = 0; j < length; j++)
                        {
                            if (j == i)
                            {
                                manip(j, srcValues[ii], dstValues[j], ref editor.Values[j]);
                                i = ++ii == count ? length : srcIndices[ii];
                            }
                            else
                                editor.Values[j] = dstValues[j];
                        }
                    }
                    res = editor.Commit();
                }
            }
            else
            {
                // dst is non-empty sparse
                int dstCount = dstValues.Length;
                var dstIndices = dst.GetIndices();
                Contracts.Assert(dstCount > 0);
                if (srcValues.Length == 0)
                {
                    var editor = VBufferEditor.Create(ref res, length, dstCount);
                    if (outer)
                    {
                        for (int jj = 0; jj < dstCount; jj++)
                        {
                            int j = dstIndices[jj];
                            editor.Indices[jj] = j;
                            manip(j, default(TSrc), dstValues[jj], ref editor.Values[jj]);
                        }
                    }
                    else
                    {
                        for (int jj = 0; jj < dstCount; jj++)
                        {
                            editor.Indices[jj] = dstIndices[jj];
                            editor.Values[jj] = dstValues[jj];
                        }
                    }
                    res = editor.Commit();
                }
                else if (src.IsDense)
                {
                    // res will be dense.
                    var editor = VBufferEditor.Create(ref res, length);
                    int jj = 0;
                    int j = dstIndices[jj];
                    for (int i = 0; i < length; i++)
                    {
                        if (i == j)
                        {
                            manip(i, srcValues[i], dstValues[jj], ref editor.Values[i]);
                            j = ++jj == dstCount ? length : dstIndices[jj];
                        }
                        else
                            manip(i, srcValues[i], default(TDst), ref editor.Values[i]);
                    }
                    res = editor.Commit();
                }
                else
                {
                    // Both src and dst are non-empty sparse.
                    Contracts.Assert(srcValues.Length > 0);

                    // Find the count of result, which is the size of the union of the indices set of src and dst.
                    int resCount = dstCount;
                    var srcIndices = src.GetIndices();
                    for (int ii = 0, jj = 0; ii < srcValues.Length; ii++)
                    {
                        int i = srcIndices[ii];
                        while (jj < dstValues.Length && dstIndices[jj] < i)
                            jj++;
                        if (jj == dstValues.Length)
                        {
                            resCount += srcValues.Length - ii;
                            break;
                        }
                        if (dstIndices[jj] == i)
                            jj++;
                        else
                            resCount++;
                    }

                    Contracts.Assert(0 < resCount && resCount <= length);
                    Contracts.Assert(resCount <= srcValues.Length + dstCount);
                    Contracts.Assert(srcValues.Length <= resCount);
                    Contracts.Assert(dstCount <= resCount);

                    if (resCount == length)
                    {
                        // result will become dense.
                        // This is unnecessary -- falling through to the sparse code will
                        // actually handle this case just fine -- but it is more efficient.
                        Densify(ref dst);
                        ApplyWithCoreCopy(in src, ref dst, ref res, manip, outer);
                    }
                    else
                    {
                        var editor = VBufferEditor.Create(ref res, length, resCount);

                        int ii = 0;
                        int i = srcIndices[ii];
                        int jj = 0;
                        int j = dstIndices[jj];

                        for (int kk = 0; kk < resCount; kk++)
                        {
                            Contracts.Assert(i < length || j < length);
                            if (i == j)
                            {
                                // Slot (i == j) both defined in src and dst. Apply manip.
                                editor.Indices[kk] = i;
                                manip(i, srcValues[ii], dstValues[jj], ref editor.Values[kk]);
                                i = ++ii == srcValues.Length ? length : srcIndices[ii];
                                j = ++jj == dstCount ? length : dstIndices[jj];
                            }
                            else if (i < j)
                            {
                                // Slot i defined only in src, but not in dst. Apply manip.
                                editor.Indices[kk] = i;
                                manip(i, srcValues[ii], default(TDst), ref editor.Values[kk]);
                                i = ++ii == srcValues.Length ? length : srcIndices[ii];
                            }
                            else
                            {
                                // Slot j defined only in dst, but not in src. Apply manip if outer.
                                // Otherwise just copy.
                                editor.Indices[kk] = j;
                                // REVIEW: Should we move checking of outer outside the loop?
                                if (outer)
                                    manip(j, default(TSrc), dstValues[jj], ref editor.Values[kk]);
                                else
                                    editor.Values[kk] = dstValues[jj];
                                j = ++jj == dstCount ? length : dstIndices[jj];
                            }
                        }

                        Contracts.Assert(ii == srcValues.Length && jj == dstCount);
                        Contracts.Assert(i == length && j == length);
                        res = editor.Commit();
                    }
                }
            }
        }

        /// <summary>
        /// Applies a function to explicitly defined elements in a vector <paramref name="src"/>,
        /// storing the result in <paramref name="dst"/>, overwriting any of its existing contents.
        /// The contents of <paramref name="dst"/> do not affect calculation. If you instead wish
        /// to calculate a function that reads and writes <paramref name="dst"/>, see
        /// <see cref="ApplyWith{TSrc,TDst}"/> and <see cref="ApplyWithEitherDefined{TSrc,TDst}"/>. Post-operation,
        /// <paramref name="dst"/> will be dense iff <paramref name="src"/> is dense.
        /// </summary>
        /// <seealso cref="ApplyWith{TSrc,TDst}"/>
        /// <seealso cref="ApplyWithEitherDefined{TSrc,TDst}"/>
        public static void ApplyIntoEitherDefined<TSrc, TDst>(in VBuffer<TSrc> src, ref VBuffer<TDst> dst, Func<int, TSrc, TDst> func)
        {
            Contracts.CheckValue(func, nameof(func));

            var srcValues = src.GetValues();

            // REVIEW: The analogous WritableVector method insisted on
            // equal lengths, but I don't care here.
            if (srcValues.Length == 0)
            {
                Resize(ref dst, src.Length, 0);
                return;
            }
            var editor = VBufferEditor.Create(ref dst,
                src.Length,
                srcValues.Length,
                maxValuesCapacity: src.Length);
            Span<TDst> values = editor.Values;
            if (src.IsDense)
            {
                for (int i = 0; i < src.Length; ++i)
                    values[i] = func(i, srcValues[i]);
            }
            else
            {
                Span<int> indices = editor.Indices;
                var srcIndices = src.GetIndices();
                srcIndices.CopyTo(indices);
                for (int i = 0; i < srcValues.Length; ++i)
                    values[i] = func(srcIndices[i], srcValues[i]);
            }
            dst = editor.Commit();
        }

        /// <summary>
        /// Applies a function <paramref name="func"/> to two vectors, storing the result in
        /// <paramref name="dst"/>, whose existing contents are discarded and overwritten. The
        /// function is called for every index value that appears in either <paramref name="a"/>
        /// or <paramref name="b"/>. If either of the two inputs is dense, the output will
        /// necessarily be dense. Otherwise, if both are sparse, the output will be sparse iff
        /// there is any slot that is not explicitly represented in either vector.
        /// </summary>
        public static void ApplyInto<TSrc1, TSrc2, TDst>(in VBuffer<TSrc1> a, in VBuffer<TSrc2> b, ref VBuffer<TDst> dst, Func<int, TSrc1, TSrc2, TDst> func)
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

            var aValues = a.GetValues();
            var bValues = b.GetValues();
            if (aValues.Length == 0 && bValues.Length == 0)
            {
                // Case 1. Output will be empty.
                Resize(ref dst, a.Length, 0);
                return;
            }

            int aI = 0;
            int bI = 0;
            ReadOnlySpan<int> aIndices;
            ReadOnlySpan<int> bIndices;
            VBufferEditor<TDst> editor;
            if (a.IsDense || b.IsDense)
            {
                // Case 2. One of the two inputs is dense. The output will be dense.
                editor = VBufferEditor.Create(ref dst, a.Length);
                if (!a.IsDense)
                {
                    // a is sparse, b is dense
                    aIndices = a.GetIndices();
                    for (int i = 0; i < b.Length; i++)
                    {
                        TSrc1 aVal = (aI < aIndices.Length && i == aIndices[aI]) ? aValues[aI++] : default(TSrc1);
                        editor.Values[i] = func(i, aVal, bValues[i]);
                    }
                }
                else if (!b.IsDense)
                {
                    // b is sparse, a is dense
                    bIndices = b.GetIndices();
                    for (int i = 0; i < a.Length; i++)
                    {
                        TSrc2 bVal = (bI < bIndices.Length && i == bIndices[bI]) ? bValues[bI++] : default(TSrc2);
                        editor.Values[i] = func(i, aValues[i], bVal);
                    }
                }
                else
                {
                    // both dense
                    for (int i = 0; i < a.Length; i++)
                        editor.Values[i] = func(i, aValues[i], bValues[i]);
                }
                dst = editor.Commit();
                return;
            }

            // a, b both sparse.
            int newCount = 0;
            aIndices = a.GetIndices();
            bIndices = b.GetIndices();
            while (aI < aIndices.Length && bI < bIndices.Length)
            {
                int aCompB = aIndices[aI] - bIndices[bI];
                if (aCompB <= 0) // a is no larger than b.
                    aI++;
                if (aCompB >= 0) // b is no larger than a.
                    bI++;
                newCount++;
            }

            if (aI < aIndices.Length)
                newCount += aIndices.Length - aI;
            if (bI < bIndices.Length)
                newCount += bIndices.Length - bI;

            // REVIEW: Worth optimizing the newCount == a.Length case?
            // Probably not...

            editor = VBufferEditor.Create(ref dst, a.Length, newCount, requireIndicesOnDense: true);
            Span<int> indices = editor.Indices;

            if (newCount == bValues.Length)
            {
                if (newCount == aValues.Length)
                {
                    // Case 3, a and b actually have the same indices!
                    aIndices.CopyTo(indices);
                    for (aI = 0; aI < aValues.Length; aI++)
                    {
                        Contracts.Assert(aIndices[aI] == bIndices[aI]);
                        editor.Values[aI] = func(aIndices[aI], aValues[aI], bValues[aI]);
                    }
                }
                else
                {
                    // Case 4, a's indices are a subset of b's.
                    bIndices.CopyTo(indices);
                    aI = 0;
                    for (bI = 0; aI < aValues.Length && bI < bValues.Length; bI++)
                    {
                        Contracts.Assert(aIndices[aI] >= bIndices[bI]);
                        TSrc1 aVal = aIndices[aI] == bIndices[bI] ? aValues[aI++] : default(TSrc1);
                        editor.Values[bI] = func(bIndices[bI], aVal, bValues[bI]);
                    }
                    for (; bI < bValues.Length; bI++)
                        editor.Values[bI] = func(bIndices[bI], default(TSrc1), bValues[bI]);
                }
            }
            else if (newCount == aValues.Length)
            {
                // Case 5, b's indices are a subset of a's.
                aIndices.CopyTo(indices);
                bI = 0;
                for (aI = 0; bI < bValues.Length && aI < aValues.Length; aI++)
                {
                    Contracts.Assert(bIndices[bI] >= aIndices[aI]);
                    TSrc2 bVal = aIndices[aI] == bIndices[bI] ? bValues[bI++] : default(TSrc2);
                    editor.Values[aI] = func(aIndices[aI], aValues[aI], bVal);
                }
                for (; aI < aValues.Length; aI++)
                    editor.Values[aI] = func(aIndices[aI], aValues[aI], default(TSrc2));
            }
            else
            {
                // Case 6, neither a nor b's indices are a subset of the other.
                int newI = aI = bI = 0;
                TSrc1 aVal = default(TSrc1);
                TSrc2 bVal = default(TSrc2);
                while (aI < aIndices.Length && bI < bIndices.Length)
                {
                    int aCompB = aIndices[aI] - bIndices[bI];
                    int index = 0;

                    if (aCompB < 0)
                    {
                        index = aIndices[aI];
                        aVal = aValues[aI++];
                        bVal = default(TSrc2);
                    }
                    else if (aCompB > 0)
                    {
                        index = bIndices[bI];
                        aVal = default(TSrc1);
                        bVal = bValues[bI++];
                    }
                    else
                    {
                        index = aIndices[aI];
                        Contracts.Assert(index == bIndices[bI]);
                        aVal = aValues[aI++];
                        bVal = bValues[bI++];
                    }
                    editor.Values[newI] = func(index, aVal, bVal);
                    indices[newI++] = index;
                }

                for (; aI < aIndices.Length; aI++)
                {
                    int index = aIndices[aI];
                    editor.Values[newI] = func(index, aValues[aI], default(TSrc2));
                    indices[newI++] = index;
                }

                for (; bI < bIndices.Length; bI++)
                {
                    int index = bIndices[bI];
                    editor.Values[newI] = func(index, default(TSrc1), bValues[bI]);
                    indices[newI++] = index;
                }
            }
            dst = editor.Commit();
        }

        /// <summary>
        /// Copy from a source list to the given VBuffer destination.
        /// </summary>
        public static void Copy<T>(List<T> src, ref VBuffer<T> dst, int length)
        {
            Contracts.CheckParam(0 <= length && length <= Utils.Size(src), nameof(length));
            var editor = VBufferEditor.Create(ref dst, length);
            if (length > 0)
            {
                // List<T>.CopyTo should have an overload for Span - https://github.com/dotnet/corefx/issues/33006
                for (int i = 0; i < length; i++)
                {
                    editor.Values[i] = src[i];
                }
            }
            dst = editor.Commit();
        }

        /// <summary>
        /// Updates the logical length and number of physical values to be represented in
        /// <paramref name="dst"/>, while preserving the underlying buffers.
        /// </summary>
        public static void Resize<T>(ref VBuffer<T> dst, int newLogicalLength, int? valuesCount = null)
        {
            dst = VBufferEditor.Create(ref dst, newLogicalLength, valuesCount)
                .Commit();
        }
    }
}
