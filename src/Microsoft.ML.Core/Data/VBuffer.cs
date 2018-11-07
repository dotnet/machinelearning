// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// A buffer that supports both dense and sparse representations. This is the
    /// representation type for all VectorType instances. When an instance of this
    /// is passed to a row cursor getter, the callee is free to take ownership of
    /// and re-use the arrays (Values and Indices).
    /// </summary>
    public readonly struct VBuffer<T>
    {
        /// <summary>
        /// The logical length of the buffer.
        /// </summary>
        public readonly int Length;

        /// <summary>
        /// The number of items explicitly represented. This is == Length when the representation
        /// is dense and &lt; Length when sparse.
        /// </summary>
        public readonly int Count;

        /// <summary>
        /// The values. Only the first Count of these are valid.
        /// </summary>
        public readonly T[] Values;

        /// <summary>
        /// The indices. For a dense representation, this array is not used. For a sparse representation
        /// it is parallel to values and specifies the logical indices for the corresponding values.
        /// </summary>
        public readonly int[] Indices;

        /// <summary>
        /// The explicitly represented values.
        /// </summary>
        public ReadOnlySpan<T> GetValues() => Values.AsSpan(0, Count);

        /// <summary>
        /// The indices. For a dense representation, this array is not used. For a sparse representation
        /// it is parallel to values and specifies the logical indices for the corresponding values.
        /// </summary>
        /// <remarks>
        /// For example, if GetIndices() returns [3, 5] and GetValues() produces [98, 76], this VBuffer
        /// stands for a vector with:
        ///  - non-zeros values 98 and 76 respectively at the 4th and 6th coordinates
        ///  - zeros at all other coordinates
        /// </remarks>
        public ReadOnlySpan<int> GetIndices() => IsDense ? default : Indices.AsSpan(0, Count);

        /// <summary>
        /// Equivalent to Count == Length.
        /// </summary>
        public bool IsDense
        {
            get
            {
                Contracts.Assert(Count <= Length);
                return Count == Length;
            }
        }

        /// <summary>
        /// Construct a dense representation with unused Indices array.
        /// </summary>
        public VBuffer(int length, T[] values, int[] indices = null)
        {
            Contracts.CheckParam(length >= 0, nameof(length));
            Contracts.CheckParam(Utils.Size(values) >= length, nameof(values));
            Contracts.CheckValueOrNull(indices);

            Length = length;
            Count = length;
            Values = values;
            Indices = indices;
        }

        /// <summary>
        /// Construct a possibly sparse representation.
        /// </summary>
        public VBuffer(int length, int count, T[] values, int[] indices)
        {
            Contracts.CheckParam(length >= 0, nameof(length));
            Contracts.CheckParam(0 <= count && count <= length, nameof(count));
            Contracts.CheckParam(Utils.Size(values) >= count, nameof(values));
            Contracts.CheckParam(count == length || Utils.Size(indices) >= count, nameof(indices));

#if DEBUG // REVIEW: This validation should really use "Checks" and be in release code, but it is not cheap.
            if (0 < count && count < length)
            {
                int cur = indices[0];
                int res = cur;
                for (int i = 1; i < count; ++i)
                {
                    int tmp = cur;
                    cur = indices[i];
                    res |= cur - tmp - 1; // Make sure the gap is not negative.
                    res |= cur;
                }
                Contracts.Assert(res >= 0 && cur < length);
            }
#endif

            Length = length;
            Count = count;
            Values = values;
            Indices = indices;
        }

        /// <summary>
        /// Copy from this buffer to the given destination, forcing a dense representation.
        /// </summary>
        public void CopyToDense(ref VBuffer<T> dst)
        {
            var values = dst.Values;
            if (Utils.Size(values) < Length)
                values = new T[Length];

            if (!IsDense)
                CopyTo(values);
            else if (Length > 0)
                Array.Copy(Values, values, Length);
            dst = new VBuffer<T>(Length, values, dst.Indices);
        }

        /// <summary>
        /// Copy from this buffer to the given destination.
        /// </summary>
        public void CopyTo(ref VBuffer<T> dst)
        {
            var values = dst.Values;
            var indices = dst.Indices;
            if (IsDense)
            {
                if (Length > 0)
                {
                    if (Utils.Size(values) < Length)
                        values = new T[Length];
                    Array.Copy(Values, values, Length);
                }
                dst = new VBuffer<T>(Length, values, indices);
                Contracts.Assert(dst.IsDense);
            }
            else
            {
                if (Count > 0)
                {
                    if (Utils.Size(values) < Count)
                        values = new T[Count];
                    if (Utils.Size(indices) < Count)
                        indices = new int[Count];
                    Array.Copy(Values, values, Count);
                    Array.Copy(Indices, indices, Count);
                }
                dst = new VBuffer<T>(Length, Count, values, indices);
            }
        }

        /// <summary>
        /// Copy a range of values from this buffer to the given destination.
        /// </summary>
        public void CopyTo(ref VBuffer<T> dst, int srcMin, int length)
        {
            Contracts.Check(0 <= srcMin && srcMin <= Length, "srcMin");
            Contracts.Check(0 <= length && srcMin <= Length - length, "length");
            var values = dst.Values;
            var indices = dst.Indices;
            if (IsDense)
            {
                if (length > 0)
                {
                    if (Utils.Size(values) < length)
                        values = new T[length];
                    Array.Copy(Values, srcMin, values, 0, length);
                }
                dst = new VBuffer<T>(length, values, indices);
                Contracts.Assert(dst.IsDense);
            }
            else
            {
                int copyCount = 0;
                if (Count > 0)
                {
                    int copyMin = Indices.FindIndexSorted(0, Count, srcMin);
                    int copyLim = Indices.FindIndexSorted(copyMin, Count, srcMin + length);
                    Contracts.Assert(copyMin <= copyLim);
                    copyCount = copyLim - copyMin;
                    if (copyCount > 0)
                    {
                        if (Utils.Size(values) < copyCount)
                            values = new T[copyCount];
                        Array.Copy(Values, copyMin, values, 0, copyCount);
                        if (copyCount < length)
                        {
                            if (Utils.Size(indices) < copyCount)
                                indices = new int[copyCount];
                            for (int i = 0; i < copyCount; ++i)
                                indices[i] = Indices[i + copyMin] - srcMin;
                        }
                    }
                }
                dst = new VBuffer<T>(length, copyCount, values, indices);
            }
        }

        /// <summary>
        /// Copy from this buffer to the given destination, making sure to explicitly include the
        /// first count indices in indicesInclude. Note that indicesInclude should be sorted
        /// with each index less than this.Length. Note that this can make the destination be
        /// dense even if "this" is sparse.
        /// </summary>
        public void CopyTo(ref VBuffer<T> dst, int[] indicesInclude, int count)
        {
            Contracts.CheckParam(count >= 0, nameof(count));
            Contracts.CheckParam(Utils.Size(indicesInclude) >= count, nameof(indicesInclude));
            Contracts.CheckParam(Utils.Size(indicesInclude) <= Length, nameof(indicesInclude));

            // REVIEW: Ideally we should Check that indicesInclude is sorted and in range. Would that
            // check be too expensive?
#if DEBUG
            int prev = -1;
            for (int i = 0; i < count; i++)
            {
                Contracts.Assert(prev < indicesInclude[i]);
                prev = indicesInclude[i];
            }
            Contracts.Assert(prev < Length);
#endif

            if (IsDense || count == 0)
            {
                CopyTo(ref dst);
                return;
            }

            if (count >= Length / 2 || Count >= Length / 2)
            {
                CopyToDense(ref dst);
                return;
            }

            var indices = dst.Indices;
            var values = dst.Values;
            if (Count == 0)
            {
                // No values in "this".
                if (Utils.Size(indices) < count)
                    indices = new int[count];
                Array.Copy(indicesInclude, indices, count);
                if (Utils.Size(values) < count)
                    values = new T[count];
                else
                    Array.Clear(values, 0, count);
                dst = new VBuffer<T>(Length, count, values, indices);
                return;
            }

            int size = 0;
            int max = count + Count;
            Contracts.Assert(max < Length);
            int ii1;
            int ii2;
            if (max >= Length / 2 || Utils.Size(values) < max || Utils.Size(indices) < max)
            {
                // Compute the needed size.
                ii1 = 0;
                ii2 = 0;
                for (; ; )
                {
                    Contracts.Assert(ii1 < Count);
                    Contracts.Assert(ii2 < count);
                    size++;
                    int diff = Indices[ii1] - indicesInclude[ii2];
                    if (diff == 0)
                    {
                        ii1++;
                        ii2++;
                        if (ii1 >= Count)
                        {
                            size += count - ii2;
                            break;
                        }
                        if (ii2 >= count)
                        {
                            size += Count - ii1;
                            break;
                        }
                    }
                    else if (diff < 0)
                    {
                        if (++ii1 >= Count)
                        {
                            size += count - ii2;
                            break;
                        }
                    }
                    else
                    {
                        if (++ii2 >= count)
                        {
                            size += Count - ii1;
                            break;
                        }
                    }
                }
                Contracts.Assert(size >= count && size >= Count);

                if (size == Count)
                {
                    CopyTo(ref dst);
                    return;
                }

                if (size >= Length / 2)
                {
                    CopyToDense(ref dst);
                    return;
                }

                if (Utils.Size(values) < size)
                    values = new T[size];
                if (Utils.Size(indices) < size)
                    indices = new int[size];
                max = size;
            }

            int ii = 0;
            ii1 = 0;
            ii2 = 0;
            for (; ; )
            {
                Contracts.Assert(ii < max);
                Contracts.Assert(ii1 < Count);
                Contracts.Assert(ii2 < count);
                int i1 = Indices[ii1];
                int i2 = indicesInclude[ii2];
                if (i1 <= i2)
                {
                    indices[ii] = i1;
                    values[ii] = Values[ii1];
                    ii++;
                    if (i1 == i2)
                        ii2++;
                    if (++ii1 >= Count)
                    {
                        if (ii2 >= count)
                            break;
                        Array.Clear(values, ii, count - ii2);
                        Array.Copy(indicesInclude, ii2, indices, ii, count - ii2);
                        ii += count - ii2;
                        break;
                    }
                    if (ii2 >= count)
                    {
                        Array.Copy(Values, ii1, values, ii, Count - ii1);
                        Array.Copy(Indices, ii1, indices, ii, Count - ii1);
                        ii += Count - ii1;
                        break;
                    }
                }
                else
                {
                    indices[ii] = i2;
                    values[ii] = default(T);
                    ii++;
                    if (++ii2 >= count)
                    {
                        Array.Copy(Values, ii1, values, ii, Count - ii1);
                        Array.Copy(Indices, ii1, indices, ii, Count - ii1);
                        ii += Count - ii1;
                        break;
                    }
                }
            }
            Contracts.Assert(size == ii || size == 0);

            dst = new VBuffer<T>(Length, ii, values, indices);
        }

        /// <summary>
        /// Copy from this buffer to the given destination array. This "densifies".
        /// </summary>
        public void CopyTo(T[] dst)
        {
            CopyTo(dst, 0);
        }

        public void CopyTo(T[] dst, int ivDst, T defaultValue = default(T))
        {
            Contracts.CheckParam(0 <= ivDst && ivDst <= Utils.Size(dst) - Length, nameof(dst), "dst is not large enough");

            if (Length == 0)
                return;
            if (IsDense)
            {
                Array.Copy(Values, 0, dst, ivDst, Length);
                return;
            }

            if (Count == 0)
            {
                Array.Clear(dst, ivDst, Length);
                return;
            }

            int iv = 0;
            for (int islot = 0; islot < Count; islot++)
            {
                int slot = Indices[islot];
                Contracts.Assert(slot >= iv);
                while (iv < slot)
                    dst[ivDst + iv++] = defaultValue;
                Contracts.Assert(iv == slot);
                dst[ivDst + iv++] = Values[islot];
            }
            while (iv < Length)
                dst[ivDst + iv++] = defaultValue;
        }

        /// <summary>
        /// Copy from a section of a source array to the given destination.
        /// </summary>
        public static void Copy(T[] src, int srcIndex, ref VBuffer<T> dst, int length)
        {
            Contracts.CheckParam(0 <= length && length <= Utils.Size(src), nameof(length));
            Contracts.CheckParam(0 <= srcIndex && srcIndex <= Utils.Size(src) - length, nameof(srcIndex));
            var values = dst.Values;
            if (length > 0)
            {
                if (Utils.Size(values) < length)
                    values = new T[length];
                Array.Copy(src, srcIndex, values, 0, length);
            }
            dst = new VBuffer<T>(length, values, dst.Indices);
        }

        public IEnumerable<KeyValuePair<int, T>> Items(bool all = false)
        {
            return VBufferUtils.Items(Values, Indices, Length, Count, all);
        }

        public IEnumerable<T> DenseValues()
        {
            return VBufferUtils.DenseValues(Values, Indices, Length, Count);
        }

        public void GetItemOrDefault(int slot, ref T dst)
        {
            Contracts.CheckParam(0 <= slot && slot < Length, nameof(slot));

            int index;
            if (IsDense)
                dst = Values[slot];
            else if (Count > 0 && Indices.TryFindIndexSorted(0, Count, slot, out index))
                dst = Values[index];
            else
                dst = default(T);
        }

        public T GetItemOrDefault(int slot)
        {
            Contracts.CheckParam(0 <= slot && slot < Length, nameof(slot));

            int index;
            if (IsDense)
                return Values[slot];
            if (Count > 0 && Indices.TryFindIndexSorted(0, Count, slot, out index))
                return Values[index];
            return default(T);
        }

        public override string ToString()
            => IsDense ? $"Dense vector of size {Length}" : $"Sparse vector of size {Length}, {Count} explicit values";
    }
}
