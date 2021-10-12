// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.ML.Internal.DataView;
using Microsoft.ML.Internal.Utilities;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// A buffer that supports both dense and sparse representations. This is the representation type for all
    /// <see cref="VectorDataViewType"/> instances. The explicitly defined values of this vector are exposed through
    /// <see cref="GetValues"/> and, if not dense, <see cref="GetIndices"/>.
    /// </summary>
    /// <remarks>
    /// This structure is by itself immutable, but to enable buffer editing including re-use of the internal buffers,
    /// a mutable variant <see cref="VBufferEditor{T}"/> can be accessed through <see cref="VBuffer{T}"/>.
    ///
    /// Throughout the code, we make the assumption that a sparse <see cref="VBuffer{T}"/> is logically equivalent to
    /// a dense <see cref="VBuffer{T}"/> with the default value for <typeparamref name="T"/> filling in the default values.
    /// </remarks>
    /// <typeparam name="T">The type of the vector. There are no compile-time restrictions on what this could be, but
    /// this code and practically all code that uses <see cref="VBuffer{T}"/> makes the assumption that an assignment of
    /// a value is sufficient to make a completely independent copy of it. So, for example, this means that a buffer of
    /// buffers is not possible. But, things like <see cref="int"/>, <see cref="float"/>, and <see
    /// cref="ReadOnlyMemory{Char}"/>, are totally fine.</typeparam>
    public readonly struct VBuffer<T>
    {
        /// <summary>
        /// The internal re-usable array of values.
        /// </summary>
        private readonly T[] _values;

        /// <summary>
        /// The internal re-usable array of indices.
        /// </summary>
        private readonly int[] _indices;

        /// <summary>
        /// The number of items explicitly represented. This equals <see cref="Length"/> when the representation
        /// is dense and less than <see cref="Length"/> when sparse.
        /// </summary>
        private readonly int _count;

        /// <summary>
        /// The logical length of the buffer.
        /// </summary>
        /// <remarks>
        /// Note that if this vector <see cref="IsDense"/>, then this will be the same as the <see cref="ReadOnlySpan{T}.Length"/>
        /// as returned from <see cref="GetValues"/>, since all values are explicitly represented in a dense representation. If
        /// this is a sparse representation, then that <see cref="ReadOnlySpan{T}.Length"/> will be somewhat shorter, as this
        /// field contains the number of both explicit and implicit entries.
        /// </remarks>
        public readonly int Length;

        /// <summary>
        /// The explicitly represented values. When this <see cref="IsDense"/>, the <see cref="ReadOnlySpan{T}.Length"/>
        /// of the returned value will equal <see cref="Length"/>, and otherwise will have length less than
        /// <see cref="Length"/>.
        /// </summary>
        public ReadOnlySpan<T> GetValues() => _values.AsSpan(0, _count);

        /// <summary>
        /// The indices. For a dense representation, this array is not used, and will return the default "empty" span.
        /// For a sparse representation it is parallel to that returned from <see cref="GetValues"/> and specifies the
        /// logical indices for the corresponding values, in increasing order, between 0 inclusive and
        /// <see cref="Length"/> exclusive, corresponding to all explicitly defined values. All values at unspecified
        /// indices should be treated as being implicitly defined with the default value of <typeparamref name="T"/>.
        /// </summary>
        /// <remarks>
        /// To give one example, if <see cref="GetIndices"/> returns [3, 5] and <see cref="GetValues"/>() produces [98, 76],
        /// this <see cref="VBuffer{T}"/> stands for a vector with non-zero values 98 and 76 respectively at the 4th and 6th
        /// coordinates, and zeros at all other indices. (Zero, because that is the default value for all .NET numeric
        /// types.)
        /// </remarks>
        public ReadOnlySpan<int> GetIndices() => IsDense ? default : _indices.AsSpan(0, _count);

        /// <summary>
        /// Gets a value indicating whether every logical element is explicitly represented in the buffer.
        /// </summary>
        public bool IsDense
        {
            get
            {
                Debug.Assert(_count <= Length);
                return _count == Length;
            }
        }

        /// <summary>
        /// Construct a dense representation. The <paramref name="indices"/> array is often unspecified, but if
        /// specified it should be considered a buffer to be held on to, to be possibly used.
        /// </summary>
        /// <param name="length">The logical length of the resulting instance.</param>
        /// <param name="values">
        /// The values to be used. This must be at least as long as <paramref name="length"/>. If
        /// <paramref name="length"/> is 0, it is legal for this to be <see langword="null"/>. The constructed buffer
        /// takes ownership of this array.
        /// </param>
        /// <param name="indices">
        /// The internal indices buffer. Because this constructor is for dense representations
        /// this will not be immediately useful, but it does provide a buffer to be potentially reused to avoid
        /// allocation. This is mostly non-null in situations where you want to produce a dense
        /// <see cref="VBuffer{T}"/>, but you happen to have an indices array "left over" and you don't want to
        /// needlessly lose.
        /// </param>
        /// <remarks>
        /// The resulting structure takes ownership of the passed in arrays, so they should not be used for
        /// other purposes in the future.
        /// </remarks>
        public VBuffer(int length, T[] values, int[] indices = null)
        {
            Contracts.CheckParam(length >= 0, nameof(length));
            Contracts.CheckParam(ArrayUtils.Size(values) >= length, nameof(values));

            Length = length;
            _count = length;
            _values = values;
            _indices = indices;
        }

        /// <summary>
        /// Construct a possibly sparse vector representation.
        /// </summary>
        /// <param name="length">The length of the constructed buffer.</param>
        /// <param name="count">The count of explicit entries. This must be between 0 and <paramref name="length"/>, both
        /// inclusive. If it equals <paramref name="length"/> the result is a dense vector, and if less this will be a
        /// sparse vector.</param>
        /// <param name="values">
        /// The values to be used. This must be at least as long as <paramref name="count"/>. If
        /// <paramref name="count"/> is 0, it is legal for this to be <see langword="null"/>.
        /// </param>
        /// <param name="indices">The indices to be used. If we are constructing a dense representation, or
        /// <paramref name="count"/> is 0, this can be <see langword="null"/>. Otherwise, this must be at least as long
        /// as <paramref name="count"/>.</param>
        /// <remarks>The resulting structure takes ownership of the passed in arrays, so they should not be used for
        /// other purposes in the future.</remarks>
        public VBuffer(int length, int count, T[] values, int[] indices)
        {
            Contracts.CheckParam(length >= 0, nameof(length));
            Contracts.CheckParam(0 <= count && count <= length, nameof(count));
            Contracts.CheckParam(ArrayUtils.Size(values) >= count, nameof(values));
            Contracts.CheckParam(count == length || ArrayUtils.Size(indices) >= count, nameof(indices));

#if DEBUG // REVIEW: This validation should really use "Checks" and be in release code, but it is not cheap, so for practical reasons we must forego it.
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
                Debug.Assert(res >= 0 && cur < length);
            }
#endif

            Length = length;
            _count = count;
            _values = values;
            _indices = indices;
        }

        /// <summary>
        /// Copy from this buffer to the given destination, forcing a dense representation.
        /// </summary>
        /// <param name="destination">The destination buffer. After the copy, this will have <see cref="VBuffer{T}.Length"/>
        /// of <see cref="Length"/>.</param>
        public void CopyToDense(ref VBuffer<T> destination)
        {
            // create a dense editor
            var editor = VBufferEditor.Create(ref destination, Length);

            if (!IsDense)
                CopyTo(editor.Values);
            else if (Length > 0)
                _values.AsSpan(0, Length).CopyTo(editor.Values);
            destination = editor.Commit();
        }

        /// <summary>
        /// Copy from this buffer to the given destination.
        /// </summary>
        /// <param name="destination">The destination buffer. After the copy, this will have <see cref="VBuffer{T}.Length"/>
        /// of <see cref="Length"/>.</param>
        public void CopyTo(ref VBuffer<T> destination)
        {
            var editor = VBufferEditor.Create(ref destination, Length, _count);
            if (IsDense)
            {
                if (Length > 0)
                {
                    _values.AsSpan(0, Length).CopyTo(editor.Values);
                }
                destination = editor.Commit();
                Debug.Assert(destination.IsDense);
            }
            else
            {
                if (_count > 0)
                {
                    _values.AsSpan(0, _count).CopyTo(editor.Values);
                    _indices.AsSpan(0, _count).CopyTo(editor.Indices);
                }
                destination = editor.Commit();
            }
        }

        /// <summary>
        /// Copy a range of values from this buffer to the given destination.
        /// </summary>
        /// <param name="destination">The destination buffer. After the copy, this will have <see cref="VBuffer{T}.Length"/>
        /// of <paramref name="length"/>.</param>
        /// <param name="sourceIndex">The minimum inclusive index to start copying from this vector.</param>
        /// <param name="length">The logical number of values to copy from this vector into <paramref name="destination"/>.</param>
        public void CopyTo(ref VBuffer<T> destination, int sourceIndex, int length)
        {
            Contracts.CheckParam(0 <= sourceIndex && sourceIndex <= Length, nameof(sourceIndex));
            Contracts.CheckParam(0 <= length && sourceIndex <= Length - length, nameof(length));

            if (IsDense)
            {
                var editor = VBufferEditor.Create(ref destination, length, length);
                if (length > 0)
                {
                    _values.AsSpan(sourceIndex, length).CopyTo(editor.Values);
                }
                destination = editor.Commit();
                Debug.Assert(destination.IsDense);
            }
            else
            {
                int copyCount = 0;
                if (_count > 0)
                {
                    int copyMin = ArrayUtils.FindIndexSorted(_indices, 0, _count, sourceIndex);
                    int copyLim = ArrayUtils.FindIndexSorted(_indices, copyMin, _count, sourceIndex + length);
                    Debug.Assert(copyMin <= copyLim);
                    copyCount = copyLim - copyMin;
                    var editor = VBufferEditor.Create(ref destination, length, copyCount);
                    if (copyCount > 0)
                    {
                        _values.AsSpan(copyMin, copyCount).CopyTo(editor.Values);
                        if (copyCount < length)
                        {
                            for (int i = 0; i < copyCount; ++i)
                                editor.Indices[i] = _indices[i + copyMin] - sourceIndex;
                        }
                    }
                    destination = editor.Commit();
                }
                else
                {
                    var editor = VBufferEditor.Create(ref destination, length, copyCount);
                    destination = editor.Commit();
                }
            }
        }

        /// <summary>
        /// Copy from this buffer to the given destination span. This "densifies."
        /// </summary>
        /// <param name="destination">The destination buffer. This <see cref="Span{T}.Length"/> must have least <see cref="Length"/>.</param>
        public void CopyTo(Span<T> destination)
        {
            CopyTo(destination, 0);
        }

        /// <summary>
        /// Copy from this buffer to the given destination span, starting at the specified index. This "densifies."
        /// </summary>
        /// <param name="destination">The destination buffer. This <see cref="Span{T}.Length"/> must be at least <see cref="Length"/>
        /// plus <paramref name="destinationIndex"/>.</param>
        /// <param name="destinationIndex">The starting index of <paramref name="destination"/> at which to start copying.</param>
        /// <param name="defaultValue">The value to fill in for the implicit sparse entries. This is a potential exception to
        /// general expectation of sparse <see cref="VBuffer{T}"/> that the implicit sparse entries have the default value
        /// of <typeparamref name="T"/>.</param>
        public void CopyTo(Span<T> destination, int destinationIndex, T defaultValue = default(T))
        {
            Contracts.CheckParam(0 <= destinationIndex && destinationIndex <= destination.Length - Length,
                nameof(destination), "Not large enough to hold these values.");

            if (Length == 0)
                return;
            if (IsDense)
            {
                _values.AsSpan(0, Length).CopyTo(destination.Slice(destinationIndex));
                return;
            }

            if (_count == 0)
            {
                destination.Slice(destinationIndex, Length).Clear();
                return;
            }

            int iv = 0;
            for (int islot = 0; islot < _count; islot++)
            {
                int slot = _indices[islot];
                Debug.Assert(slot >= iv);
                while (iv < slot)
                    destination[destinationIndex + iv++] = defaultValue;
                Debug.Assert(iv == slot);
                destination[destinationIndex + iv++] = _values[islot];
            }
            while (iv < Length)
                destination[destinationIndex + iv++] = defaultValue;
        }

        /// <summary>
        /// Copy from a section of a source array to the given destination.
        /// </summary>
        public static void Copy(T[] source, int sourceIndex, ref VBuffer<T> destination, int length)
        {
            Contracts.CheckParam(0 <= length && length <= ArrayUtils.Size(source), nameof(length));
            Contracts.CheckParam(0 <= sourceIndex && sourceIndex <= ArrayUtils.Size(source) - length, nameof(sourceIndex));
            var editor = VBufferEditor.Create(ref destination, length, length);
            if (length > 0)
            {
                source.AsSpan(sourceIndex, length).CopyTo(editor.Values);
            }
            destination = editor.Commit();
        }

        /// <summary>
        /// Returns the joint list of all index/value pairs.
        /// </summary>
        /// <param name="all">
        /// If <see langword="true"/> all pairs, even those implicit values of a sparse representation,
        /// will be returned, with the implicit values having the default value, as is appropriate. If left
        /// <see langword="false"/> then only explicitly defined values are returned.
        /// </param>
        /// <returns>The index/value pairs.</returns>
        public IEnumerable<KeyValuePair<int, T>> Items(bool all = false)
        {
            return Items(_values, _indices, Length, _count, all);
        }

        /// <summary>
        /// Returns an enumerable with <see cref="Length"/> items, representing the values.
        /// </summary>
        public IEnumerable<T> DenseValues()
        {
            return DenseValues(_values, _indices, Length, _count);
        }

        /// <summary>
        /// Gets the item stored in this structure. In the case of a dense vector this is a simple lookup.
        /// In the case of a sparse vector, it will try to find the entry with that index, and set <paramref name="destination"/>
        /// to that stored value, or if no such value was found, assign it the default value.
        /// </summary>
        /// <remarks>
        /// In the case where <see cref="IsDense"/> is <see langword="true"/>, this will take constant time since it an
        /// directly lookup. For sparse vectors, however, because it must perform a bisection search on the indices to
        /// find the appropriate value, that takes logarithmic time with respect to the number of explicitly represented
        /// items, which is to say, the <see cref="ReadOnlySpan{Int32}.Length"/> of the return value of <see cref="GetIndices"/>.
        ///
        /// For that reason, a single completely isolated lookup, since constructing <see cref="ReadOnlySpan{T}"/> as
        /// <see cref="GetValues"/> does is not a free operation, it may be more efficient to use this method. However
        /// if one is doing a more involved computation involving many operations, it may be faster to utilize
        /// <see cref="GetValues"/> and, if appropriate, <see cref="GetIndices"/> directly.
        /// </remarks>
        /// <param name="index">The index, which must be a non-negative number less than <see cref="Length"/>.</param>
        /// <param name="destination">The value stored at that index, or if this is a sparse vector where this is an implicit
        /// entry, the default value for <typeparamref name="T"/>.</param>
        public void GetItemOrDefault(int index, ref T destination)
        {
            Contracts.CheckParam(0 <= index && index < Length, nameof(index));

            if (IsDense)
                destination = _values[index];
            else if (_count > 0 && ArrayUtils.TryFindIndexSorted(_indices, 0, _count, index, out int bufferIndex))
                destination = _values[bufferIndex];
            else
                destination = default(T);
        }

        /// <summary>
        /// A variant of <see cref="GetItemOrDefault(int, ref T)"/> that returns the value instead of passing it
        /// back using a reference parameter.
        /// </summary>
        /// <param name="index">The index, which must be a non-negative number less than <see cref="Length"/>.</param>
        /// <returns>The value stored at that index, or if this is a sparse vector where this is an implicit
        /// entry, the default value for <typeparamref name="T"/>.</returns>
        public T GetItemOrDefault(int index)
        {
            Contracts.CheckParam(0 <= index && index < Length, nameof(index));

            if (IsDense)
                return _values[index];
            if (_count > 0 && ArrayUtils.TryFindIndexSorted(_indices, 0, _count, index, out int bufferIndex))
                return _values[bufferIndex];
            return default(T);
        }

        public override string ToString()
            => IsDense ? $"Dense vector of size {Length}" : $"Sparse vector of size {Length}, {_count} explicit values";

        internal VBufferEditor<T> GetEditor()
        {
            return GetEditor(Length, _count);
        }

        internal VBufferEditor<T> GetEditor(
            int newLogicalLength,
            int? valuesCount,
            int? maxValuesCapacity = null,
            bool keepOldOnResize = false,
            bool requireIndicesOnDense = false)
        {
            Contracts.CheckParam(newLogicalLength >= 0, nameof(newLogicalLength), "Must be non-negative.");
            Contracts.CheckParam(valuesCount == null || valuesCount.Value >= 0, nameof(valuesCount),
                "If specified, must be non-negative.");
            Contracts.CheckParam(valuesCount == null || valuesCount.Value <= newLogicalLength, nameof(valuesCount),
                "If specified, must be no greater than " + nameof(newLogicalLength));

            int logicalValuesCount = valuesCount ?? newLogicalLength;

            int maxCapacity = maxValuesCapacity ?? ArrayUtils.ArrayMaxSize;

            T[] values = _values;
            bool createdNewValues;
            ArrayUtils.EnsureSize(ref values, logicalValuesCount, maxCapacity, keepOldOnResize, out createdNewValues);

            int[] indices = _indices;
            bool isDense = newLogicalLength == logicalValuesCount;
            bool createdNewIndices;
            if (isDense && !requireIndicesOnDense)
            {
                createdNewIndices = false;
            }
            else
            {
                ArrayUtils.EnsureSize(ref indices, logicalValuesCount, maxCapacity, keepOldOnResize, out createdNewIndices);
            }

            return new VBufferEditor<T>(
                newLogicalLength,
                logicalValuesCount,
                values,
                indices,
                requireIndicesOnDense,
                createdNewValues,
                createdNewIndices);
        }

        /// <summary>
        /// A helper method that gives us an iterable over the items given the fields from a <see cref="VBuffer{T}"/>.
        /// Note that we have this in a separate utility class, rather than in its more natural location of
        /// <see cref="VBuffer{T}"/> itself, due to a bug in the C++/CLI compiler. (DevDiv 1097919:
        /// [C++/CLI] Nested generic types are not correctly imported from metadata). So, if we want to use
        /// <see cref="VBuffer{T}"/> in C++/CLI projects, we cannot have a generic struct with a nested class
        /// that has the outer struct type as a field.
        /// </summary>
        private static IEnumerable<KeyValuePair<int, T>> Items(T[] values, int[] indices, int length, int count, bool all)
        {
            Debug.Assert(0 <= count && count <= ArrayUtils.Size(values));
            Debug.Assert(count <= length);
            Debug.Assert(count == length || count <= ArrayUtils.Size(indices));

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
                    Debug.Assert(slotCur < slot && slot < length);
                    while (++slotCur < slot)
                        yield return new KeyValuePair<int, T>(slotCur, default(T));
                    Debug.Assert(slotCur == slot);
                    yield return new KeyValuePair<int, T>(slotCur, values[i]);
                }
                Debug.Assert(slotCur < length);
                while (++slotCur < length)
                    yield return new KeyValuePair<int, T>(slotCur, default(T));
            }
        }

        private static IEnumerable<T> DenseValues(T[] values, int[] indices, int length, int count)
        {
            Debug.Assert(0 <= count && count <= ArrayUtils.Size(values));
            Debug.Assert(count <= length);
            Debug.Assert(count == length || count <= ArrayUtils.Size(indices));

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
                    Debug.Assert(slotCur < slot && slot < length);
                    while (++slotCur < slot)
                        yield return default(T);
                    Debug.Assert(slotCur == slot);
                    yield return values[i];
                }
                Debug.Assert(slotCur < length);
                while (++slotCur < length)
                    yield return default(T);
            }
        }
    }
}
