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
    /// A buffer that supports both dense and sparse representations. This is the
    /// representation type for all VectorType instances. When an instance of this
    /// is passed to a row cursor getter, the callee is free to take ownership of
    /// and re-use the arrays (Values and Indices).
    /// </summary>
    public readonly struct VBuffer<T>
    {
        private readonly T[] _values;
        private readonly int[] _indices;

        /// <summary>
        /// The number of items explicitly represented. This is == Length when the representation
        /// is dense and &lt; Length when sparse.
        /// </summary>
        private readonly int _count;

        /// <summary>
        /// The logical length of the buffer.
        /// </summary>
        public readonly int Length;

        /// <summary>
        /// The explicitly represented values.
        /// </summary>
        public ReadOnlySpan<T> GetValues() => _values.AsSpan(0, _count);

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
        public ReadOnlySpan<int> GetIndices() => IsDense ? default : _indices.AsSpan(0, _count);

        /// <summary>
        /// Gets a value indicating whether every logical element is explicitly
        /// represented in the buffer.
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
        /// Construct a dense representation with unused Indices array.
        /// </summary>
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
        /// Construct a possibly sparse representation.
        /// </summary>
        public VBuffer(int length, int count, T[] values, int[] indices)
        {
            Contracts.CheckParam(length >= 0, nameof(length));
            Contracts.CheckParam(0 <= count && count <= length, nameof(count));
            Contracts.CheckParam(ArrayUtils.Size(values) >= count, nameof(values));
            Contracts.CheckParam(count == length || ArrayUtils.Size(indices) >= count, nameof(indices));

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
        public void CopyToDense(ref VBuffer<T> dst)
        {
            // create a dense editor
            var editor = VBufferEditor.Create(ref dst, Length);

            if (!IsDense)
                CopyTo(editor.Values);
            else if (Length > 0)
                _values.AsSpan(0, Length).CopyTo(editor.Values);
            dst = editor.Commit();
        }

        /// <summary>
        /// Copy from this buffer to the given destination.
        /// </summary>
        public void CopyTo(ref VBuffer<T> dst)
        {
            var editor = VBufferEditor.Create(ref dst, Length, _count);
            if (IsDense)
            {
                if (Length > 0)
                {
                    _values.AsSpan(0, Length).CopyTo(editor.Values);
                }
                dst = editor.Commit();
                Debug.Assert(dst.IsDense);
            }
            else
            {
                if (_count > 0)
                {
                    _values.AsSpan(0, _count).CopyTo(editor.Values);
                    _indices.AsSpan(0, _count).CopyTo(editor.Indices);
                }
                dst = editor.Commit();
            }
        }

        /// <summary>
        /// Copy a range of values from this buffer to the given destination.
        /// </summary>
        public void CopyTo(ref VBuffer<T> dst, int srcMin, int length)
        {
            Contracts.Check(0 <= srcMin && srcMin <= Length, "srcMin");
            Contracts.Check(0 <= length && srcMin <= Length - length, "length");

            if (IsDense)
            {
                var editor = VBufferEditor.Create(ref dst, length, length);
                if (length > 0)
                {
                    _values.AsSpan(srcMin, length).CopyTo(editor.Values);
                }
                dst = editor.Commit();
                Debug.Assert(dst.IsDense);
            }
            else
            {
                int copyCount = 0;
                if (_count > 0)
                {
                    int copyMin = ArrayUtils.FindIndexSorted(_indices, 0, _count, srcMin);
                    int copyLim = ArrayUtils.FindIndexSorted(_indices, copyMin, _count, srcMin + length);
                    Debug.Assert(copyMin <= copyLim);
                    copyCount = copyLim - copyMin;
                    var editor = VBufferEditor.Create(ref dst, length, copyCount);
                    if (copyCount > 0)
                    {
                        _values.AsSpan(copyMin, copyCount).CopyTo(editor.Values);
                        if (copyCount < length)
                        {
                            for (int i = 0; i < copyCount; ++i)
                                editor.Indices[i] = _indices[i + copyMin] - srcMin;
                        }
                    }
                    dst = editor.Commit();
                }
                else
                {
                    var editor = VBufferEditor.Create(ref dst, length, copyCount);
                    dst = editor.Commit();
                }
            }
        }

        /// <summary>
        /// Copy from this buffer to the given destination array. This "densifies".
        /// </summary>
        public void CopyTo(Span<T> dst)
        {
            CopyTo(dst, 0);
        }

        public void CopyTo(Span<T> dst, int ivDst, T defaultValue = default(T))
        {
            Contracts.CheckParam(0 <= ivDst && ivDst <= dst.Length - Length, nameof(dst), "dst is not large enough");

            if (Length == 0)
                return;
            if (IsDense)
            {
                _values.AsSpan(0, Length).CopyTo(dst.Slice(ivDst));
                return;
            }

            if (_count == 0)
            {
                dst.Slice(ivDst, Length).Clear();
                return;
            }

            int iv = 0;
            for (int islot = 0; islot < _count; islot++)
            {
                int slot = _indices[islot];
                Debug.Assert(slot >= iv);
                while (iv < slot)
                    dst[ivDst + iv++] = defaultValue;
                Debug.Assert(iv == slot);
                dst[ivDst + iv++] = _values[islot];
            }
            while (iv < Length)
                dst[ivDst + iv++] = defaultValue;
        }

        /// <summary>
        /// Copy from a section of a source array to the given destination.
        /// </summary>
        public static void Copy(T[] src, int srcIndex, ref VBuffer<T> dst, int length)
        {
            Contracts.CheckParam(0 <= length && length <= ArrayUtils.Size(src), nameof(length));
            Contracts.CheckParam(0 <= srcIndex && srcIndex <= ArrayUtils.Size(src) - length, nameof(srcIndex));
            var editor = VBufferEditor.Create(ref dst, length, length);
            if (length > 0)
            {
                src.AsSpan(srcIndex, length).CopyTo(editor.Values);
            }
            dst = editor.Commit();
        }

        public IEnumerable<KeyValuePair<int, T>> Items(bool all = false)
        {
            return Items(_values, _indices, Length, _count, all);
        }

        public IEnumerable<T> DenseValues()
        {
            return DenseValues(_values, _indices, Length, _count);
        }

        public void GetItemOrDefault(int slot, ref T dst)
        {
            Contracts.CheckParam(0 <= slot && slot < Length, nameof(slot));

            int index;
            if (IsDense)
                dst = _values[slot];
            else if (_count > 0 && ArrayUtils.TryFindIndexSorted(_indices, 0, _count, slot, out index))
                dst = _values[index];
            else
                dst = default(T);
        }

        public T GetItemOrDefault(int slot)
        {
            Contracts.CheckParam(0 <= slot && slot < Length, nameof(slot));

            int index;
            if (IsDense)
                return _values[slot];
            if (_count > 0 && ArrayUtils.TryFindIndexSorted(_indices, 0, _count, slot, out index))
                return _values[index];
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
            Contracts.CheckParam(newLogicalLength >= 0, nameof(newLogicalLength));
            Contracts.CheckParam(valuesCount == null || valuesCount.Value <= newLogicalLength, nameof(valuesCount));

            valuesCount = valuesCount ?? newLogicalLength;

            int maxCapacity = maxValuesCapacity ?? ArrayUtils.ArrayMaxSize;

            T[] values = _values;
            bool createdNewValues;
            ArrayUtils.EnsureSize(ref values, valuesCount.Value, maxCapacity, keepOldOnResize, out createdNewValues);

            int[] indices = _indices;
            bool isDense = newLogicalLength == valuesCount.Value;
            bool createdNewIndices;
            if (isDense && !requireIndicesOnDense)
            {
                createdNewIndices = false;
            }
            else
            {
                ArrayUtils.EnsureSize(ref indices, valuesCount.Value, maxCapacity, keepOldOnResize, out createdNewIndices);
            }

            return new VBufferEditor<T>(
                newLogicalLength,
                valuesCount.Value,
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