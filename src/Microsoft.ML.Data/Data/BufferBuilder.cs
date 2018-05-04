// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    using Conditional = System.Diagnostics.ConditionalAttribute;

    /// <summary>
    /// Helper base class for building feature vectors (sparse or dense). Note that this is abstract
    /// with some of the esoteric stuff "protected" instead of "public". This is so callees can't
    /// abuse an instance of it.
    /// </summary>
    public sealed class BufferBuilder<T>
    {
        // Don't walk more than this many items when doing an insert.
        private const int InsertThreshold = 20;

        private readonly Combiner<T> _comb;

        // _length is the logical length of the feature set. Valid indices are non-negative
        // indices less than _length.
        private int _length;

        // Whether we're currently using a dense representation. In that case, _indices is not used.
        private bool _dense;

        // When we're currently using a sparse representation, this indicates whether the indices are
        // known to be sorted (increasing, with no duplicates). We'll do a small amount of work on each
        // AddFeature call to maintain _sorted.
        private bool _sorted;

        // When in a sparse representation, _count is the number of active values in _values and _indices.
        // When dense, _count == _length (see AssertValid).
        private int _count;

        // The indices and values. When sparse, these are parallel with _count active entries. When dense,
        // _indices is not used and the values are stored in the first _length elements of _values.
        private int[] _indices;
        private T[] _values;

        // When invoking a component to add features in a particular range, these can be set to the base
        // index and length of the component's feature index range. Then the component doesn't need to
        // know about the rest of the feature index space.
        private int _ifeatCur;
        private int _cfeatCur;

        public bool IsEmpty => _count == 0;

        public int Length => _length;

        public BufferBuilder(Combiner<T> comb)
        {
            Contracts.AssertValue(comb);
            _comb = comb;

            // _values and _indices are resized as needed. This initialization ensures that _values
            // is never null. Note that _indices starts as null. This is because if we use a dense
            // representation (specified in ResetImpl), then _indices isn't even needed so no point
            // pre-allocating.
            _values = new T[8];
        }

        [Conditional("DEBUG")]
        private void AssertValid()
        {
#if DEBUG
            Contracts.Assert(_count >= 0);
            Contracts.AssertValue(_values);
            Contracts.Assert(_values.Length >= _count);
            Contracts.Assert(0 <= _ifeatCur & 0 <= _cfeatCur & _ifeatCur <= _length - _cfeatCur);

            if (_dense)
                Contracts.Assert(_count == _length);
            else
            {
                // ResetImpl forces _indices != null and _length > 0.
                Contracts.Assert(_indices != null || _length == 0);
                Contracts.Assert(Utils.Size(_indices) >= _count);

                // If we have no more than InsertThreshold items, we always keep things sorted.
                Contracts.Assert(_sorted | _count > InsertThreshold);
            }
#endif
        }

        public static BufferBuilder<T> CreateDefault()
        {
            if (typeof(T) == typeof(DvText))
                return (BufferBuilder<T>)(object)new BufferBuilder<DvText>(TextCombiner.Instance);
            if (typeof(T) == typeof(float))
                return (BufferBuilder<T>)(object)new BufferBuilder<float>(FloatAdder.Instance);
            throw Contracts.Except($"Unrecognized type '{typeof(T)}' for default {nameof(BufferBuilder<T>)}");
        }

        /// <summary>
        /// This resets the FeatureSet to be used again. This functionality is for memory
        /// efficiency - we can keep pools of these to be re-used.
        /// Dense indicates whether this should start out dense. It can, of course,
        /// become dense when it makes sense to do so.
        /// </summary>
        private void ResetImpl(int length, bool dense)
        {
            Contracts.Assert(length > 0);
            _length = length;
            _dense = false;
            _sorted = true;
            _count = 0;
            _ifeatCur = 0;
            _cfeatCur = 0;

            if (dense)
            {
                if (_values.Length < _length)
                    _values = new T[_length];
                else
                    Array.Clear(_values, 0, _length);
                _dense = true;
                _count = _length;
            }
            else if (_indices == null)
                _indices = new int[8];

            AssertValid();
        }

        /// <summary>
        /// This sets the active sub-range of the feature index space. For example, when asking
        /// a feature handler to add features, we call this so the feature handler can use zero-based
        /// indexing for the features it is generating. This also prohibits the feature handler from
        /// messing with a different index range. Note that this is protected so a non-abstract derived
        /// type can choose how to use it, but a feature handler can't directly mess with the active
        /// range.
        /// </summary>
        /// <param name="ifeat">The min feature index of the active range</param>
        /// <param name="cfeat">The number of feature indices in the active range</param>
        private void SetActiveRangeImpl(int ifeat, int cfeat)
        {
            AssertValid();
            Contracts.Assert(0 <= ifeat & 0 <= cfeat & ifeat <= _length - cfeat);
            _ifeatCur = ifeat;
            _cfeatCur = cfeat;
            AssertValid();
        }

        /// <summary>
        /// Adds a feature to the current active range. If the index is a duplicate, this adds the
        /// given value to any previously provided value(s).
        /// </summary>
        public void AddFeature(int index, T value)
        {
            AssertValid();
            Contracts.Assert(0 <= index & index < _cfeatCur);

            // Ignore default values.
            if (_comb.IsDefault(value))
                return;

            // Adjust the index.
            index += _ifeatCur;

            if (_dense)
            {
                _comb.Combine(ref _values[index], value);
                return;
            }

            // ResetImpl ensures that _indices is non-null when _dense is false.
            Contracts.Assert(_indices != null);
            if (!_sorted)
            {
                if (_count < _length)
                {
                    // Make room.
                    if (_values.Length <= _count)
                        Array.Resize(ref _values, Math.Max(Math.Min(_length, checked(_count * 2)), 8));
                    if (_indices.Length <= _count)
                        Array.Resize(ref _indices, Math.Max(Math.Min(_length, checked(_count * 2)), 8));

                    _values[_count] = value;
                    _indices[_count] = index;
                    _count++;
                    return;
                }

                SortAndSumDups();
                if (_dense)
                {
                    _comb.Combine(ref _values[index], value);
                    return;
                }
            }
            Contracts.Assert(_sorted);

            if (_count >= _length / 2 - 1)
            {
                MakeDense();
                _comb.Combine(ref _values[index], value);
                return;
            }

            // Make room.
            if (_values.Length <= _count)
                Array.Resize(ref _values, Math.Max(Math.Min(_length, checked(_count * 2)), 8));
            if (_indices.Length <= _count)
                Array.Resize(ref _indices, Math.Max(Math.Min(_length, checked(_count * 2)), 8));

            if (_count >= InsertThreshold && _indices[_count - InsertThreshold] > index)
            {
                _values[_count] = value;
                _indices[_count] = index;
                _count++;
                _sorted = false;
                return;
            }

            // Insert this one. Find the right place.
            // REVIEW: Should we ever use binary search?
            int ivDst = _count;
            for (;;)
            {
                if (--ivDst < 0)
                    break;
                int diff = _indices[ivDst] - index;
                if (diff <= 0)
                {
                    if (diff < 0)
                        break;

                    // entry is found, increment the value
                    _comb.Combine(ref _values[ivDst], value);
                    return;
                }
            }
            Contracts.Assert(ivDst < 0 || _indices[ivDst] < index);
            ivDst++;
            Contracts.Assert(ivDst == _count || _indices[ivDst] > index);

            // Value goes here at ivDst. Shift others up.
            for (int i = _count; --i >= ivDst;)
            {
                _indices[i + 1] = _indices[i];
                _values[i + 1] = _values[i];
            }
            _indices[ivDst] = index;
            _values[ivDst] = value;
            _count++;
        }

        /// <summary>
        /// Sort the indices/values (by index) and sum the values for duplicate indices. This asserts that
        /// _sorted is false and _dense is false. It also asserts that _count > 1.
        /// </summary>
        private void SortAndSumDups()
        {
            AssertValid();
            Contracts.Assert(!_sorted);
            Contracts.Assert(!_dense);
            Contracts.Assert(_count > 1);

            // REVIEW: Ideally this would be a stable sort.
            Array.Sort(_indices, _values, 0, _count);

            int ivSrc = 0;
            int ivDst = 0;
            for (;;)
            {
                if (ivSrc >= _count)
                {
                    _count = 0;
                    _sorted = true;
                    AssertValid();
                    return;
                }
                if (!_comb.IsDefault(_values[ivSrc]))
                    break;
            }
            Contracts.Assert(ivSrc < _count && !_comb.IsDefault(_values[ivSrc]));

            _values[ivDst] = _values[ivSrc];
            _indices[ivDst++] = _indices[ivSrc];
            while (++ivSrc < _count)
            {
                Contracts.Assert(ivDst <= ivSrc);
                if (_indices[ivDst - 1] == _indices[ivSrc])
                {
                    _comb.Combine(ref _values[ivDst - 1], _values[ivSrc]);
                    continue;
                }

                if (ivDst < ivSrc)
                {
                    // Copy down
                    _indices[ivDst] = _indices[ivSrc];
                    _values[ivDst] = _values[ivSrc];
                }
                ivDst++;
            }
            Contracts.Assert(0 < ivDst & ivDst <= _count);
            _count = ivDst;
            _sorted = true;
            AssertValid();

            if (_count >= _length / 2)
                MakeDense();
        }

        /// <summary>
        /// Convert a sorted non-dense representation to dense.
        /// </summary>
        private void MakeDense()
        {
            AssertValid();
            Contracts.Assert(!_dense);
            Contracts.Assert(_sorted);

            if (_values.Length < _length)
                Array.Resize(ref _values, _length);

            int ivDst = _length;
            int iivSrc = _count;
            while (--iivSrc >= 0)
            {
                int index = _indices[iivSrc];
                Contracts.Assert(ivDst > index);
                while (--ivDst > index)
                    _values[ivDst] = default(T);
                Contracts.Assert(ivDst == index);
                _values[ivDst] = _values[iivSrc];
            }
            while (--ivDst >= 0)
                _values[ivDst] = default(T);

            _dense = true;
            _count = _length;
        }

        /// <summary>
        /// Try to get the value for the given feature. Returns false if the feature index is not found.
        /// Note that this respects the "active range", just as AddFeature does.
        /// </summary>
        public bool TryGetFeature(int index, out T v)
        {
            AssertValid();
            Contracts.Assert(0 <= index & index < _cfeatCur);

            int ifeat = index + _ifeatCur;
            if (_dense)
            {
                v = _values[ifeat];
                return true;
            }

            // Make sure the indices are sorted.
            if (!_sorted)
            {
                SortAndSumDups();
                if (_dense)
                {
                    v = _values[ifeat];
                    return true;
                }
            }

            int iv = Utils.FindIndexSorted(_indices, 0, _count, ifeat);
            Contracts.Assert(iv == 0 || ifeat > _indices[iv - 1]);

            if (iv < _count)
            {
                Contracts.Assert(ifeat <= _indices[iv]);
                if (ifeat == _indices[iv])
                {
                    v = _values[iv];
                    return true;
                }
            }

            v = default(T);
            return false;
        }

        private void GetResult(ref T[] values, ref int[] indices, out int count, out int length)
        {
            if (_count == 0)
            {
                count = 0;
                length = _length;
                return;
            }

            if (!_dense)
            {
                if (!_sorted)
                    SortAndSumDups();
                if (!_dense && _count >= _length / 2)
                    MakeDense();
            }

            if (_dense)
            {
                if (Utils.Size(values) < _length)
                    values = new T[_length];
                Array.Copy(_values, values, _length);
                count = _length;
                length = _length;
            }
            else
            {
                Contracts.Assert(_count < _length);
                if (Utils.Size(values) < _count)
                    values = new T[_count];
                if (Utils.Size(indices) < _count)
                    indices = new int[_count];
                Array.Copy(_values, values, _count);
                Array.Copy(_indices, indices, _count);
                count = _count;
                length = _length;
            }
        }

        public void Reset(int length, bool dense)
        {
            ResetImpl(length, dense);
            SetActiveRangeImpl(0, length);
        }

        public void AddFeatures(int index, ref VBuffer<T> buffer)
        {
            Contracts.Check(0 <= index && index <= _length - buffer.Length);

            int count = buffer.Count;
            if (count == 0)
                return;

            var values = buffer.Values;
            if (buffer.IsDense)
            {
                Contracts.Assert(count == buffer.Length);
                if (_dense)
                {
                    for (int i = 0; i < count; i++)
                        _comb.Combine(ref _values[index + i], values[i]);
                }
                else
                {
                    // REVIEW: Optimize this.
                    for (int i = 0; i < count; i++)
                        AddFeature(index + i, values[i]);
                }
            }
            else
            {
                // REVIEW: Validate indices!
                var indices = buffer.Indices;
                if (_dense)
                {
                    for (int i = 0; i < count; i++)
                        _comb.Combine(ref _values[index + indices[i]], values[i]);
                }
                else
                {
                    // REVIEW: Optimize this.
                    for (int i = 0; i < count; i++)
                        AddFeature(index + indices[i], values[i]);
                }
            }
        }

        public void GetResult(ref VBuffer<T> buffer)
        {
            var values = buffer.Values;
            var indices = buffer.Indices;

            if (IsEmpty)
            {
                buffer = new VBuffer<T>(_length, 0, values, indices);
                return;
            }

            int count;
            int length;
            GetResult(ref values, ref indices, out count, out length);
            Contracts.Assert(0 <= count && count <= length);

            if (count == length)
                buffer = new VBuffer<T>(length, values, indices);
            else
                buffer = new VBuffer<T>(length, count, values, indices);
        }
    }
}
