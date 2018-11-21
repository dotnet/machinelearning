// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    using Conditional = System.Diagnostics.ConditionalAttribute;

    /// <summary>
    /// A fixed-length circular array. Items are added at the end. If the array is full, adding
    ///  an item will result in discarding the least recently added item.
    /// </summary>
    public sealed class FixedSizeQueue<T>
    {
        private readonly T[] _array;
        private int _startIndex;
        private int _count;

        public FixedSizeQueue(int capacity)
        {
            Contracts.Assert(capacity > 0, "Array capacity should be greater than zero");
            _array = new T[capacity];
            AssertValid();
        }

        [Conditional("DEBUG")]
        private void AssertValid()
        {
            Contracts.Assert(Utils.Size(_array) >= 0);
            Contracts.Assert(0 <= _startIndex & _startIndex < _array.Length);
            Contracts.Assert(0 <= _count & _count <= _array.Length);
        }

        public int Count
        {
            get
            {
                AssertValid();
                return _count;
            }
        }

        public int Capacity
        {
            get
            {
                AssertValid();
                return _array.Length;
            }
        }

        public bool IsFull
        {
            get
            {
                AssertValid();
                return _count == _array.Length;
            }
        }

        public T this[int index]
        {
            get
            {
                AssertValid();
                Contracts.Assert(index >= 0 & index < _count);
                return _array[(_startIndex + index) % _array.Length];
            }
        }

        public void AddLast(T item)
        {
            AssertValid();
            if (_count == _array.Length)
            {
                // Replace least recently added item (found at _startIndex)
                _array[_startIndex] = item;
                _startIndex = (_startIndex + 1) % _array.Length;
            }
            else
            {
                _array[(_startIndex + _count) % _array.Length] = item;
                _count++;
            }

            AssertValid();
        }

        public T PeekFirst()
        {
            AssertValid();
            Contracts.Assert(_count != 0, "Array is empty");
            return _array[_startIndex];
        }

        public T PeekLast()
        {
            AssertValid();
            Contracts.Assert(_count != 0, "Array is empty");
            return _array[(_startIndex + _count - 1) % _array.Length];
        }

        public T RemoveFirst()
        {
            AssertValid();
            Contracts.Assert(_count != 0, "Array is empty");
            T item = _array[_startIndex];
            _array[_startIndex] = default(T);
            _startIndex = (_startIndex + 1) % _array.Length;
            _count--;
            AssertValid();
            return item;
        }

        public T RemoveLast()
        {
            AssertValid();
            Contracts.Assert(_count != 0, "Array is empty");
            int lastIndex = (_startIndex + _count - 1) % _array.Length;
            T item = _array[lastIndex];
            _array[lastIndex] = default(T);
            _count--;
            AssertValid();
            return item;
        }

        public void Clear()
        {
            AssertValid();
            for (int i = 0; i < _count; i++)
                _array[(_startIndex + i) % _array.Length] = default(T);
            _startIndex = 0;
            _count = 0;
            AssertValid();
        }
    }
}
