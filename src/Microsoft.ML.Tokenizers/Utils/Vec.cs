// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Tokenizers
{
    internal struct Vec<T>
    {
        private const int DefaultCapacity = 10;
        private int _count;
        private T[]? _buffer;

        public ref T this[int index]
        {
            get
            {
                if (index >= _count)
                {
                    throw new ArgumentOutOfRangeException(nameof(index), $"{index} is out of range");
                }
                return ref _buffer![index];
            }
        }

        public Vec()
        {
            _count = 0;
            _buffer = null;
        }

        public Vec(int capacity)
        {
            _count = 0;
            _buffer = new T[capacity];
        }

        public int Capacity => _buffer is null ? 0 : _buffer.Length;
        public int Count => _count;

        public void Push(T t)
        {
            if (_buffer is null)
            {
                _buffer = new T[DefaultCapacity];
                _buffer[0] = t;
                _count = 1;
                return;
            }

            if (_buffer.Length <= _count)
            {
                Array.Resize(ref _buffer, _buffer.Length << 1);
            }

            _buffer[_count++] = t;
        }

        public void Remove(int index)
        {
            if (index >= _count || _buffer is null)
            {
                return;
            }

            for (int i = index; i < _count - 1; i++)
            {
                _buffer[i] = _buffer[i + 1];
            }

            _count--;
        }

        public void Clear() => _count = 0;
    }
}
