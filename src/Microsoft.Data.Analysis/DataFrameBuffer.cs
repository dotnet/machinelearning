// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Microsoft.Data.Analysis
{
    /// <summary>
    /// A basic mutable store to hold values in a DataFrame column. Supports wrapping with an ArrowBuffer
    /// </summary>
    /// <typeparam name="T"></typeparam>
    internal class DataFrameBuffer<T> : ReadOnlyDataFrameBuffer<T>
        where T : unmanaged
    {
        private const int MinCapacity = 8;

        private Memory<byte> _memory;

        public override ReadOnlyMemory<byte> ReadOnlyBuffer => _memory;

        public Memory<byte> Buffer
        {
            get => _memory;
        }

        public Span<T> Span
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => (MemoryMarshal.Cast<byte, T>(Buffer.Span)).Slice(0, Length);
        }

        public Span<T> RawSpan
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => MemoryMarshal.Cast<byte, T>(Buffer.Span);
        }

        public DataFrameBuffer(int capacity = 0)
        {
            if ((long)capacity > MaxCapacity)
            {
                throw new ArgumentException($"{capacity} exceeds buffer capacity", nameof(capacity));
            }

            _memory = new byte[Math.Max(capacity, MinCapacity) * Size];
        }

        internal DataFrameBuffer(ReadOnlyMemory<byte> buffer, int length)
        {
            _memory = new byte[buffer.Length];
            buffer.CopyTo(_memory);
            Length = length;
        }

        public void Append(T value)
        {
            EnsureCapacity(1);

            RawSpan[Length] = value;
            Length++;
        }

        public void IncreaseSize(int numberOfValues)
        {
            EnsureCapacity(numberOfValues);
            Length += numberOfValues;
        }

        public void EnsureCapacity(int numberOfValues)
        {
            long newLength = Length + (long)numberOfValues;
            if (newLength > MaxCapacity)
            {
                throw new ArgumentException("Current buffer is full", nameof(numberOfValues));
            }

            if (newLength > Capacity)
            {
                //Double buffer size, but not higher than MaxByteCapacity
                var doubledSize = (int)Math.Min((long)ReadOnlyBuffer.Length * 2, ArrayUtility.ArrayMaxSize);
                var newCapacity = Math.Max(newLength * Size, doubledSize);

                var memory = new Memory<byte>(new byte[newCapacity]);
                _memory.CopyTo(memory);
                _memory = memory;
            }
        }

        internal override T this[int index]
        {
            set
            {
                if (index >= Length)
                    throw new ArgumentOutOfRangeException(nameof(index));

                RawSpan[index] = value;
            }
        }

        internal static DataFrameBuffer<T> GetMutableBuffer(ReadOnlyDataFrameBuffer<T> buffer)
        {
            DataFrameBuffer<T> mutableBuffer = buffer as DataFrameBuffer<T>;
            if (mutableBuffer == null)
            {
                mutableBuffer = new DataFrameBuffer<T>(buffer.ReadOnlyBuffer, buffer.Length);
            }
            return mutableBuffer;
        }
    }
}
