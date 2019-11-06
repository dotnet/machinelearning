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
        where T : struct
    {
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

        public DataFrameBuffer(int numberOfValues = 8) : base(numberOfValues) { }

        internal DataFrameBuffer(ReadOnlyMemory<byte> buffer, int length) : base(buffer, length)
        {
            _memory = new byte[buffer.Length];
            buffer.CopyTo(_memory);
        }

        public void Append(T value)
        {
            if (Length == MaxCapacity)
            {
                throw new ArgumentException("Current buffer is full", nameof(value));
            }
            EnsureCapacity(1);
            if (Length < MaxCapacity)
                ++Length;
            Span[Length - 1] = value;
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
                var newCapacity = Math.Max(newLength * Size, ReadOnlyBuffer.Length * 2);
                var memory = new Memory<byte>(new byte[newCapacity]);
                _memory.CopyTo(memory);
                _memory = memory;
            }
        }

        internal override T this[int index]
        {
            set
            {
                if (index > Length)
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
                mutableBuffer.Length = buffer.Length;
            }
            return mutableBuffer;
        }
    }
}
