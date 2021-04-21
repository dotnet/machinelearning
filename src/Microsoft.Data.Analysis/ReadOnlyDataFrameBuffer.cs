// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace Microsoft.Data.Analysis
{
    /// <summary>
    /// A basic immutable store to hold values in a DataFrame column. Supports wrapping with an ArrowBuffer
    /// </summary>
    /// <typeparam name="T"></typeparam>
    internal class ReadOnlyDataFrameBuffer<T>
        where T : struct
    {
        private readonly ReadOnlyMemory<byte> _readOnlyBuffer;

        public virtual ReadOnlyMemory<byte> ReadOnlyBuffer => _readOnlyBuffer;

        public ReadOnlyMemory<T> ReadOnlyMemory => RawReadOnlyMemory.Slice(0, Length);

        public ReadOnlyMemory<T> RawReadOnlyMemory
        {
            get
            {
                ReadOnlyMemory<byte> memory = ReadOnlyBuffer;
                return Unsafe.As<ReadOnlyMemory<byte>, ReadOnlyMemory<T>>(ref memory);
            }
        }

        protected static int Size = Unsafe.SizeOf<T>();

        protected int Capacity => ReadOnlyBuffer.Length / Size;


        public static int MaxCapacity => Int32.MaxValue / Size;

        public ReadOnlySpan<T> ReadOnlySpan
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => (MemoryMarshal.Cast<byte, T>(ReadOnlyBuffer.Span)).Slice(0, Length);
        }

        public int Length { get; internal set; }

        public ReadOnlyDataFrameBuffer(int numberOfValues = 8)
        {
            if ((long)numberOfValues * Size > MaxCapacity)
            {
                throw new ArgumentException($"{numberOfValues} exceeds buffer capacity", nameof(numberOfValues));
            }
            _readOnlyBuffer = new byte[numberOfValues * Size];
        }

        public ReadOnlyDataFrameBuffer(ReadOnlyMemory<byte> buffer, int length)
        {
            _readOnlyBuffer = buffer;
            Length = length;
        }

        internal virtual T this[int index]
        {
            get
            {
                if (index > Length)
                    throw new ArgumentOutOfRangeException(nameof(index));
                return ReadOnlySpan[index];
            }
            set => throw new NotSupportedException();
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            ReadOnlySpan<T> span = ReadOnlySpan;
            for (int i = 0; i < Length; i++)
            {
                sb.Append(span[i]).Append(" ");
            }
            return sb.ToString();
        }

    }
}
