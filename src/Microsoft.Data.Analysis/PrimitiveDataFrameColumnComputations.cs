

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from PrimitiveColumnComputations.tt. Do not modify directly

using System;
using System.Collections.Generic;

namespace Microsoft.Data.Analysis
{
    internal interface IPrimitiveColumnComputation<T>
        where T : struct
    {
        void Abs(PrimitiveColumnContainer<T> column);
        void All(PrimitiveColumnContainer<T> column, out bool ret);
        void Any(PrimitiveColumnContainer<T> column, out bool ret);
        void CumulativeMax(PrimitiveColumnContainer<T> column);
        void CumulativeMax(PrimitiveColumnContainer<T> column, IEnumerable<long> rows);
        void CumulativeMin(PrimitiveColumnContainer<T> column);
        void CumulativeMin(PrimitiveColumnContainer<T> column, IEnumerable<long> rows);
        void CumulativeProduct(PrimitiveColumnContainer<T> column);
        void CumulativeProduct(PrimitiveColumnContainer<T> column, IEnumerable<long> rows);
        void CumulativeSum(PrimitiveColumnContainer<T> column);
        void CumulativeSum(PrimitiveColumnContainer<T> column, IEnumerable<long> rows);
        void Max(PrimitiveColumnContainer<T> column, out T ret);
        void Max(PrimitiveColumnContainer<T> column, IEnumerable<long> rows, out T ret);
        void Min(PrimitiveColumnContainer<T> column, out T ret);
        void Min(PrimitiveColumnContainer<T> column, IEnumerable<long> rows, out T ret);
        void Product(PrimitiveColumnContainer<T> column, out T ret);
        void Product(PrimitiveColumnContainer<T> column, IEnumerable<long> rows, out T ret);
        void Sum(PrimitiveColumnContainer<T> column, out T ret);
        void Sum(PrimitiveColumnContainer<T> column, IEnumerable<long> rows, out T ret);
        void Round(PrimitiveColumnContainer<T> column);
    }

    internal static class PrimitiveColumnComputation<T>
        where T : struct
    {
        public static IPrimitiveColumnComputation<T> Instance { get; } = PrimitiveColumnComputation.GetComputation<T>();
    }

    internal static class PrimitiveColumnComputation
    {
        public static IPrimitiveColumnComputation<T> GetComputation<T>()
            where T : struct
        {
            if (typeof(T) == typeof(bool))
            {
                return (IPrimitiveColumnComputation<T>)new BoolComputation();
            }
            else if (typeof(T) == typeof(byte))
            {
                return (IPrimitiveColumnComputation<T>)new ByteComputation();
            }
            else if (typeof(T) == typeof(char))
            {
                return (IPrimitiveColumnComputation<T>)new CharComputation();
            }
            else if (typeof(T) == typeof(decimal))
            {
                return (IPrimitiveColumnComputation<T>)new DecimalComputation();
            }
            else if (typeof(T) == typeof(double))
            {
                return (IPrimitiveColumnComputation<T>)new DoubleComputation();
            }
            else if (typeof(T) == typeof(float))
            {
                return (IPrimitiveColumnComputation<T>)new FloatComputation();
            }
            else if (typeof(T) == typeof(int))
            {
                return (IPrimitiveColumnComputation<T>)new IntComputation();
            }
            else if (typeof(T) == typeof(long))
            {
                return (IPrimitiveColumnComputation<T>)new LongComputation();
            }
            else if (typeof(T) == typeof(sbyte))
            {
                return (IPrimitiveColumnComputation<T>)new SByteComputation();
            }
            else if (typeof(T) == typeof(short))
            {
                return (IPrimitiveColumnComputation<T>)new ShortComputation();
            }
            else if (typeof(T) == typeof(uint))
            {
                return (IPrimitiveColumnComputation<T>)new UIntComputation();
            }
            else if (typeof(T) == typeof(ulong))
            {
                return (IPrimitiveColumnComputation<T>)new ULongComputation();
            }
            else if (typeof(T) == typeof(ushort))
            {
                return (IPrimitiveColumnComputation<T>)new UShortComputation();
            }
            throw new NotSupportedException();
        }
    }

    internal class BoolComputation : IPrimitiveColumnComputation<bool>
    {
        public void Abs(PrimitiveColumnContainer<bool> column)
        {
            throw new NotSupportedException();
        }

        public void All(PrimitiveColumnContainer<bool> column, out bool ret)
        {
            ret = true;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var span = buffer.ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    if (span[i] == false)
                    {
                        ret = false;
                        return;
                    }
                }
            }
        }

        public void Any(PrimitiveColumnContainer<bool> column, out bool ret)
        {
            ret = false;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var span = buffer.ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    if (span[i] == true)
                    {
                        ret = true;
                        return;
                    }
                }
            }
        }

        public void CumulativeMax(PrimitiveColumnContainer<bool> column)
        {
            throw new NotSupportedException();
        }

        public void CumulativeMax(PrimitiveColumnContainer<bool> column, IEnumerable<long> rows)
        {
            throw new NotSupportedException();
        }

        public void CumulativeMin(PrimitiveColumnContainer<bool> column)
        {
            throw new NotSupportedException();
        }

        public void CumulativeMin(PrimitiveColumnContainer<bool> column, IEnumerable<long> rows)
        {
            throw new NotSupportedException();
        }

        public void CumulativeProduct(PrimitiveColumnContainer<bool> column)
        {
            throw new NotSupportedException();
        }

        public void CumulativeProduct(PrimitiveColumnContainer<bool> column, IEnumerable<long> rows)
        {
            throw new NotSupportedException();
        }

        public void CumulativeSum(PrimitiveColumnContainer<bool> column)
        {
            throw new NotSupportedException();
        }

        public void CumulativeSum(PrimitiveColumnContainer<bool> column, IEnumerable<long> rows)
        {
            throw new NotSupportedException();
        }

        public void Max(PrimitiveColumnContainer<bool> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void Max(PrimitiveColumnContainer<bool> column, IEnumerable<long> rows, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void Min(PrimitiveColumnContainer<bool> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void Min(PrimitiveColumnContainer<bool> column, IEnumerable<long> rows, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void Product(PrimitiveColumnContainer<bool> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void Product(PrimitiveColumnContainer<bool> column, IEnumerable<long> rows, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void Sum(PrimitiveColumnContainer<bool> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void Sum(PrimitiveColumnContainer<bool> column, IEnumerable<long> rows, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void Round(PrimitiveColumnContainer<bool> column)
        {
            throw new NotSupportedException();
        }

    }
    internal class ByteComputation : IPrimitiveColumnComputation<byte>
    {
        public void Abs(PrimitiveColumnContainer<byte> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (byte)(Math.Abs((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void All(PrimitiveColumnContainer<byte> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void Any(PrimitiveColumnContainer<byte> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void CumulativeMax(PrimitiveColumnContainer<byte> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (byte)(Math.Max(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMax(PrimitiveColumnContainer<byte> column, IEnumerable<long> rows)
        {
            var ret = default(byte);
            var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<byte>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (byte)Math.Max(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<byte> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (byte)(Math.Min(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<byte> column, IEnumerable<long> rows)
        {
            var ret = default(byte);
            var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<byte>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (byte)Math.Min(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<byte> column)
        {
            var ret = (byte)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (byte)(readOnlySpan[i] * ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<byte> column, IEnumerable<long> rows)
        {
            var ret = default(byte);
            var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<byte>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((byte)((span[(int)row]) * ret));
                span[(int)row] = ret;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<byte> column)
        {
            var ret = (byte)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (byte)(readOnlySpan[i] + ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<byte> column, IEnumerable<long> rows)
        {
            var ret = default(byte);
            var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<byte>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((byte)((span[(int)row]) + ret));
                span[(int)row] = ret;
            }
        }

        public void Max(PrimitiveColumnContainer<byte> column, out byte ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (byte)(Math.Max(readOnlySpan[i], ret));
                }
            }
        }

        public void Max(PrimitiveColumnContainer<byte> column, IEnumerable<long> rows, out byte ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<byte>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (byte)(Math.Max(readOnlySpan[(int)row], ret));
            }
        }

        public void Min(PrimitiveColumnContainer<byte> column, out byte ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (byte)(Math.Min(readOnlySpan[i], ret));
                }
            }
        }

        public void Min(PrimitiveColumnContainer<byte> column, IEnumerable<long> rows, out byte ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<byte>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (byte)(Math.Min(readOnlySpan[(int)row], ret));
            }
        }

        public void Product(PrimitiveColumnContainer<byte> column, out byte ret)
        {
            ret = (byte)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (byte)(readOnlySpan[i] * ret);
                }
            }
        }

        public void Product(PrimitiveColumnContainer<byte> column, IEnumerable<long> rows, out byte ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<byte>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((byte)(readOnlySpan[(int)row] * ret));
            }
        }

        public void Sum(PrimitiveColumnContainer<byte> column, out byte ret)
        {
            ret = (byte)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (byte)(readOnlySpan[i] + ret);
                }
            }
        }

        public void Sum(PrimitiveColumnContainer<byte> column, IEnumerable<long> rows, out byte ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<byte>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((byte)(readOnlySpan[(int)row] + ret));
            }
        }

        public void Round(PrimitiveColumnContainer<byte> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (byte)(Math.Round((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

    }
    internal class CharComputation : IPrimitiveColumnComputation<char>
    {
        public void Abs(PrimitiveColumnContainer<char> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (char)(Math.Abs((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void All(PrimitiveColumnContainer<char> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void Any(PrimitiveColumnContainer<char> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void CumulativeMax(PrimitiveColumnContainer<char> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (char)(Math.Max(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMax(PrimitiveColumnContainer<char> column, IEnumerable<long> rows)
        {
            var ret = default(char);
            var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<char>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (char)Math.Max(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<char> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (char)(Math.Min(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<char> column, IEnumerable<long> rows)
        {
            var ret = default(char);
            var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<char>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (char)Math.Min(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<char> column)
        {
            var ret = (char)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (char)(readOnlySpan[i] * ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<char> column, IEnumerable<long> rows)
        {
            var ret = default(char);
            var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<char>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((char)((span[(int)row]) * ret));
                span[(int)row] = ret;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<char> column)
        {
            var ret = (char)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (char)(readOnlySpan[i] + ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<char> column, IEnumerable<long> rows)
        {
            var ret = default(char);
            var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<char>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((char)((span[(int)row]) + ret));
                span[(int)row] = ret;
            }
        }

        public void Max(PrimitiveColumnContainer<char> column, out char ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (char)(Math.Max(readOnlySpan[i], ret));
                }
            }
        }

        public void Max(PrimitiveColumnContainer<char> column, IEnumerable<long> rows, out char ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<char>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (char)(Math.Max(readOnlySpan[(int)row], ret));
            }
        }

        public void Min(PrimitiveColumnContainer<char> column, out char ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (char)(Math.Min(readOnlySpan[i], ret));
                }
            }
        }

        public void Min(PrimitiveColumnContainer<char> column, IEnumerable<long> rows, out char ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<char>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (char)(Math.Min(readOnlySpan[(int)row], ret));
            }
        }

        public void Product(PrimitiveColumnContainer<char> column, out char ret)
        {
            ret = (char)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (char)(readOnlySpan[i] * ret);
                }
            }
        }

        public void Product(PrimitiveColumnContainer<char> column, IEnumerable<long> rows, out char ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<char>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((char)(readOnlySpan[(int)row] * ret));
            }
        }

        public void Sum(PrimitiveColumnContainer<char> column, out char ret)
        {
            ret = (char)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (char)(readOnlySpan[i] + ret);
                }
            }
        }

        public void Sum(PrimitiveColumnContainer<char> column, IEnumerable<long> rows, out char ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<char>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((char)(readOnlySpan[(int)row] + ret));
            }
        }

        public void Round(PrimitiveColumnContainer<char> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (char)(Math.Round((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

    }
    internal class DecimalComputation : IPrimitiveColumnComputation<decimal>
    {
        public void Abs(PrimitiveColumnContainer<decimal> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (decimal)(Math.Abs((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void All(PrimitiveColumnContainer<decimal> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void Any(PrimitiveColumnContainer<decimal> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void CumulativeMax(PrimitiveColumnContainer<decimal> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (decimal)(Math.Max(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMax(PrimitiveColumnContainer<decimal> column, IEnumerable<long> rows)
        {
            var ret = default(decimal);
            var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<decimal>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (decimal)Math.Max(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<decimal> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (decimal)(Math.Min(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<decimal> column, IEnumerable<long> rows)
        {
            var ret = default(decimal);
            var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<decimal>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (decimal)Math.Min(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<decimal> column)
        {
            var ret = (decimal)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (decimal)(readOnlySpan[i] * ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<decimal> column, IEnumerable<long> rows)
        {
            var ret = default(decimal);
            var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<decimal>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((decimal)((span[(int)row]) * ret));
                span[(int)row] = ret;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<decimal> column)
        {
            var ret = (decimal)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (decimal)(readOnlySpan[i] + ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<decimal> column, IEnumerable<long> rows)
        {
            var ret = default(decimal);
            var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<decimal>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((decimal)((span[(int)row]) + ret));
                span[(int)row] = ret;
            }
        }

        public void Max(PrimitiveColumnContainer<decimal> column, out decimal ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (decimal)(Math.Max(readOnlySpan[i], ret));
                }
            }
        }

        public void Max(PrimitiveColumnContainer<decimal> column, IEnumerable<long> rows, out decimal ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<decimal>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (decimal)(Math.Max(readOnlySpan[(int)row], ret));
            }
        }

        public void Min(PrimitiveColumnContainer<decimal> column, out decimal ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (decimal)(Math.Min(readOnlySpan[i], ret));
                }
            }
        }

        public void Min(PrimitiveColumnContainer<decimal> column, IEnumerable<long> rows, out decimal ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<decimal>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (decimal)(Math.Min(readOnlySpan[(int)row], ret));
            }
        }

        public void Product(PrimitiveColumnContainer<decimal> column, out decimal ret)
        {
            ret = (decimal)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (decimal)(readOnlySpan[i] * ret);
                }
            }
        }

        public void Product(PrimitiveColumnContainer<decimal> column, IEnumerable<long> rows, out decimal ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<decimal>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((decimal)(readOnlySpan[(int)row] * ret));
            }
        }

        public void Sum(PrimitiveColumnContainer<decimal> column, out decimal ret)
        {
            ret = (decimal)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (decimal)(readOnlySpan[i] + ret);
                }
            }
        }

        public void Sum(PrimitiveColumnContainer<decimal> column, IEnumerable<long> rows, out decimal ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<decimal>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((decimal)(readOnlySpan[(int)row] + ret));
            }
        }

        public void Round(PrimitiveColumnContainer<decimal> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (decimal)(Math.Round((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

    }
    internal class DoubleComputation : IPrimitiveColumnComputation<double>
    {
        public void Abs(PrimitiveColumnContainer<double> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (double)(Math.Abs((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void All(PrimitiveColumnContainer<double> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void Any(PrimitiveColumnContainer<double> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void CumulativeMax(PrimitiveColumnContainer<double> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (double)(Math.Max(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMax(PrimitiveColumnContainer<double> column, IEnumerable<long> rows)
        {
            var ret = default(double);
            var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<double>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (double)Math.Max(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<double> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (double)(Math.Min(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<double> column, IEnumerable<long> rows)
        {
            var ret = default(double);
            var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<double>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (double)Math.Min(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<double> column)
        {
            var ret = (double)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (double)(readOnlySpan[i] * ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<double> column, IEnumerable<long> rows)
        {
            var ret = default(double);
            var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<double>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((double)((span[(int)row]) * ret));
                span[(int)row] = ret;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<double> column)
        {
            var ret = (double)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (double)(readOnlySpan[i] + ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<double> column, IEnumerable<long> rows)
        {
            var ret = default(double);
            var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<double>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((double)((span[(int)row]) + ret));
                span[(int)row] = ret;
            }
        }

        public void Max(PrimitiveColumnContainer<double> column, out double ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (double)(Math.Max(readOnlySpan[i], ret));
                }
            }
        }

        public void Max(PrimitiveColumnContainer<double> column, IEnumerable<long> rows, out double ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<double>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (double)(Math.Max(readOnlySpan[(int)row], ret));
            }
        }

        public void Min(PrimitiveColumnContainer<double> column, out double ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (double)(Math.Min(readOnlySpan[i], ret));
                }
            }
        }

        public void Min(PrimitiveColumnContainer<double> column, IEnumerable<long> rows, out double ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<double>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (double)(Math.Min(readOnlySpan[(int)row], ret));
            }
        }

        public void Product(PrimitiveColumnContainer<double> column, out double ret)
        {
            ret = (double)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (double)(readOnlySpan[i] * ret);
                }
            }
        }

        public void Product(PrimitiveColumnContainer<double> column, IEnumerable<long> rows, out double ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<double>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((double)(readOnlySpan[(int)row] * ret));
            }
        }

        public void Sum(PrimitiveColumnContainer<double> column, out double ret)
        {
            ret = (double)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (double)(readOnlySpan[i] + ret);
                }
            }
        }

        public void Sum(PrimitiveColumnContainer<double> column, IEnumerable<long> rows, out double ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<double>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((double)(readOnlySpan[(int)row] + ret));
            }
        }

        public void Round(PrimitiveColumnContainer<double> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (double)(Math.Round((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

    }
    internal class FloatComputation : IPrimitiveColumnComputation<float>
    {
        public void Abs(PrimitiveColumnContainer<float> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (float)(Math.Abs((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void All(PrimitiveColumnContainer<float> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void Any(PrimitiveColumnContainer<float> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void CumulativeMax(PrimitiveColumnContainer<float> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (float)(Math.Max(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMax(PrimitiveColumnContainer<float> column, IEnumerable<long> rows)
        {
            var ret = default(float);
            var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<float>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (float)Math.Max(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<float> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (float)(Math.Min(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<float> column, IEnumerable<long> rows)
        {
            var ret = default(float);
            var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<float>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (float)Math.Min(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<float> column)
        {
            var ret = (float)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (float)(readOnlySpan[i] * ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<float> column, IEnumerable<long> rows)
        {
            var ret = default(float);
            var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<float>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((float)((span[(int)row]) * ret));
                span[(int)row] = ret;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<float> column)
        {
            var ret = (float)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (float)(readOnlySpan[i] + ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<float> column, IEnumerable<long> rows)
        {
            var ret = default(float);
            var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<float>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((float)((span[(int)row]) + ret));
                span[(int)row] = ret;
            }
        }

        public void Max(PrimitiveColumnContainer<float> column, out float ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (float)(Math.Max(readOnlySpan[i], ret));
                }
            }
        }

        public void Max(PrimitiveColumnContainer<float> column, IEnumerable<long> rows, out float ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<float>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (float)(Math.Max(readOnlySpan[(int)row], ret));
            }
        }

        public void Min(PrimitiveColumnContainer<float> column, out float ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (float)(Math.Min(readOnlySpan[i], ret));
                }
            }
        }

        public void Min(PrimitiveColumnContainer<float> column, IEnumerable<long> rows, out float ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<float>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (float)(Math.Min(readOnlySpan[(int)row], ret));
            }
        }

        public void Product(PrimitiveColumnContainer<float> column, out float ret)
        {
            ret = (float)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (float)(readOnlySpan[i] * ret);
                }
            }
        }

        public void Product(PrimitiveColumnContainer<float> column, IEnumerable<long> rows, out float ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<float>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((float)(readOnlySpan[(int)row] * ret));
            }
        }

        public void Sum(PrimitiveColumnContainer<float> column, out float ret)
        {
            ret = (float)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (float)(readOnlySpan[i] + ret);
                }
            }
        }

        public void Sum(PrimitiveColumnContainer<float> column, IEnumerable<long> rows, out float ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<float>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((float)(readOnlySpan[(int)row] + ret));
            }
        }

        public void Round(PrimitiveColumnContainer<float> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (float)(Math.Round((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

    }
    internal class IntComputation : IPrimitiveColumnComputation<int>
    {
        public void Abs(PrimitiveColumnContainer<int> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (int)(Math.Abs((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void All(PrimitiveColumnContainer<int> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void Any(PrimitiveColumnContainer<int> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void CumulativeMax(PrimitiveColumnContainer<int> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (int)(Math.Max(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMax(PrimitiveColumnContainer<int> column, IEnumerable<long> rows)
        {
            var ret = default(int);
            var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<int>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (int)Math.Max(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<int> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (int)(Math.Min(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<int> column, IEnumerable<long> rows)
        {
            var ret = default(int);
            var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<int>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (int)Math.Min(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<int> column)
        {
            var ret = (int)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (int)(readOnlySpan[i] * ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<int> column, IEnumerable<long> rows)
        {
            var ret = default(int);
            var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<int>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((int)((span[(int)row]) * ret));
                span[(int)row] = ret;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<int> column)
        {
            var ret = (int)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (int)(readOnlySpan[i] + ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<int> column, IEnumerable<long> rows)
        {
            var ret = default(int);
            var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<int>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((int)((span[(int)row]) + ret));
                span[(int)row] = ret;
            }
        }

        public void Max(PrimitiveColumnContainer<int> column, out int ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (int)(Math.Max(readOnlySpan[i], ret));
                }
            }
        }

        public void Max(PrimitiveColumnContainer<int> column, IEnumerable<long> rows, out int ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<int>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (int)(Math.Max(readOnlySpan[(int)row], ret));
            }
        }

        public void Min(PrimitiveColumnContainer<int> column, out int ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (int)(Math.Min(readOnlySpan[i], ret));
                }
            }
        }

        public void Min(PrimitiveColumnContainer<int> column, IEnumerable<long> rows, out int ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<int>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (int)(Math.Min(readOnlySpan[(int)row], ret));
            }
        }

        public void Product(PrimitiveColumnContainer<int> column, out int ret)
        {
            ret = (int)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (int)(readOnlySpan[i] * ret);
                }
            }
        }

        public void Product(PrimitiveColumnContainer<int> column, IEnumerable<long> rows, out int ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<int>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((int)(readOnlySpan[(int)row] * ret));
            }
        }

        public void Sum(PrimitiveColumnContainer<int> column, out int ret)
        {
            ret = (int)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (int)(readOnlySpan[i] + ret);
                }
            }
        }

        public void Sum(PrimitiveColumnContainer<int> column, IEnumerable<long> rows, out int ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<int>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((int)(readOnlySpan[(int)row] + ret));
            }
        }

        public void Round(PrimitiveColumnContainer<int> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (int)(Math.Round((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

    }
    internal class LongComputation : IPrimitiveColumnComputation<long>
    {
        public void Abs(PrimitiveColumnContainer<long> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (long)(Math.Abs((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void All(PrimitiveColumnContainer<long> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void Any(PrimitiveColumnContainer<long> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void CumulativeMax(PrimitiveColumnContainer<long> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (long)(Math.Max(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMax(PrimitiveColumnContainer<long> column, IEnumerable<long> rows)
        {
            var ret = default(long);
            var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<long>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (long)Math.Max(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<long> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (long)(Math.Min(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<long> column, IEnumerable<long> rows)
        {
            var ret = default(long);
            var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<long>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (long)Math.Min(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<long> column)
        {
            var ret = (long)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (long)(readOnlySpan[i] * ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<long> column, IEnumerable<long> rows)
        {
            var ret = default(long);
            var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<long>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((long)((span[(int)row]) * ret));
                span[(int)row] = ret;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<long> column)
        {
            var ret = (long)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (long)(readOnlySpan[i] + ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<long> column, IEnumerable<long> rows)
        {
            var ret = default(long);
            var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<long>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((long)((span[(int)row]) + ret));
                span[(int)row] = ret;
            }
        }

        public void Max(PrimitiveColumnContainer<long> column, out long ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (long)(Math.Max(readOnlySpan[i], ret));
                }
            }
        }

        public void Max(PrimitiveColumnContainer<long> column, IEnumerable<long> rows, out long ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<long>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (long)(Math.Max(readOnlySpan[(int)row], ret));
            }
        }

        public void Min(PrimitiveColumnContainer<long> column, out long ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (long)(Math.Min(readOnlySpan[i], ret));
                }
            }
        }

        public void Min(PrimitiveColumnContainer<long> column, IEnumerable<long> rows, out long ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<long>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (long)(Math.Min(readOnlySpan[(int)row], ret));
            }
        }

        public void Product(PrimitiveColumnContainer<long> column, out long ret)
        {
            ret = (long)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (long)(readOnlySpan[i] * ret);
                }
            }
        }

        public void Product(PrimitiveColumnContainer<long> column, IEnumerable<long> rows, out long ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<long>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((long)(readOnlySpan[(int)row] * ret));
            }
        }

        public void Sum(PrimitiveColumnContainer<long> column, out long ret)
        {
            ret = (long)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (long)(readOnlySpan[i] + ret);
                }
            }
        }

        public void Sum(PrimitiveColumnContainer<long> column, IEnumerable<long> rows, out long ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<long>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((long)(readOnlySpan[(int)row] + ret));
            }
        }

        public void Round(PrimitiveColumnContainer<long> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (long)(Math.Round((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

    }
    internal class SByteComputation : IPrimitiveColumnComputation<sbyte>
    {
        public void Abs(PrimitiveColumnContainer<sbyte> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (sbyte)(Math.Abs((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void All(PrimitiveColumnContainer<sbyte> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void Any(PrimitiveColumnContainer<sbyte> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void CumulativeMax(PrimitiveColumnContainer<sbyte> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (sbyte)(Math.Max(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMax(PrimitiveColumnContainer<sbyte> column, IEnumerable<long> rows)
        {
            var ret = default(sbyte);
            var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<sbyte>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (sbyte)Math.Max(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<sbyte> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (sbyte)(Math.Min(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<sbyte> column, IEnumerable<long> rows)
        {
            var ret = default(sbyte);
            var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<sbyte>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (sbyte)Math.Min(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<sbyte> column)
        {
            var ret = (sbyte)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (sbyte)(readOnlySpan[i] * ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<sbyte> column, IEnumerable<long> rows)
        {
            var ret = default(sbyte);
            var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<sbyte>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((sbyte)((span[(int)row]) * ret));
                span[(int)row] = ret;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<sbyte> column)
        {
            var ret = (sbyte)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (sbyte)(readOnlySpan[i] + ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<sbyte> column, IEnumerable<long> rows)
        {
            var ret = default(sbyte);
            var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<sbyte>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((sbyte)((span[(int)row]) + ret));
                span[(int)row] = ret;
            }
        }

        public void Max(PrimitiveColumnContainer<sbyte> column, out sbyte ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (sbyte)(Math.Max(readOnlySpan[i], ret));
                }
            }
        }

        public void Max(PrimitiveColumnContainer<sbyte> column, IEnumerable<long> rows, out sbyte ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<sbyte>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (sbyte)(Math.Max(readOnlySpan[(int)row], ret));
            }
        }

        public void Min(PrimitiveColumnContainer<sbyte> column, out sbyte ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (sbyte)(Math.Min(readOnlySpan[i], ret));
                }
            }
        }

        public void Min(PrimitiveColumnContainer<sbyte> column, IEnumerable<long> rows, out sbyte ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<sbyte>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (sbyte)(Math.Min(readOnlySpan[(int)row], ret));
            }
        }

        public void Product(PrimitiveColumnContainer<sbyte> column, out sbyte ret)
        {
            ret = (sbyte)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (sbyte)(readOnlySpan[i] * ret);
                }
            }
        }

        public void Product(PrimitiveColumnContainer<sbyte> column, IEnumerable<long> rows, out sbyte ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<sbyte>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((sbyte)(readOnlySpan[(int)row] * ret));
            }
        }

        public void Sum(PrimitiveColumnContainer<sbyte> column, out sbyte ret)
        {
            ret = (sbyte)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (sbyte)(readOnlySpan[i] + ret);
                }
            }
        }

        public void Sum(PrimitiveColumnContainer<sbyte> column, IEnumerable<long> rows, out sbyte ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<sbyte>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((sbyte)(readOnlySpan[(int)row] + ret));
            }
        }

        public void Round(PrimitiveColumnContainer<sbyte> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (sbyte)(Math.Round((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

    }
    internal class ShortComputation : IPrimitiveColumnComputation<short>
    {
        public void Abs(PrimitiveColumnContainer<short> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (short)(Math.Abs((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void All(PrimitiveColumnContainer<short> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void Any(PrimitiveColumnContainer<short> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void CumulativeMax(PrimitiveColumnContainer<short> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (short)(Math.Max(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMax(PrimitiveColumnContainer<short> column, IEnumerable<long> rows)
        {
            var ret = default(short);
            var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<short>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (short)Math.Max(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<short> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (short)(Math.Min(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<short> column, IEnumerable<long> rows)
        {
            var ret = default(short);
            var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<short>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (short)Math.Min(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<short> column)
        {
            var ret = (short)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (short)(readOnlySpan[i] * ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<short> column, IEnumerable<long> rows)
        {
            var ret = default(short);
            var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<short>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((short)((span[(int)row]) * ret));
                span[(int)row] = ret;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<short> column)
        {
            var ret = (short)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (short)(readOnlySpan[i] + ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<short> column, IEnumerable<long> rows)
        {
            var ret = default(short);
            var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<short>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((short)((span[(int)row]) + ret));
                span[(int)row] = ret;
            }
        }

        public void Max(PrimitiveColumnContainer<short> column, out short ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (short)(Math.Max(readOnlySpan[i], ret));
                }
            }
        }

        public void Max(PrimitiveColumnContainer<short> column, IEnumerable<long> rows, out short ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<short>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (short)(Math.Max(readOnlySpan[(int)row], ret));
            }
        }

        public void Min(PrimitiveColumnContainer<short> column, out short ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (short)(Math.Min(readOnlySpan[i], ret));
                }
            }
        }

        public void Min(PrimitiveColumnContainer<short> column, IEnumerable<long> rows, out short ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<short>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (short)(Math.Min(readOnlySpan[(int)row], ret));
            }
        }

        public void Product(PrimitiveColumnContainer<short> column, out short ret)
        {
            ret = (short)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (short)(readOnlySpan[i] * ret);
                }
            }
        }

        public void Product(PrimitiveColumnContainer<short> column, IEnumerable<long> rows, out short ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<short>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((short)(readOnlySpan[(int)row] * ret));
            }
        }

        public void Sum(PrimitiveColumnContainer<short> column, out short ret)
        {
            ret = (short)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (short)(readOnlySpan[i] + ret);
                }
            }
        }

        public void Sum(PrimitiveColumnContainer<short> column, IEnumerable<long> rows, out short ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<short>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((short)(readOnlySpan[(int)row] + ret));
            }
        }

        public void Round(PrimitiveColumnContainer<short> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (short)(Math.Round((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

    }
    internal class UIntComputation : IPrimitiveColumnComputation<uint>
    {
        public void Abs(PrimitiveColumnContainer<uint> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (uint)(Math.Abs((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void All(PrimitiveColumnContainer<uint> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void Any(PrimitiveColumnContainer<uint> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void CumulativeMax(PrimitiveColumnContainer<uint> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (uint)(Math.Max(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMax(PrimitiveColumnContainer<uint> column, IEnumerable<long> rows)
        {
            var ret = default(uint);
            var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<uint>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (uint)Math.Max(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<uint> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (uint)(Math.Min(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<uint> column, IEnumerable<long> rows)
        {
            var ret = default(uint);
            var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<uint>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (uint)Math.Min(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<uint> column)
        {
            var ret = (uint)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (uint)(readOnlySpan[i] * ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<uint> column, IEnumerable<long> rows)
        {
            var ret = default(uint);
            var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<uint>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((uint)((span[(int)row]) * ret));
                span[(int)row] = ret;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<uint> column)
        {
            var ret = (uint)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (uint)(readOnlySpan[i] + ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<uint> column, IEnumerable<long> rows)
        {
            var ret = default(uint);
            var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<uint>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((uint)((span[(int)row]) + ret));
                span[(int)row] = ret;
            }
        }

        public void Max(PrimitiveColumnContainer<uint> column, out uint ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (uint)(Math.Max(readOnlySpan[i], ret));
                }
            }
        }

        public void Max(PrimitiveColumnContainer<uint> column, IEnumerable<long> rows, out uint ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<uint>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (uint)(Math.Max(readOnlySpan[(int)row], ret));
            }
        }

        public void Min(PrimitiveColumnContainer<uint> column, out uint ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (uint)(Math.Min(readOnlySpan[i], ret));
                }
            }
        }

        public void Min(PrimitiveColumnContainer<uint> column, IEnumerable<long> rows, out uint ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<uint>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (uint)(Math.Min(readOnlySpan[(int)row], ret));
            }
        }

        public void Product(PrimitiveColumnContainer<uint> column, out uint ret)
        {
            ret = (uint)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (uint)(readOnlySpan[i] * ret);
                }
            }
        }

        public void Product(PrimitiveColumnContainer<uint> column, IEnumerable<long> rows, out uint ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<uint>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((uint)(readOnlySpan[(int)row] * ret));
            }
        }

        public void Sum(PrimitiveColumnContainer<uint> column, out uint ret)
        {
            ret = (uint)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (uint)(readOnlySpan[i] + ret);
                }
            }
        }

        public void Sum(PrimitiveColumnContainer<uint> column, IEnumerable<long> rows, out uint ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<uint>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((uint)(readOnlySpan[(int)row] + ret));
            }
        }

        public void Round(PrimitiveColumnContainer<uint> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (uint)(Math.Round((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

    }
    internal class ULongComputation : IPrimitiveColumnComputation<ulong>
    {
        public void Abs(PrimitiveColumnContainer<ulong> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (ulong)(Math.Abs((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void All(PrimitiveColumnContainer<ulong> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void Any(PrimitiveColumnContainer<ulong> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void CumulativeMax(PrimitiveColumnContainer<ulong> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (ulong)(Math.Max(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMax(PrimitiveColumnContainer<ulong> column, IEnumerable<long> rows)
        {
            var ret = default(ulong);
            var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<ulong>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (ulong)Math.Max(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<ulong> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (ulong)(Math.Min(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<ulong> column, IEnumerable<long> rows)
        {
            var ret = default(ulong);
            var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<ulong>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (ulong)Math.Min(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<ulong> column)
        {
            var ret = (ulong)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (ulong)(readOnlySpan[i] * ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<ulong> column, IEnumerable<long> rows)
        {
            var ret = default(ulong);
            var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<ulong>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((ulong)((span[(int)row]) * ret));
                span[(int)row] = ret;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<ulong> column)
        {
            var ret = (ulong)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (ulong)(readOnlySpan[i] + ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<ulong> column, IEnumerable<long> rows)
        {
            var ret = default(ulong);
            var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<ulong>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((ulong)((span[(int)row]) + ret));
                span[(int)row] = ret;
            }
        }

        public void Max(PrimitiveColumnContainer<ulong> column, out ulong ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (ulong)(Math.Max(readOnlySpan[i], ret));
                }
            }
        }

        public void Max(PrimitiveColumnContainer<ulong> column, IEnumerable<long> rows, out ulong ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<ulong>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (ulong)(Math.Max(readOnlySpan[(int)row], ret));
            }
        }

        public void Min(PrimitiveColumnContainer<ulong> column, out ulong ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (ulong)(Math.Min(readOnlySpan[i], ret));
                }
            }
        }

        public void Min(PrimitiveColumnContainer<ulong> column, IEnumerable<long> rows, out ulong ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<ulong>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (ulong)(Math.Min(readOnlySpan[(int)row], ret));
            }
        }

        public void Product(PrimitiveColumnContainer<ulong> column, out ulong ret)
        {
            ret = (ulong)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (ulong)(readOnlySpan[i] * ret);
                }
            }
        }

        public void Product(PrimitiveColumnContainer<ulong> column, IEnumerable<long> rows, out ulong ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<ulong>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((ulong)(readOnlySpan[(int)row] * ret));
            }
        }

        public void Sum(PrimitiveColumnContainer<ulong> column, out ulong ret)
        {
            ret = (ulong)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (ulong)(readOnlySpan[i] + ret);
                }
            }
        }

        public void Sum(PrimitiveColumnContainer<ulong> column, IEnumerable<long> rows, out ulong ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<ulong>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((ulong)(readOnlySpan[(int)row] + ret));
            }
        }

        public void Round(PrimitiveColumnContainer<ulong> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (ulong)(Math.Round((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

    }
    internal class UShortComputation : IPrimitiveColumnComputation<ushort>
    {
        public void Abs(PrimitiveColumnContainer<ushort> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (ushort)(Math.Abs((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void All(PrimitiveColumnContainer<ushort> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void Any(PrimitiveColumnContainer<ushort> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void CumulativeMax(PrimitiveColumnContainer<ushort> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (ushort)(Math.Max(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMax(PrimitiveColumnContainer<ushort> column, IEnumerable<long> rows)
        {
            var ret = default(ushort);
            var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<ushort>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (ushort)Math.Max(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<ushort> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (ushort)(Math.Min(readOnlySpan[i], ret));
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<ushort> column, IEnumerable<long> rows)
        {
            var ret = default(ushort);
            var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<ushort>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (ushort)Math.Min(span[(int)row], ret);
                span[(int)row] = ret;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<ushort> column)
        {
            var ret = (ushort)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (ushort)(readOnlySpan[i] * ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<ushort> column, IEnumerable<long> rows)
        {
            var ret = default(ushort);
            var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<ushort>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((ushort)((span[(int)row]) * ret));
                span[(int)row] = ret;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<ushort> column)
        {
            var ret = (ushort)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (ushort)(readOnlySpan[i] + ret);
                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeSum(PrimitiveColumnContainer<ushort> column, IEnumerable<long> rows)
        {
            var ret = default(ushort);
            var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<ushort>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = span[(int)row];
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((ushort)((span[(int)row]) + ret));
                span[(int)row] = ret;
            }
        }

        public void Max(PrimitiveColumnContainer<ushort> column, out ushort ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (ushort)(Math.Max(readOnlySpan[i], ret));
                }
            }
        }

        public void Max(PrimitiveColumnContainer<ushort> column, IEnumerable<long> rows, out ushort ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<ushort>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (ushort)(Math.Max(readOnlySpan[(int)row], ret));
            }
        }

        public void Min(PrimitiveColumnContainer<ushort> column, out ushort ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (ushort)(Math.Min(readOnlySpan[i], ret));
                }
            }
        }

        public void Min(PrimitiveColumnContainer<ushort> column, IEnumerable<long> rows, out ushort ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<ushort>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = (ushort)(Math.Min(readOnlySpan[(int)row], ret));
            }
        }

        public void Product(PrimitiveColumnContainer<ushort> column, out ushort ret)
        {
            ret = (ushort)1;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (ushort)(readOnlySpan[i] * ret);
                }
            }
        }

        public void Product(PrimitiveColumnContainer<ushort> column, IEnumerable<long> rows, out ushort ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<ushort>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((ushort)(readOnlySpan[(int)row] * ret));
            }
        }

        public void Sum(PrimitiveColumnContainer<ushort> column, out ushort ret)
        {
            ret = (ushort)0;
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    ret = (ushort)(readOnlySpan[i] + ret);
                }
            }
        }

        public void Sum(PrimitiveColumnContainer<ushort> column, IEnumerable<long> rows, out ushort ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<ushort>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;
                ret = checked((ushort)(readOnlySpan[(int)row] + ret));
            }
        }

        public void Round(PrimitiveColumnContainer<ushort> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    mutableSpan[i] = (ushort)(Math.Round((decimal)mutableSpan[i]));
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

    }
}
