// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.Data.Analysis
{
    internal class DateTimeComputation : IPrimitiveColumnComputation<DateTime>
    {
        public void Abs(PrimitiveColumnContainer<DateTime> column)
        {
            throw new NotSupportedException();
        }

        public void All(PrimitiveColumnContainer<DateTime> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void Any(PrimitiveColumnContainer<DateTime> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void CumulativeMax(PrimitiveColumnContainer<DateTime> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<DateTime>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    var val = readOnlySpan[i];

                    if (val > ret)
                    {
                        ret = val;
                    }

                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMax(PrimitiveColumnContainer<DateTime> column, IEnumerable<long> rows)
        {
            var ret = default(DateTime);
            var mutableBuffer = DataFrameBuffer<DateTime>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<DateTime>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<DateTime>.GetMutableBuffer(column.Buffers[bufferIndex]);
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
                    mutableBuffer = DataFrameBuffer<DateTime>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;

                var val = span[(int)row];

                if (val > ret)
                {
                    ret = val;
                }

                span[(int)row] = ret;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<DateTime> column)
        {
            var ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<DateTime>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    var val = readOnlySpan[i];

                    if (val < ret)
                    {
                        ret = val;
                    }

                    mutableSpan[i] = ret;
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        public void CumulativeMin(PrimitiveColumnContainer<DateTime> column, IEnumerable<long> rows)
        {
            var ret = default(DateTime);
            var mutableBuffer = DataFrameBuffer<DateTime>.GetMutableBuffer(column.Buffers[0]);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<DateTime>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();
            if (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = DataFrameBuffer<DateTime>.GetMutableBuffer(column.Buffers[bufferIndex]);
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
                    mutableBuffer = DataFrameBuffer<DateTime>.GetMutableBuffer(column.Buffers[bufferIndex]);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;

                var val = span[(int)row];

                if (val < ret)
                {
                    ret = val;
                }

                span[(int)row] = ret;
            }
        }

        public void CumulativeProduct(PrimitiveColumnContainer<DateTime> column)
        {
            throw new NotSupportedException();
        }

        public void CumulativeProduct(PrimitiveColumnContainer<DateTime> column, IEnumerable<long> rows)
        {
            throw new NotSupportedException();
        }

        public void CumulativeSum(PrimitiveColumnContainer<DateTime> column)
        {
            throw new NotSupportedException();
        }

        public void CumulativeSum(PrimitiveColumnContainer<DateTime> column, IEnumerable<long> rows)
        {
            throw new NotSupportedException();
        }

        public void Max(PrimitiveColumnContainer<DateTime> column, out DateTime ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    var val = readOnlySpan[i];

                    if (val > ret)
                    {
                        ret = val;
                    }
                }
            }
        }

        public void Max(PrimitiveColumnContainer<DateTime> column, IEnumerable<long> rows, out DateTime ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<DateTime>.MaxCapacity;
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

                var val = readOnlySpan[(int)row];

                if (val > ret)
                {
                    ret = val;
                }
            }
        }

        public void Min(PrimitiveColumnContainer<DateTime> column, out DateTime ret)
        {
            ret = column.Buffers[0].ReadOnlySpan[0];
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var readOnlySpan = buffer.ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    var val = readOnlySpan[i];

                    if (val < ret)
                    {
                        ret = val;
                    }
                }
            }
        }

        public void Min(PrimitiveColumnContainer<DateTime> column, IEnumerable<long> rows, out DateTime ret)
        {
            ret = default;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<DateTime>.MaxCapacity;
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

                var val = readOnlySpan[(int)row];

                if (val < ret)
                {
                    ret = val;
                }
            }
        }

        public void Product(PrimitiveColumnContainer<DateTime> column, out DateTime ret)
        {
            throw new NotSupportedException();
        }

        public void Product(PrimitiveColumnContainer<DateTime> column, IEnumerable<long> rows, out DateTime ret)
        {
            throw new NotSupportedException();
        }

        public void Sum(PrimitiveColumnContainer<DateTime> column, out DateTime ret)
        {
            throw new NotSupportedException();
        }

        public void Sum(PrimitiveColumnContainer<DateTime> column, IEnumerable<long> rows, out DateTime ret)
        {
            throw new NotSupportedException();
        }

        public void Round(PrimitiveColumnContainer<DateTime> column)
        {
            throw new NotSupportedException();
        }

    }

    internal class DateTimeArithmetic : IPrimitiveDataFrameColumnArithmetic<DateTime>
    {
        public void Add(PrimitiveColumnContainer<DateTime> left, PrimitiveColumnContainer<DateTime> right)
        {
            throw new NotSupportedException();
        }
        public void Add(PrimitiveColumnContainer<DateTime> column, DateTime scalar)
        {
            throw new NotSupportedException();
        }

        public void Add(DateTime scalar, PrimitiveColumnContainer<DateTime> column)
        {
            throw new NotSupportedException();
        }
        public void Subtract(PrimitiveColumnContainer<DateTime> left, PrimitiveColumnContainer<DateTime> right)
        {
            throw new NotSupportedException();
        }
        public void Subtract(PrimitiveColumnContainer<DateTime> column, DateTime scalar)
        {
            throw new NotSupportedException();
        }

        public void Subtract(DateTime scalar, PrimitiveColumnContainer<DateTime> column)
        {
            throw new NotSupportedException();
        }
        public void Multiply(PrimitiveColumnContainer<DateTime> left, PrimitiveColumnContainer<DateTime> right)
        {
            throw new NotSupportedException();
        }
        public void Multiply(PrimitiveColumnContainer<DateTime> column, DateTime scalar)
        {
            throw new NotSupportedException();
        }

        public void Multiply(DateTime scalar, PrimitiveColumnContainer<DateTime> column)
        {
            throw new NotSupportedException();
        }
        public void Divide(PrimitiveColumnContainer<DateTime> left, PrimitiveColumnContainer<DateTime> right)
        {
            throw new NotSupportedException();
        }
        public void Divide(PrimitiveColumnContainer<DateTime> column, DateTime scalar)
        {
            throw new NotSupportedException();
        }

        public void Divide(DateTime scalar, PrimitiveColumnContainer<DateTime> column)
        {
            throw new NotSupportedException();
        }
        public void Modulo(PrimitiveColumnContainer<DateTime> left, PrimitiveColumnContainer<DateTime> right)
        {
            throw new NotSupportedException();
        }
        public void Modulo(PrimitiveColumnContainer<DateTime> column, DateTime scalar)
        {
            throw new NotSupportedException();
        }

        public void Modulo(DateTime scalar, PrimitiveColumnContainer<DateTime> column)
        {
            throw new NotSupportedException();
        }
        public void And(PrimitiveColumnContainer<DateTime> left, PrimitiveColumnContainer<DateTime> right)
        {
            throw new NotSupportedException();
        }
        public void And(PrimitiveColumnContainer<DateTime> column, DateTime scalar)
        {
            throw new NotSupportedException();
        }

        public void And(DateTime scalar, PrimitiveColumnContainer<DateTime> column)
        {
            throw new NotSupportedException();
        }
        public void Or(PrimitiveColumnContainer<DateTime> left, PrimitiveColumnContainer<DateTime> right)
        {
            throw new NotSupportedException();
        }
        public void Or(PrimitiveColumnContainer<DateTime> column, DateTime scalar)
        {
            throw new NotSupportedException();
        }

        public void Or(DateTime scalar, PrimitiveColumnContainer<DateTime> column)
        {
            throw new NotSupportedException();
        }
        public void Xor(PrimitiveColumnContainer<DateTime> left, PrimitiveColumnContainer<DateTime> right)
        {
            throw new NotSupportedException();
        }
        public void Xor(PrimitiveColumnContainer<DateTime> column, DateTime scalar)
        {
            throw new NotSupportedException();
        }

        public void Xor(DateTime scalar, PrimitiveColumnContainer<DateTime> column)
        {
            throw new NotSupportedException();
        }

        public void LeftShift(PrimitiveColumnContainer<DateTime> column, int value)
        {
            throw new NotSupportedException();
        }
        public void RightShift(PrimitiveColumnContainer<DateTime> column, int value)
        {
            throw new NotSupportedException();
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<DateTime> left, PrimitiveColumnContainer<DateTime> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<DateTime>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == otherSpan[i]);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<DateTime> column, DateTime scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<DateTime>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == scalar);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<DateTime> left, PrimitiveColumnContainer<DateTime> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<DateTime>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != otherSpan[i]);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<DateTime> column, DateTime scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<DateTime>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != scalar);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<DateTime> left, PrimitiveColumnContainer<DateTime> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<DateTime>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<DateTime> column, DateTime scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<DateTime>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= scalar);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<DateTime> left, PrimitiveColumnContainer<DateTime> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<DateTime>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<DateTime> column, DateTime scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<DateTime>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= scalar);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<DateTime> left, PrimitiveColumnContainer<DateTime> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<DateTime>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<DateTime> column, DateTime scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<DateTime>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > scalar);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<DateTime> left, PrimitiveColumnContainer<DateTime> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<DateTime>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<DateTime> column, DateTime scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<DateTime>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < scalar);
                }
            }
        }
    }
}
