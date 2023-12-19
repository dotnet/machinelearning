// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
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

        public void Max(PrimitiveColumnContainer<DateTime> column, out DateTime? ret)
        {
            var maxDate = DateTime.MinValue;
            bool hasMaxValue = false;

            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var readOnlySpan = column.Buffers[b].ReadOnlySpan;
                var bitmapSpan = column.NullBitMapBuffers[b].ReadOnlySpan;
                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    //Check if bit is not set (value is null) - skip
                    if (!BitUtility.IsValid(bitmapSpan, i))
                        continue;

                    var val = readOnlySpan[i];

                    if (val > maxDate)
                    {
                        maxDate = val;
                        hasMaxValue = true;
                    }
                }
            }

            ret = hasMaxValue ? maxDate : null;
        }

        public void Max(PrimitiveColumnContainer<DateTime> column, IEnumerable<long> rows, out DateTime? ret)
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

        public void Min(PrimitiveColumnContainer<DateTime> column, out DateTime? ret)
        {
            var minDate = DateTime.MaxValue;
            bool hasMinValue = false;

            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var readOnlySpan = column.Buffers[b].ReadOnlySpan;
                var bitmapSpan = column.NullBitMapBuffers[b].ReadOnlySpan;

                for (int i = 0; i < readOnlySpan.Length; i++)
                {
                    //Check if bit is not set (value is null) - skip
                    if (!BitUtility.IsValid(bitmapSpan, i))
                        continue;

                    var val = readOnlySpan[i];

                    if (val < minDate)
                    {
                        minDate = val;
                        hasMinValue = true;
                    }
                }
            }

            ret = hasMinValue ? minDate : null;
        }

        public void Min(PrimitiveColumnContainer<DateTime> column, IEnumerable<long> rows, out DateTime? ret)
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

        public void Product(PrimitiveColumnContainer<DateTime> column, out DateTime? ret)
        {
            throw new NotSupportedException();
        }

        public void Product(PrimitiveColumnContainer<DateTime> column, IEnumerable<long> rows, out DateTime? ret)
        {
            throw new NotSupportedException();
        }

        public void Sum(PrimitiveColumnContainer<DateTime> column, out DateTime? ret)
        {
            throw new NotSupportedException();
        }

        public void Sum(PrimitiveColumnContainer<DateTime> column, IEnumerable<long> rows, out DateTime? ret)
        {
            throw new NotSupportedException();
        }

        public void Round(PrimitiveColumnContainer<DateTime> column)
        {
            throw new NotSupportedException();
        }

    }
}
