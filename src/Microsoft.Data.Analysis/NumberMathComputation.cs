

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from PrimitiveColumnComputations.tt. Do not modify directly

using System;
using System.Collections.Generic;
using System.Runtime.Versioning;

namespace Microsoft.Data.Analysis
{
    [RequiresPreviewFeatures]
    internal class NumberMathComputation<T> : IPrimitiveColumnComputation<T>
        where T : unmanaged, INumber<T>
    {
        public void Abs(PrimitiveColumnContainer<T> column)
        {
            Apply(column, T.Abs);
        }

        public void All(PrimitiveColumnContainer<T> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void Any(PrimitiveColumnContainer<T> column, out bool ret)
        {
            throw new NotSupportedException();
        }

        public void CumulativeMax(PrimitiveColumnContainer<T> column)
        {
            CumulativeApply(column, T.Max);
        }

        public void CumulativeMax(PrimitiveColumnContainer<T> column, IEnumerable<long> rows)
        {
            CumulativeApply(column, T.Max, rows);
        }

        public void CumulativeMin(PrimitiveColumnContainer<T> column)
        {
            CumulativeApply(column, T.Min);
        }

        public void CumulativeMin(PrimitiveColumnContainer<T> column, IEnumerable<long> rows)
        {
            CumulativeApply(column, T.Min, rows);
        }

        private T Multiply(T left, T right) => left * right;

        public void CumulativeProduct(PrimitiveColumnContainer<T> column)
        {
            CumulativeApply(column, Multiply, T.One);
        }

        public void CumulativeProduct(PrimitiveColumnContainer<T> column, IEnumerable<long> rows)
        {
            CumulativeApply(column, Multiply, rows);
        }

        private T Add(T left, T right) => left + right;
        public void CumulativeSum(PrimitiveColumnContainer<T> column)
        {
            CumulativeApply(column, Add, T.Zero);
        }

        public void CumulativeSum(PrimitiveColumnContainer<T> column, IEnumerable<long> rows)
        {
            CumulativeApply(column, Add, rows);
        }

        public void Max(PrimitiveColumnContainer<T> column, out T ret)
        {
            ret = CalculateReduction(column, T.Max, column[0].Value);
        }

        public void Max(PrimitiveColumnContainer<T> column, IEnumerable<long> rows, out T ret)
        {
            ret = CalculateReduction(column, T.Max, rows);
        }

        public void Min(PrimitiveColumnContainer<T> column, out T ret)
        {
            ret = CalculateReduction(column, T.Min, column[0].Value);
        }

        public void Min(PrimitiveColumnContainer<T> column, IEnumerable<long> rows, out T ret)
        {

            ret = CalculateReduction(column, T.Min, rows);
        }

        public void Product(PrimitiveColumnContainer<T> column, out T ret)
        {
            ret = CalculateReduction(column, Multiply, T.One);
        }

        public void Product(PrimitiveColumnContainer<T> column, IEnumerable<long> rows, out T ret)
        {
            ret = CalculateReduction(column, Multiply, rows);
        }

        public void Sum(PrimitiveColumnContainer<T> column, out T ret)
        {
            ret = CalculateReduction(column, Add, T.Zero);
        }

        public void Sum(PrimitiveColumnContainer<T> column, IEnumerable<long> rows, out T ret)
        {
            ret = CalculateReduction(column, Add, rows);
        }

        public virtual void Round(PrimitiveColumnContainer<T> column)
        {
            // do nothing
        }

        public override bool Equals(object obj)
        {
            return base.Equals(obj);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        public override string ToString()
        {
            return base.ToString();
        }

        protected void Apply(PrimitiveColumnContainer<T> column, Func<T, T> func)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;

                var nullBitMapBuffer = column.NullBitMapBuffers[b].ReadOnlySpan;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    if (column.IsValid(nullBitMapBuffer, i))
                    {
                        mutableSpan[i] = func(mutableSpan[i]);
                    }
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        protected void CumulativeApply(PrimitiveColumnContainer<T> column, Func<T, T, T> func)
        {
            CumulativeApply(column, func, column.Buffers[0].ReadOnlySpan[0]);
        }

        protected void CumulativeApply(PrimitiveColumnContainer<T> column, Func<T, T, T> func, T startingValue)
        {
            T ret = startingValue;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                var mutableSpan = mutableBuffer.Span;

                var nullBitMapBuffer = column.NullBitMapBuffers[b].ReadOnlySpan;
                for (int i = 0; i < mutableSpan.Length; i++)
                {
                    if (column.IsValid(nullBitMapBuffer, i))
                    {
                        ret = func(mutableSpan[i], ret);
                        mutableSpan[i] = ret;
                    }
                }
                column.Buffers[b] = mutableBuffer;
            }
        }

        protected void CumulativeApply(PrimitiveColumnContainer<T> column, Func<T, T, T> func, IEnumerable<long> rows)
        {
            T ret = T.Zero;
            var mutableBuffer = column.Buffers.GetOrCreateMutable(0);
            var nullBitMap = column.NullBitMapBuffers.GetOrCreateMutable(0);
            var span = mutableBuffer.Span;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<T>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();

            bool isValid = false;
            while (!isValid && enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = column.Buffers.GetOrCreateMutable(bufferIndex);
                    nullBitMap = column.NullBitMapBuffers.GetOrCreateMutable(bufferIndex);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }

                row -= minRange;
                if (column.IsValid(nullBitMap.Span, (int)row))
                {
                    isValid = true;
                    ret = span[(int)row];
                }
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    mutableBuffer = column.Buffers.GetOrCreateMutable(bufferIndex);
                    nullBitMap = column.NullBitMapBuffers.GetOrCreateMutable(bufferIndex);
                    span = mutableBuffer.Span;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }

                row -= minRange;
                if (column.IsValid(nullBitMapBuffer, i))
                {
                    ret = func(ret, readonlySpan[i]);
                    span[(int)row] = ret;
                }
            }
        }


        protected T CalculateReduction(PrimitiveColumnContainer<T> column, Func<T, T, T> func, T startValue)
        {
            var ret = startValue;

            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var readonlySpan = column.Buffers[b].ReadOnlySpan;
                var nullBitMapBuffer = column.NullBitMapBuffers[b].ReadOnlySpan;
                for (int i = 0; i < readonlySpan.Length; i++)
                {
                    if (column.IsValid(nullBitMapBuffer, i))
                    {
                        ret = func(ret, readonlySpan[i]);
                    }
                }
            }
            return ret;
        }

        protected T CalculateReduction(PrimitiveColumnContainer<T> column, Func<T, T, T> func, IEnumerable<long> rows)
        {
            var ret = T.Zero;
            var readOnlySpan = column.Buffers[0].ReadOnlySpan;
            var readOnlyNullBitMap = column.NullBitMapBuffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<T>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();

            bool isValid = false;
            while (!isValid && enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    readOnlyNullBitMap = column.NullBitMapBuffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;

                if (column.IsValid(readOnlyNullBitMap, (int)row))
                {
                    isValid = true;
                    ret = readOnlySpan[(int)row];
                }
            }

            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    readOnlySpan = column.Buffers[bufferIndex].ReadOnlySpan;
                    readOnlyNullBitMap = column.NullBitMapBuffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;

                ret = checked(func(readOnlySpan[(int)row], ret));
            }

            return ret;
        }
    }
}
