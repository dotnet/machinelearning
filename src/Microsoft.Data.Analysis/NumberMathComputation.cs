

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from PrimitiveColumnComputations.tt. Do not modify directly

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
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
            CumulativeApply(column, T.Max, column.Buffers[0].ReadOnlySpan[0]);
        }

        public void CumulativeMax(PrimitiveColumnContainer<T> column, IEnumerable<long> rows)
        {
            CumulativeApply(column, T.Max, rows);
        }

        public void CumulativeMin(PrimitiveColumnContainer<T> column)
        {
            CumulativeApply(column, T.Min, column.Buffers[0].ReadOnlySpan[0]);
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

        public void Max(PrimitiveColumnContainer<T> column, out T? ret)
        {
            ret = CalculateReduction(column, T.Max);
        }

        public void Max(PrimitiveColumnContainer<T> column, IEnumerable<long> rows, out T? ret)
        {
            ret = CalculateReduction(column, T.Max, rows);
        }

        public void Min(PrimitiveColumnContainer<T> column, out T? ret)
        {
            ret = CalculateReduction(column, T.Min);
        }

        public void Min(PrimitiveColumnContainer<T> column, IEnumerable<long> rows, out T? ret)
        {

            ret = CalculateReduction(column, T.Min, rows);
        }

        public void Product(PrimitiveColumnContainer<T> column, out T? ret)
        {
            ret = CalculateReduction(column, Multiply);
        }

        public void Product(PrimitiveColumnContainer<T> column, IEnumerable<long> rows, out T? ret)
        {
            ret = CalculateReduction(column, Multiply, rows);
        }

        public void Sum(PrimitiveColumnContainer<T> column, out T? ret)
        {
            ret = CalculateReduction(column, Add);
        }

        public void Sum(PrimitiveColumnContainer<T> column, IEnumerable<long> rows, out T? ret)
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
                var buffer = column.Buffers.GetOrCreateMutable(b).Span;
                var bitmap = column.NullBitMapBuffers[b].ReadOnlySpan;
                for (int i = 0; i < buffer.Length; i++)
                {
                    if (BitmapHelper.IsValid(bitmap, i))
                    {
                        buffer[i] = func(buffer[i]);
                    }
                }
            }
        }

        protected void CumulativeApply(PrimitiveColumnContainer<T> column, Func<T, T, T> func, T startingValue)
        {
            T ret = startingValue;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers.GetOrCreateMutable(b).Span;
                var bitmap = column.NullBitMapBuffers[b].ReadOnlySpan;
                for (int i = 0; i < buffer.Length; i++)
                {
                    if (BitmapHelper.IsValid(bitmap, i))
                    {
                        ret = func(buffer[i], ret);
                        buffer[i] = ret;
                    }
                }
            }
        }

        protected T? CalculateReduction(PrimitiveColumnContainer<T> column, Func<T, T, T> func)
        {
            T? ret = null;
            bool isInitialized = false;

            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b].ReadOnlySpan;
                var bitMap = column.NullBitMapBuffers[b].ReadOnlySpan;
                for (int i = 0; i < buffer.Length; i++)
                {
                    if (BitmapHelper.IsValid(bitMap, i))
                    {
                        if (!isInitialized)
                        {
                            isInitialized = true;
                            ret = buffer[i];
                        }
                        else
                        {
                            ret = checked(func(ret.Value, buffer[i]));
                        }
                    }
                }
            }
            return ret;
        }

        protected void CumulativeApply(PrimitiveColumnContainer<T> column, Func<T, T, T> func, IEnumerable<long> rows)
        {
            T ret = T.Zero;
            var buffer = column.Buffers.GetOrCreateMutable(0).Span;
            var bitmap = column.NullBitMapBuffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<T>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();

            bool isInitialized = false;
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    buffer = column.Buffers.GetOrCreateMutable(bufferIndex).Span;
                    bitmap = column.NullBitMapBuffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }

                row -= minRange;
                if (BitmapHelper.IsValid(bitmap, (int)row))
                {
                    if (!isInitialized)
                    {
                        isInitialized = true;
                        ret = buffer[(int)row];
                    }
                    else
                    {
                        ret = func(ret, buffer[(int)row]);
                        buffer[(int)row] = ret;
                    }
                }
            }
        }

        protected T CalculateReduction(PrimitiveColumnContainer<T> column, Func<T, T, T> func, IEnumerable<long> rows)
        {
            var ret = T.Zero;
            var buffer = column.Buffers[0].ReadOnlySpan;
            var bitMap = column.NullBitMapBuffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = ReadOnlyDataFrameBuffer<T>.MaxCapacity;
            long maxCapacity = maxRange;
            IEnumerator<long> enumerator = rows.GetEnumerator();

            bool isInitialized = false;
            while (enumerator.MoveNext())
            {
                long row = enumerator.Current;
                if (row < minRange || row >= maxRange)
                {
                    int bufferIndex = (int)(row / maxCapacity);
                    buffer = column.Buffers[bufferIndex].ReadOnlySpan;
                    bitMap = column.NullBitMapBuffers[bufferIndex].ReadOnlySpan;
                    minRange = checked(bufferIndex * maxCapacity);
                    maxRange = checked((bufferIndex + 1) * maxCapacity);
                }
                row -= minRange;

                if (BitmapHelper.IsValid(bitMap, (int)row))
                {
                    if (!isInitialized)
                    {
                        isInitialized = true;
                        ret = buffer[(int)row];
                    }
                    else
                    {
                        ret = checked(func(ret, buffer[(int)row]));
                    }
                }
            }

            return ret;
        }

        public PrimitiveColumnContainer<U> CreateTruncating<U>(PrimitiveColumnContainer<T> column)
            where U : unmanaged, INumber<U>
        {
            var ret = new PrimitiveColumnContainer<U>();
            foreach (ReadOnlyDataFrameBuffer<T> buffer in column.Buffers)
            {
                ret.Length += buffer.Length;
                DataFrameBuffer<U> newBuffer = new DataFrameBuffer<U>();
                ret.Buffers.Add(newBuffer);
                newBuffer.EnsureCapacity(buffer.Length);
                ReadOnlySpan<T> span = buffer.ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    newBuffer.Append(U.CreateTruncating<T>(span[i]));
                }
            }
            ret.NullBitMapBuffers = column.CloneNullBitMapBuffers();
            ret.NullCount = column.NullCount;
            return ret;
        }
    }
}
