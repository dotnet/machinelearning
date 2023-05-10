

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from PrimitiveDataFrameColumnArithmetic.tt. Do not modify directly

using System;

namespace Microsoft.Data.Analysis
{

    internal static class PrimitiveDataFrameColumnArithmetic<T>
        where T : unmanaged
    {
        public static IPrimitiveDataFrameColumnArithmetic<T> Instance { get; } = PrimitiveDataFrameColumnArithmetic.GetArithmetic<T>();
    }

    internal static class PrimitiveDataFrameColumnArithmetic
    {
        public static IPrimitiveDataFrameColumnArithmetic<T> GetArithmetic<T>()
            where T : unmanaged
        {
            if (typeof(T) == typeof(bool))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new BoolArithmetic();
            }
            else if (typeof(T) == typeof(byte))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new NumericDataFrameColumnArithmetic<byte>();
            }
            else if (typeof(T) == typeof(char))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new NumericDataFrameColumnArithmetic<char>();
            }
            else if (typeof(T) == typeof(decimal))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new DecimalArithmetic();
            }
            else if (typeof(T) == typeof(double))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new DoubleArithmetic();
            }
            else if (typeof(T) == typeof(float))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new FloatArithmetic();
            }
            else if (typeof(T) == typeof(int))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new NumericDataFrameColumnArithmetic<int>();
            }
            else if (typeof(T) == typeof(long))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new NumericDataFrameColumnArithmetic<long>();
            }
            else if (typeof(T) == typeof(sbyte))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new NumericDataFrameColumnArithmetic<sbyte>();
            }
            else if (typeof(T) == typeof(short))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new NumericDataFrameColumnArithmetic<short>();
            }
            else if (typeof(T) == typeof(uint))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new NumericDataFrameColumnArithmetic<uint>();
            }
            else if (typeof(T) == typeof(ulong))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new NumericDataFrameColumnArithmetic<ulong>();
            }
            else if (typeof(T) == typeof(ushort))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new NumericDataFrameColumnArithmetic<ushort>();
            }
            else if (typeof(T) == typeof(DateTime))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new DateTimeArithmetic();
            }
            throw new NotSupportedException();
        }
    }

    internal class BoolArithmetic : IPrimitiveDataFrameColumnArithmetic<bool>
    {
        public void Add(PrimitiveColumnContainer<bool> left, PrimitiveColumnContainer<bool> right)
        {
            throw new NotSupportedException();
        }
        public void Add(PrimitiveColumnContainer<bool> column, bool scalar)
        {
            throw new NotSupportedException();
        }
        public void Add(bool scalar, PrimitiveColumnContainer<bool> column)
        {
            throw new NotSupportedException();
        }
        public void Subtract(PrimitiveColumnContainer<bool> left, PrimitiveColumnContainer<bool> right)
        {
            throw new NotSupportedException();
        }
        public void Subtract(PrimitiveColumnContainer<bool> column, bool scalar)
        {
            throw new NotSupportedException();
        }
        public void Subtract(bool scalar, PrimitiveColumnContainer<bool> column)
        {
            throw new NotSupportedException();
        }
        public void Multiply(PrimitiveColumnContainer<bool> left, PrimitiveColumnContainer<bool> right)
        {
            throw new NotSupportedException();
        }
        public void Multiply(PrimitiveColumnContainer<bool> column, bool scalar)
        {
            throw new NotSupportedException();
        }
        public void Multiply(bool scalar, PrimitiveColumnContainer<bool> column)
        {
            throw new NotSupportedException();
        }
        public void Divide(PrimitiveColumnContainer<bool> left, PrimitiveColumnContainer<bool> right)
        {
            throw new NotSupportedException();
        }
        public void Divide(PrimitiveColumnContainer<bool> column, bool scalar)
        {
            throw new NotSupportedException();
        }
        public void Divide(bool scalar, PrimitiveColumnContainer<bool> column)
        {
            throw new NotSupportedException();
        }
        public void Modulo(PrimitiveColumnContainer<bool> left, PrimitiveColumnContainer<bool> right)
        {
            throw new NotSupportedException();
        }
        public void Modulo(PrimitiveColumnContainer<bool> column, bool scalar)
        {
            throw new NotSupportedException();
        }
        public void Modulo(bool scalar, PrimitiveColumnContainer<bool> column)
        {
            throw new NotSupportedException();
        }
        public void And(PrimitiveColumnContainer<bool> left, PrimitiveColumnContainer<bool> right)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<bool>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (bool)(span[i] & otherSpan[i]);
                }
            }
        }
        public void And(PrimitiveColumnContainer<bool> column, bool scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<bool>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (bool)(span[i] & scalar);
                }
            }
        }
        public void And(bool scalar, PrimitiveColumnContainer<bool> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<bool>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (bool)(scalar & span[i]);
                }
            }
        }
        public void Or(PrimitiveColumnContainer<bool> left, PrimitiveColumnContainer<bool> right)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<bool>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (bool)(span[i] | otherSpan[i]);
                }
            }
        }
        public void Or(PrimitiveColumnContainer<bool> column, bool scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<bool>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (bool)(span[i] | scalar);
                }
            }
        }
        public void Or(bool scalar, PrimitiveColumnContainer<bool> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<bool>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (bool)(scalar | span[i]);
                }
            }
        }
        public void Xor(PrimitiveColumnContainer<bool> left, PrimitiveColumnContainer<bool> right)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<bool>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (bool)(span[i] ^ otherSpan[i]);
                }
            }
        }
        public void Xor(PrimitiveColumnContainer<bool> column, bool scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<bool>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (bool)(span[i] ^ scalar);
                }
            }
        }
        public void Xor(bool scalar, PrimitiveColumnContainer<bool> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<bool>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (bool)(scalar ^ span[i]);
                }
            }
        }
        public void LeftShift(PrimitiveColumnContainer<bool> column, int value)
        {
            throw new NotSupportedException();
        }
        public void RightShift(PrimitiveColumnContainer<bool> column, int value)
        {
            throw new NotSupportedException();
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<bool> left, PrimitiveColumnContainer<bool> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<bool>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == otherSpan[i]);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<bool> column, bool scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<bool>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == scalar);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<bool> left, PrimitiveColumnContainer<bool> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<bool>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != otherSpan[i]);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<bool> column, bool scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<bool>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != scalar);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<bool> left, PrimitiveColumnContainer<bool> right, PrimitiveColumnContainer<bool> ret)
        {
            throw new NotSupportedException();
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<bool> column, bool scalar, PrimitiveColumnContainer<bool> ret)
        {
            throw new NotSupportedException();
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<bool> left, PrimitiveColumnContainer<bool> right, PrimitiveColumnContainer<bool> ret)
        {
            throw new NotSupportedException();
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<bool> column, bool scalar, PrimitiveColumnContainer<bool> ret)
        {
            throw new NotSupportedException();
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<bool> left, PrimitiveColumnContainer<bool> right, PrimitiveColumnContainer<bool> ret)
        {
            throw new NotSupportedException();
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<bool> column, bool scalar, PrimitiveColumnContainer<bool> ret)
        {
            throw new NotSupportedException();
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<bool> left, PrimitiveColumnContainer<bool> right, PrimitiveColumnContainer<bool> ret)
        {
            throw new NotSupportedException();
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<bool> column, bool scalar, PrimitiveColumnContainer<bool> ret)
        {
            throw new NotSupportedException();
        }
    }

    internal class DecimalArithmetic : IPrimitiveDataFrameColumnArithmetic<decimal>
    {
        public void Add(PrimitiveColumnContainer<decimal> left, PrimitiveColumnContainer<decimal> right)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(span[i] + otherSpan[i]);
                }
            }
        }
        public void Add(PrimitiveColumnContainer<decimal> column, decimal scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(span[i] + scalar);
                }
            }
        }
        public void Add(decimal scalar, PrimitiveColumnContainer<decimal> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(scalar + span[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<decimal> left, PrimitiveColumnContainer<decimal> right)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(span[i] - otherSpan[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<decimal> column, decimal scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(span[i] - scalar);
                }
            }
        }
        public void Subtract(decimal scalar, PrimitiveColumnContainer<decimal> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(scalar - span[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<decimal> left, PrimitiveColumnContainer<decimal> right)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(span[i] * otherSpan[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<decimal> column, decimal scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(span[i] * scalar);
                }
            }
        }
        public void Multiply(decimal scalar, PrimitiveColumnContainer<decimal> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(scalar * span[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<decimal> left, PrimitiveColumnContainer<decimal> right)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(span[i] / otherSpan[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<decimal> column, decimal scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(span[i] / scalar);
                }
            }
        }
        public void Divide(decimal scalar, PrimitiveColumnContainer<decimal> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(scalar / span[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<decimal> left, PrimitiveColumnContainer<decimal> right)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(span[i] % otherSpan[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<decimal> column, decimal scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(span[i] % scalar);
                }
            }
        }
        public void Modulo(decimal scalar, PrimitiveColumnContainer<decimal> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(scalar % span[i]);
                }
            }
        }
        public void And(PrimitiveColumnContainer<decimal> left, PrimitiveColumnContainer<decimal> right)
        {
            throw new NotSupportedException();
        }
        public void And(PrimitiveColumnContainer<decimal> column, decimal scalar)
        {
            throw new NotSupportedException();
        }
        public void And(decimal scalar, PrimitiveColumnContainer<decimal> column)
        {
            throw new NotSupportedException();
        }
        public void Or(PrimitiveColumnContainer<decimal> left, PrimitiveColumnContainer<decimal> right)
        {
            throw new NotSupportedException();
        }
        public void Or(PrimitiveColumnContainer<decimal> column, decimal scalar)
        {
            throw new NotSupportedException();
        }
        public void Or(decimal scalar, PrimitiveColumnContainer<decimal> column)
        {
            throw new NotSupportedException();
        }
        public void Xor(PrimitiveColumnContainer<decimal> left, PrimitiveColumnContainer<decimal> right)
        {
            throw new NotSupportedException();
        }
        public void Xor(PrimitiveColumnContainer<decimal> column, decimal scalar)
        {
            throw new NotSupportedException();
        }
        public void Xor(decimal scalar, PrimitiveColumnContainer<decimal> column)
        {
            throw new NotSupportedException();
        }
        public void LeftShift(PrimitiveColumnContainer<decimal> column, int value)
        {
            throw new NotSupportedException();
        }
        public void RightShift(PrimitiveColumnContainer<decimal> column, int value)
        {
            throw new NotSupportedException();
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<decimal> left, PrimitiveColumnContainer<decimal> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == otherSpan[i]);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<decimal> column, decimal scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == scalar);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<decimal> left, PrimitiveColumnContainer<decimal> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != otherSpan[i]);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<decimal> column, decimal scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != scalar);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<decimal> left, PrimitiveColumnContainer<decimal> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<decimal> column, decimal scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= scalar);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<decimal> left, PrimitiveColumnContainer<decimal> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<decimal> column, decimal scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= scalar);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<decimal> left, PrimitiveColumnContainer<decimal> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<decimal> column, decimal scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > scalar);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<decimal> left, PrimitiveColumnContainer<decimal> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<decimal> column, decimal scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<decimal>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < scalar);
                }
            }
        }
    }
    internal class DoubleArithmetic : IPrimitiveDataFrameColumnArithmetic<double>
    {
        public void Add(PrimitiveColumnContainer<double> left, PrimitiveColumnContainer<double> right)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(span[i] + otherSpan[i]);
                }
            }
        }
        public void Add(PrimitiveColumnContainer<double> column, double scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(span[i] + scalar);
                }
            }
        }
        public void Add(double scalar, PrimitiveColumnContainer<double> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(scalar + span[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<double> left, PrimitiveColumnContainer<double> right)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(span[i] - otherSpan[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<double> column, double scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(span[i] - scalar);
                }
            }
        }
        public void Subtract(double scalar, PrimitiveColumnContainer<double> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(scalar - span[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<double> left, PrimitiveColumnContainer<double> right)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(span[i] * otherSpan[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<double> column, double scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(span[i] * scalar);
                }
            }
        }
        public void Multiply(double scalar, PrimitiveColumnContainer<double> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(scalar * span[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<double> left, PrimitiveColumnContainer<double> right)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(span[i] / otherSpan[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<double> column, double scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(span[i] / scalar);
                }
            }
        }
        public void Divide(double scalar, PrimitiveColumnContainer<double> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(scalar / span[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<double> left, PrimitiveColumnContainer<double> right)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(span[i] % otherSpan[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<double> column, double scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(span[i] % scalar);
                }
            }
        }
        public void Modulo(double scalar, PrimitiveColumnContainer<double> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(scalar % span[i]);
                }
            }
        }
        public void And(PrimitiveColumnContainer<double> left, PrimitiveColumnContainer<double> right)
        {
            throw new NotSupportedException();
        }
        public void And(PrimitiveColumnContainer<double> column, double scalar)
        {
            throw new NotSupportedException();
        }
        public void And(double scalar, PrimitiveColumnContainer<double> column)
        {
            throw new NotSupportedException();
        }
        public void Or(PrimitiveColumnContainer<double> left, PrimitiveColumnContainer<double> right)
        {
            throw new NotSupportedException();
        }
        public void Or(PrimitiveColumnContainer<double> column, double scalar)
        {
            throw new NotSupportedException();
        }
        public void Or(double scalar, PrimitiveColumnContainer<double> column)
        {
            throw new NotSupportedException();
        }
        public void Xor(PrimitiveColumnContainer<double> left, PrimitiveColumnContainer<double> right)
        {
            throw new NotSupportedException();
        }
        public void Xor(PrimitiveColumnContainer<double> column, double scalar)
        {
            throw new NotSupportedException();
        }
        public void Xor(double scalar, PrimitiveColumnContainer<double> column)
        {
            throw new NotSupportedException();
        }
        public void LeftShift(PrimitiveColumnContainer<double> column, int value)
        {
            throw new NotSupportedException();
        }
        public void RightShift(PrimitiveColumnContainer<double> column, int value)
        {
            throw new NotSupportedException();
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<double> left, PrimitiveColumnContainer<double> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == otherSpan[i]);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<double> column, double scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == scalar);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<double> left, PrimitiveColumnContainer<double> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != otherSpan[i]);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<double> column, double scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != scalar);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<double> left, PrimitiveColumnContainer<double> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<double> column, double scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= scalar);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<double> left, PrimitiveColumnContainer<double> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<double> column, double scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= scalar);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<double> left, PrimitiveColumnContainer<double> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<double> column, double scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > scalar);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<double> left, PrimitiveColumnContainer<double> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<double> column, double scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<double>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < scalar);
                }
            }
        }
    }
    internal class FloatArithmetic : IPrimitiveDataFrameColumnArithmetic<float>
    {
        public void Add(PrimitiveColumnContainer<float> left, PrimitiveColumnContainer<float> right)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(span[i] + otherSpan[i]);
                }
            }
        }
        public void Add(PrimitiveColumnContainer<float> column, float scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(span[i] + scalar);
                }
            }
        }
        public void Add(float scalar, PrimitiveColumnContainer<float> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(scalar + span[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<float> left, PrimitiveColumnContainer<float> right)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(span[i] - otherSpan[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<float> column, float scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(span[i] - scalar);
                }
            }
        }
        public void Subtract(float scalar, PrimitiveColumnContainer<float> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(scalar - span[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<float> left, PrimitiveColumnContainer<float> right)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(span[i] * otherSpan[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<float> column, float scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(span[i] * scalar);
                }
            }
        }
        public void Multiply(float scalar, PrimitiveColumnContainer<float> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(scalar * span[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<float> left, PrimitiveColumnContainer<float> right)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(span[i] / otherSpan[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<float> column, float scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(span[i] / scalar);
                }
            }
        }
        public void Divide(float scalar, PrimitiveColumnContainer<float> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(scalar / span[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<float> left, PrimitiveColumnContainer<float> right)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(span[i] % otherSpan[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<float> column, float scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(span[i] % scalar);
                }
            }
        }
        public void Modulo(float scalar, PrimitiveColumnContainer<float> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(scalar % span[i]);
                }
            }
        }
        public void And(PrimitiveColumnContainer<float> left, PrimitiveColumnContainer<float> right)
        {
            throw new NotSupportedException();
        }
        public void And(PrimitiveColumnContainer<float> column, float scalar)
        {
            throw new NotSupportedException();
        }
        public void And(float scalar, PrimitiveColumnContainer<float> column)
        {
            throw new NotSupportedException();
        }
        public void Or(PrimitiveColumnContainer<float> left, PrimitiveColumnContainer<float> right)
        {
            throw new NotSupportedException();
        }
        public void Or(PrimitiveColumnContainer<float> column, float scalar)
        {
            throw new NotSupportedException();
        }
        public void Or(float scalar, PrimitiveColumnContainer<float> column)
        {
            throw new NotSupportedException();
        }
        public void Xor(PrimitiveColumnContainer<float> left, PrimitiveColumnContainer<float> right)
        {
            throw new NotSupportedException();
        }
        public void Xor(PrimitiveColumnContainer<float> column, float scalar)
        {
            throw new NotSupportedException();
        }
        public void Xor(float scalar, PrimitiveColumnContainer<float> column)
        {
            throw new NotSupportedException();
        }
        public void LeftShift(PrimitiveColumnContainer<float> column, int value)
        {
            throw new NotSupportedException();
        }
        public void RightShift(PrimitiveColumnContainer<float> column, int value)
        {
            throw new NotSupportedException();
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<float> left, PrimitiveColumnContainer<float> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == otherSpan[i]);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<float> column, float scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == scalar);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<float> left, PrimitiveColumnContainer<float> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != otherSpan[i]);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<float> column, float scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != scalar);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<float> left, PrimitiveColumnContainer<float> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<float> column, float scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= scalar);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<float> left, PrimitiveColumnContainer<float> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<float> column, float scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= scalar);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<float> left, PrimitiveColumnContainer<float> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<float> column, float scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > scalar);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<float> left, PrimitiveColumnContainer<float> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<float> column, float scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<float>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < scalar);
                }
            }
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
            throw new NotSupportedException();
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<DateTime> column, DateTime scalar, PrimitiveColumnContainer<bool> ret)
        {
            throw new NotSupportedException();
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<DateTime> left, PrimitiveColumnContainer<DateTime> right, PrimitiveColumnContainer<bool> ret)
        {
            throw new NotSupportedException();
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<DateTime> column, DateTime scalar, PrimitiveColumnContainer<bool> ret)
        {
            throw new NotSupportedException();
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<DateTime> left, PrimitiveColumnContainer<DateTime> right, PrimitiveColumnContainer<bool> ret)
        {
            throw new NotSupportedException();
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<DateTime> column, DateTime scalar, PrimitiveColumnContainer<bool> ret)
        {
            throw new NotSupportedException();
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<DateTime> left, PrimitiveColumnContainer<DateTime> right, PrimitiveColumnContainer<bool> ret)
        {
            throw new NotSupportedException();
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<DateTime> column, DateTime scalar, PrimitiveColumnContainer<bool> ret)
        {
            throw new NotSupportedException();
        }
    }
}
