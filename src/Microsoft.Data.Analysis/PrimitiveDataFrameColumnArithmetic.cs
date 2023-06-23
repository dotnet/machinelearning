

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
                return (IPrimitiveDataFrameColumnArithmetic<T>)new NumberDataFrameColumnArithmetic<byte>();
            }
            else if (typeof(T) == typeof(char))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new NumberDataFrameColumnArithmetic<char>();
            }
            else if (typeof(T) == typeof(decimal))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new FloatingPointDataFrameColumnArithmetic<decimal>();
            }
            else if (typeof(T) == typeof(double))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new FloatingPointDataFrameColumnArithmetic<double>();
            }
            else if (typeof(T) == typeof(float))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new FloatingPointDataFrameColumnArithmetic<float>();
            }
            else if (typeof(T) == typeof(int))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new NumberDataFrameColumnArithmetic<int>();
            }
            else if (typeof(T) == typeof(long))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new NumberDataFrameColumnArithmetic<long>();
            }
            else if (typeof(T) == typeof(sbyte))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new NumberDataFrameColumnArithmetic<sbyte>();
            }
            else if (typeof(T) == typeof(short))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new NumberDataFrameColumnArithmetic<short>();
            }
            else if (typeof(T) == typeof(uint))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new NumberDataFrameColumnArithmetic<uint>();
            }
            else if (typeof(T) == typeof(ulong))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new NumberDataFrameColumnArithmetic<ulong>();
            }
            else if (typeof(T) == typeof(ushort))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new NumberDataFrameColumnArithmetic<ushort>();
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
                var mutableBuffer = left.Buffers.GetOrCreateMutable(b);
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
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
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
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
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
                var mutableBuffer = left.Buffers.GetOrCreateMutable(b);
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
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
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
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
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
                var mutableBuffer = left.Buffers.GetOrCreateMutable(b);
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
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
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
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
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
                var span = left.Buffers[b].ReadOnlySpan;
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
                var span = column.Buffers[b].ReadOnlySpan;
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
                var span = left.Buffers[b].ReadOnlySpan;
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
                var span = column.Buffers[b].ReadOnlySpan;
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
                var span = left.Buffers[b].ReadOnlySpan;
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
                var span = column.Buffers[b].ReadOnlySpan;
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
                var span = left.Buffers[b].ReadOnlySpan;
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
                var span = column.Buffers[b].ReadOnlySpan;
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
