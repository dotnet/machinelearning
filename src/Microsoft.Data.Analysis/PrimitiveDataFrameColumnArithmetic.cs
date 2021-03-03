

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from PrimitiveDataFrameColumnArithmetic.tt. Do not modify directly

using System;

namespace Microsoft.Data.Analysis
{
    internal interface IPrimitiveDataFrameColumnArithmetic<T>
        where T : struct
    {
       void Add(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right);
       void Add(PrimitiveColumnContainer<T> column, T scalar);
       void Add(T scalar, PrimitiveColumnContainer<T> column);
       void Subtract(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right);
       void Subtract(PrimitiveColumnContainer<T> column, T scalar);
       void Subtract(T scalar, PrimitiveColumnContainer<T> column);
       void Multiply(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right);
       void Multiply(PrimitiveColumnContainer<T> column, T scalar);
       void Multiply(T scalar, PrimitiveColumnContainer<T> column);
       void Divide(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right);
       void Divide(PrimitiveColumnContainer<T> column, T scalar);
       void Divide(T scalar, PrimitiveColumnContainer<T> column);
       void Modulo(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right);
       void Modulo(PrimitiveColumnContainer<T> column, T scalar);
       void Modulo(T scalar, PrimitiveColumnContainer<T> column);
       void And(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right);
       void And(PrimitiveColumnContainer<T> column, T scalar);
       void And(T scalar, PrimitiveColumnContainer<T> column);
       void Or(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right);
       void Or(PrimitiveColumnContainer<T> column, T scalar);
       void Or(T scalar, PrimitiveColumnContainer<T> column);
       void Xor(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right);
       void Xor(PrimitiveColumnContainer<T> column, T scalar);
       void Xor(T scalar, PrimitiveColumnContainer<T> column);
       void LeftShift(PrimitiveColumnContainer<T> column, int value);
       void RightShift(PrimitiveColumnContainer<T> column, int value);
       void ElementwiseEquals(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right, PrimitiveColumnContainer<bool> ret);
       void ElementwiseEquals(PrimitiveColumnContainer<T> column, T scalar, PrimitiveColumnContainer<bool> ret);
       void ElementwiseNotEquals(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right, PrimitiveColumnContainer<bool> ret);
       void ElementwiseNotEquals(PrimitiveColumnContainer<T> column, T scalar, PrimitiveColumnContainer<bool> ret);
       void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right, PrimitiveColumnContainer<bool> ret);
       void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<T> column, T scalar, PrimitiveColumnContainer<bool> ret);
       void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right, PrimitiveColumnContainer<bool> ret);
       void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<T> column, T scalar, PrimitiveColumnContainer<bool> ret);
       void ElementwiseGreaterThan(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right, PrimitiveColumnContainer<bool> ret);
       void ElementwiseGreaterThan(PrimitiveColumnContainer<T> column, T scalar, PrimitiveColumnContainer<bool> ret);
       void ElementwiseLessThan(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right, PrimitiveColumnContainer<bool> ret);
       void ElementwiseLessThan(PrimitiveColumnContainer<T> column, T scalar, PrimitiveColumnContainer<bool> ret);
    }

    internal static class PrimitiveDataFrameColumnArithmetic<T>
        where T : struct
    {
        public static IPrimitiveDataFrameColumnArithmetic<T> Instance { get; } = PrimitiveDataFrameColumnArithmetic.GetArithmetic<T>();
    }

    internal static class PrimitiveDataFrameColumnArithmetic
    {
        public static IPrimitiveDataFrameColumnArithmetic<T> GetArithmetic<T>()
            where T : struct
        {
            if (typeof(T) == typeof(bool))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new BoolArithmetic();
            }
            else if (typeof(T) == typeof(byte))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new ByteArithmetic();
            }
            else if (typeof(T) == typeof(char))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new CharArithmetic();
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
                return (IPrimitiveDataFrameColumnArithmetic<T>)new IntArithmetic();
            }
            else if (typeof(T) == typeof(long))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new LongArithmetic();
            }
            else if (typeof(T) == typeof(sbyte))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new SByteArithmetic();
            }
            else if (typeof(T) == typeof(short))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new ShortArithmetic();
            }
            else if (typeof(T) == typeof(uint))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new UIntArithmetic();
            }
            else if (typeof(T) == typeof(ulong))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new ULongArithmetic();
            }
            else if (typeof(T) == typeof(ushort))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new UShortArithmetic();
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
    internal class ByteArithmetic : IPrimitiveDataFrameColumnArithmetic<byte>
    {
        public void Add(PrimitiveColumnContainer<byte> left, PrimitiveColumnContainer<byte> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] + otherSpan[i]);
                }
            }
        }
        public void Add(PrimitiveColumnContainer<byte> column, byte scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] + scalar);
                }
            }
        }
 
        public void Add(byte scalar, PrimitiveColumnContainer<byte> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (byte)(scalar + span[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<byte> left, PrimitiveColumnContainer<byte> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] - otherSpan[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<byte> column, byte scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] - scalar);
                }
            }
        }
 
        public void Subtract(byte scalar, PrimitiveColumnContainer<byte> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (byte)(scalar - span[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<byte> left, PrimitiveColumnContainer<byte> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] * otherSpan[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<byte> column, byte scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] * scalar);
                }
            }
        }
 
        public void Multiply(byte scalar, PrimitiveColumnContainer<byte> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (byte)(scalar * span[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<byte> left, PrimitiveColumnContainer<byte> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] / otherSpan[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<byte> column, byte scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] / scalar);
                }
            }
        }
 
        public void Divide(byte scalar, PrimitiveColumnContainer<byte> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (byte)(scalar / span[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<byte> left, PrimitiveColumnContainer<byte> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] % otherSpan[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<byte> column, byte scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] % scalar);
                }
            }
        }
 
        public void Modulo(byte scalar, PrimitiveColumnContainer<byte> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (byte)(scalar % span[i]);
                }
            }
        }
        public void And(PrimitiveColumnContainer<byte> left, PrimitiveColumnContainer<byte> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] & otherSpan[i]);
                }
            }
        }
        public void And(PrimitiveColumnContainer<byte> column, byte scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] & scalar);
                }
            }
        }
 
        public void And(byte scalar, PrimitiveColumnContainer<byte> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (byte)(scalar & span[i]);
                }
            }
        }
        public void Or(PrimitiveColumnContainer<byte> left, PrimitiveColumnContainer<byte> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] | otherSpan[i]);
                }
            }
        }
        public void Or(PrimitiveColumnContainer<byte> column, byte scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] | scalar);
                }
            }
        }
 
        public void Or(byte scalar, PrimitiveColumnContainer<byte> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (byte)(scalar | span[i]);
                }
            }
        }
        public void Xor(PrimitiveColumnContainer<byte> left, PrimitiveColumnContainer<byte> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] ^ otherSpan[i]);
                }
            }
        }
        public void Xor(PrimitiveColumnContainer<byte> column, byte scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] ^ scalar);
                }
            }
        }
 
        public void Xor(byte scalar, PrimitiveColumnContainer<byte> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (byte)(scalar ^ span[i]);
                }
            }
        }
        public void LeftShift(PrimitiveColumnContainer<byte> column, int value)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] << value);
                }
            }
        }
        public void RightShift(PrimitiveColumnContainer<byte> column, int value)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] >> value);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<byte> left, PrimitiveColumnContainer<byte> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == otherSpan[i]);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<byte> column, byte scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == scalar);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<byte> left, PrimitiveColumnContainer<byte> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != otherSpan[i]);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<byte> column, byte scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != scalar);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<byte> left, PrimitiveColumnContainer<byte> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<byte> column, byte scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= scalar);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<byte> left, PrimitiveColumnContainer<byte> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<byte> column, byte scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= scalar);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<byte> left, PrimitiveColumnContainer<byte> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<byte> column, byte scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > scalar);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<byte> left, PrimitiveColumnContainer<byte> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<byte> column, byte scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<byte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < scalar);
                }
            }
        }
    }
    internal class CharArithmetic : IPrimitiveDataFrameColumnArithmetic<char>
    {
        public void Add(PrimitiveColumnContainer<char> left, PrimitiveColumnContainer<char> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] + otherSpan[i]);
                }
            }
        }
        public void Add(PrimitiveColumnContainer<char> column, char scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] + scalar);
                }
            }
        }
 
        public void Add(char scalar, PrimitiveColumnContainer<char> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (char)(scalar + span[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<char> left, PrimitiveColumnContainer<char> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] - otherSpan[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<char> column, char scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] - scalar);
                }
            }
        }
 
        public void Subtract(char scalar, PrimitiveColumnContainer<char> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (char)(scalar - span[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<char> left, PrimitiveColumnContainer<char> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] * otherSpan[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<char> column, char scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] * scalar);
                }
            }
        }
 
        public void Multiply(char scalar, PrimitiveColumnContainer<char> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (char)(scalar * span[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<char> left, PrimitiveColumnContainer<char> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] / otherSpan[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<char> column, char scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] / scalar);
                }
            }
        }
 
        public void Divide(char scalar, PrimitiveColumnContainer<char> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (char)(scalar / span[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<char> left, PrimitiveColumnContainer<char> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] % otherSpan[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<char> column, char scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] % scalar);
                }
            }
        }
 
        public void Modulo(char scalar, PrimitiveColumnContainer<char> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (char)(scalar % span[i]);
                }
            }
        }
        public void And(PrimitiveColumnContainer<char> left, PrimitiveColumnContainer<char> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] & otherSpan[i]);
                }
            }
        }
        public void And(PrimitiveColumnContainer<char> column, char scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] & scalar);
                }
            }
        }
 
        public void And(char scalar, PrimitiveColumnContainer<char> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (char)(scalar & span[i]);
                }
            }
        }
        public void Or(PrimitiveColumnContainer<char> left, PrimitiveColumnContainer<char> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] | otherSpan[i]);
                }
            }
        }
        public void Or(PrimitiveColumnContainer<char> column, char scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] | scalar);
                }
            }
        }
 
        public void Or(char scalar, PrimitiveColumnContainer<char> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (char)(scalar | span[i]);
                }
            }
        }
        public void Xor(PrimitiveColumnContainer<char> left, PrimitiveColumnContainer<char> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] ^ otherSpan[i]);
                }
            }
        }
        public void Xor(PrimitiveColumnContainer<char> column, char scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] ^ scalar);
                }
            }
        }
 
        public void Xor(char scalar, PrimitiveColumnContainer<char> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (char)(scalar ^ span[i]);
                }
            }
        }
        public void LeftShift(PrimitiveColumnContainer<char> column, int value)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] << value);
                }
            }
        }
        public void RightShift(PrimitiveColumnContainer<char> column, int value)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] >> value);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<char> left, PrimitiveColumnContainer<char> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == otherSpan[i]);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<char> column, char scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == scalar);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<char> left, PrimitiveColumnContainer<char> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != otherSpan[i]);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<char> column, char scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != scalar);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<char> left, PrimitiveColumnContainer<char> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<char> column, char scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= scalar);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<char> left, PrimitiveColumnContainer<char> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<char> column, char scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= scalar);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<char> left, PrimitiveColumnContainer<char> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<char> column, char scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > scalar);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<char> left, PrimitiveColumnContainer<char> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<char> column, char scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<char>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < scalar);
                }
            }
        }
    }
    internal class DecimalArithmetic : IPrimitiveDataFrameColumnArithmetic<decimal>
    {
        public void Add(PrimitiveColumnContainer<decimal> left, PrimitiveColumnContainer<decimal> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
            for (int b = 0 ; b < left.Buffers.Count; b++)
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
            for (int b = 0 ; b < column.Buffers.Count; b++)
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
    internal class IntArithmetic : IPrimitiveDataFrameColumnArithmetic<int>
    {
        public void Add(PrimitiveColumnContainer<int> left, PrimitiveColumnContainer<int> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] + otherSpan[i]);
                }
            }
        }
        public void Add(PrimitiveColumnContainer<int> column, int scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] + scalar);
                }
            }
        }
 
        public void Add(int scalar, PrimitiveColumnContainer<int> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (int)(scalar + span[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<int> left, PrimitiveColumnContainer<int> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] - otherSpan[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<int> column, int scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] - scalar);
                }
            }
        }
 
        public void Subtract(int scalar, PrimitiveColumnContainer<int> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (int)(scalar - span[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<int> left, PrimitiveColumnContainer<int> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] * otherSpan[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<int> column, int scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] * scalar);
                }
            }
        }
 
        public void Multiply(int scalar, PrimitiveColumnContainer<int> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (int)(scalar * span[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<int> left, PrimitiveColumnContainer<int> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] / otherSpan[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<int> column, int scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] / scalar);
                }
            }
        }
 
        public void Divide(int scalar, PrimitiveColumnContainer<int> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (int)(scalar / span[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<int> left, PrimitiveColumnContainer<int> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] % otherSpan[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<int> column, int scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] % scalar);
                }
            }
        }
 
        public void Modulo(int scalar, PrimitiveColumnContainer<int> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (int)(scalar % span[i]);
                }
            }
        }
        public void And(PrimitiveColumnContainer<int> left, PrimitiveColumnContainer<int> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] & otherSpan[i]);
                }
            }
        }
        public void And(PrimitiveColumnContainer<int> column, int scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] & scalar);
                }
            }
        }
 
        public void And(int scalar, PrimitiveColumnContainer<int> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (int)(scalar & span[i]);
                }
            }
        }
        public void Or(PrimitiveColumnContainer<int> left, PrimitiveColumnContainer<int> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] | otherSpan[i]);
                }
            }
        }
        public void Or(PrimitiveColumnContainer<int> column, int scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] | scalar);
                }
            }
        }
 
        public void Or(int scalar, PrimitiveColumnContainer<int> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (int)(scalar | span[i]);
                }
            }
        }
        public void Xor(PrimitiveColumnContainer<int> left, PrimitiveColumnContainer<int> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] ^ otherSpan[i]);
                }
            }
        }
        public void Xor(PrimitiveColumnContainer<int> column, int scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] ^ scalar);
                }
            }
        }
 
        public void Xor(int scalar, PrimitiveColumnContainer<int> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (int)(scalar ^ span[i]);
                }
            }
        }
        public void LeftShift(PrimitiveColumnContainer<int> column, int value)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] << value);
                }
            }
        }
        public void RightShift(PrimitiveColumnContainer<int> column, int value)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] >> value);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<int> left, PrimitiveColumnContainer<int> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == otherSpan[i]);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<int> column, int scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == scalar);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<int> left, PrimitiveColumnContainer<int> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != otherSpan[i]);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<int> column, int scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != scalar);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<int> left, PrimitiveColumnContainer<int> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<int> column, int scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= scalar);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<int> left, PrimitiveColumnContainer<int> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<int> column, int scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= scalar);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<int> left, PrimitiveColumnContainer<int> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<int> column, int scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > scalar);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<int> left, PrimitiveColumnContainer<int> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<int> column, int scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<int>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < scalar);
                }
            }
        }
    }
    internal class LongArithmetic : IPrimitiveDataFrameColumnArithmetic<long>
    {
        public void Add(PrimitiveColumnContainer<long> left, PrimitiveColumnContainer<long> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] + otherSpan[i]);
                }
            }
        }
        public void Add(PrimitiveColumnContainer<long> column, long scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] + scalar);
                }
            }
        }
 
        public void Add(long scalar, PrimitiveColumnContainer<long> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (long)(scalar + span[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<long> left, PrimitiveColumnContainer<long> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] - otherSpan[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<long> column, long scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] - scalar);
                }
            }
        }
 
        public void Subtract(long scalar, PrimitiveColumnContainer<long> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (long)(scalar - span[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<long> left, PrimitiveColumnContainer<long> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] * otherSpan[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<long> column, long scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] * scalar);
                }
            }
        }
 
        public void Multiply(long scalar, PrimitiveColumnContainer<long> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (long)(scalar * span[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<long> left, PrimitiveColumnContainer<long> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] / otherSpan[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<long> column, long scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] / scalar);
                }
            }
        }
 
        public void Divide(long scalar, PrimitiveColumnContainer<long> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (long)(scalar / span[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<long> left, PrimitiveColumnContainer<long> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] % otherSpan[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<long> column, long scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] % scalar);
                }
            }
        }
 
        public void Modulo(long scalar, PrimitiveColumnContainer<long> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (long)(scalar % span[i]);
                }
            }
        }
        public void And(PrimitiveColumnContainer<long> left, PrimitiveColumnContainer<long> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] & otherSpan[i]);
                }
            }
        }
        public void And(PrimitiveColumnContainer<long> column, long scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] & scalar);
                }
            }
        }
 
        public void And(long scalar, PrimitiveColumnContainer<long> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (long)(scalar & span[i]);
                }
            }
        }
        public void Or(PrimitiveColumnContainer<long> left, PrimitiveColumnContainer<long> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] | otherSpan[i]);
                }
            }
        }
        public void Or(PrimitiveColumnContainer<long> column, long scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] | scalar);
                }
            }
        }
 
        public void Or(long scalar, PrimitiveColumnContainer<long> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (long)(scalar | span[i]);
                }
            }
        }
        public void Xor(PrimitiveColumnContainer<long> left, PrimitiveColumnContainer<long> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] ^ otherSpan[i]);
                }
            }
        }
        public void Xor(PrimitiveColumnContainer<long> column, long scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] ^ scalar);
                }
            }
        }
 
        public void Xor(long scalar, PrimitiveColumnContainer<long> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (long)(scalar ^ span[i]);
                }
            }
        }
        public void LeftShift(PrimitiveColumnContainer<long> column, int value)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] << value);
                }
            }
        }
        public void RightShift(PrimitiveColumnContainer<long> column, int value)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] >> value);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<long> left, PrimitiveColumnContainer<long> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == otherSpan[i]);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<long> column, long scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == scalar);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<long> left, PrimitiveColumnContainer<long> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != otherSpan[i]);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<long> column, long scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != scalar);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<long> left, PrimitiveColumnContainer<long> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<long> column, long scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= scalar);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<long> left, PrimitiveColumnContainer<long> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<long> column, long scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= scalar);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<long> left, PrimitiveColumnContainer<long> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<long> column, long scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > scalar);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<long> left, PrimitiveColumnContainer<long> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<long> column, long scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<long>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < scalar);
                }
            }
        }
    }
    internal class SByteArithmetic : IPrimitiveDataFrameColumnArithmetic<sbyte>
    {
        public void Add(PrimitiveColumnContainer<sbyte> left, PrimitiveColumnContainer<sbyte> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] + otherSpan[i]);
                }
            }
        }
        public void Add(PrimitiveColumnContainer<sbyte> column, sbyte scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] + scalar);
                }
            }
        }
 
        public void Add(sbyte scalar, PrimitiveColumnContainer<sbyte> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (sbyte)(scalar + span[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<sbyte> left, PrimitiveColumnContainer<sbyte> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] - otherSpan[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<sbyte> column, sbyte scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] - scalar);
                }
            }
        }
 
        public void Subtract(sbyte scalar, PrimitiveColumnContainer<sbyte> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (sbyte)(scalar - span[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<sbyte> left, PrimitiveColumnContainer<sbyte> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] * otherSpan[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<sbyte> column, sbyte scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] * scalar);
                }
            }
        }
 
        public void Multiply(sbyte scalar, PrimitiveColumnContainer<sbyte> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (sbyte)(scalar * span[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<sbyte> left, PrimitiveColumnContainer<sbyte> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] / otherSpan[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<sbyte> column, sbyte scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] / scalar);
                }
            }
        }
 
        public void Divide(sbyte scalar, PrimitiveColumnContainer<sbyte> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (sbyte)(scalar / span[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<sbyte> left, PrimitiveColumnContainer<sbyte> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] % otherSpan[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<sbyte> column, sbyte scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] % scalar);
                }
            }
        }
 
        public void Modulo(sbyte scalar, PrimitiveColumnContainer<sbyte> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (sbyte)(scalar % span[i]);
                }
            }
        }
        public void And(PrimitiveColumnContainer<sbyte> left, PrimitiveColumnContainer<sbyte> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] & otherSpan[i]);
                }
            }
        }
        public void And(PrimitiveColumnContainer<sbyte> column, sbyte scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] & scalar);
                }
            }
        }
 
        public void And(sbyte scalar, PrimitiveColumnContainer<sbyte> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (sbyte)(scalar & span[i]);
                }
            }
        }
        public void Or(PrimitiveColumnContainer<sbyte> left, PrimitiveColumnContainer<sbyte> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] | otherSpan[i]);
                }
            }
        }
        public void Or(PrimitiveColumnContainer<sbyte> column, sbyte scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] | scalar);
                }
            }
        }
 
        public void Or(sbyte scalar, PrimitiveColumnContainer<sbyte> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (sbyte)(scalar | span[i]);
                }
            }
        }
        public void Xor(PrimitiveColumnContainer<sbyte> left, PrimitiveColumnContainer<sbyte> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] ^ otherSpan[i]);
                }
            }
        }
        public void Xor(PrimitiveColumnContainer<sbyte> column, sbyte scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] ^ scalar);
                }
            }
        }
 
        public void Xor(sbyte scalar, PrimitiveColumnContainer<sbyte> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (sbyte)(scalar ^ span[i]);
                }
            }
        }
        public void LeftShift(PrimitiveColumnContainer<sbyte> column, int value)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] << value);
                }
            }
        }
        public void RightShift(PrimitiveColumnContainer<sbyte> column, int value)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] >> value);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<sbyte> left, PrimitiveColumnContainer<sbyte> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == otherSpan[i]);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<sbyte> column, sbyte scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == scalar);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<sbyte> left, PrimitiveColumnContainer<sbyte> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != otherSpan[i]);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<sbyte> column, sbyte scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != scalar);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<sbyte> left, PrimitiveColumnContainer<sbyte> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<sbyte> column, sbyte scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= scalar);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<sbyte> left, PrimitiveColumnContainer<sbyte> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<sbyte> column, sbyte scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= scalar);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<sbyte> left, PrimitiveColumnContainer<sbyte> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<sbyte> column, sbyte scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > scalar);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<sbyte> left, PrimitiveColumnContainer<sbyte> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<sbyte> column, sbyte scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<sbyte>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < scalar);
                }
            }
        }
    }
    internal class ShortArithmetic : IPrimitiveDataFrameColumnArithmetic<short>
    {
        public void Add(PrimitiveColumnContainer<short> left, PrimitiveColumnContainer<short> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] + otherSpan[i]);
                }
            }
        }
        public void Add(PrimitiveColumnContainer<short> column, short scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] + scalar);
                }
            }
        }
 
        public void Add(short scalar, PrimitiveColumnContainer<short> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (short)(scalar + span[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<short> left, PrimitiveColumnContainer<short> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] - otherSpan[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<short> column, short scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] - scalar);
                }
            }
        }
 
        public void Subtract(short scalar, PrimitiveColumnContainer<short> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (short)(scalar - span[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<short> left, PrimitiveColumnContainer<short> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] * otherSpan[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<short> column, short scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] * scalar);
                }
            }
        }
 
        public void Multiply(short scalar, PrimitiveColumnContainer<short> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (short)(scalar * span[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<short> left, PrimitiveColumnContainer<short> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] / otherSpan[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<short> column, short scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] / scalar);
                }
            }
        }
 
        public void Divide(short scalar, PrimitiveColumnContainer<short> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (short)(scalar / span[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<short> left, PrimitiveColumnContainer<short> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] % otherSpan[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<short> column, short scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] % scalar);
                }
            }
        }
 
        public void Modulo(short scalar, PrimitiveColumnContainer<short> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (short)(scalar % span[i]);
                }
            }
        }
        public void And(PrimitiveColumnContainer<short> left, PrimitiveColumnContainer<short> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] & otherSpan[i]);
                }
            }
        }
        public void And(PrimitiveColumnContainer<short> column, short scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] & scalar);
                }
            }
        }
 
        public void And(short scalar, PrimitiveColumnContainer<short> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (short)(scalar & span[i]);
                }
            }
        }
        public void Or(PrimitiveColumnContainer<short> left, PrimitiveColumnContainer<short> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] | otherSpan[i]);
                }
            }
        }
        public void Or(PrimitiveColumnContainer<short> column, short scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] | scalar);
                }
            }
        }
 
        public void Or(short scalar, PrimitiveColumnContainer<short> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (short)(scalar | span[i]);
                }
            }
        }
        public void Xor(PrimitiveColumnContainer<short> left, PrimitiveColumnContainer<short> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] ^ otherSpan[i]);
                }
            }
        }
        public void Xor(PrimitiveColumnContainer<short> column, short scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] ^ scalar);
                }
            }
        }
 
        public void Xor(short scalar, PrimitiveColumnContainer<short> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (short)(scalar ^ span[i]);
                }
            }
        }
        public void LeftShift(PrimitiveColumnContainer<short> column, int value)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] << value);
                }
            }
        }
        public void RightShift(PrimitiveColumnContainer<short> column, int value)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] >> value);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<short> left, PrimitiveColumnContainer<short> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == otherSpan[i]);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<short> column, short scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == scalar);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<short> left, PrimitiveColumnContainer<short> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != otherSpan[i]);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<short> column, short scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != scalar);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<short> left, PrimitiveColumnContainer<short> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<short> column, short scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= scalar);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<short> left, PrimitiveColumnContainer<short> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<short> column, short scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= scalar);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<short> left, PrimitiveColumnContainer<short> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<short> column, short scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > scalar);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<short> left, PrimitiveColumnContainer<short> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<short> column, short scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<short>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < scalar);
                }
            }
        }
    }
    internal class UIntArithmetic : IPrimitiveDataFrameColumnArithmetic<uint>
    {
        public void Add(PrimitiveColumnContainer<uint> left, PrimitiveColumnContainer<uint> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] + otherSpan[i]);
                }
            }
        }
        public void Add(PrimitiveColumnContainer<uint> column, uint scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] + scalar);
                }
            }
        }
 
        public void Add(uint scalar, PrimitiveColumnContainer<uint> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (uint)(scalar + span[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<uint> left, PrimitiveColumnContainer<uint> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] - otherSpan[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<uint> column, uint scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] - scalar);
                }
            }
        }
 
        public void Subtract(uint scalar, PrimitiveColumnContainer<uint> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (uint)(scalar - span[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<uint> left, PrimitiveColumnContainer<uint> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] * otherSpan[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<uint> column, uint scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] * scalar);
                }
            }
        }
 
        public void Multiply(uint scalar, PrimitiveColumnContainer<uint> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (uint)(scalar * span[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<uint> left, PrimitiveColumnContainer<uint> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] / otherSpan[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<uint> column, uint scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] / scalar);
                }
            }
        }
 
        public void Divide(uint scalar, PrimitiveColumnContainer<uint> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (uint)(scalar / span[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<uint> left, PrimitiveColumnContainer<uint> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] % otherSpan[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<uint> column, uint scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] % scalar);
                }
            }
        }
 
        public void Modulo(uint scalar, PrimitiveColumnContainer<uint> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (uint)(scalar % span[i]);
                }
            }
        }
        public void And(PrimitiveColumnContainer<uint> left, PrimitiveColumnContainer<uint> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] & otherSpan[i]);
                }
            }
        }
        public void And(PrimitiveColumnContainer<uint> column, uint scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] & scalar);
                }
            }
        }
 
        public void And(uint scalar, PrimitiveColumnContainer<uint> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (uint)(scalar & span[i]);
                }
            }
        }
        public void Or(PrimitiveColumnContainer<uint> left, PrimitiveColumnContainer<uint> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] | otherSpan[i]);
                }
            }
        }
        public void Or(PrimitiveColumnContainer<uint> column, uint scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] | scalar);
                }
            }
        }
 
        public void Or(uint scalar, PrimitiveColumnContainer<uint> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (uint)(scalar | span[i]);
                }
            }
        }
        public void Xor(PrimitiveColumnContainer<uint> left, PrimitiveColumnContainer<uint> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] ^ otherSpan[i]);
                }
            }
        }
        public void Xor(PrimitiveColumnContainer<uint> column, uint scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] ^ scalar);
                }
            }
        }
 
        public void Xor(uint scalar, PrimitiveColumnContainer<uint> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (uint)(scalar ^ span[i]);
                }
            }
        }
        public void LeftShift(PrimitiveColumnContainer<uint> column, int value)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] << value);
                }
            }
        }
        public void RightShift(PrimitiveColumnContainer<uint> column, int value)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] >> value);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<uint> left, PrimitiveColumnContainer<uint> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == otherSpan[i]);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<uint> column, uint scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == scalar);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<uint> left, PrimitiveColumnContainer<uint> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != otherSpan[i]);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<uint> column, uint scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != scalar);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<uint> left, PrimitiveColumnContainer<uint> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<uint> column, uint scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= scalar);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<uint> left, PrimitiveColumnContainer<uint> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<uint> column, uint scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= scalar);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<uint> left, PrimitiveColumnContainer<uint> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<uint> column, uint scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > scalar);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<uint> left, PrimitiveColumnContainer<uint> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<uint> column, uint scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<uint>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < scalar);
                }
            }
        }
    }
    internal class ULongArithmetic : IPrimitiveDataFrameColumnArithmetic<ulong>
    {
        public void Add(PrimitiveColumnContainer<ulong> left, PrimitiveColumnContainer<ulong> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] + otherSpan[i]);
                }
            }
        }
        public void Add(PrimitiveColumnContainer<ulong> column, ulong scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] + scalar);
                }
            }
        }
 
        public void Add(ulong scalar, PrimitiveColumnContainer<ulong> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (ulong)(scalar + span[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<ulong> left, PrimitiveColumnContainer<ulong> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] - otherSpan[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<ulong> column, ulong scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] - scalar);
                }
            }
        }
 
        public void Subtract(ulong scalar, PrimitiveColumnContainer<ulong> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (ulong)(scalar - span[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<ulong> left, PrimitiveColumnContainer<ulong> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] * otherSpan[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<ulong> column, ulong scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] * scalar);
                }
            }
        }
 
        public void Multiply(ulong scalar, PrimitiveColumnContainer<ulong> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (ulong)(scalar * span[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<ulong> left, PrimitiveColumnContainer<ulong> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] / otherSpan[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<ulong> column, ulong scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] / scalar);
                }
            }
        }
 
        public void Divide(ulong scalar, PrimitiveColumnContainer<ulong> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (ulong)(scalar / span[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<ulong> left, PrimitiveColumnContainer<ulong> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] % otherSpan[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<ulong> column, ulong scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] % scalar);
                }
            }
        }
 
        public void Modulo(ulong scalar, PrimitiveColumnContainer<ulong> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (ulong)(scalar % span[i]);
                }
            }
        }
        public void And(PrimitiveColumnContainer<ulong> left, PrimitiveColumnContainer<ulong> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] & otherSpan[i]);
                }
            }
        }
        public void And(PrimitiveColumnContainer<ulong> column, ulong scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] & scalar);
                }
            }
        }
 
        public void And(ulong scalar, PrimitiveColumnContainer<ulong> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (ulong)(scalar & span[i]);
                }
            }
        }
        public void Or(PrimitiveColumnContainer<ulong> left, PrimitiveColumnContainer<ulong> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] | otherSpan[i]);
                }
            }
        }
        public void Or(PrimitiveColumnContainer<ulong> column, ulong scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] | scalar);
                }
            }
        }
 
        public void Or(ulong scalar, PrimitiveColumnContainer<ulong> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (ulong)(scalar | span[i]);
                }
            }
        }
        public void Xor(PrimitiveColumnContainer<ulong> left, PrimitiveColumnContainer<ulong> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] ^ otherSpan[i]);
                }
            }
        }
        public void Xor(PrimitiveColumnContainer<ulong> column, ulong scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] ^ scalar);
                }
            }
        }
 
        public void Xor(ulong scalar, PrimitiveColumnContainer<ulong> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (ulong)(scalar ^ span[i]);
                }
            }
        }
        public void LeftShift(PrimitiveColumnContainer<ulong> column, int value)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] << value);
                }
            }
        }
        public void RightShift(PrimitiveColumnContainer<ulong> column, int value)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] >> value);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<ulong> left, PrimitiveColumnContainer<ulong> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == otherSpan[i]);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<ulong> column, ulong scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == scalar);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<ulong> left, PrimitiveColumnContainer<ulong> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != otherSpan[i]);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<ulong> column, ulong scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != scalar);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<ulong> left, PrimitiveColumnContainer<ulong> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<ulong> column, ulong scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= scalar);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<ulong> left, PrimitiveColumnContainer<ulong> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<ulong> column, ulong scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= scalar);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<ulong> left, PrimitiveColumnContainer<ulong> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<ulong> column, ulong scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > scalar);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<ulong> left, PrimitiveColumnContainer<ulong> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<ulong> column, ulong scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ulong>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < scalar);
                }
            }
        }
    }
    internal class UShortArithmetic : IPrimitiveDataFrameColumnArithmetic<ushort>
    {
        public void Add(PrimitiveColumnContainer<ushort> left, PrimitiveColumnContainer<ushort> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] + otherSpan[i]);
                }
            }
        }
        public void Add(PrimitiveColumnContainer<ushort> column, ushort scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] + scalar);
                }
            }
        }
 
        public void Add(ushort scalar, PrimitiveColumnContainer<ushort> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (ushort)(scalar + span[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<ushort> left, PrimitiveColumnContainer<ushort> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] - otherSpan[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<ushort> column, ushort scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] - scalar);
                }
            }
        }
 
        public void Subtract(ushort scalar, PrimitiveColumnContainer<ushort> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (ushort)(scalar - span[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<ushort> left, PrimitiveColumnContainer<ushort> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] * otherSpan[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<ushort> column, ushort scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] * scalar);
                }
            }
        }
 
        public void Multiply(ushort scalar, PrimitiveColumnContainer<ushort> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (ushort)(scalar * span[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<ushort> left, PrimitiveColumnContainer<ushort> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] / otherSpan[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<ushort> column, ushort scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] / scalar);
                }
            }
        }
 
        public void Divide(ushort scalar, PrimitiveColumnContainer<ushort> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (ushort)(scalar / span[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<ushort> left, PrimitiveColumnContainer<ushort> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] % otherSpan[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<ushort> column, ushort scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] % scalar);
                }
            }
        }
 
        public void Modulo(ushort scalar, PrimitiveColumnContainer<ushort> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (ushort)(scalar % span[i]);
                }
            }
        }
        public void And(PrimitiveColumnContainer<ushort> left, PrimitiveColumnContainer<ushort> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] & otherSpan[i]);
                }
            }
        }
        public void And(PrimitiveColumnContainer<ushort> column, ushort scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] & scalar);
                }
            }
        }
 
        public void And(ushort scalar, PrimitiveColumnContainer<ushort> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (ushort)(scalar & span[i]);
                }
            }
        }
        public void Or(PrimitiveColumnContainer<ushort> left, PrimitiveColumnContainer<ushort> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] | otherSpan[i]);
                }
            }
        }
        public void Or(PrimitiveColumnContainer<ushort> column, ushort scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] | scalar);
                }
            }
        }
 
        public void Or(ushort scalar, PrimitiveColumnContainer<ushort> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (ushort)(scalar | span[i]);
                }
            }
        }
        public void Xor(PrimitiveColumnContainer<ushort> left, PrimitiveColumnContainer<ushort> right)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] ^ otherSpan[i]);
                }
            }
        }
        public void Xor(PrimitiveColumnContainer<ushort> column, ushort scalar)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] ^ scalar);
                }
            }
        }
 
        public void Xor(ushort scalar, PrimitiveColumnContainer<ushort> column)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b]; 
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++) 
                {
                    span[i] = (ushort)(scalar ^ span[i]);
                }
            }
        }
        public void LeftShift(PrimitiveColumnContainer<ushort> column, int value)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] << value);
                }
            }
        }
        public void RightShift(PrimitiveColumnContainer<ushort> column, int value)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] >> value);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<ushort> left, PrimitiveColumnContainer<ushort> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == otherSpan[i]);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<ushort> column, ushort scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == scalar);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<ushort> left, PrimitiveColumnContainer<ushort> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != otherSpan[i]);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<ushort> column, ushort scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != scalar);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<ushort> left, PrimitiveColumnContainer<ushort> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<ushort> column, ushort scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= scalar);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<ushort> left, PrimitiveColumnContainer<ushort> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<ushort> column, ushort scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= scalar);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<ushort> left, PrimitiveColumnContainer<ushort> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<ushort> column, ushort scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > scalar);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<ushort> left, PrimitiveColumnContainer<ushort> right, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<ushort> column, ushort scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0 ; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<ushort>.GetMutableBuffer(buffer);
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
