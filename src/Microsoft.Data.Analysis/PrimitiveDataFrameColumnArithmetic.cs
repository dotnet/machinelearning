
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from PrimitiveDataFrameColumnArithmetic.tt. Do not modify directly

using System;

namespace Microsoft.Data.Analysis
{
    internal interface IPrimitiveDataFrameColumnArithmetic<T>
        where T : unmanaged
    {
        void Add(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> result);
        void Add(PrimitiveColumnContainer<T> column, T scalar);
        void Add(T scalar, PrimitiveColumnContainer<T> column);
        void Subtract(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> result);
        void Subtract(PrimitiveColumnContainer<T> column, T scalar);
        void Subtract(T scalar, PrimitiveColumnContainer<T> column);
        void Multiply(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> result);
        void Multiply(PrimitiveColumnContainer<T> column, T scalar);
        void Multiply(T scalar, PrimitiveColumnContainer<T> column);
        void Divide(ReadOnlySpan<T> span, ReadOnlySpan<T> otherSpan, ReadOnlySpan<byte> otherValiditySpan, Span<T> resultSpan, Span<byte> returnValiditySpan);
        void Divide(PrimitiveColumnContainer<T> column, T scalar);
        void Divide(T scalar, PrimitiveColumnContainer<T> column);
        void Modulo(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> result);
        void Modulo(PrimitiveColumnContainer<T> column, T scalar);
        void Modulo(T scalar, PrimitiveColumnContainer<T> column);
        void And(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> result);
        void And(PrimitiveColumnContainer<T> column, T scalar);
        void And(T scalar, PrimitiveColumnContainer<T> column);
        void Or(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> result);
        void Or(PrimitiveColumnContainer<T> column, T scalar);
        void Or(T scalar, PrimitiveColumnContainer<T> column);
        void Xor(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> result);
        void Xor(PrimitiveColumnContainer<T> column, T scalar);
        void Xor(T scalar, PrimitiveColumnContainer<T> column);
        void LeftShift(PrimitiveColumnContainer<T> column, int value);
        void RightShift(PrimitiveColumnContainer<T> column, int value);
        PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right);
        PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<T> column, T scalar);
        PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right);
        PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<T> column, T scalar);
        PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right);
        PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<T> column, T scalar);
        PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right);
        PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<T> column, T scalar);
        PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right);
        PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<T> column, T scalar);
        PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right);
        PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<T> column, T scalar);
    }

    internal class PrimitiveDataFrameColumnArithmetic<T> : IPrimitiveDataFrameColumnArithmetic<T>
        where T : unmanaged
    {
        public static IPrimitiveDataFrameColumnArithmetic<T> Instance { get; } = PrimitiveDataFrameColumnArithmetic.GetArithmetic<T>();

        public virtual void Add(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> result) => throw new NotSupportedException();
        public virtual void Add(PrimitiveColumnContainer<T> column, T scalar) => throw new NotSupportedException();
        public virtual void Add(T scalar, PrimitiveColumnContainer<T> column) => throw new NotSupportedException();
        public virtual void Subtract(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> result) => throw new NotSupportedException();
        public virtual void Subtract(PrimitiveColumnContainer<T> column, T scalar) => throw new NotSupportedException();
        public virtual void Subtract(T scalar, PrimitiveColumnContainer<T> column) => throw new NotSupportedException();
        public virtual void Multiply(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> result) => throw new NotSupportedException();
        public virtual void Multiply(PrimitiveColumnContainer<T> column, T scalar) => throw new NotSupportedException();
        public virtual void Multiply(T scalar, PrimitiveColumnContainer<T> column) => throw new NotSupportedException();
        public virtual void Divide(ReadOnlySpan<T> span, ReadOnlySpan<T> otherSpan, ReadOnlySpan<byte> otherValiditySpan, Span<T> resultSpan, Span<byte> returnValiditySpan) => throw new NotSupportedException();
        public virtual void Divide(PrimitiveColumnContainer<T> column, T scalar) => throw new NotSupportedException();
        public virtual void Divide(T scalar, PrimitiveColumnContainer<T> column) => throw new NotSupportedException();
        public virtual void Modulo(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> result) => throw new NotSupportedException();
        public virtual void Modulo(PrimitiveColumnContainer<T> column, T scalar) => throw new NotSupportedException();
        public virtual void Modulo(T scalar, PrimitiveColumnContainer<T> column) => throw new NotSupportedException();
        public virtual void And(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> result) => throw new NotSupportedException();
        public virtual void And(PrimitiveColumnContainer<T> column, T scalar) => throw new NotSupportedException();
        public virtual void And(T scalar, PrimitiveColumnContainer<T> column) => throw new NotSupportedException();
        public virtual void Or(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> result) => throw new NotSupportedException();
        public virtual void Or(PrimitiveColumnContainer<T> column, T scalar) => throw new NotSupportedException();
        public virtual void Or(T scalar, PrimitiveColumnContainer<T> column) => throw new NotSupportedException();
        public virtual void Xor(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> result) => throw new NotSupportedException();
        public virtual void Xor(PrimitiveColumnContainer<T> column, T scalar) => throw new NotSupportedException();
        public virtual void Xor(T scalar, PrimitiveColumnContainer<T> column) => throw new NotSupportedException();
        public virtual void LeftShift(PrimitiveColumnContainer<T> column, int value) => throw new NotSupportedException();
        public virtual void RightShift(PrimitiveColumnContainer<T> column, int value) => throw new NotSupportedException();
        public virtual PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right) => throw new NotSupportedException();
        public virtual PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<T> column, T scalar) => throw new NotSupportedException();
        public virtual PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right) => throw new NotSupportedException();
        public virtual PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<T> column, T scalar) => throw new NotSupportedException();
        public virtual PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right) => throw new NotSupportedException();
        public virtual PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<T> column, T scalar) => throw new NotSupportedException();
        public virtual PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right) => throw new NotSupportedException();
        public virtual PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<T> column, T scalar) => throw new NotSupportedException();
        public virtual PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right) => throw new NotSupportedException();
        public virtual PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<T> column, T scalar) => throw new NotSupportedException();
        public virtual PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right) => throw new NotSupportedException();
        public virtual PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<T> column, T scalar) => throw new NotSupportedException();

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
            else if (typeof(T) == typeof(DateTime))
            {
                return (IPrimitiveDataFrameColumnArithmetic<T>)new DateTimeArithmetic();
            }
            throw new NotSupportedException();
        }
    }
    internal class BoolArithmetic : PrimitiveDataFrameColumnArithmetic<bool>
    {
        public override void And(ReadOnlySpan<bool> span, ReadOnlySpan<bool> otherSpan, Span<bool> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (bool)(span[i] & otherSpan[i]);
            }
        }

        public override void And(PrimitiveColumnContainer<bool> column, bool scalar)
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

        public override void And(bool scalar, PrimitiveColumnContainer<bool> column)
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
        public override void Or(ReadOnlySpan<bool> span, ReadOnlySpan<bool> otherSpan, Span<bool> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (bool)(span[i] | otherSpan[i]);
            }
        }

        public override void Or(PrimitiveColumnContainer<bool> column, bool scalar)
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

        public override void Or(bool scalar, PrimitiveColumnContainer<bool> column)
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
        public override void Xor(ReadOnlySpan<bool> span, ReadOnlySpan<bool> otherSpan, Span<bool> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (bool)(span[i] ^ otherSpan[i]);
            }
        }

        public override void Xor(PrimitiveColumnContainer<bool> column, bool scalar)
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

        public override void Xor(bool scalar, PrimitiveColumnContainer<bool> column)
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

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<bool> left, PrimitiveColumnContainer<bool> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<bool> column, bool scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<bool> left, PrimitiveColumnContainer<bool> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<bool> column, bool scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != scalar);
                }
            }
            return ret;
        }
    }
    internal class ByteArithmetic : PrimitiveDataFrameColumnArithmetic<byte>
    {
        public override void Add(ReadOnlySpan<byte> span, ReadOnlySpan<byte> otherSpan, Span<byte> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (byte)(span[i] + otherSpan[i]);
            }
        }

        public override void Add(PrimitiveColumnContainer<byte> column, byte scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] + scalar);
                }
            }
        }

        public override void Add(byte scalar, PrimitiveColumnContainer<byte> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(scalar + span[i]);
                }
            }
        }
        public override void Subtract(ReadOnlySpan<byte> span, ReadOnlySpan<byte> otherSpan, Span<byte> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (byte)(span[i] - otherSpan[i]);
            }
        }

        public override void Subtract(PrimitiveColumnContainer<byte> column, byte scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] - scalar);
                }
            }
        }

        public override void Subtract(byte scalar, PrimitiveColumnContainer<byte> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(scalar - span[i]);
                }
            }
        }
        public override void Multiply(ReadOnlySpan<byte> span, ReadOnlySpan<byte> otherSpan, Span<byte> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (byte)(span[i] * otherSpan[i]);
            }
        }

        public override void Multiply(PrimitiveColumnContainer<byte> column, byte scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] * scalar);
                }
            }
        }

        public override void Multiply(byte scalar, PrimitiveColumnContainer<byte> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(scalar * span[i]);
                }
            }
        }
        public override void Divide(ReadOnlySpan<byte> span, ReadOnlySpan<byte> otherSpan, ReadOnlySpan<byte> otherValiditySpan, Span<byte> resultSpan, Span<byte> resultValiditySpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                if (BitmapHelper.IsValid(otherValiditySpan, i))
                    resultSpan[i] = (byte)(span[i] / otherSpan[i]);
                else
                    BitmapHelper.ClearBit(resultValiditySpan, i);
            }
        }

        public override void Divide(PrimitiveColumnContainer<byte> column, byte scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] / scalar);
                }
            }
        }

        public override void Divide(byte scalar, PrimitiveColumnContainer<byte> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(scalar / span[i]);
                }
            }
        }
        public override void Modulo(ReadOnlySpan<byte> span, ReadOnlySpan<byte> otherSpan, Span<byte> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (byte)(span[i] % otherSpan[i]);
            }
        }

        public override void Modulo(PrimitiveColumnContainer<byte> column, byte scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] % scalar);
                }
            }
        }

        public override void Modulo(byte scalar, PrimitiveColumnContainer<byte> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(scalar % span[i]);
                }
            }
        }
        public override void And(ReadOnlySpan<byte> span, ReadOnlySpan<byte> otherSpan, Span<byte> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (byte)(span[i] & otherSpan[i]);
            }
        }

        public override void And(PrimitiveColumnContainer<byte> column, byte scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] & scalar);
                }
            }
        }

        public override void And(byte scalar, PrimitiveColumnContainer<byte> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(scalar & span[i]);
                }
            }
        }
        public override void Or(ReadOnlySpan<byte> span, ReadOnlySpan<byte> otherSpan, Span<byte> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (byte)(span[i] | otherSpan[i]);
            }
        }

        public override void Or(PrimitiveColumnContainer<byte> column, byte scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] | scalar);
                }
            }
        }

        public override void Or(byte scalar, PrimitiveColumnContainer<byte> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(scalar | span[i]);
                }
            }
        }
        public override void Xor(ReadOnlySpan<byte> span, ReadOnlySpan<byte> otherSpan, Span<byte> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (byte)(span[i] ^ otherSpan[i]);
            }
        }

        public override void Xor(PrimitiveColumnContainer<byte> column, byte scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] ^ scalar);
                }
            }
        }

        public override void Xor(byte scalar, PrimitiveColumnContainer<byte> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(scalar ^ span[i]);
                }
            }
        }

        public override void LeftShift(PrimitiveColumnContainer<byte> column, int value)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] << value);
                }
            }
        }

        public override void RightShift(PrimitiveColumnContainer<byte> column, int value)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (byte)(span[i] >> value);
                }
            }
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<byte> left, PrimitiveColumnContainer<byte> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<byte> column, byte scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<byte> left, PrimitiveColumnContainer<byte> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<byte> column, byte scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<byte> left, PrimitiveColumnContainer<byte> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<byte> column, byte scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<byte> left, PrimitiveColumnContainer<byte> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<byte> column, byte scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<byte> left, PrimitiveColumnContainer<byte> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<byte> column, byte scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<byte> left, PrimitiveColumnContainer<byte> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<byte> column, byte scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < scalar);
                }
            }
            return ret;
        }
    }
    internal class CharArithmetic : PrimitiveDataFrameColumnArithmetic<char>
    {
        public override void Add(ReadOnlySpan<char> span, ReadOnlySpan<char> otherSpan, Span<char> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (char)(span[i] + otherSpan[i]);
            }
        }

        public override void Add(PrimitiveColumnContainer<char> column, char scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] + scalar);
                }
            }
        }

        public override void Add(char scalar, PrimitiveColumnContainer<char> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(scalar + span[i]);
                }
            }
        }
        public override void Subtract(ReadOnlySpan<char> span, ReadOnlySpan<char> otherSpan, Span<char> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (char)(span[i] - otherSpan[i]);
            }
        }

        public override void Subtract(PrimitiveColumnContainer<char> column, char scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] - scalar);
                }
            }
        }

        public override void Subtract(char scalar, PrimitiveColumnContainer<char> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(scalar - span[i]);
                }
            }
        }
        public override void Multiply(ReadOnlySpan<char> span, ReadOnlySpan<char> otherSpan, Span<char> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (char)(span[i] * otherSpan[i]);
            }
        }

        public override void Multiply(PrimitiveColumnContainer<char> column, char scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] * scalar);
                }
            }
        }

        public override void Multiply(char scalar, PrimitiveColumnContainer<char> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(scalar * span[i]);
                }
            }
        }
        public override void Divide(ReadOnlySpan<char> span, ReadOnlySpan<char> otherSpan, ReadOnlySpan<byte> otherValiditySpan, Span<char> resultSpan, Span<byte> resultValiditySpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                if (BitmapHelper.IsValid(otherValiditySpan, i))
                    resultSpan[i] = (char)(span[i] / otherSpan[i]);
                else
                    BitmapHelper.ClearBit(resultValiditySpan, i);
            }
        }

        public override void Divide(PrimitiveColumnContainer<char> column, char scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] / scalar);
                }
            }
        }

        public override void Divide(char scalar, PrimitiveColumnContainer<char> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(scalar / span[i]);
                }
            }
        }
        public override void Modulo(ReadOnlySpan<char> span, ReadOnlySpan<char> otherSpan, Span<char> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (char)(span[i] % otherSpan[i]);
            }
        }

        public override void Modulo(PrimitiveColumnContainer<char> column, char scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] % scalar);
                }
            }
        }

        public override void Modulo(char scalar, PrimitiveColumnContainer<char> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(scalar % span[i]);
                }
            }
        }
        public override void And(ReadOnlySpan<char> span, ReadOnlySpan<char> otherSpan, Span<char> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (char)(span[i] & otherSpan[i]);
            }
        }

        public override void And(PrimitiveColumnContainer<char> column, char scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] & scalar);
                }
            }
        }

        public override void And(char scalar, PrimitiveColumnContainer<char> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(scalar & span[i]);
                }
            }
        }
        public override void Or(ReadOnlySpan<char> span, ReadOnlySpan<char> otherSpan, Span<char> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (char)(span[i] | otherSpan[i]);
            }
        }

        public override void Or(PrimitiveColumnContainer<char> column, char scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] | scalar);
                }
            }
        }

        public override void Or(char scalar, PrimitiveColumnContainer<char> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(scalar | span[i]);
                }
            }
        }
        public override void Xor(ReadOnlySpan<char> span, ReadOnlySpan<char> otherSpan, Span<char> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (char)(span[i] ^ otherSpan[i]);
            }
        }

        public override void Xor(PrimitiveColumnContainer<char> column, char scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] ^ scalar);
                }
            }
        }

        public override void Xor(char scalar, PrimitiveColumnContainer<char> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(scalar ^ span[i]);
                }
            }
        }

        public override void LeftShift(PrimitiveColumnContainer<char> column, int value)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] << value);
                }
            }
        }

        public override void RightShift(PrimitiveColumnContainer<char> column, int value)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (char)(span[i] >> value);
                }
            }
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<char> left, PrimitiveColumnContainer<char> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<char> column, char scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<char> left, PrimitiveColumnContainer<char> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<char> column, char scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<char> left, PrimitiveColumnContainer<char> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<char> column, char scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<char> left, PrimitiveColumnContainer<char> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<char> column, char scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<char> left, PrimitiveColumnContainer<char> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<char> column, char scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<char> left, PrimitiveColumnContainer<char> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<char> column, char scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < scalar);
                }
            }
            return ret;
        }
    }
    internal class DecimalArithmetic : PrimitiveDataFrameColumnArithmetic<decimal>
    {
        public override void Add(ReadOnlySpan<decimal> span, ReadOnlySpan<decimal> otherSpan, Span<decimal> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (decimal)(span[i] + otherSpan[i]);
            }
        }

        public override void Add(PrimitiveColumnContainer<decimal> column, decimal scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(span[i] + scalar);
                }
            }
        }

        public override void Add(decimal scalar, PrimitiveColumnContainer<decimal> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(scalar + span[i]);
                }
            }
        }
        public override void Subtract(ReadOnlySpan<decimal> span, ReadOnlySpan<decimal> otherSpan, Span<decimal> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (decimal)(span[i] - otherSpan[i]);
            }
        }

        public override void Subtract(PrimitiveColumnContainer<decimal> column, decimal scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(span[i] - scalar);
                }
            }
        }

        public override void Subtract(decimal scalar, PrimitiveColumnContainer<decimal> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(scalar - span[i]);
                }
            }
        }
        public override void Multiply(ReadOnlySpan<decimal> span, ReadOnlySpan<decimal> otherSpan, Span<decimal> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (decimal)(span[i] * otherSpan[i]);
            }
        }

        public override void Multiply(PrimitiveColumnContainer<decimal> column, decimal scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(span[i] * scalar);
                }
            }
        }

        public override void Multiply(decimal scalar, PrimitiveColumnContainer<decimal> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(scalar * span[i]);
                }
            }
        }
        public override void Divide(ReadOnlySpan<decimal> span, ReadOnlySpan<decimal> otherSpan, ReadOnlySpan<byte> otherValiditySpan, Span<decimal> resultSpan, Span<byte> resultValiditySpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                if (BitmapHelper.IsValid(otherValiditySpan, i))
                    resultSpan[i] = (decimal)(span[i] / otherSpan[i]);
                else
                    BitmapHelper.ClearBit(resultValiditySpan, i);
            }
        }

        public override void Divide(PrimitiveColumnContainer<decimal> column, decimal scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(span[i] / scalar);
                }
            }
        }

        public override void Divide(decimal scalar, PrimitiveColumnContainer<decimal> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(scalar / span[i]);
                }
            }
        }
        public override void Modulo(ReadOnlySpan<decimal> span, ReadOnlySpan<decimal> otherSpan, Span<decimal> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (decimal)(span[i] % otherSpan[i]);
            }
        }

        public override void Modulo(PrimitiveColumnContainer<decimal> column, decimal scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(span[i] % scalar);
                }
            }
        }

        public override void Modulo(decimal scalar, PrimitiveColumnContainer<decimal> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (decimal)(scalar % span[i]);
                }
            }
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<decimal> left, PrimitiveColumnContainer<decimal> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<decimal> column, decimal scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<decimal> left, PrimitiveColumnContainer<decimal> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<decimal> column, decimal scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<decimal> left, PrimitiveColumnContainer<decimal> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<decimal> column, decimal scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<decimal> left, PrimitiveColumnContainer<decimal> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<decimal> column, decimal scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<decimal> left, PrimitiveColumnContainer<decimal> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<decimal> column, decimal scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<decimal> left, PrimitiveColumnContainer<decimal> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<decimal> column, decimal scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < scalar);
                }
            }
            return ret;
        }
    }
    internal class DoubleArithmetic : PrimitiveDataFrameColumnArithmetic<double>
    {
        public override void Add(ReadOnlySpan<double> span, ReadOnlySpan<double> otherSpan, Span<double> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (double)(span[i] + otherSpan[i]);
            }
        }

        public override void Add(PrimitiveColumnContainer<double> column, double scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(span[i] + scalar);
                }
            }
        }

        public override void Add(double scalar, PrimitiveColumnContainer<double> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(scalar + span[i]);
                }
            }
        }
        public override void Subtract(ReadOnlySpan<double> span, ReadOnlySpan<double> otherSpan, Span<double> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (double)(span[i] - otherSpan[i]);
            }
        }

        public override void Subtract(PrimitiveColumnContainer<double> column, double scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(span[i] - scalar);
                }
            }
        }

        public override void Subtract(double scalar, PrimitiveColumnContainer<double> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(scalar - span[i]);
                }
            }
        }
        public override void Multiply(ReadOnlySpan<double> span, ReadOnlySpan<double> otherSpan, Span<double> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (double)(span[i] * otherSpan[i]);
            }
        }

        public override void Multiply(PrimitiveColumnContainer<double> column, double scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(span[i] * scalar);
                }
            }
        }

        public override void Multiply(double scalar, PrimitiveColumnContainer<double> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(scalar * span[i]);
                }
            }
        }
        public override void Divide(ReadOnlySpan<double> span, ReadOnlySpan<double> otherSpan, ReadOnlySpan<byte> otherValiditySpan, Span<double> resultSpan, Span<byte> resultValiditySpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                if (BitmapHelper.IsValid(otherValiditySpan, i))
                    resultSpan[i] = (double)(span[i] / otherSpan[i]);
                else
                    BitmapHelper.ClearBit(resultValiditySpan, i);
            }
        }

        public override void Divide(PrimitiveColumnContainer<double> column, double scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(span[i] / scalar);
                }
            }
        }

        public override void Divide(double scalar, PrimitiveColumnContainer<double> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(scalar / span[i]);
                }
            }
        }
        public override void Modulo(ReadOnlySpan<double> span, ReadOnlySpan<double> otherSpan, Span<double> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (double)(span[i] % otherSpan[i]);
            }
        }

        public override void Modulo(PrimitiveColumnContainer<double> column, double scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(span[i] % scalar);
                }
            }
        }

        public override void Modulo(double scalar, PrimitiveColumnContainer<double> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (double)(scalar % span[i]);
                }
            }
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<double> left, PrimitiveColumnContainer<double> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<double> column, double scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<double> left, PrimitiveColumnContainer<double> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<double> column, double scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<double> left, PrimitiveColumnContainer<double> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<double> column, double scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<double> left, PrimitiveColumnContainer<double> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<double> column, double scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<double> left, PrimitiveColumnContainer<double> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<double> column, double scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<double> left, PrimitiveColumnContainer<double> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<double> column, double scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < scalar);
                }
            }
            return ret;
        }
    }
    internal class FloatArithmetic : PrimitiveDataFrameColumnArithmetic<float>
    {
        public override void Add(ReadOnlySpan<float> span, ReadOnlySpan<float> otherSpan, Span<float> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (float)(span[i] + otherSpan[i]);
            }
        }

        public override void Add(PrimitiveColumnContainer<float> column, float scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(span[i] + scalar);
                }
            }
        }

        public override void Add(float scalar, PrimitiveColumnContainer<float> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(scalar + span[i]);
                }
            }
        }
        public override void Subtract(ReadOnlySpan<float> span, ReadOnlySpan<float> otherSpan, Span<float> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (float)(span[i] - otherSpan[i]);
            }
        }

        public override void Subtract(PrimitiveColumnContainer<float> column, float scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(span[i] - scalar);
                }
            }
        }

        public override void Subtract(float scalar, PrimitiveColumnContainer<float> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(scalar - span[i]);
                }
            }
        }
        public override void Multiply(ReadOnlySpan<float> span, ReadOnlySpan<float> otherSpan, Span<float> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (float)(span[i] * otherSpan[i]);
            }
        }

        public override void Multiply(PrimitiveColumnContainer<float> column, float scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(span[i] * scalar);
                }
            }
        }

        public override void Multiply(float scalar, PrimitiveColumnContainer<float> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(scalar * span[i]);
                }
            }
        }
        public override void Divide(ReadOnlySpan<float> span, ReadOnlySpan<float> otherSpan, ReadOnlySpan<byte> otherValiditySpan, Span<float> resultSpan, Span<byte> resultValiditySpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                if (BitmapHelper.IsValid(otherValiditySpan, i))
                    resultSpan[i] = (float)(span[i] / otherSpan[i]);
                else
                    BitmapHelper.ClearBit(resultValiditySpan, i);
            }
        }

        public override void Divide(PrimitiveColumnContainer<float> column, float scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(span[i] / scalar);
                }
            }
        }

        public override void Divide(float scalar, PrimitiveColumnContainer<float> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(scalar / span[i]);
                }
            }
        }
        public override void Modulo(ReadOnlySpan<float> span, ReadOnlySpan<float> otherSpan, Span<float> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (float)(span[i] % otherSpan[i]);
            }
        }

        public override void Modulo(PrimitiveColumnContainer<float> column, float scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(span[i] % scalar);
                }
            }
        }

        public override void Modulo(float scalar, PrimitiveColumnContainer<float> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(scalar % span[i]);
                }
            }
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<float> left, PrimitiveColumnContainer<float> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<float> column, float scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<float> left, PrimitiveColumnContainer<float> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<float> column, float scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<float> left, PrimitiveColumnContainer<float> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<float> column, float scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<float> left, PrimitiveColumnContainer<float> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<float> column, float scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<float> left, PrimitiveColumnContainer<float> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<float> column, float scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<float> left, PrimitiveColumnContainer<float> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<float> column, float scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < scalar);
                }
            }
            return ret;
        }
    }
    internal class IntArithmetic : PrimitiveDataFrameColumnArithmetic<int>
    {
        public override void Add(ReadOnlySpan<int> span, ReadOnlySpan<int> otherSpan, Span<int> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (int)(span[i] + otherSpan[i]);
            }
        }

        public override void Add(PrimitiveColumnContainer<int> column, int scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] + scalar);
                }
            }
        }

        public override void Add(int scalar, PrimitiveColumnContainer<int> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(scalar + span[i]);
                }
            }
        }
        public override void Subtract(ReadOnlySpan<int> span, ReadOnlySpan<int> otherSpan, Span<int> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (int)(span[i] - otherSpan[i]);
            }
        }

        public override void Subtract(PrimitiveColumnContainer<int> column, int scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] - scalar);
                }
            }
        }

        public override void Subtract(int scalar, PrimitiveColumnContainer<int> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(scalar - span[i]);
                }
            }
        }
        public override void Multiply(ReadOnlySpan<int> span, ReadOnlySpan<int> otherSpan, Span<int> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (int)(span[i] * otherSpan[i]);
            }
        }

        public override void Multiply(PrimitiveColumnContainer<int> column, int scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] * scalar);
                }
            }
        }

        public override void Multiply(int scalar, PrimitiveColumnContainer<int> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(scalar * span[i]);
                }
            }
        }
        public override void Divide(ReadOnlySpan<int> span, ReadOnlySpan<int> otherSpan, ReadOnlySpan<byte> otherValiditySpan, Span<int> resultSpan, Span<byte> resultValiditySpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                if (BitmapHelper.IsValid(otherValiditySpan, i))
                    resultSpan[i] = (int)(span[i] / otherSpan[i]);
                else
                    BitmapHelper.ClearBit(resultValiditySpan, i);
            }
        }

        public override void Divide(PrimitiveColumnContainer<int> column, int scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] / scalar);
                }
            }
        }

        public override void Divide(int scalar, PrimitiveColumnContainer<int> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(scalar / span[i]);
                }
            }
        }
        public override void Modulo(ReadOnlySpan<int> span, ReadOnlySpan<int> otherSpan, Span<int> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (int)(span[i] % otherSpan[i]);
            }
        }

        public override void Modulo(PrimitiveColumnContainer<int> column, int scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] % scalar);
                }
            }
        }

        public override void Modulo(int scalar, PrimitiveColumnContainer<int> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(scalar % span[i]);
                }
            }
        }
        public override void And(ReadOnlySpan<int> span, ReadOnlySpan<int> otherSpan, Span<int> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (int)(span[i] & otherSpan[i]);
            }
        }

        public override void And(PrimitiveColumnContainer<int> column, int scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] & scalar);
                }
            }
        }

        public override void And(int scalar, PrimitiveColumnContainer<int> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(scalar & span[i]);
                }
            }
        }
        public override void Or(ReadOnlySpan<int> span, ReadOnlySpan<int> otherSpan, Span<int> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (int)(span[i] | otherSpan[i]);
            }
        }

        public override void Or(PrimitiveColumnContainer<int> column, int scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] | scalar);
                }
            }
        }

        public override void Or(int scalar, PrimitiveColumnContainer<int> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(scalar | span[i]);
                }
            }
        }
        public override void Xor(ReadOnlySpan<int> span, ReadOnlySpan<int> otherSpan, Span<int> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (int)(span[i] ^ otherSpan[i]);
            }
        }

        public override void Xor(PrimitiveColumnContainer<int> column, int scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] ^ scalar);
                }
            }
        }

        public override void Xor(int scalar, PrimitiveColumnContainer<int> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(scalar ^ span[i]);
                }
            }
        }

        public override void LeftShift(PrimitiveColumnContainer<int> column, int value)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] << value);
                }
            }
        }

        public override void RightShift(PrimitiveColumnContainer<int> column, int value)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (int)(span[i] >> value);
                }
            }
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<int> left, PrimitiveColumnContainer<int> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<int> column, int scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<int> left, PrimitiveColumnContainer<int> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<int> column, int scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<int> left, PrimitiveColumnContainer<int> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<int> column, int scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<int> left, PrimitiveColumnContainer<int> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<int> column, int scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<int> left, PrimitiveColumnContainer<int> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<int> column, int scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<int> left, PrimitiveColumnContainer<int> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<int> column, int scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < scalar);
                }
            }
            return ret;
        }
    }
    internal class LongArithmetic : PrimitiveDataFrameColumnArithmetic<long>
    {
        public override void Add(ReadOnlySpan<long> span, ReadOnlySpan<long> otherSpan, Span<long> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (long)(span[i] + otherSpan[i]);
            }
        }

        public override void Add(PrimitiveColumnContainer<long> column, long scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] + scalar);
                }
            }
        }

        public override void Add(long scalar, PrimitiveColumnContainer<long> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(scalar + span[i]);
                }
            }
        }
        public override void Subtract(ReadOnlySpan<long> span, ReadOnlySpan<long> otherSpan, Span<long> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (long)(span[i] - otherSpan[i]);
            }
        }

        public override void Subtract(PrimitiveColumnContainer<long> column, long scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] - scalar);
                }
            }
        }

        public override void Subtract(long scalar, PrimitiveColumnContainer<long> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(scalar - span[i]);
                }
            }
        }
        public override void Multiply(ReadOnlySpan<long> span, ReadOnlySpan<long> otherSpan, Span<long> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (long)(span[i] * otherSpan[i]);
            }
        }

        public override void Multiply(PrimitiveColumnContainer<long> column, long scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] * scalar);
                }
            }
        }

        public override void Multiply(long scalar, PrimitiveColumnContainer<long> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(scalar * span[i]);
                }
            }
        }
        public override void Divide(ReadOnlySpan<long> span, ReadOnlySpan<long> otherSpan, ReadOnlySpan<byte> otherValiditySpan, Span<long> resultSpan, Span<byte> resultValiditySpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                if (BitmapHelper.IsValid(otherValiditySpan, i))
                    resultSpan[i] = (long)(span[i] / otherSpan[i]);
                else
                    BitmapHelper.ClearBit(resultValiditySpan, i);
            }
        }

        public override void Divide(PrimitiveColumnContainer<long> column, long scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] / scalar);
                }
            }
        }

        public override void Divide(long scalar, PrimitiveColumnContainer<long> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(scalar / span[i]);
                }
            }
        }
        public override void Modulo(ReadOnlySpan<long> span, ReadOnlySpan<long> otherSpan, Span<long> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (long)(span[i] % otherSpan[i]);
            }
        }

        public override void Modulo(PrimitiveColumnContainer<long> column, long scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] % scalar);
                }
            }
        }

        public override void Modulo(long scalar, PrimitiveColumnContainer<long> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(scalar % span[i]);
                }
            }
        }
        public override void And(ReadOnlySpan<long> span, ReadOnlySpan<long> otherSpan, Span<long> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (long)(span[i] & otherSpan[i]);
            }
        }

        public override void And(PrimitiveColumnContainer<long> column, long scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] & scalar);
                }
            }
        }

        public override void And(long scalar, PrimitiveColumnContainer<long> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(scalar & span[i]);
                }
            }
        }
        public override void Or(ReadOnlySpan<long> span, ReadOnlySpan<long> otherSpan, Span<long> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (long)(span[i] | otherSpan[i]);
            }
        }

        public override void Or(PrimitiveColumnContainer<long> column, long scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] | scalar);
                }
            }
        }

        public override void Or(long scalar, PrimitiveColumnContainer<long> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(scalar | span[i]);
                }
            }
        }
        public override void Xor(ReadOnlySpan<long> span, ReadOnlySpan<long> otherSpan, Span<long> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (long)(span[i] ^ otherSpan[i]);
            }
        }

        public override void Xor(PrimitiveColumnContainer<long> column, long scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] ^ scalar);
                }
            }
        }

        public override void Xor(long scalar, PrimitiveColumnContainer<long> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(scalar ^ span[i]);
                }
            }
        }

        public override void LeftShift(PrimitiveColumnContainer<long> column, int value)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] << value);
                }
            }
        }

        public override void RightShift(PrimitiveColumnContainer<long> column, int value)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (long)(span[i] >> value);
                }
            }
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<long> left, PrimitiveColumnContainer<long> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<long> column, long scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<long> left, PrimitiveColumnContainer<long> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<long> column, long scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<long> left, PrimitiveColumnContainer<long> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<long> column, long scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<long> left, PrimitiveColumnContainer<long> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<long> column, long scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<long> left, PrimitiveColumnContainer<long> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<long> column, long scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<long> left, PrimitiveColumnContainer<long> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<long> column, long scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < scalar);
                }
            }
            return ret;
        }
    }
    internal class SByteArithmetic : PrimitiveDataFrameColumnArithmetic<sbyte>
    {
        public override void Add(ReadOnlySpan<sbyte> span, ReadOnlySpan<sbyte> otherSpan, Span<sbyte> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (sbyte)(span[i] + otherSpan[i]);
            }
        }

        public override void Add(PrimitiveColumnContainer<sbyte> column, sbyte scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] + scalar);
                }
            }
        }

        public override void Add(sbyte scalar, PrimitiveColumnContainer<sbyte> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(scalar + span[i]);
                }
            }
        }
        public override void Subtract(ReadOnlySpan<sbyte> span, ReadOnlySpan<sbyte> otherSpan, Span<sbyte> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (sbyte)(span[i] - otherSpan[i]);
            }
        }

        public override void Subtract(PrimitiveColumnContainer<sbyte> column, sbyte scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] - scalar);
                }
            }
        }

        public override void Subtract(sbyte scalar, PrimitiveColumnContainer<sbyte> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(scalar - span[i]);
                }
            }
        }
        public override void Multiply(ReadOnlySpan<sbyte> span, ReadOnlySpan<sbyte> otherSpan, Span<sbyte> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (sbyte)(span[i] * otherSpan[i]);
            }
        }

        public override void Multiply(PrimitiveColumnContainer<sbyte> column, sbyte scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] * scalar);
                }
            }
        }

        public override void Multiply(sbyte scalar, PrimitiveColumnContainer<sbyte> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(scalar * span[i]);
                }
            }
        }
        public override void Divide(ReadOnlySpan<sbyte> span, ReadOnlySpan<sbyte> otherSpan, ReadOnlySpan<byte> otherValiditySpan, Span<sbyte> resultSpan, Span<byte> resultValiditySpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                if (BitmapHelper.IsValid(otherValiditySpan, i))
                    resultSpan[i] = (sbyte)(span[i] / otherSpan[i]);
                else
                    BitmapHelper.ClearBit(resultValiditySpan, i);
            }
        }

        public override void Divide(PrimitiveColumnContainer<sbyte> column, sbyte scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] / scalar);
                }
            }
        }

        public override void Divide(sbyte scalar, PrimitiveColumnContainer<sbyte> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(scalar / span[i]);
                }
            }
        }
        public override void Modulo(ReadOnlySpan<sbyte> span, ReadOnlySpan<sbyte> otherSpan, Span<sbyte> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (sbyte)(span[i] % otherSpan[i]);
            }
        }

        public override void Modulo(PrimitiveColumnContainer<sbyte> column, sbyte scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] % scalar);
                }
            }
        }

        public override void Modulo(sbyte scalar, PrimitiveColumnContainer<sbyte> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(scalar % span[i]);
                }
            }
        }
        public override void And(ReadOnlySpan<sbyte> span, ReadOnlySpan<sbyte> otherSpan, Span<sbyte> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (sbyte)(span[i] & otherSpan[i]);
            }
        }

        public override void And(PrimitiveColumnContainer<sbyte> column, sbyte scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] & scalar);
                }
            }
        }

        public override void And(sbyte scalar, PrimitiveColumnContainer<sbyte> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(scalar & span[i]);
                }
            }
        }
        public override void Or(ReadOnlySpan<sbyte> span, ReadOnlySpan<sbyte> otherSpan, Span<sbyte> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (sbyte)(span[i] | otherSpan[i]);
            }
        }

        public override void Or(PrimitiveColumnContainer<sbyte> column, sbyte scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] | scalar);
                }
            }
        }

        public override void Or(sbyte scalar, PrimitiveColumnContainer<sbyte> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(scalar | span[i]);
                }
            }
        }
        public override void Xor(ReadOnlySpan<sbyte> span, ReadOnlySpan<sbyte> otherSpan, Span<sbyte> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (sbyte)(span[i] ^ otherSpan[i]);
            }
        }

        public override void Xor(PrimitiveColumnContainer<sbyte> column, sbyte scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] ^ scalar);
                }
            }
        }

        public override void Xor(sbyte scalar, PrimitiveColumnContainer<sbyte> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(scalar ^ span[i]);
                }
            }
        }

        public override void LeftShift(PrimitiveColumnContainer<sbyte> column, int value)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] << value);
                }
            }
        }

        public override void RightShift(PrimitiveColumnContainer<sbyte> column, int value)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (sbyte)(span[i] >> value);
                }
            }
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<sbyte> left, PrimitiveColumnContainer<sbyte> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<sbyte> column, sbyte scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<sbyte> left, PrimitiveColumnContainer<sbyte> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<sbyte> column, sbyte scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<sbyte> left, PrimitiveColumnContainer<sbyte> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<sbyte> column, sbyte scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<sbyte> left, PrimitiveColumnContainer<sbyte> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<sbyte> column, sbyte scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<sbyte> left, PrimitiveColumnContainer<sbyte> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<sbyte> column, sbyte scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<sbyte> left, PrimitiveColumnContainer<sbyte> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<sbyte> column, sbyte scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < scalar);
                }
            }
            return ret;
        }
    }
    internal class ShortArithmetic : PrimitiveDataFrameColumnArithmetic<short>
    {
        public override void Add(ReadOnlySpan<short> span, ReadOnlySpan<short> otherSpan, Span<short> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (short)(span[i] + otherSpan[i]);
            }
        }

        public override void Add(PrimitiveColumnContainer<short> column, short scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] + scalar);
                }
            }
        }

        public override void Add(short scalar, PrimitiveColumnContainer<short> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(scalar + span[i]);
                }
            }
        }
        public override void Subtract(ReadOnlySpan<short> span, ReadOnlySpan<short> otherSpan, Span<short> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (short)(span[i] - otherSpan[i]);
            }
        }

        public override void Subtract(PrimitiveColumnContainer<short> column, short scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] - scalar);
                }
            }
        }

        public override void Subtract(short scalar, PrimitiveColumnContainer<short> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(scalar - span[i]);
                }
            }
        }
        public override void Multiply(ReadOnlySpan<short> span, ReadOnlySpan<short> otherSpan, Span<short> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (short)(span[i] * otherSpan[i]);
            }
        }

        public override void Multiply(PrimitiveColumnContainer<short> column, short scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] * scalar);
                }
            }
        }

        public override void Multiply(short scalar, PrimitiveColumnContainer<short> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(scalar * span[i]);
                }
            }
        }
        public override void Divide(ReadOnlySpan<short> span, ReadOnlySpan<short> otherSpan, ReadOnlySpan<byte> otherValiditySpan, Span<short> resultSpan, Span<byte> resultValiditySpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                if (BitmapHelper.IsValid(otherValiditySpan, i))
                    resultSpan[i] = (short)(span[i] / otherSpan[i]);
                else
                    BitmapHelper.ClearBit(resultValiditySpan, i);
            }
        }

        public override void Divide(PrimitiveColumnContainer<short> column, short scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] / scalar);
                }
            }
        }

        public override void Divide(short scalar, PrimitiveColumnContainer<short> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(scalar / span[i]);
                }
            }
        }
        public override void Modulo(ReadOnlySpan<short> span, ReadOnlySpan<short> otherSpan, Span<short> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (short)(span[i] % otherSpan[i]);
            }
        }

        public override void Modulo(PrimitiveColumnContainer<short> column, short scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] % scalar);
                }
            }
        }

        public override void Modulo(short scalar, PrimitiveColumnContainer<short> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(scalar % span[i]);
                }
            }
        }
        public override void And(ReadOnlySpan<short> span, ReadOnlySpan<short> otherSpan, Span<short> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (short)(span[i] & otherSpan[i]);
            }
        }

        public override void And(PrimitiveColumnContainer<short> column, short scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] & scalar);
                }
            }
        }

        public override void And(short scalar, PrimitiveColumnContainer<short> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(scalar & span[i]);
                }
            }
        }
        public override void Or(ReadOnlySpan<short> span, ReadOnlySpan<short> otherSpan, Span<short> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (short)(span[i] | otherSpan[i]);
            }
        }

        public override void Or(PrimitiveColumnContainer<short> column, short scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] | scalar);
                }
            }
        }

        public override void Or(short scalar, PrimitiveColumnContainer<short> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(scalar | span[i]);
                }
            }
        }
        public override void Xor(ReadOnlySpan<short> span, ReadOnlySpan<short> otherSpan, Span<short> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (short)(span[i] ^ otherSpan[i]);
            }
        }

        public override void Xor(PrimitiveColumnContainer<short> column, short scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] ^ scalar);
                }
            }
        }

        public override void Xor(short scalar, PrimitiveColumnContainer<short> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(scalar ^ span[i]);
                }
            }
        }

        public override void LeftShift(PrimitiveColumnContainer<short> column, int value)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] << value);
                }
            }
        }

        public override void RightShift(PrimitiveColumnContainer<short> column, int value)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (short)(span[i] >> value);
                }
            }
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<short> left, PrimitiveColumnContainer<short> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<short> column, short scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<short> left, PrimitiveColumnContainer<short> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<short> column, short scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<short> left, PrimitiveColumnContainer<short> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<short> column, short scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<short> left, PrimitiveColumnContainer<short> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<short> column, short scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<short> left, PrimitiveColumnContainer<short> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<short> column, short scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<short> left, PrimitiveColumnContainer<short> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<short> column, short scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < scalar);
                }
            }
            return ret;
        }
    }
    internal class UIntArithmetic : PrimitiveDataFrameColumnArithmetic<uint>
    {
        public override void Add(ReadOnlySpan<uint> span, ReadOnlySpan<uint> otherSpan, Span<uint> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (uint)(span[i] + otherSpan[i]);
            }
        }

        public override void Add(PrimitiveColumnContainer<uint> column, uint scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] + scalar);
                }
            }
        }

        public override void Add(uint scalar, PrimitiveColumnContainer<uint> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(scalar + span[i]);
                }
            }
        }
        public override void Subtract(ReadOnlySpan<uint> span, ReadOnlySpan<uint> otherSpan, Span<uint> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (uint)(span[i] - otherSpan[i]);
            }
        }

        public override void Subtract(PrimitiveColumnContainer<uint> column, uint scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] - scalar);
                }
            }
        }

        public override void Subtract(uint scalar, PrimitiveColumnContainer<uint> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(scalar - span[i]);
                }
            }
        }
        public override void Multiply(ReadOnlySpan<uint> span, ReadOnlySpan<uint> otherSpan, Span<uint> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (uint)(span[i] * otherSpan[i]);
            }
        }

        public override void Multiply(PrimitiveColumnContainer<uint> column, uint scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] * scalar);
                }
            }
        }

        public override void Multiply(uint scalar, PrimitiveColumnContainer<uint> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(scalar * span[i]);
                }
            }
        }
        public override void Divide(ReadOnlySpan<uint> span, ReadOnlySpan<uint> otherSpan, ReadOnlySpan<byte> otherValiditySpan, Span<uint> resultSpan, Span<byte> resultValiditySpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                if (BitmapHelper.IsValid(otherValiditySpan, i))
                    resultSpan[i] = (uint)(span[i] / otherSpan[i]);
                else
                    BitmapHelper.ClearBit(resultValiditySpan, i);
            }
        }

        public override void Divide(PrimitiveColumnContainer<uint> column, uint scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] / scalar);
                }
            }
        }

        public override void Divide(uint scalar, PrimitiveColumnContainer<uint> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(scalar / span[i]);
                }
            }
        }
        public override void Modulo(ReadOnlySpan<uint> span, ReadOnlySpan<uint> otherSpan, Span<uint> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (uint)(span[i] % otherSpan[i]);
            }
        }

        public override void Modulo(PrimitiveColumnContainer<uint> column, uint scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] % scalar);
                }
            }
        }

        public override void Modulo(uint scalar, PrimitiveColumnContainer<uint> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(scalar % span[i]);
                }
            }
        }
        public override void And(ReadOnlySpan<uint> span, ReadOnlySpan<uint> otherSpan, Span<uint> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (uint)(span[i] & otherSpan[i]);
            }
        }

        public override void And(PrimitiveColumnContainer<uint> column, uint scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] & scalar);
                }
            }
        }

        public override void And(uint scalar, PrimitiveColumnContainer<uint> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(scalar & span[i]);
                }
            }
        }
        public override void Or(ReadOnlySpan<uint> span, ReadOnlySpan<uint> otherSpan, Span<uint> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (uint)(span[i] | otherSpan[i]);
            }
        }

        public override void Or(PrimitiveColumnContainer<uint> column, uint scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] | scalar);
                }
            }
        }

        public override void Or(uint scalar, PrimitiveColumnContainer<uint> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(scalar | span[i]);
                }
            }
        }
        public override void Xor(ReadOnlySpan<uint> span, ReadOnlySpan<uint> otherSpan, Span<uint> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (uint)(span[i] ^ otherSpan[i]);
            }
        }

        public override void Xor(PrimitiveColumnContainer<uint> column, uint scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] ^ scalar);
                }
            }
        }

        public override void Xor(uint scalar, PrimitiveColumnContainer<uint> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(scalar ^ span[i]);
                }
            }
        }

        public override void LeftShift(PrimitiveColumnContainer<uint> column, int value)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] << value);
                }
            }
        }

        public override void RightShift(PrimitiveColumnContainer<uint> column, int value)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (uint)(span[i] >> value);
                }
            }
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<uint> left, PrimitiveColumnContainer<uint> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<uint> column, uint scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<uint> left, PrimitiveColumnContainer<uint> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<uint> column, uint scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<uint> left, PrimitiveColumnContainer<uint> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<uint> column, uint scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<uint> left, PrimitiveColumnContainer<uint> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<uint> column, uint scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<uint> left, PrimitiveColumnContainer<uint> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<uint> column, uint scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<uint> left, PrimitiveColumnContainer<uint> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<uint> column, uint scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < scalar);
                }
            }
            return ret;
        }
    }
    internal class ULongArithmetic : PrimitiveDataFrameColumnArithmetic<ulong>
    {
        public override void Add(ReadOnlySpan<ulong> span, ReadOnlySpan<ulong> otherSpan, Span<ulong> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (ulong)(span[i] + otherSpan[i]);
            }
        }

        public override void Add(PrimitiveColumnContainer<ulong> column, ulong scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] + scalar);
                }
            }
        }

        public override void Add(ulong scalar, PrimitiveColumnContainer<ulong> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(scalar + span[i]);
                }
            }
        }
        public override void Subtract(ReadOnlySpan<ulong> span, ReadOnlySpan<ulong> otherSpan, Span<ulong> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (ulong)(span[i] - otherSpan[i]);
            }
        }

        public override void Subtract(PrimitiveColumnContainer<ulong> column, ulong scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] - scalar);
                }
            }
        }

        public override void Subtract(ulong scalar, PrimitiveColumnContainer<ulong> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(scalar - span[i]);
                }
            }
        }
        public override void Multiply(ReadOnlySpan<ulong> span, ReadOnlySpan<ulong> otherSpan, Span<ulong> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (ulong)(span[i] * otherSpan[i]);
            }
        }

        public override void Multiply(PrimitiveColumnContainer<ulong> column, ulong scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] * scalar);
                }
            }
        }

        public override void Multiply(ulong scalar, PrimitiveColumnContainer<ulong> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(scalar * span[i]);
                }
            }
        }
        public override void Divide(ReadOnlySpan<ulong> span, ReadOnlySpan<ulong> otherSpan, ReadOnlySpan<byte> otherValiditySpan, Span<ulong> resultSpan, Span<byte> resultValiditySpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                if (BitmapHelper.IsValid(otherValiditySpan, i))
                    resultSpan[i] = (ulong)(span[i] / otherSpan[i]);
                else
                    BitmapHelper.ClearBit(resultValiditySpan, i);
            }
        }

        public override void Divide(PrimitiveColumnContainer<ulong> column, ulong scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] / scalar);
                }
            }
        }

        public override void Divide(ulong scalar, PrimitiveColumnContainer<ulong> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(scalar / span[i]);
                }
            }
        }
        public override void Modulo(ReadOnlySpan<ulong> span, ReadOnlySpan<ulong> otherSpan, Span<ulong> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (ulong)(span[i] % otherSpan[i]);
            }
        }

        public override void Modulo(PrimitiveColumnContainer<ulong> column, ulong scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] % scalar);
                }
            }
        }

        public override void Modulo(ulong scalar, PrimitiveColumnContainer<ulong> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(scalar % span[i]);
                }
            }
        }
        public override void And(ReadOnlySpan<ulong> span, ReadOnlySpan<ulong> otherSpan, Span<ulong> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (ulong)(span[i] & otherSpan[i]);
            }
        }

        public override void And(PrimitiveColumnContainer<ulong> column, ulong scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] & scalar);
                }
            }
        }

        public override void And(ulong scalar, PrimitiveColumnContainer<ulong> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(scalar & span[i]);
                }
            }
        }
        public override void Or(ReadOnlySpan<ulong> span, ReadOnlySpan<ulong> otherSpan, Span<ulong> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (ulong)(span[i] | otherSpan[i]);
            }
        }

        public override void Or(PrimitiveColumnContainer<ulong> column, ulong scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] | scalar);
                }
            }
        }

        public override void Or(ulong scalar, PrimitiveColumnContainer<ulong> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(scalar | span[i]);
                }
            }
        }
        public override void Xor(ReadOnlySpan<ulong> span, ReadOnlySpan<ulong> otherSpan, Span<ulong> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (ulong)(span[i] ^ otherSpan[i]);
            }
        }

        public override void Xor(PrimitiveColumnContainer<ulong> column, ulong scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] ^ scalar);
                }
            }
        }

        public override void Xor(ulong scalar, PrimitiveColumnContainer<ulong> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(scalar ^ span[i]);
                }
            }
        }

        public override void LeftShift(PrimitiveColumnContainer<ulong> column, int value)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] << value);
                }
            }
        }

        public override void RightShift(PrimitiveColumnContainer<ulong> column, int value)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ulong)(span[i] >> value);
                }
            }
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<ulong> left, PrimitiveColumnContainer<ulong> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<ulong> column, ulong scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<ulong> left, PrimitiveColumnContainer<ulong> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<ulong> column, ulong scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<ulong> left, PrimitiveColumnContainer<ulong> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<ulong> column, ulong scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<ulong> left, PrimitiveColumnContainer<ulong> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<ulong> column, ulong scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<ulong> left, PrimitiveColumnContainer<ulong> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<ulong> column, ulong scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<ulong> left, PrimitiveColumnContainer<ulong> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<ulong> column, ulong scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < scalar);
                }
            }
            return ret;
        }
    }
    internal class UShortArithmetic : PrimitiveDataFrameColumnArithmetic<ushort>
    {
        public override void Add(ReadOnlySpan<ushort> span, ReadOnlySpan<ushort> otherSpan, Span<ushort> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (ushort)(span[i] + otherSpan[i]);
            }
        }

        public override void Add(PrimitiveColumnContainer<ushort> column, ushort scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] + scalar);
                }
            }
        }

        public override void Add(ushort scalar, PrimitiveColumnContainer<ushort> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(scalar + span[i]);
                }
            }
        }
        public override void Subtract(ReadOnlySpan<ushort> span, ReadOnlySpan<ushort> otherSpan, Span<ushort> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (ushort)(span[i] - otherSpan[i]);
            }
        }

        public override void Subtract(PrimitiveColumnContainer<ushort> column, ushort scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] - scalar);
                }
            }
        }

        public override void Subtract(ushort scalar, PrimitiveColumnContainer<ushort> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(scalar - span[i]);
                }
            }
        }
        public override void Multiply(ReadOnlySpan<ushort> span, ReadOnlySpan<ushort> otherSpan, Span<ushort> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (ushort)(span[i] * otherSpan[i]);
            }
        }

        public override void Multiply(PrimitiveColumnContainer<ushort> column, ushort scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] * scalar);
                }
            }
        }

        public override void Multiply(ushort scalar, PrimitiveColumnContainer<ushort> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(scalar * span[i]);
                }
            }
        }
        public override void Divide(ReadOnlySpan<ushort> span, ReadOnlySpan<ushort> otherSpan, ReadOnlySpan<byte> otherValiditySpan, Span<ushort> resultSpan, Span<byte> resultValiditySpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                if (BitmapHelper.IsValid(otherValiditySpan, i))
                    resultSpan[i] = (ushort)(span[i] / otherSpan[i]);
                else
                    BitmapHelper.ClearBit(resultValiditySpan, i);
            }
        }

        public override void Divide(PrimitiveColumnContainer<ushort> column, ushort scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] / scalar);
                }
            }
        }

        public override void Divide(ushort scalar, PrimitiveColumnContainer<ushort> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(scalar / span[i]);
                }
            }
        }
        public override void Modulo(ReadOnlySpan<ushort> span, ReadOnlySpan<ushort> otherSpan, Span<ushort> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (ushort)(span[i] % otherSpan[i]);
            }
        }

        public override void Modulo(PrimitiveColumnContainer<ushort> column, ushort scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] % scalar);
                }
            }
        }

        public override void Modulo(ushort scalar, PrimitiveColumnContainer<ushort> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(scalar % span[i]);
                }
            }
        }
        public override void And(ReadOnlySpan<ushort> span, ReadOnlySpan<ushort> otherSpan, Span<ushort> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (ushort)(span[i] & otherSpan[i]);
            }
        }

        public override void And(PrimitiveColumnContainer<ushort> column, ushort scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] & scalar);
                }
            }
        }

        public override void And(ushort scalar, PrimitiveColumnContainer<ushort> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(scalar & span[i]);
                }
            }
        }
        public override void Or(ReadOnlySpan<ushort> span, ReadOnlySpan<ushort> otherSpan, Span<ushort> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (ushort)(span[i] | otherSpan[i]);
            }
        }

        public override void Or(PrimitiveColumnContainer<ushort> column, ushort scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] | scalar);
                }
            }
        }

        public override void Or(ushort scalar, PrimitiveColumnContainer<ushort> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(scalar | span[i]);
                }
            }
        }
        public override void Xor(ReadOnlySpan<ushort> span, ReadOnlySpan<ushort> otherSpan, Span<ushort> resultSpan)
        {
            for (int i = 0; i < span.Length; i++)
            {
                resultSpan[i] = (ushort)(span[i] ^ otherSpan[i]);
            }
        }

        public override void Xor(PrimitiveColumnContainer<ushort> column, ushort scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] ^ scalar);
                }
            }
        }

        public override void Xor(ushort scalar, PrimitiveColumnContainer<ushort> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(scalar ^ span[i]);
                }
            }
        }

        public override void LeftShift(PrimitiveColumnContainer<ushort> column, int value)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] << value);
                }
            }
        }

        public override void RightShift(PrimitiveColumnContainer<ushort> column, int value)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var mutableBuffer = column.Buffers.GetOrCreateMutable(b);
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (ushort)(span[i] >> value);
                }
            }
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<ushort> left, PrimitiveColumnContainer<ushort> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<ushort> column, ushort scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<ushort> left, PrimitiveColumnContainer<ushort> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<ushort> column, ushort scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<ushort> left, PrimitiveColumnContainer<ushort> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<ushort> column, ushort scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] >= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<ushort> left, PrimitiveColumnContainer<ushort> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<ushort> column, ushort scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] <= scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<ushort> left, PrimitiveColumnContainer<ushort> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<ushort> column, ushort scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] > scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<ushort> left, PrimitiveColumnContainer<ushort> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<ushort> column, ushort scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] < scalar);
                }
            }
            return ret;
        }
    }
    internal class DateTimeArithmetic : PrimitiveDataFrameColumnArithmetic<DateTime>
    {

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<DateTime> left, PrimitiveColumnContainer<DateTime> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<DateTime> column, DateTime scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] == scalar);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<DateTime> left, PrimitiveColumnContainer<DateTime> right)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(left.Length);
            long index = 0;
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var span = left.Buffers[b].ReadOnlySpan;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != otherSpan[i]);
                }
            }
            return ret;
        }

        public override PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<DateTime> column, DateTime scalar)
        {
            PrimitiveColumnContainer<bool> ret = new PrimitiveColumnContainer<bool>(column.Length);
            long index = 0;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var span = column.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[index++] = (span[i] != scalar);
                }
            }
            return ret;
        }
    }
}
