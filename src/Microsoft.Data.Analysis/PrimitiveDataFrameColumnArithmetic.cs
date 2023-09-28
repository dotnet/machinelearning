
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
        void HandleOperation(BinaryOperation operation, Span<T> left, Span<byte> leftValidity, ReadOnlySpan<T> right, ReadOnlySpan<byte> rightValidity);

        void HandleOperation(BinaryScalarOperation operation, Span<T> left, T right);
        void HandleOperation(BinaryScalarOperation operation, T left, Span<T> right, ReadOnlySpan<byte> rightValidity);

        void HandleOperation(BinaryIntOperation operation, Span<T> left, int right);

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

        public virtual void HandleOperation(BinaryOperation operation, Span<T> left, Span<byte> leftValidity, ReadOnlySpan<T> right, ReadOnlySpan<byte> rightValidity)
        {
            if (operation == BinaryOperation.Divide)
            {
                Divide(left, leftValidity, right, rightValidity);
                return;
            }
            else if (operation == BinaryOperation.Add)
                Add(left, right);
            else if (operation == BinaryOperation.Subtract)
                Subtract(left, right);
            else if (operation == BinaryOperation.Multiply)
                Multiply(left, right);
            else if (operation == BinaryOperation.Modulo)
                Modulo(left, right);
            else if (operation == BinaryOperation.And)
                And(left, right);
            else if (operation == BinaryOperation.Or)
                Or(left, right);
            else if (operation == BinaryOperation.Xor)
                Xor(left, right);

            BitmapHelper.ElementwiseAnd(leftValidity, rightValidity, leftValidity);
        }

        public virtual void HandleOperation(BinaryScalarOperation operation, Span<T> left, T right)
        {
            switch (operation)
            {
                case BinaryScalarOperation.Add:
                    Add(left, right);
                    break;
                case BinaryScalarOperation.Subtract:
                    Subtract(left, right);
                    break;
                case BinaryScalarOperation.Multiply:
                    Multiply(left, right);
                    break;
                case BinaryScalarOperation.Divide:
                    Divide(left, right);
                    break;
                case BinaryScalarOperation.Modulo:
                    Modulo(left, right);
                    break;
                case BinaryScalarOperation.And:
                    And(left, right);
                    break;
                case BinaryScalarOperation.Or:
                    Or(left, right);
                    break;
                case BinaryScalarOperation.Xor:
                    Xor(left, right);
                    break;
            }
        }

        public virtual void HandleOperation(BinaryScalarOperation operation, T left, Span<T> right, ReadOnlySpan<byte> rightValidity)
        {
            if (operation == BinaryScalarOperation.Divide)
            {
                Divide(left, right, rightValidity);
                return;
            }
            else if (operation == BinaryScalarOperation.Add)
                Add(left, right);
            else if (operation == BinaryScalarOperation.Subtract)
                Subtract(left, right);
            else if (operation == BinaryScalarOperation.Multiply)
                Multiply(left, right);
            else if (operation == BinaryScalarOperation.Modulo)
                Modulo(left, right);
            else if (operation == BinaryScalarOperation.And)
                And(left, right);
            else if (operation == BinaryScalarOperation.Or)
                Or(left, right);
            else if (operation == BinaryScalarOperation.Xor)
                Xor(left, right);
        }

        public virtual void HandleOperation(BinaryIntOperation operation, Span<T> left, int right)
        {
            switch (operation)
            {
                case BinaryIntOperation.LeftShift:
                    LeftShift(left, right);
                    break;
                case BinaryIntOperation.RightShift:
                    RightShift(left, right);
                    break;
            }
        }

        public virtual void Add(Span<T> left, ReadOnlySpan<T> right) => throw new NotSupportedException();
        public virtual void Add(Span<T> left, T scalar) => throw new NotSupportedException();
        public virtual void Add(T left, Span<T> right) => throw new NotSupportedException();
        public virtual void Subtract(Span<T> left, ReadOnlySpan<T> right) => throw new NotSupportedException();
        public virtual void Subtract(Span<T> left, T scalar) => throw new NotSupportedException();
        public virtual void Subtract(T left, Span<T> right) => throw new NotSupportedException();
        public virtual void Multiply(Span<T> left, ReadOnlySpan<T> right) => throw new NotSupportedException();
        public virtual void Multiply(Span<T> left, T scalar) => throw new NotSupportedException();
        public virtual void Multiply(T left, Span<T> right) => throw new NotSupportedException();
        public virtual void Divide(Span<T> left, Span<byte> leftValidity, ReadOnlySpan<T> right, ReadOnlySpan<byte> rightValidity) => throw new NotSupportedException();
        public virtual void Divide(Span<T> left, T scalar) => throw new NotSupportedException();
        public virtual void Divide(T left, Span<T> right, ReadOnlySpan<byte> rightValidity) => throw new NotSupportedException();
        public virtual void Modulo(Span<T> left, ReadOnlySpan<T> right) => throw new NotSupportedException();
        public virtual void Modulo(Span<T> left, T scalar) => throw new NotSupportedException();
        public virtual void Modulo(T left, Span<T> right) => throw new NotSupportedException();
        public virtual void And(Span<T> left, ReadOnlySpan<T> right) => throw new NotSupportedException();
        public virtual void And(Span<T> left, T scalar) => throw new NotSupportedException();
        public virtual void And(T left, Span<T> right) => throw new NotSupportedException();
        public virtual void Or(Span<T> left, ReadOnlySpan<T> right) => throw new NotSupportedException();
        public virtual void Or(Span<T> left, T scalar) => throw new NotSupportedException();
        public virtual void Or(T left, Span<T> right) => throw new NotSupportedException();
        public virtual void Xor(Span<T> left, ReadOnlySpan<T> right) => throw new NotSupportedException();
        public virtual void Xor(Span<T> left, T scalar) => throw new NotSupportedException();
        public virtual void Xor(T left, Span<T> right) => throw new NotSupportedException();
        public virtual void LeftShift(Span<T> left, int right) => throw new NotSupportedException();
        public virtual void RightShift(Span<T> left, int right) => throw new NotSupportedException();
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
                return (IPrimitiveDataFrameColumnArithmetic<T>)new BoolArithmetic();
            else if (typeof(T) == typeof(byte))
                return (IPrimitiveDataFrameColumnArithmetic<T>)new ByteArithmetic();
            else if (typeof(T) == typeof(char))
                return (IPrimitiveDataFrameColumnArithmetic<T>)new CharArithmetic();
            else if (typeof(T) == typeof(decimal))
                return (IPrimitiveDataFrameColumnArithmetic<T>)new DecimalArithmetic();
            else if (typeof(T) == typeof(double))
                return (IPrimitiveDataFrameColumnArithmetic<T>)new DoubleArithmetic();
            else if (typeof(T) == typeof(float))
                return (IPrimitiveDataFrameColumnArithmetic<T>)new FloatArithmetic();
            else if (typeof(T) == typeof(int))
                return (IPrimitiveDataFrameColumnArithmetic<T>)new IntArithmetic();
            else if (typeof(T) == typeof(long))
                return (IPrimitiveDataFrameColumnArithmetic<T>)new LongArithmetic();
            else if (typeof(T) == typeof(sbyte))
                return (IPrimitiveDataFrameColumnArithmetic<T>)new SByteArithmetic();
            else if (typeof(T) == typeof(short))
                return (IPrimitiveDataFrameColumnArithmetic<T>)new ShortArithmetic();
            else if (typeof(T) == typeof(uint))
                return (IPrimitiveDataFrameColumnArithmetic<T>)new UIntArithmetic();
            else if (typeof(T) == typeof(ulong))
                return (IPrimitiveDataFrameColumnArithmetic<T>)new ULongArithmetic();
            else if (typeof(T) == typeof(ushort))
                return (IPrimitiveDataFrameColumnArithmetic<T>)new UShortArithmetic();
            else if (typeof(T) == typeof(DateTime))
                return (IPrimitiveDataFrameColumnArithmetic<T>)new DateTimeArithmetic();
            throw new NotSupportedException();
        }
    }

    internal class BoolArithmetic : PrimitiveDataFrameColumnArithmetic<bool>
    {

        public override void And(Span<bool> left, ReadOnlySpan<bool> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (bool)(left[i] & right[i]);
        }

        public override void And(Span<bool> left, bool right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (bool)(left[i] & right);
        }

        public override void And(bool left, Span<bool> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (bool)(left & right[i]);
        }
        public override void Or(Span<bool> left, ReadOnlySpan<bool> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (bool)(left[i] | right[i]);
        }

        public override void Or(Span<bool> left, bool right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (bool)(left[i] | right);
        }

        public override void Or(bool left, Span<bool> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (bool)(left | right[i]);
        }
        public override void Xor(Span<bool> left, ReadOnlySpan<bool> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (bool)(left[i] ^ right[i]);
        }

        public override void Xor(Span<bool> left, bool right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (bool)(left[i] ^ right);
        }

        public override void Xor(bool left, Span<bool> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (bool)(left ^ right[i]);
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

        public override void Add(Span<byte> left, ReadOnlySpan<byte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] + right[i]);
        }

        public override void Add(Span<byte> left, byte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] + right);
        }

        public override void Add(byte left, Span<byte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (byte)(left + right[i]);
        }
        public override void Subtract(Span<byte> left, ReadOnlySpan<byte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] - right[i]);
        }

        public override void Subtract(Span<byte> left, byte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] - right);
        }

        public override void Subtract(byte left, Span<byte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (byte)(left - right[i]);
        }
        public override void Multiply(Span<byte> left, ReadOnlySpan<byte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] * right[i]);
        }

        public override void Multiply(Span<byte> left, byte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] * right);
        }

        public override void Multiply(byte left, Span<byte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (byte)(left * right[i]);
        }
        public override void Divide(Span<byte> left, Span<byte> leftValidity, ReadOnlySpan<byte> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    left[i] = (byte)(left[i] / right[i]);
                else
                    BitmapHelper.ClearBit(leftValidity, i);
            }
        }

        public override void Divide(Span<byte> left, byte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] / right);
        }

        public override void Divide(byte left, Span<byte> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    right[i] = (byte)(left / right[i]);
            }
        }
        public override void Modulo(Span<byte> left, ReadOnlySpan<byte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] % right[i]);
        }

        public override void Modulo(Span<byte> left, byte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] % right);
        }

        public override void Modulo(byte left, Span<byte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (byte)(left % right[i]);
        }
        public override void And(Span<byte> left, ReadOnlySpan<byte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] & right[i]);
        }

        public override void And(Span<byte> left, byte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] & right);
        }

        public override void And(byte left, Span<byte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (byte)(left & right[i]);
        }
        public override void Or(Span<byte> left, ReadOnlySpan<byte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] | right[i]);
        }

        public override void Or(Span<byte> left, byte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] | right);
        }

        public override void Or(byte left, Span<byte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (byte)(left | right[i]);
        }
        public override void Xor(Span<byte> left, ReadOnlySpan<byte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] ^ right[i]);
        }

        public override void Xor(Span<byte> left, byte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] ^ right);
        }

        public override void Xor(byte left, Span<byte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (byte)(left ^ right[i]);
        }

        public override void LeftShift(Span<byte> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] << right);
        }

        public override void RightShift(Span<byte> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] >> right);
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

        public override void Add(Span<char> left, ReadOnlySpan<char> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] + right[i]);
        }

        public override void Add(Span<char> left, char right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] + right);
        }

        public override void Add(char left, Span<char> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (char)(left + right[i]);
        }
        public override void Subtract(Span<char> left, ReadOnlySpan<char> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] - right[i]);
        }

        public override void Subtract(Span<char> left, char right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] - right);
        }

        public override void Subtract(char left, Span<char> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (char)(left - right[i]);
        }
        public override void Multiply(Span<char> left, ReadOnlySpan<char> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] * right[i]);
        }

        public override void Multiply(Span<char> left, char right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] * right);
        }

        public override void Multiply(char left, Span<char> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (char)(left * right[i]);
        }
        public override void Divide(Span<char> left, Span<byte> leftValidity, ReadOnlySpan<char> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    left[i] = (char)(left[i] / right[i]);
                else
                    BitmapHelper.ClearBit(leftValidity, i);
            }
        }

        public override void Divide(Span<char> left, char right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] / right);
        }

        public override void Divide(char left, Span<char> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    right[i] = (char)(left / right[i]);
            }
        }
        public override void Modulo(Span<char> left, ReadOnlySpan<char> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] % right[i]);
        }

        public override void Modulo(Span<char> left, char right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] % right);
        }

        public override void Modulo(char left, Span<char> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (char)(left % right[i]);
        }
        public override void And(Span<char> left, ReadOnlySpan<char> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] & right[i]);
        }

        public override void And(Span<char> left, char right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] & right);
        }

        public override void And(char left, Span<char> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (char)(left & right[i]);
        }
        public override void Or(Span<char> left, ReadOnlySpan<char> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] | right[i]);
        }

        public override void Or(Span<char> left, char right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] | right);
        }

        public override void Or(char left, Span<char> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (char)(left | right[i]);
        }
        public override void Xor(Span<char> left, ReadOnlySpan<char> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] ^ right[i]);
        }

        public override void Xor(Span<char> left, char right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] ^ right);
        }

        public override void Xor(char left, Span<char> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (char)(left ^ right[i]);
        }

        public override void LeftShift(Span<char> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] << right);
        }

        public override void RightShift(Span<char> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] >> right);
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

        public override void Add(Span<decimal> left, ReadOnlySpan<decimal> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (decimal)(left[i] + right[i]);
        }

        public override void Add(Span<decimal> left, decimal right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (decimal)(left[i] + right);
        }

        public override void Add(decimal left, Span<decimal> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (decimal)(left + right[i]);
        }
        public override void Subtract(Span<decimal> left, ReadOnlySpan<decimal> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (decimal)(left[i] - right[i]);
        }

        public override void Subtract(Span<decimal> left, decimal right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (decimal)(left[i] - right);
        }

        public override void Subtract(decimal left, Span<decimal> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (decimal)(left - right[i]);
        }
        public override void Multiply(Span<decimal> left, ReadOnlySpan<decimal> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (decimal)(left[i] * right[i]);
        }

        public override void Multiply(Span<decimal> left, decimal right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (decimal)(left[i] * right);
        }

        public override void Multiply(decimal left, Span<decimal> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (decimal)(left * right[i]);
        }
        public override void Divide(Span<decimal> left, Span<byte> leftValidity, ReadOnlySpan<decimal> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    left[i] = (decimal)(left[i] / right[i]);
                else
                    BitmapHelper.ClearBit(leftValidity, i);
            }
        }

        public override void Divide(Span<decimal> left, decimal right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (decimal)(left[i] / right);
        }

        public override void Divide(decimal left, Span<decimal> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    right[i] = (decimal)(left / right[i]);
            }
        }
        public override void Modulo(Span<decimal> left, ReadOnlySpan<decimal> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (decimal)(left[i] % right[i]);
        }

        public override void Modulo(Span<decimal> left, decimal right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (decimal)(left[i] % right);
        }

        public override void Modulo(decimal left, Span<decimal> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (decimal)(left % right[i]);
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

        public override void Add(Span<double> left, ReadOnlySpan<double> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (double)(left[i] + right[i]);
        }

        public override void Add(Span<double> left, double right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (double)(left[i] + right);
        }

        public override void Add(double left, Span<double> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (double)(left + right[i]);
        }
        public override void Subtract(Span<double> left, ReadOnlySpan<double> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (double)(left[i] - right[i]);
        }

        public override void Subtract(Span<double> left, double right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (double)(left[i] - right);
        }

        public override void Subtract(double left, Span<double> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (double)(left - right[i]);
        }
        public override void Multiply(Span<double> left, ReadOnlySpan<double> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (double)(left[i] * right[i]);
        }

        public override void Multiply(Span<double> left, double right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (double)(left[i] * right);
        }

        public override void Multiply(double left, Span<double> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (double)(left * right[i]);
        }
        public override void Divide(Span<double> left, Span<byte> leftValidity, ReadOnlySpan<double> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    left[i] = (double)(left[i] / right[i]);
                else
                    BitmapHelper.ClearBit(leftValidity, i);
            }
        }

        public override void Divide(Span<double> left, double right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (double)(left[i] / right);
        }

        public override void Divide(double left, Span<double> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    right[i] = (double)(left / right[i]);
            }
        }
        public override void Modulo(Span<double> left, ReadOnlySpan<double> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (double)(left[i] % right[i]);
        }

        public override void Modulo(Span<double> left, double right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (double)(left[i] % right);
        }

        public override void Modulo(double left, Span<double> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (double)(left % right[i]);
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

        public override void Add(Span<float> left, ReadOnlySpan<float> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (float)(left[i] + right[i]);
        }

        public override void Add(Span<float> left, float right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (float)(left[i] + right);
        }

        public override void Add(float left, Span<float> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (float)(left + right[i]);
        }
        public override void Subtract(Span<float> left, ReadOnlySpan<float> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (float)(left[i] - right[i]);
        }

        public override void Subtract(Span<float> left, float right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (float)(left[i] - right);
        }

        public override void Subtract(float left, Span<float> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (float)(left - right[i]);
        }
        public override void Multiply(Span<float> left, ReadOnlySpan<float> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (float)(left[i] * right[i]);
        }

        public override void Multiply(Span<float> left, float right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (float)(left[i] * right);
        }

        public override void Multiply(float left, Span<float> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (float)(left * right[i]);
        }
        public override void Divide(Span<float> left, Span<byte> leftValidity, ReadOnlySpan<float> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    left[i] = (float)(left[i] / right[i]);
                else
                    BitmapHelper.ClearBit(leftValidity, i);
            }
        }

        public override void Divide(Span<float> left, float right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (float)(left[i] / right);
        }

        public override void Divide(float left, Span<float> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    right[i] = (float)(left / right[i]);
            }
        }
        public override void Modulo(Span<float> left, ReadOnlySpan<float> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (float)(left[i] % right[i]);
        }

        public override void Modulo(Span<float> left, float right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (float)(left[i] % right);
        }

        public override void Modulo(float left, Span<float> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (float)(left % right[i]);
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

        public override void Add(Span<int> left, ReadOnlySpan<int> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] + right[i]);
        }

        public override void Add(Span<int> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] + right);
        }

        public override void Add(int left, Span<int> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (int)(left + right[i]);
        }
        public override void Subtract(Span<int> left, ReadOnlySpan<int> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] - right[i]);
        }

        public override void Subtract(Span<int> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] - right);
        }

        public override void Subtract(int left, Span<int> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (int)(left - right[i]);
        }
        public override void Multiply(Span<int> left, ReadOnlySpan<int> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] * right[i]);
        }

        public override void Multiply(Span<int> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] * right);
        }

        public override void Multiply(int left, Span<int> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (int)(left * right[i]);
        }
        public override void Divide(Span<int> left, Span<byte> leftValidity, ReadOnlySpan<int> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    left[i] = (int)(left[i] / right[i]);
                else
                    BitmapHelper.ClearBit(leftValidity, i);
            }
        }

        public override void Divide(Span<int> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] / right);
        }

        public override void Divide(int left, Span<int> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    right[i] = (int)(left / right[i]);
            }
        }
        public override void Modulo(Span<int> left, ReadOnlySpan<int> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] % right[i]);
        }

        public override void Modulo(Span<int> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] % right);
        }

        public override void Modulo(int left, Span<int> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (int)(left % right[i]);
        }
        public override void And(Span<int> left, ReadOnlySpan<int> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] & right[i]);
        }

        public override void And(Span<int> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] & right);
        }

        public override void And(int left, Span<int> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (int)(left & right[i]);
        }
        public override void Or(Span<int> left, ReadOnlySpan<int> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] | right[i]);
        }

        public override void Or(Span<int> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] | right);
        }

        public override void Or(int left, Span<int> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (int)(left | right[i]);
        }
        public override void Xor(Span<int> left, ReadOnlySpan<int> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] ^ right[i]);
        }

        public override void Xor(Span<int> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] ^ right);
        }

        public override void Xor(int left, Span<int> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (int)(left ^ right[i]);
        }

        public override void LeftShift(Span<int> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] << right);
        }

        public override void RightShift(Span<int> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] >> right);
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

        public override void Add(Span<long> left, ReadOnlySpan<long> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] + right[i]);
        }

        public override void Add(Span<long> left, long right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] + right);
        }

        public override void Add(long left, Span<long> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (long)(left + right[i]);
        }
        public override void Subtract(Span<long> left, ReadOnlySpan<long> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] - right[i]);
        }

        public override void Subtract(Span<long> left, long right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] - right);
        }

        public override void Subtract(long left, Span<long> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (long)(left - right[i]);
        }
        public override void Multiply(Span<long> left, ReadOnlySpan<long> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] * right[i]);
        }

        public override void Multiply(Span<long> left, long right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] * right);
        }

        public override void Multiply(long left, Span<long> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (long)(left * right[i]);
        }
        public override void Divide(Span<long> left, Span<byte> leftValidity, ReadOnlySpan<long> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    left[i] = (long)(left[i] / right[i]);
                else
                    BitmapHelper.ClearBit(leftValidity, i);
            }
        }

        public override void Divide(Span<long> left, long right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] / right);
        }

        public override void Divide(long left, Span<long> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    right[i] = (long)(left / right[i]);
            }
        }
        public override void Modulo(Span<long> left, ReadOnlySpan<long> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] % right[i]);
        }

        public override void Modulo(Span<long> left, long right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] % right);
        }

        public override void Modulo(long left, Span<long> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (long)(left % right[i]);
        }
        public override void And(Span<long> left, ReadOnlySpan<long> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] & right[i]);
        }

        public override void And(Span<long> left, long right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] & right);
        }

        public override void And(long left, Span<long> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (long)(left & right[i]);
        }
        public override void Or(Span<long> left, ReadOnlySpan<long> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] | right[i]);
        }

        public override void Or(Span<long> left, long right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] | right);
        }

        public override void Or(long left, Span<long> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (long)(left | right[i]);
        }
        public override void Xor(Span<long> left, ReadOnlySpan<long> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] ^ right[i]);
        }

        public override void Xor(Span<long> left, long right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] ^ right);
        }

        public override void Xor(long left, Span<long> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (long)(left ^ right[i]);
        }

        public override void LeftShift(Span<long> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] << right);
        }

        public override void RightShift(Span<long> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] >> right);
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

        public override void Add(Span<sbyte> left, ReadOnlySpan<sbyte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] + right[i]);
        }

        public override void Add(Span<sbyte> left, sbyte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] + right);
        }

        public override void Add(sbyte left, Span<sbyte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (sbyte)(left + right[i]);
        }
        public override void Subtract(Span<sbyte> left, ReadOnlySpan<sbyte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] - right[i]);
        }

        public override void Subtract(Span<sbyte> left, sbyte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] - right);
        }

        public override void Subtract(sbyte left, Span<sbyte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (sbyte)(left - right[i]);
        }
        public override void Multiply(Span<sbyte> left, ReadOnlySpan<sbyte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] * right[i]);
        }

        public override void Multiply(Span<sbyte> left, sbyte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] * right);
        }

        public override void Multiply(sbyte left, Span<sbyte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (sbyte)(left * right[i]);
        }
        public override void Divide(Span<sbyte> left, Span<byte> leftValidity, ReadOnlySpan<sbyte> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    left[i] = (sbyte)(left[i] / right[i]);
                else
                    BitmapHelper.ClearBit(leftValidity, i);
            }
        }

        public override void Divide(Span<sbyte> left, sbyte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] / right);
        }

        public override void Divide(sbyte left, Span<sbyte> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    right[i] = (sbyte)(left / right[i]);
            }
        }
        public override void Modulo(Span<sbyte> left, ReadOnlySpan<sbyte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] % right[i]);
        }

        public override void Modulo(Span<sbyte> left, sbyte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] % right);
        }

        public override void Modulo(sbyte left, Span<sbyte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (sbyte)(left % right[i]);
        }
        public override void And(Span<sbyte> left, ReadOnlySpan<sbyte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] & right[i]);
        }

        public override void And(Span<sbyte> left, sbyte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] & right);
        }

        public override void And(sbyte left, Span<sbyte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (sbyte)(left & right[i]);
        }
        public override void Or(Span<sbyte> left, ReadOnlySpan<sbyte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] | right[i]);
        }

        public override void Or(Span<sbyte> left, sbyte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] | right);
        }

        public override void Or(sbyte left, Span<sbyte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (sbyte)(left | right[i]);
        }
        public override void Xor(Span<sbyte> left, ReadOnlySpan<sbyte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] ^ right[i]);
        }

        public override void Xor(Span<sbyte> left, sbyte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] ^ right);
        }

        public override void Xor(sbyte left, Span<sbyte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (sbyte)(left ^ right[i]);
        }

        public override void LeftShift(Span<sbyte> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] << right);
        }

        public override void RightShift(Span<sbyte> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] >> right);
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

        public override void Add(Span<short> left, ReadOnlySpan<short> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] + right[i]);
        }

        public override void Add(Span<short> left, short right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] + right);
        }

        public override void Add(short left, Span<short> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (short)(left + right[i]);
        }
        public override void Subtract(Span<short> left, ReadOnlySpan<short> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] - right[i]);
        }

        public override void Subtract(Span<short> left, short right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] - right);
        }

        public override void Subtract(short left, Span<short> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (short)(left - right[i]);
        }
        public override void Multiply(Span<short> left, ReadOnlySpan<short> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] * right[i]);
        }

        public override void Multiply(Span<short> left, short right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] * right);
        }

        public override void Multiply(short left, Span<short> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (short)(left * right[i]);
        }
        public override void Divide(Span<short> left, Span<byte> leftValidity, ReadOnlySpan<short> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    left[i] = (short)(left[i] / right[i]);
                else
                    BitmapHelper.ClearBit(leftValidity, i);
            }
        }

        public override void Divide(Span<short> left, short right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] / right);
        }

        public override void Divide(short left, Span<short> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    right[i] = (short)(left / right[i]);
            }
        }
        public override void Modulo(Span<short> left, ReadOnlySpan<short> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] % right[i]);
        }

        public override void Modulo(Span<short> left, short right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] % right);
        }

        public override void Modulo(short left, Span<short> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (short)(left % right[i]);
        }
        public override void And(Span<short> left, ReadOnlySpan<short> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] & right[i]);
        }

        public override void And(Span<short> left, short right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] & right);
        }

        public override void And(short left, Span<short> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (short)(left & right[i]);
        }
        public override void Or(Span<short> left, ReadOnlySpan<short> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] | right[i]);
        }

        public override void Or(Span<short> left, short right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] | right);
        }

        public override void Or(short left, Span<short> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (short)(left | right[i]);
        }
        public override void Xor(Span<short> left, ReadOnlySpan<short> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] ^ right[i]);
        }

        public override void Xor(Span<short> left, short right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] ^ right);
        }

        public override void Xor(short left, Span<short> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (short)(left ^ right[i]);
        }

        public override void LeftShift(Span<short> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] << right);
        }

        public override void RightShift(Span<short> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] >> right);
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

        public override void Add(Span<uint> left, ReadOnlySpan<uint> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] + right[i]);
        }

        public override void Add(Span<uint> left, uint right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] + right);
        }

        public override void Add(uint left, Span<uint> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (uint)(left + right[i]);
        }
        public override void Subtract(Span<uint> left, ReadOnlySpan<uint> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] - right[i]);
        }

        public override void Subtract(Span<uint> left, uint right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] - right);
        }

        public override void Subtract(uint left, Span<uint> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (uint)(left - right[i]);
        }
        public override void Multiply(Span<uint> left, ReadOnlySpan<uint> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] * right[i]);
        }

        public override void Multiply(Span<uint> left, uint right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] * right);
        }

        public override void Multiply(uint left, Span<uint> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (uint)(left * right[i]);
        }
        public override void Divide(Span<uint> left, Span<byte> leftValidity, ReadOnlySpan<uint> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    left[i] = (uint)(left[i] / right[i]);
                else
                    BitmapHelper.ClearBit(leftValidity, i);
            }
        }

        public override void Divide(Span<uint> left, uint right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] / right);
        }

        public override void Divide(uint left, Span<uint> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    right[i] = (uint)(left / right[i]);
            }
        }
        public override void Modulo(Span<uint> left, ReadOnlySpan<uint> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] % right[i]);
        }

        public override void Modulo(Span<uint> left, uint right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] % right);
        }

        public override void Modulo(uint left, Span<uint> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (uint)(left % right[i]);
        }
        public override void And(Span<uint> left, ReadOnlySpan<uint> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] & right[i]);
        }

        public override void And(Span<uint> left, uint right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] & right);
        }

        public override void And(uint left, Span<uint> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (uint)(left & right[i]);
        }
        public override void Or(Span<uint> left, ReadOnlySpan<uint> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] | right[i]);
        }

        public override void Or(Span<uint> left, uint right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] | right);
        }

        public override void Or(uint left, Span<uint> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (uint)(left | right[i]);
        }
        public override void Xor(Span<uint> left, ReadOnlySpan<uint> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] ^ right[i]);
        }

        public override void Xor(Span<uint> left, uint right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] ^ right);
        }

        public override void Xor(uint left, Span<uint> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (uint)(left ^ right[i]);
        }

        public override void LeftShift(Span<uint> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] << right);
        }

        public override void RightShift(Span<uint> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] >> right);
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

        public override void Add(Span<ulong> left, ReadOnlySpan<ulong> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] + right[i]);
        }

        public override void Add(Span<ulong> left, ulong right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] + right);
        }

        public override void Add(ulong left, Span<ulong> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ulong)(left + right[i]);
        }
        public override void Subtract(Span<ulong> left, ReadOnlySpan<ulong> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] - right[i]);
        }

        public override void Subtract(Span<ulong> left, ulong right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] - right);
        }

        public override void Subtract(ulong left, Span<ulong> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ulong)(left - right[i]);
        }
        public override void Multiply(Span<ulong> left, ReadOnlySpan<ulong> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] * right[i]);
        }

        public override void Multiply(Span<ulong> left, ulong right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] * right);
        }

        public override void Multiply(ulong left, Span<ulong> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ulong)(left * right[i]);
        }
        public override void Divide(Span<ulong> left, Span<byte> leftValidity, ReadOnlySpan<ulong> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    left[i] = (ulong)(left[i] / right[i]);
                else
                    BitmapHelper.ClearBit(leftValidity, i);
            }
        }

        public override void Divide(Span<ulong> left, ulong right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] / right);
        }

        public override void Divide(ulong left, Span<ulong> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    right[i] = (ulong)(left / right[i]);
            }
        }
        public override void Modulo(Span<ulong> left, ReadOnlySpan<ulong> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] % right[i]);
        }

        public override void Modulo(Span<ulong> left, ulong right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] % right);
        }

        public override void Modulo(ulong left, Span<ulong> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ulong)(left % right[i]);
        }
        public override void And(Span<ulong> left, ReadOnlySpan<ulong> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] & right[i]);
        }

        public override void And(Span<ulong> left, ulong right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] & right);
        }

        public override void And(ulong left, Span<ulong> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ulong)(left & right[i]);
        }
        public override void Or(Span<ulong> left, ReadOnlySpan<ulong> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] | right[i]);
        }

        public override void Or(Span<ulong> left, ulong right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] | right);
        }

        public override void Or(ulong left, Span<ulong> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ulong)(left | right[i]);
        }
        public override void Xor(Span<ulong> left, ReadOnlySpan<ulong> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] ^ right[i]);
        }

        public override void Xor(Span<ulong> left, ulong right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] ^ right);
        }

        public override void Xor(ulong left, Span<ulong> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ulong)(left ^ right[i]);
        }

        public override void LeftShift(Span<ulong> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] << right);
        }

        public override void RightShift(Span<ulong> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] >> right);
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

        public override void Add(Span<ushort> left, ReadOnlySpan<ushort> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] + right[i]);
        }

        public override void Add(Span<ushort> left, ushort right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] + right);
        }

        public override void Add(ushort left, Span<ushort> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ushort)(left + right[i]);
        }
        public override void Subtract(Span<ushort> left, ReadOnlySpan<ushort> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] - right[i]);
        }

        public override void Subtract(Span<ushort> left, ushort right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] - right);
        }

        public override void Subtract(ushort left, Span<ushort> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ushort)(left - right[i]);
        }
        public override void Multiply(Span<ushort> left, ReadOnlySpan<ushort> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] * right[i]);
        }

        public override void Multiply(Span<ushort> left, ushort right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] * right);
        }

        public override void Multiply(ushort left, Span<ushort> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ushort)(left * right[i]);
        }
        public override void Divide(Span<ushort> left, Span<byte> leftValidity, ReadOnlySpan<ushort> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    left[i] = (ushort)(left[i] / right[i]);
                else
                    BitmapHelper.ClearBit(leftValidity, i);
            }
        }

        public override void Divide(Span<ushort> left, ushort right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] / right);
        }

        public override void Divide(ushort left, Span<ushort> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitmapHelper.IsValid(rightValidity, i))
                    right[i] = (ushort)(left / right[i]);
            }
        }
        public override void Modulo(Span<ushort> left, ReadOnlySpan<ushort> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] % right[i]);
        }

        public override void Modulo(Span<ushort> left, ushort right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] % right);
        }

        public override void Modulo(ushort left, Span<ushort> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ushort)(left % right[i]);
        }
        public override void And(Span<ushort> left, ReadOnlySpan<ushort> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] & right[i]);
        }

        public override void And(Span<ushort> left, ushort right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] & right);
        }

        public override void And(ushort left, Span<ushort> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ushort)(left & right[i]);
        }
        public override void Or(Span<ushort> left, ReadOnlySpan<ushort> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] | right[i]);
        }

        public override void Or(Span<ushort> left, ushort right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] | right);
        }

        public override void Or(ushort left, Span<ushort> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ushort)(left | right[i]);
        }
        public override void Xor(Span<ushort> left, ReadOnlySpan<ushort> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] ^ right[i]);
        }

        public override void Xor(Span<ushort> left, ushort right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] ^ right);
        }

        public override void Xor(ushort left, Span<ushort> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ushort)(left ^ right[i]);
        }

        public override void LeftShift(Span<ushort> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] << right);
        }

        public override void RightShift(Span<ushort> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] >> right);
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
