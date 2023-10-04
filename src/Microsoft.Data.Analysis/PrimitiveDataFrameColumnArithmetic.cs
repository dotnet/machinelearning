
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from PrimitiveDataFrameColumnArithmetic.tt. Do not modify directly

using System;

namespace Microsoft.Data.Analysis
{
    internal class PrimitiveDataFrameColumnArithmetic<T> : IPrimitiveDataFrameColumnArithmetic<T>
        where T : unmanaged
    {
        public static IPrimitiveDataFrameColumnArithmetic<T> Instance { get; } = PrimitiveDataFrameColumnArithmetic.GetArithmetic<T>();

        public void HandleOperation(BinaryOperation operation, Span<T> left, Span<byte> leftValidity, ReadOnlySpan<T> right, ReadOnlySpan<byte> rightValidity)
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

            BitUtility.ElementwiseAnd(leftValidity, rightValidity, leftValidity);
        }

        public void HandleOperation(BinaryOperation operation, Span<T> left, T right)
        {
            switch (operation)
            {
                case BinaryOperation.Add:
                    Add(left, right);
                    break;
                case BinaryOperation.Subtract:
                    Subtract(left, right);
                    break;
                case BinaryOperation.Multiply:
                    Multiply(left, right);
                    break;
                case BinaryOperation.Divide:
                    Divide(left, right);
                    break;
                case BinaryOperation.Modulo:
                    Modulo(left, right);
                    break;
                case BinaryOperation.And:
                    And(left, right);
                    break;
                case BinaryOperation.Or:
                    Or(left, right);
                    break;
                case BinaryOperation.Xor:
                    Xor(left, right);
                    break;
            }
        }

        public void HandleOperation(BinaryOperation operation, T left, Span<T> right, ReadOnlySpan<byte> rightValidity)
        {
            if (operation == BinaryOperation.Divide)
            {
                Divide(left, right, rightValidity);
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
        }

        public void HandleOperation(BinaryIntOperation operation, Span<T> left, int right)
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

        public void HandleOperation(ComparisonOperation operation, ReadOnlySpan<T> left, ReadOnlySpan<T> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            switch (operation)
            {
                case ComparisonOperation.ElementwiseEquals:
                    ElementwiseEquals(left, right, result, offset);
                    break;
                case ComparisonOperation.ElementwiseNotEquals:
                    ElementwiseNotEquals(left, right, result, offset);
                    break;
                case ComparisonOperation.ElementwiseGreaterThanOrEqual:
                    ElementwiseGreaterThanOrEqual(left, right, result, offset);
                    break;
                case ComparisonOperation.ElementwiseLessThanOrEqual:
                    ElementwiseLessThanOrEqual(left, right, result, offset);
                    break;
                case ComparisonOperation.ElementwiseGreaterThan:
                    ElementwiseGreaterThan(left, right, result, offset);
                    break;
                case ComparisonOperation.ElementwiseLessThan:
                    ElementwiseLessThan(left, right, result, offset);
                    break;
            }
        }

        public void HandleOperation(ComparisonOperation operation, ReadOnlySpan<T> left, T right, PrimitiveColumnContainer<bool> result, long offset)
        {
            switch (operation)
            {
                case ComparisonOperation.ElementwiseEquals:
                    ElementwiseEquals(left, right, result, offset);
                    break;
                case ComparisonOperation.ElementwiseNotEquals:
                    ElementwiseNotEquals(left, right, result, offset);
                    break;
                case ComparisonOperation.ElementwiseGreaterThanOrEqual:
                    ElementwiseGreaterThanOrEqual(left, right, result, offset);
                    break;
                case ComparisonOperation.ElementwiseLessThanOrEqual:
                    ElementwiseLessThanOrEqual(left, right, result, offset);
                    break;
                case ComparisonOperation.ElementwiseGreaterThan:
                    ElementwiseGreaterThan(left, right, result, offset);
                    break;
                case ComparisonOperation.ElementwiseLessThan:
                    ElementwiseLessThan(left, right, result, offset);
                    break;
            }
        }

        protected virtual void Add(Span<T> left, ReadOnlySpan<T> right) => throw new NotSupportedException();
        protected virtual void Add(Span<T> left, T scalar) => throw new NotSupportedException();
        protected virtual void Add(T left, Span<T> right) => throw new NotSupportedException();
        protected virtual void Subtract(Span<T> left, ReadOnlySpan<T> right) => throw new NotSupportedException();
        protected virtual void Subtract(Span<T> left, T scalar) => throw new NotSupportedException();
        protected virtual void Subtract(T left, Span<T> right) => throw new NotSupportedException();
        protected virtual void Multiply(Span<T> left, ReadOnlySpan<T> right) => throw new NotSupportedException();
        protected virtual void Multiply(Span<T> left, T scalar) => throw new NotSupportedException();
        protected virtual void Multiply(T left, Span<T> right) => throw new NotSupportedException();
        protected virtual void Divide(Span<T> left, Span<byte> leftValidity, ReadOnlySpan<T> right, ReadOnlySpan<byte> rightValidity) => throw new NotSupportedException();
        protected virtual void Divide(Span<T> left, T scalar) => throw new NotSupportedException();
        protected virtual void Divide(T left, Span<T> right, ReadOnlySpan<byte> rightValidity) => throw new NotSupportedException();
        protected virtual void Modulo(Span<T> left, ReadOnlySpan<T> right) => throw new NotSupportedException();
        protected virtual void Modulo(Span<T> left, T scalar) => throw new NotSupportedException();
        protected virtual void Modulo(T left, Span<T> right) => throw new NotSupportedException();
        protected virtual void And(Span<T> left, ReadOnlySpan<T> right) => throw new NotSupportedException();
        protected virtual void And(Span<T> left, T scalar) => throw new NotSupportedException();
        protected virtual void And(T left, Span<T> right) => throw new NotSupportedException();
        protected virtual void Or(Span<T> left, ReadOnlySpan<T> right) => throw new NotSupportedException();
        protected virtual void Or(Span<T> left, T scalar) => throw new NotSupportedException();
        protected virtual void Or(T left, Span<T> right) => throw new NotSupportedException();
        protected virtual void Xor(Span<T> left, ReadOnlySpan<T> right) => throw new NotSupportedException();
        protected virtual void Xor(Span<T> left, T scalar) => throw new NotSupportedException();
        protected virtual void Xor(T left, Span<T> right) => throw new NotSupportedException();
        protected virtual void LeftShift(Span<T> left, int right) => throw new NotSupportedException();
        protected virtual void RightShift(Span<T> left, int right) => throw new NotSupportedException();
        protected virtual void ElementwiseEquals(ReadOnlySpan<T> left, ReadOnlySpan<T> right, PrimitiveColumnContainer<bool> result, long offset) => throw new NotSupportedException();
        protected virtual void ElementwiseEquals(ReadOnlySpan<T> left, T right, PrimitiveColumnContainer<bool> result, long offset) => throw new NotSupportedException();
        protected virtual void ElementwiseNotEquals(ReadOnlySpan<T> left, ReadOnlySpan<T> right, PrimitiveColumnContainer<bool> result, long offset) => throw new NotSupportedException();
        protected virtual void ElementwiseNotEquals(ReadOnlySpan<T> left, T right, PrimitiveColumnContainer<bool> result, long offset) => throw new NotSupportedException();
        protected virtual void ElementwiseGreaterThanOrEqual(ReadOnlySpan<T> left, ReadOnlySpan<T> right, PrimitiveColumnContainer<bool> result, long offset) => throw new NotSupportedException();
        protected virtual void ElementwiseGreaterThanOrEqual(ReadOnlySpan<T> left, T right, PrimitiveColumnContainer<bool> result, long offset) => throw new NotSupportedException();
        protected virtual void ElementwiseLessThanOrEqual(ReadOnlySpan<T> left, ReadOnlySpan<T> right, PrimitiveColumnContainer<bool> result, long offset) => throw new NotSupportedException();
        protected virtual void ElementwiseLessThanOrEqual(ReadOnlySpan<T> left, T right, PrimitiveColumnContainer<bool> result, long offset) => throw new NotSupportedException();
        protected virtual void ElementwiseGreaterThan(ReadOnlySpan<T> left, ReadOnlySpan<T> right, PrimitiveColumnContainer<bool> result, long offset) => throw new NotSupportedException();
        protected virtual void ElementwiseGreaterThan(ReadOnlySpan<T> left, T right, PrimitiveColumnContainer<bool> result, long offset) => throw new NotSupportedException();
        protected virtual void ElementwiseLessThan(ReadOnlySpan<T> left, ReadOnlySpan<T> right, PrimitiveColumnContainer<bool> result, long offset) => throw new NotSupportedException();
        protected virtual void ElementwiseLessThan(ReadOnlySpan<T> left, T right, PrimitiveColumnContainer<bool> result, long offset) => throw new NotSupportedException();

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

        protected override void And(Span<bool> left, ReadOnlySpan<bool> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (bool)(left[i] & right[i]);
        }

        protected override void And(Span<bool> left, bool right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (bool)(left[i] & right);
        }

        protected override void And(bool left, Span<bool> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (bool)(left & right[i]);
        }

        protected override void Or(Span<bool> left, ReadOnlySpan<bool> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (bool)(left[i] | right[i]);
        }

        protected override void Or(Span<bool> left, bool right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (bool)(left[i] | right);
        }

        protected override void Or(bool left, Span<bool> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (bool)(left | right[i]);
        }

        protected override void Xor(Span<bool> left, ReadOnlySpan<bool> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (bool)(left[i] ^ right[i]);
        }

        protected override void Xor(Span<bool> left, bool right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (bool)(left[i] ^ right);
        }

        protected override void Xor(bool left, Span<bool> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (bool)(left ^ right[i]);
        }

        protected override void ElementwiseEquals(ReadOnlySpan<bool> left, ReadOnlySpan<bool> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right[i]);
            }
        }

        protected override void ElementwiseEquals(ReadOnlySpan<bool> left, bool right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<bool> left, ReadOnlySpan<bool> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right[i]);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<bool> left, bool right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right);
            }
        }
    }
    internal class ByteArithmetic : PrimitiveDataFrameColumnArithmetic<byte>
    {

        protected override void Add(Span<byte> left, ReadOnlySpan<byte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] + right[i]);
        }

        protected override void Add(Span<byte> left, byte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] + right);
        }

        protected override void Add(byte left, Span<byte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (byte)(left + right[i]);
        }

        protected override void Subtract(Span<byte> left, ReadOnlySpan<byte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] - right[i]);
        }

        protected override void Subtract(Span<byte> left, byte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] - right);
        }

        protected override void Subtract(byte left, Span<byte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (byte)(left - right[i]);
        }

        protected override void Multiply(Span<byte> left, ReadOnlySpan<byte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] * right[i]);
        }

        protected override void Multiply(Span<byte> left, byte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] * right);
        }

        protected override void Multiply(byte left, Span<byte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (byte)(left * right[i]);
        }

        protected override void Divide(Span<byte> left, Span<byte> leftValidity, ReadOnlySpan<byte> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    left[i] = (byte)(left[i] / right[i]);
                else
                    BitUtility.ClearBit(leftValidity, i);
            }
        }

        protected override void Divide(Span<byte> left, byte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] / right);
        }

        protected override void Divide(byte left, Span<byte> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    right[i] = (byte)(left / right[i]);
            }
        }

        protected override void Modulo(Span<byte> left, ReadOnlySpan<byte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] % right[i]);
        }

        protected override void Modulo(Span<byte> left, byte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] % right);
        }

        protected override void Modulo(byte left, Span<byte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (byte)(left % right[i]);
        }

        protected override void And(Span<byte> left, ReadOnlySpan<byte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] & right[i]);
        }

        protected override void And(Span<byte> left, byte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] & right);
        }

        protected override void And(byte left, Span<byte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (byte)(left & right[i]);
        }

        protected override void Or(Span<byte> left, ReadOnlySpan<byte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] | right[i]);
        }

        protected override void Or(Span<byte> left, byte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] | right);
        }

        protected override void Or(byte left, Span<byte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (byte)(left | right[i]);
        }

        protected override void Xor(Span<byte> left, ReadOnlySpan<byte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] ^ right[i]);
        }

        protected override void Xor(Span<byte> left, byte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] ^ right);
        }

        protected override void Xor(byte left, Span<byte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (byte)(left ^ right[i]);
        }

        protected override void LeftShift(Span<byte> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] << right);
        }

        protected override void RightShift(Span<byte> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (byte)(left[i] >> right);
        }

        protected override void ElementwiseEquals(ReadOnlySpan<byte> left, ReadOnlySpan<byte> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right[i]);
            }
        }

        protected override void ElementwiseEquals(ReadOnlySpan<byte> left, byte right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<byte> left, ReadOnlySpan<byte> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right[i]);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<byte> left, byte right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<byte> left, ReadOnlySpan<byte> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right[i]);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<byte> left, byte right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<byte> left, ReadOnlySpan<byte> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right[i]);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<byte> left, byte right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<byte> left, ReadOnlySpan<byte> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right[i]);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<byte> left, byte right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<byte> left, ReadOnlySpan<byte> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right[i]);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<byte> left, byte right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right);
            }
        }
    }
    internal class CharArithmetic : PrimitiveDataFrameColumnArithmetic<char>
    {

        protected override void Add(Span<char> left, ReadOnlySpan<char> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] + right[i]);
        }

        protected override void Add(Span<char> left, char right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] + right);
        }

        protected override void Add(char left, Span<char> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (char)(left + right[i]);
        }

        protected override void Subtract(Span<char> left, ReadOnlySpan<char> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] - right[i]);
        }

        protected override void Subtract(Span<char> left, char right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] - right);
        }

        protected override void Subtract(char left, Span<char> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (char)(left - right[i]);
        }

        protected override void Multiply(Span<char> left, ReadOnlySpan<char> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] * right[i]);
        }

        protected override void Multiply(Span<char> left, char right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] * right);
        }

        protected override void Multiply(char left, Span<char> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (char)(left * right[i]);
        }

        protected override void Divide(Span<char> left, Span<byte> leftValidity, ReadOnlySpan<char> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    left[i] = (char)(left[i] / right[i]);
                else
                    BitUtility.ClearBit(leftValidity, i);
            }
        }

        protected override void Divide(Span<char> left, char right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] / right);
        }

        protected override void Divide(char left, Span<char> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    right[i] = (char)(left / right[i]);
            }
        }

        protected override void Modulo(Span<char> left, ReadOnlySpan<char> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] % right[i]);
        }

        protected override void Modulo(Span<char> left, char right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] % right);
        }

        protected override void Modulo(char left, Span<char> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (char)(left % right[i]);
        }

        protected override void And(Span<char> left, ReadOnlySpan<char> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] & right[i]);
        }

        protected override void And(Span<char> left, char right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] & right);
        }

        protected override void And(char left, Span<char> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (char)(left & right[i]);
        }

        protected override void Or(Span<char> left, ReadOnlySpan<char> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] | right[i]);
        }

        protected override void Or(Span<char> left, char right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] | right);
        }

        protected override void Or(char left, Span<char> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (char)(left | right[i]);
        }

        protected override void Xor(Span<char> left, ReadOnlySpan<char> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] ^ right[i]);
        }

        protected override void Xor(Span<char> left, char right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] ^ right);
        }

        protected override void Xor(char left, Span<char> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (char)(left ^ right[i]);
        }

        protected override void LeftShift(Span<char> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] << right);
        }

        protected override void RightShift(Span<char> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (char)(left[i] >> right);
        }

        protected override void ElementwiseEquals(ReadOnlySpan<char> left, ReadOnlySpan<char> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right[i]);
            }
        }

        protected override void ElementwiseEquals(ReadOnlySpan<char> left, char right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<char> left, ReadOnlySpan<char> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right[i]);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<char> left, char right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<char> left, ReadOnlySpan<char> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right[i]);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<char> left, char right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<char> left, ReadOnlySpan<char> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right[i]);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<char> left, char right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<char> left, ReadOnlySpan<char> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right[i]);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<char> left, char right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<char> left, ReadOnlySpan<char> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right[i]);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<char> left, char right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right);
            }
        }
    }
    internal class DecimalArithmetic : PrimitiveDataFrameColumnArithmetic<decimal>
    {

        protected override void Add(Span<decimal> left, ReadOnlySpan<decimal> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (decimal)(left[i] + right[i]);
        }

        protected override void Add(Span<decimal> left, decimal right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (decimal)(left[i] + right);
        }

        protected override void Add(decimal left, Span<decimal> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (decimal)(left + right[i]);
        }

        protected override void Subtract(Span<decimal> left, ReadOnlySpan<decimal> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (decimal)(left[i] - right[i]);
        }

        protected override void Subtract(Span<decimal> left, decimal right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (decimal)(left[i] - right);
        }

        protected override void Subtract(decimal left, Span<decimal> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (decimal)(left - right[i]);
        }

        protected override void Multiply(Span<decimal> left, ReadOnlySpan<decimal> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (decimal)(left[i] * right[i]);
        }

        protected override void Multiply(Span<decimal> left, decimal right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (decimal)(left[i] * right);
        }

        protected override void Multiply(decimal left, Span<decimal> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (decimal)(left * right[i]);
        }

        protected override void Divide(Span<decimal> left, Span<byte> leftValidity, ReadOnlySpan<decimal> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    left[i] = (decimal)(left[i] / right[i]);
                else
                    BitUtility.ClearBit(leftValidity, i);
            }
        }

        protected override void Divide(Span<decimal> left, decimal right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (decimal)(left[i] / right);
        }

        protected override void Divide(decimal left, Span<decimal> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    right[i] = (decimal)(left / right[i]);
            }
        }

        protected override void Modulo(Span<decimal> left, ReadOnlySpan<decimal> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (decimal)(left[i] % right[i]);
        }

        protected override void Modulo(Span<decimal> left, decimal right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (decimal)(left[i] % right);
        }

        protected override void Modulo(decimal left, Span<decimal> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (decimal)(left % right[i]);
        }

        protected override void ElementwiseEquals(ReadOnlySpan<decimal> left, ReadOnlySpan<decimal> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right[i]);
            }
        }

        protected override void ElementwiseEquals(ReadOnlySpan<decimal> left, decimal right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<decimal> left, ReadOnlySpan<decimal> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right[i]);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<decimal> left, decimal right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<decimal> left, ReadOnlySpan<decimal> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right[i]);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<decimal> left, decimal right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<decimal> left, ReadOnlySpan<decimal> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right[i]);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<decimal> left, decimal right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<decimal> left, ReadOnlySpan<decimal> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right[i]);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<decimal> left, decimal right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<decimal> left, ReadOnlySpan<decimal> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right[i]);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<decimal> left, decimal right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right);
            }
        }
    }
    internal class DoubleArithmetic : PrimitiveDataFrameColumnArithmetic<double>
    {

        protected override void Add(Span<double> left, ReadOnlySpan<double> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (double)(left[i] + right[i]);
        }

        protected override void Add(Span<double> left, double right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (double)(left[i] + right);
        }

        protected override void Add(double left, Span<double> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (double)(left + right[i]);
        }

        protected override void Subtract(Span<double> left, ReadOnlySpan<double> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (double)(left[i] - right[i]);
        }

        protected override void Subtract(Span<double> left, double right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (double)(left[i] - right);
        }

        protected override void Subtract(double left, Span<double> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (double)(left - right[i]);
        }

        protected override void Multiply(Span<double> left, ReadOnlySpan<double> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (double)(left[i] * right[i]);
        }

        protected override void Multiply(Span<double> left, double right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (double)(left[i] * right);
        }

        protected override void Multiply(double left, Span<double> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (double)(left * right[i]);
        }

        protected override void Divide(Span<double> left, Span<byte> leftValidity, ReadOnlySpan<double> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    left[i] = (double)(left[i] / right[i]);
                else
                    BitUtility.ClearBit(leftValidity, i);
            }
        }

        protected override void Divide(Span<double> left, double right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (double)(left[i] / right);
        }

        protected override void Divide(double left, Span<double> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    right[i] = (double)(left / right[i]);
            }
        }

        protected override void Modulo(Span<double> left, ReadOnlySpan<double> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (double)(left[i] % right[i]);
        }

        protected override void Modulo(Span<double> left, double right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (double)(left[i] % right);
        }

        protected override void Modulo(double left, Span<double> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (double)(left % right[i]);
        }

        protected override void ElementwiseEquals(ReadOnlySpan<double> left, ReadOnlySpan<double> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right[i]);
            }
        }

        protected override void ElementwiseEquals(ReadOnlySpan<double> left, double right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<double> left, ReadOnlySpan<double> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right[i]);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<double> left, double right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<double> left, ReadOnlySpan<double> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right[i]);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<double> left, double right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<double> left, ReadOnlySpan<double> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right[i]);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<double> left, double right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<double> left, ReadOnlySpan<double> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right[i]);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<double> left, double right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<double> left, ReadOnlySpan<double> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right[i]);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<double> left, double right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right);
            }
        }
    }
    internal class FloatArithmetic : PrimitiveDataFrameColumnArithmetic<float>
    {

        protected override void Add(Span<float> left, ReadOnlySpan<float> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (float)(left[i] + right[i]);
        }

        protected override void Add(Span<float> left, float right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (float)(left[i] + right);
        }

        protected override void Add(float left, Span<float> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (float)(left + right[i]);
        }

        protected override void Subtract(Span<float> left, ReadOnlySpan<float> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (float)(left[i] - right[i]);
        }

        protected override void Subtract(Span<float> left, float right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (float)(left[i] - right);
        }

        protected override void Subtract(float left, Span<float> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (float)(left - right[i]);
        }

        protected override void Multiply(Span<float> left, ReadOnlySpan<float> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (float)(left[i] * right[i]);
        }

        protected override void Multiply(Span<float> left, float right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (float)(left[i] * right);
        }

        protected override void Multiply(float left, Span<float> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (float)(left * right[i]);
        }

        protected override void Divide(Span<float> left, Span<byte> leftValidity, ReadOnlySpan<float> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    left[i] = (float)(left[i] / right[i]);
                else
                    BitUtility.ClearBit(leftValidity, i);
            }
        }

        protected override void Divide(Span<float> left, float right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (float)(left[i] / right);
        }

        protected override void Divide(float left, Span<float> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    right[i] = (float)(left / right[i]);
            }
        }

        protected override void Modulo(Span<float> left, ReadOnlySpan<float> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (float)(left[i] % right[i]);
        }

        protected override void Modulo(Span<float> left, float right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (float)(left[i] % right);
        }

        protected override void Modulo(float left, Span<float> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (float)(left % right[i]);
        }

        protected override void ElementwiseEquals(ReadOnlySpan<float> left, ReadOnlySpan<float> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right[i]);
            }
        }

        protected override void ElementwiseEquals(ReadOnlySpan<float> left, float right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<float> left, ReadOnlySpan<float> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right[i]);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<float> left, float right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<float> left, ReadOnlySpan<float> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right[i]);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<float> left, float right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<float> left, ReadOnlySpan<float> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right[i]);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<float> left, float right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<float> left, ReadOnlySpan<float> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right[i]);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<float> left, float right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<float> left, ReadOnlySpan<float> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right[i]);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<float> left, float right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right);
            }
        }
    }
    internal class IntArithmetic : PrimitiveDataFrameColumnArithmetic<int>
    {

        protected override void Add(Span<int> left, ReadOnlySpan<int> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] + right[i]);
        }

        protected override void Add(Span<int> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] + right);
        }

        protected override void Add(int left, Span<int> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (int)(left + right[i]);
        }

        protected override void Subtract(Span<int> left, ReadOnlySpan<int> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] - right[i]);
        }

        protected override void Subtract(Span<int> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] - right);
        }

        protected override void Subtract(int left, Span<int> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (int)(left - right[i]);
        }

        protected override void Multiply(Span<int> left, ReadOnlySpan<int> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] * right[i]);
        }

        protected override void Multiply(Span<int> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] * right);
        }

        protected override void Multiply(int left, Span<int> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (int)(left * right[i]);
        }

        protected override void Divide(Span<int> left, Span<byte> leftValidity, ReadOnlySpan<int> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    left[i] = (int)(left[i] / right[i]);
                else
                    BitUtility.ClearBit(leftValidity, i);
            }
        }

        protected override void Divide(Span<int> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] / right);
        }

        protected override void Divide(int left, Span<int> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    right[i] = (int)(left / right[i]);
            }
        }

        protected override void Modulo(Span<int> left, ReadOnlySpan<int> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] % right[i]);
        }

        protected override void Modulo(Span<int> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] % right);
        }

        protected override void Modulo(int left, Span<int> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (int)(left % right[i]);
        }

        protected override void And(Span<int> left, ReadOnlySpan<int> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] & right[i]);
        }

        protected override void And(Span<int> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] & right);
        }

        protected override void And(int left, Span<int> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (int)(left & right[i]);
        }

        protected override void Or(Span<int> left, ReadOnlySpan<int> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] | right[i]);
        }

        protected override void Or(Span<int> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] | right);
        }

        protected override void Or(int left, Span<int> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (int)(left | right[i]);
        }

        protected override void Xor(Span<int> left, ReadOnlySpan<int> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] ^ right[i]);
        }

        protected override void Xor(Span<int> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] ^ right);
        }

        protected override void Xor(int left, Span<int> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (int)(left ^ right[i]);
        }

        protected override void LeftShift(Span<int> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] << right);
        }

        protected override void RightShift(Span<int> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (int)(left[i] >> right);
        }

        protected override void ElementwiseEquals(ReadOnlySpan<int> left, ReadOnlySpan<int> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right[i]);
            }
        }

        protected override void ElementwiseEquals(ReadOnlySpan<int> left, int right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<int> left, ReadOnlySpan<int> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right[i]);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<int> left, int right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<int> left, ReadOnlySpan<int> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right[i]);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<int> left, int right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<int> left, ReadOnlySpan<int> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right[i]);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<int> left, int right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<int> left, ReadOnlySpan<int> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right[i]);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<int> left, int right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<int> left, ReadOnlySpan<int> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right[i]);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<int> left, int right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right);
            }
        }
    }
    internal class LongArithmetic : PrimitiveDataFrameColumnArithmetic<long>
    {

        protected override void Add(Span<long> left, ReadOnlySpan<long> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] + right[i]);
        }

        protected override void Add(Span<long> left, long right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] + right);
        }

        protected override void Add(long left, Span<long> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (long)(left + right[i]);
        }

        protected override void Subtract(Span<long> left, ReadOnlySpan<long> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] - right[i]);
        }

        protected override void Subtract(Span<long> left, long right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] - right);
        }

        protected override void Subtract(long left, Span<long> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (long)(left - right[i]);
        }

        protected override void Multiply(Span<long> left, ReadOnlySpan<long> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] * right[i]);
        }

        protected override void Multiply(Span<long> left, long right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] * right);
        }

        protected override void Multiply(long left, Span<long> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (long)(left * right[i]);
        }

        protected override void Divide(Span<long> left, Span<byte> leftValidity, ReadOnlySpan<long> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    left[i] = (long)(left[i] / right[i]);
                else
                    BitUtility.ClearBit(leftValidity, i);
            }
        }

        protected override void Divide(Span<long> left, long right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] / right);
        }

        protected override void Divide(long left, Span<long> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    right[i] = (long)(left / right[i]);
            }
        }

        protected override void Modulo(Span<long> left, ReadOnlySpan<long> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] % right[i]);
        }

        protected override void Modulo(Span<long> left, long right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] % right);
        }

        protected override void Modulo(long left, Span<long> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (long)(left % right[i]);
        }

        protected override void And(Span<long> left, ReadOnlySpan<long> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] & right[i]);
        }

        protected override void And(Span<long> left, long right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] & right);
        }

        protected override void And(long left, Span<long> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (long)(left & right[i]);
        }

        protected override void Or(Span<long> left, ReadOnlySpan<long> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] | right[i]);
        }

        protected override void Or(Span<long> left, long right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] | right);
        }

        protected override void Or(long left, Span<long> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (long)(left | right[i]);
        }

        protected override void Xor(Span<long> left, ReadOnlySpan<long> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] ^ right[i]);
        }

        protected override void Xor(Span<long> left, long right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] ^ right);
        }

        protected override void Xor(long left, Span<long> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (long)(left ^ right[i]);
        }

        protected override void LeftShift(Span<long> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] << right);
        }

        protected override void RightShift(Span<long> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (long)(left[i] >> right);
        }

        protected override void ElementwiseEquals(ReadOnlySpan<long> left, ReadOnlySpan<long> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right[i]);
            }
        }

        protected override void ElementwiseEquals(ReadOnlySpan<long> left, long right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<long> left, ReadOnlySpan<long> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right[i]);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<long> left, long right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<long> left, ReadOnlySpan<long> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right[i]);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<long> left, long right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<long> left, ReadOnlySpan<long> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right[i]);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<long> left, long right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<long> left, ReadOnlySpan<long> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right[i]);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<long> left, long right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<long> left, ReadOnlySpan<long> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right[i]);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<long> left, long right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right);
            }
        }
    }
    internal class SByteArithmetic : PrimitiveDataFrameColumnArithmetic<sbyte>
    {

        protected override void Add(Span<sbyte> left, ReadOnlySpan<sbyte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] + right[i]);
        }

        protected override void Add(Span<sbyte> left, sbyte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] + right);
        }

        protected override void Add(sbyte left, Span<sbyte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (sbyte)(left + right[i]);
        }

        protected override void Subtract(Span<sbyte> left, ReadOnlySpan<sbyte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] - right[i]);
        }

        protected override void Subtract(Span<sbyte> left, sbyte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] - right);
        }

        protected override void Subtract(sbyte left, Span<sbyte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (sbyte)(left - right[i]);
        }

        protected override void Multiply(Span<sbyte> left, ReadOnlySpan<sbyte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] * right[i]);
        }

        protected override void Multiply(Span<sbyte> left, sbyte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] * right);
        }

        protected override void Multiply(sbyte left, Span<sbyte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (sbyte)(left * right[i]);
        }

        protected override void Divide(Span<sbyte> left, Span<byte> leftValidity, ReadOnlySpan<sbyte> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    left[i] = (sbyte)(left[i] / right[i]);
                else
                    BitUtility.ClearBit(leftValidity, i);
            }
        }

        protected override void Divide(Span<sbyte> left, sbyte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] / right);
        }

        protected override void Divide(sbyte left, Span<sbyte> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    right[i] = (sbyte)(left / right[i]);
            }
        }

        protected override void Modulo(Span<sbyte> left, ReadOnlySpan<sbyte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] % right[i]);
        }

        protected override void Modulo(Span<sbyte> left, sbyte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] % right);
        }

        protected override void Modulo(sbyte left, Span<sbyte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (sbyte)(left % right[i]);
        }

        protected override void And(Span<sbyte> left, ReadOnlySpan<sbyte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] & right[i]);
        }

        protected override void And(Span<sbyte> left, sbyte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] & right);
        }

        protected override void And(sbyte left, Span<sbyte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (sbyte)(left & right[i]);
        }

        protected override void Or(Span<sbyte> left, ReadOnlySpan<sbyte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] | right[i]);
        }

        protected override void Or(Span<sbyte> left, sbyte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] | right);
        }

        protected override void Or(sbyte left, Span<sbyte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (sbyte)(left | right[i]);
        }

        protected override void Xor(Span<sbyte> left, ReadOnlySpan<sbyte> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] ^ right[i]);
        }

        protected override void Xor(Span<sbyte> left, sbyte right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] ^ right);
        }

        protected override void Xor(sbyte left, Span<sbyte> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (sbyte)(left ^ right[i]);
        }

        protected override void LeftShift(Span<sbyte> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] << right);
        }

        protected override void RightShift(Span<sbyte> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (sbyte)(left[i] >> right);
        }

        protected override void ElementwiseEquals(ReadOnlySpan<sbyte> left, ReadOnlySpan<sbyte> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right[i]);
            }
        }

        protected override void ElementwiseEquals(ReadOnlySpan<sbyte> left, sbyte right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<sbyte> left, ReadOnlySpan<sbyte> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right[i]);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<sbyte> left, sbyte right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<sbyte> left, ReadOnlySpan<sbyte> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right[i]);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<sbyte> left, sbyte right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<sbyte> left, ReadOnlySpan<sbyte> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right[i]);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<sbyte> left, sbyte right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<sbyte> left, ReadOnlySpan<sbyte> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right[i]);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<sbyte> left, sbyte right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<sbyte> left, ReadOnlySpan<sbyte> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right[i]);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<sbyte> left, sbyte right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right);
            }
        }
    }
    internal class ShortArithmetic : PrimitiveDataFrameColumnArithmetic<short>
    {

        protected override void Add(Span<short> left, ReadOnlySpan<short> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] + right[i]);
        }

        protected override void Add(Span<short> left, short right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] + right);
        }

        protected override void Add(short left, Span<short> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (short)(left + right[i]);
        }

        protected override void Subtract(Span<short> left, ReadOnlySpan<short> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] - right[i]);
        }

        protected override void Subtract(Span<short> left, short right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] - right);
        }

        protected override void Subtract(short left, Span<short> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (short)(left - right[i]);
        }

        protected override void Multiply(Span<short> left, ReadOnlySpan<short> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] * right[i]);
        }

        protected override void Multiply(Span<short> left, short right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] * right);
        }

        protected override void Multiply(short left, Span<short> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (short)(left * right[i]);
        }

        protected override void Divide(Span<short> left, Span<byte> leftValidity, ReadOnlySpan<short> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    left[i] = (short)(left[i] / right[i]);
                else
                    BitUtility.ClearBit(leftValidity, i);
            }
        }

        protected override void Divide(Span<short> left, short right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] / right);
        }

        protected override void Divide(short left, Span<short> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    right[i] = (short)(left / right[i]);
            }
        }

        protected override void Modulo(Span<short> left, ReadOnlySpan<short> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] % right[i]);
        }

        protected override void Modulo(Span<short> left, short right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] % right);
        }

        protected override void Modulo(short left, Span<short> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (short)(left % right[i]);
        }

        protected override void And(Span<short> left, ReadOnlySpan<short> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] & right[i]);
        }

        protected override void And(Span<short> left, short right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] & right);
        }

        protected override void And(short left, Span<short> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (short)(left & right[i]);
        }

        protected override void Or(Span<short> left, ReadOnlySpan<short> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] | right[i]);
        }

        protected override void Or(Span<short> left, short right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] | right);
        }

        protected override void Or(short left, Span<short> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (short)(left | right[i]);
        }

        protected override void Xor(Span<short> left, ReadOnlySpan<short> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] ^ right[i]);
        }

        protected override void Xor(Span<short> left, short right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] ^ right);
        }

        protected override void Xor(short left, Span<short> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (short)(left ^ right[i]);
        }

        protected override void LeftShift(Span<short> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] << right);
        }

        protected override void RightShift(Span<short> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (short)(left[i] >> right);
        }

        protected override void ElementwiseEquals(ReadOnlySpan<short> left, ReadOnlySpan<short> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right[i]);
            }
        }

        protected override void ElementwiseEquals(ReadOnlySpan<short> left, short right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<short> left, ReadOnlySpan<short> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right[i]);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<short> left, short right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<short> left, ReadOnlySpan<short> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right[i]);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<short> left, short right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<short> left, ReadOnlySpan<short> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right[i]);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<short> left, short right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<short> left, ReadOnlySpan<short> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right[i]);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<short> left, short right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<short> left, ReadOnlySpan<short> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right[i]);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<short> left, short right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right);
            }
        }
    }
    internal class UIntArithmetic : PrimitiveDataFrameColumnArithmetic<uint>
    {

        protected override void Add(Span<uint> left, ReadOnlySpan<uint> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] + right[i]);
        }

        protected override void Add(Span<uint> left, uint right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] + right);
        }

        protected override void Add(uint left, Span<uint> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (uint)(left + right[i]);
        }

        protected override void Subtract(Span<uint> left, ReadOnlySpan<uint> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] - right[i]);
        }

        protected override void Subtract(Span<uint> left, uint right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] - right);
        }

        protected override void Subtract(uint left, Span<uint> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (uint)(left - right[i]);
        }

        protected override void Multiply(Span<uint> left, ReadOnlySpan<uint> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] * right[i]);
        }

        protected override void Multiply(Span<uint> left, uint right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] * right);
        }

        protected override void Multiply(uint left, Span<uint> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (uint)(left * right[i]);
        }

        protected override void Divide(Span<uint> left, Span<byte> leftValidity, ReadOnlySpan<uint> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    left[i] = (uint)(left[i] / right[i]);
                else
                    BitUtility.ClearBit(leftValidity, i);
            }
        }

        protected override void Divide(Span<uint> left, uint right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] / right);
        }

        protected override void Divide(uint left, Span<uint> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    right[i] = (uint)(left / right[i]);
            }
        }

        protected override void Modulo(Span<uint> left, ReadOnlySpan<uint> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] % right[i]);
        }

        protected override void Modulo(Span<uint> left, uint right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] % right);
        }

        protected override void Modulo(uint left, Span<uint> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (uint)(left % right[i]);
        }

        protected override void And(Span<uint> left, ReadOnlySpan<uint> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] & right[i]);
        }

        protected override void And(Span<uint> left, uint right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] & right);
        }

        protected override void And(uint left, Span<uint> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (uint)(left & right[i]);
        }

        protected override void Or(Span<uint> left, ReadOnlySpan<uint> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] | right[i]);
        }

        protected override void Or(Span<uint> left, uint right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] | right);
        }

        protected override void Or(uint left, Span<uint> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (uint)(left | right[i]);
        }

        protected override void Xor(Span<uint> left, ReadOnlySpan<uint> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] ^ right[i]);
        }

        protected override void Xor(Span<uint> left, uint right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] ^ right);
        }

        protected override void Xor(uint left, Span<uint> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (uint)(left ^ right[i]);
        }

        protected override void LeftShift(Span<uint> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] << right);
        }

        protected override void RightShift(Span<uint> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (uint)(left[i] >> right);
        }

        protected override void ElementwiseEquals(ReadOnlySpan<uint> left, ReadOnlySpan<uint> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right[i]);
            }
        }

        protected override void ElementwiseEquals(ReadOnlySpan<uint> left, uint right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<uint> left, ReadOnlySpan<uint> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right[i]);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<uint> left, uint right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<uint> left, ReadOnlySpan<uint> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right[i]);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<uint> left, uint right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<uint> left, ReadOnlySpan<uint> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right[i]);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<uint> left, uint right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<uint> left, ReadOnlySpan<uint> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right[i]);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<uint> left, uint right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<uint> left, ReadOnlySpan<uint> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right[i]);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<uint> left, uint right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right);
            }
        }
    }
    internal class ULongArithmetic : PrimitiveDataFrameColumnArithmetic<ulong>
    {

        protected override void Add(Span<ulong> left, ReadOnlySpan<ulong> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] + right[i]);
        }

        protected override void Add(Span<ulong> left, ulong right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] + right);
        }

        protected override void Add(ulong left, Span<ulong> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ulong)(left + right[i]);
        }

        protected override void Subtract(Span<ulong> left, ReadOnlySpan<ulong> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] - right[i]);
        }

        protected override void Subtract(Span<ulong> left, ulong right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] - right);
        }

        protected override void Subtract(ulong left, Span<ulong> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ulong)(left - right[i]);
        }

        protected override void Multiply(Span<ulong> left, ReadOnlySpan<ulong> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] * right[i]);
        }

        protected override void Multiply(Span<ulong> left, ulong right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] * right);
        }

        protected override void Multiply(ulong left, Span<ulong> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ulong)(left * right[i]);
        }

        protected override void Divide(Span<ulong> left, Span<byte> leftValidity, ReadOnlySpan<ulong> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    left[i] = (ulong)(left[i] / right[i]);
                else
                    BitUtility.ClearBit(leftValidity, i);
            }
        }

        protected override void Divide(Span<ulong> left, ulong right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] / right);
        }

        protected override void Divide(ulong left, Span<ulong> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    right[i] = (ulong)(left / right[i]);
            }
        }

        protected override void Modulo(Span<ulong> left, ReadOnlySpan<ulong> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] % right[i]);
        }

        protected override void Modulo(Span<ulong> left, ulong right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] % right);
        }

        protected override void Modulo(ulong left, Span<ulong> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ulong)(left % right[i]);
        }

        protected override void And(Span<ulong> left, ReadOnlySpan<ulong> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] & right[i]);
        }

        protected override void And(Span<ulong> left, ulong right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] & right);
        }

        protected override void And(ulong left, Span<ulong> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ulong)(left & right[i]);
        }

        protected override void Or(Span<ulong> left, ReadOnlySpan<ulong> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] | right[i]);
        }

        protected override void Or(Span<ulong> left, ulong right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] | right);
        }

        protected override void Or(ulong left, Span<ulong> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ulong)(left | right[i]);
        }

        protected override void Xor(Span<ulong> left, ReadOnlySpan<ulong> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] ^ right[i]);
        }

        protected override void Xor(Span<ulong> left, ulong right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] ^ right);
        }

        protected override void Xor(ulong left, Span<ulong> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ulong)(left ^ right[i]);
        }

        protected override void LeftShift(Span<ulong> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] << right);
        }

        protected override void RightShift(Span<ulong> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ulong)(left[i] >> right);
        }

        protected override void ElementwiseEquals(ReadOnlySpan<ulong> left, ReadOnlySpan<ulong> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right[i]);
            }
        }

        protected override void ElementwiseEquals(ReadOnlySpan<ulong> left, ulong right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<ulong> left, ReadOnlySpan<ulong> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right[i]);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<ulong> left, ulong right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<ulong> left, ReadOnlySpan<ulong> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right[i]);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<ulong> left, ulong right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<ulong> left, ReadOnlySpan<ulong> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right[i]);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<ulong> left, ulong right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<ulong> left, ReadOnlySpan<ulong> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right[i]);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<ulong> left, ulong right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<ulong> left, ReadOnlySpan<ulong> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right[i]);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<ulong> left, ulong right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right);
            }
        }
    }
    internal class UShortArithmetic : PrimitiveDataFrameColumnArithmetic<ushort>
    {

        protected override void Add(Span<ushort> left, ReadOnlySpan<ushort> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] + right[i]);
        }

        protected override void Add(Span<ushort> left, ushort right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] + right);
        }

        protected override void Add(ushort left, Span<ushort> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ushort)(left + right[i]);
        }

        protected override void Subtract(Span<ushort> left, ReadOnlySpan<ushort> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] - right[i]);
        }

        protected override void Subtract(Span<ushort> left, ushort right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] - right);
        }

        protected override void Subtract(ushort left, Span<ushort> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ushort)(left - right[i]);
        }

        protected override void Multiply(Span<ushort> left, ReadOnlySpan<ushort> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] * right[i]);
        }

        protected override void Multiply(Span<ushort> left, ushort right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] * right);
        }

        protected override void Multiply(ushort left, Span<ushort> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ushort)(left * right[i]);
        }

        protected override void Divide(Span<ushort> left, Span<byte> leftValidity, ReadOnlySpan<ushort> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < left.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    left[i] = (ushort)(left[i] / right[i]);
                else
                    BitUtility.ClearBit(leftValidity, i);
            }
        }

        protected override void Divide(Span<ushort> left, ushort right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] / right);
        }

        protected override void Divide(ushort left, Span<ushort> right, ReadOnlySpan<byte> rightValidity)
        {
            for (var i = 0; i < right.Length; i++)
            {
                if (BitUtility.IsValid(rightValidity, i))
                    right[i] = (ushort)(left / right[i]);
            }
        }

        protected override void Modulo(Span<ushort> left, ReadOnlySpan<ushort> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] % right[i]);
        }

        protected override void Modulo(Span<ushort> left, ushort right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] % right);
        }

        protected override void Modulo(ushort left, Span<ushort> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ushort)(left % right[i]);
        }

        protected override void And(Span<ushort> left, ReadOnlySpan<ushort> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] & right[i]);
        }

        protected override void And(Span<ushort> left, ushort right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] & right);
        }

        protected override void And(ushort left, Span<ushort> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ushort)(left & right[i]);
        }

        protected override void Or(Span<ushort> left, ReadOnlySpan<ushort> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] | right[i]);
        }

        protected override void Or(Span<ushort> left, ushort right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] | right);
        }

        protected override void Or(ushort left, Span<ushort> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ushort)(left | right[i]);
        }

        protected override void Xor(Span<ushort> left, ReadOnlySpan<ushort> right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] ^ right[i]);
        }

        protected override void Xor(Span<ushort> left, ushort right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] ^ right);
        }

        protected override void Xor(ushort left, Span<ushort> right)
        {
            for (var i = 0; i < right.Length; i++)
                right[i] = (ushort)(left ^ right[i]);
        }

        protected override void LeftShift(Span<ushort> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] << right);
        }

        protected override void RightShift(Span<ushort> left, int right)
        {
            for (var i = 0; i < left.Length; i++)
                left[i] = (ushort)(left[i] >> right);
        }

        protected override void ElementwiseEquals(ReadOnlySpan<ushort> left, ReadOnlySpan<ushort> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right[i]);
            }
        }

        protected override void ElementwiseEquals(ReadOnlySpan<ushort> left, ushort right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<ushort> left, ReadOnlySpan<ushort> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right[i]);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<ushort> left, ushort right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<ushort> left, ReadOnlySpan<ushort> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right[i]);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<ushort> left, ushort right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] >= right);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<ushort> left, ReadOnlySpan<ushort> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right[i]);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<ushort> left, ushort right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] <= right);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<ushort> left, ReadOnlySpan<ushort> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right[i]);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<ushort> left, ushort right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] > right);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<ushort> left, ReadOnlySpan<ushort> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right[i]);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<ushort> left, ushort right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] < right);
            }
        }
    }
    internal class DateTimeArithmetic : PrimitiveDataFrameColumnArithmetic<DateTime>
    {

        protected override void ElementwiseEquals(ReadOnlySpan<DateTime> left, ReadOnlySpan<DateTime> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right[i]);
            }
        }

        protected override void ElementwiseEquals(ReadOnlySpan<DateTime> left, DateTime right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] == right);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<DateTime> left, ReadOnlySpan<DateTime> right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right[i]);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<DateTime> left, DateTime right, PrimitiveColumnContainer<bool> result, long offset)
        {
            for (var i = 0; i < left.Length; i++)
            {
                result[i + offset] = (left[i] != right);
            }
        }
    }
}
