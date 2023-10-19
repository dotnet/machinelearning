
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from Arithmetic.tt. Do not modify directly

using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Microsoft.Data.Analysis
{
    ////////////////////////////////////////
    //Factory Class                       //
    ////////////////////////////////////////

    internal static class Arithmetic
    {
        public static IArithmetic<T> GetArithmetic<T>()
            where T : unmanaged
        {
            if (typeof(T) == typeof(bool))
                return (IArithmetic<T>)new BoolArithmetic();
            else if (typeof(T) == typeof(byte))
                return (IArithmetic<T>)new ByteArithmetic();
            else if (typeof(T) == typeof(char))
                return (IArithmetic<T>)new CharArithmetic();
            else if (typeof(T) == typeof(decimal))
                return (IArithmetic<T>)new DecimalArithmetic();
            else if (typeof(T) == typeof(double))
                return (IArithmetic<T>)new DoubleArithmetic();
            else if (typeof(T) == typeof(float))
                return (IArithmetic<T>)new FloatArithmetic();
            else if (typeof(T) == typeof(int))
                return (IArithmetic<T>)new IntArithmetic();
            else if (typeof(T) == typeof(long))
                return (IArithmetic<T>)new LongArithmetic();
            else if (typeof(T) == typeof(sbyte))
                return (IArithmetic<T>)new SByteArithmetic();
            else if (typeof(T) == typeof(short))
                return (IArithmetic<T>)new ShortArithmetic();
            else if (typeof(T) == typeof(uint))
                return (IArithmetic<T>)new UIntArithmetic();
            else if (typeof(T) == typeof(ulong))
                return (IArithmetic<T>)new ULongArithmetic();
            else if (typeof(T) == typeof(ushort))
                return (IArithmetic<T>)new UShortArithmetic();
            else if (typeof(T) == typeof(DateTime))
                return (IArithmetic<T>)new DateTimeArithmetic();
            throw new NotSupportedException();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ref Vector<T> AsVector<T>(ref T start, int offset)
            where T : struct => ref Unsafe.As<T, Vector<T>>(ref Unsafe.Add(ref start, offset));
    }


    ////////////////////////////////////////
    //Base Class for Arithmetic           //
    ////////////////////////////////////////

    internal class Arithmetic<T> : IArithmetic<T>
        where T : unmanaged
    {
        public static IArithmetic<T> Instance { get; } = Arithmetic.GetArithmetic<T>();


        //Binary operations

        public void HandleOperation(BinaryOperation operation, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
        {
            switch (operation)
            {
                case BinaryOperation.Add:
                    Add(x, y, destination);
                    break;
                case BinaryOperation.Subtract:
                    Subtract(x, y, destination);
                    break;
                case BinaryOperation.Multiply:
                    Multiply(x, y, destination);
                    break;
                case BinaryOperation.Divide:
                    Divide(x, y, destination);
                    break;
                case BinaryOperation.Modulo:
                    Modulo(x, y, destination);
                    break;
                case BinaryOperation.And:
                    And(x, y, destination);
                    break;
                case BinaryOperation.Or:
                    Or(x, y, destination);
                    break;
                case BinaryOperation.Xor:
                    Xor(x, y, destination);
                    break;
            }
        }

        public void HandleOperation(BinaryOperation operation, ReadOnlySpan<T> x, T y, Span<T> destination)
        {
            switch (operation)
            {
                case BinaryOperation.Add:
                    Add(x, y, destination);
                    break;
                case BinaryOperation.Subtract:
                    Subtract(x, y, destination);
                    break;
                case BinaryOperation.Multiply:
                    Multiply(x, y, destination);
                    break;
                case BinaryOperation.Divide:
                    Divide(x, y, destination);
                    break;
                case BinaryOperation.Modulo:
                    Modulo(x, y, destination);
                    break;
                case BinaryOperation.And:
                    And(x, y, destination);
                    break;
                case BinaryOperation.Or:
                    Or(x, y, destination);
                    break;
                case BinaryOperation.Xor:
                    Xor(x, y, destination);
                    break;
            }
        }

        public void HandleOperation(BinaryOperation operation, T x, ReadOnlySpan<T> y, Span<T> destination)
        {
            switch (operation)
            {
                case BinaryOperation.Add:
                    Add(x, y, destination);
                    break;
                case BinaryOperation.Subtract:
                    Subtract(x, y, destination);
                    break;
                case BinaryOperation.Multiply:
                    Multiply(x, y, destination);
                    break;
                case BinaryOperation.Divide:
                    Divide(x, y, destination);
                    break;
                case BinaryOperation.Modulo:
                    Modulo(x, y, destination);
                    break;
                case BinaryOperation.And:
                    And(x, y, destination);
                    break;
                case BinaryOperation.Or:
                    Or(x, y, destination);
                    break;
                case BinaryOperation.Xor:
                    Xor(x, y, destination);
                    break;
            }
        }

        public T HandleOperation(BinaryOperation operation, T x, T y)
        {
            if (operation == BinaryOperation.Divide)
                return Divide(x, y);

            if (operation == BinaryOperation.Modulo)
                return Modulo(x, y);

            throw new NotSupportedException();
        }


        //Binary Int operations

        public void HandleOperation(BinaryIntOperation operation, ReadOnlySpan<T> x, int y, Span<T> destination)
        {
            switch (operation)
            {
                case BinaryIntOperation.LeftShift:
                    LeftShift(x, y, destination);
                    break;
                case BinaryIntOperation.RightShift:
                    RightShift(x, y, destination);
                    break;
            }
        }


        //Comparison operations

        public void HandleOperation(ComparisonOperation operation, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
        {
            switch (operation)
            {
                case ComparisonOperation.ElementwiseEquals:
                    ElementwiseEquals(x, y, destination);
                    break;
                case ComparisonOperation.ElementwiseNotEquals:
                    ElementwiseNotEquals(x, y, destination);
                    break;
                case ComparisonOperation.ElementwiseGreaterThanOrEqual:
                    ElementwiseGreaterThanOrEqual(x, y, destination);
                    break;
                case ComparisonOperation.ElementwiseLessThanOrEqual:
                    ElementwiseLessThanOrEqual(x, y, destination);
                    break;
                case ComparisonOperation.ElementwiseGreaterThan:
                    ElementwiseGreaterThan(x, y, destination);
                    break;
                case ComparisonOperation.ElementwiseLessThan:
                    ElementwiseLessThan(x, y, destination);
                    break;
            }
        }

        public void HandleOperation(ComparisonOperation operation, ReadOnlySpan<T> x, T y, Span<bool> destination)
        {
            switch (operation)
            {
                case ComparisonOperation.ElementwiseEquals:
                    ElementwiseEquals(x, y, destination);
                    break;
                case ComparisonOperation.ElementwiseNotEquals:
                    ElementwiseNotEquals(x, y, destination);
                    break;
                case ComparisonOperation.ElementwiseGreaterThanOrEqual:
                    ElementwiseGreaterThanOrEqual(x, y, destination);
                    break;
                case ComparisonOperation.ElementwiseLessThanOrEqual:
                    ElementwiseLessThanOrEqual(x, y, destination);
                    break;
                case ComparisonOperation.ElementwiseGreaterThan:
                    ElementwiseGreaterThan(x, y, destination);
                    break;
                case ComparisonOperation.ElementwiseLessThan:
                    ElementwiseLessThan(x, y, destination);
                    break;
            }
        }


        //Protected methods

        protected virtual void Add(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void Add(ReadOnlySpan<T> x, T y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void Add(T x, ReadOnlySpan<T> y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void Subtract(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void Subtract(ReadOnlySpan<T> x, T y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void Subtract(T x, ReadOnlySpan<T> y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void Multiply(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void Multiply(ReadOnlySpan<T> x, T y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void Multiply(T x, ReadOnlySpan<T> y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void Divide(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void Divide(ReadOnlySpan<T> x, T y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void Divide(T x, ReadOnlySpan<T> y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void Modulo(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void Modulo(ReadOnlySpan<T> x, T y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void Modulo(T x, ReadOnlySpan<T> y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void And(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void And(ReadOnlySpan<T> x, T y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void And(T x, ReadOnlySpan<T> y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void Or(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void Or(ReadOnlySpan<T> x, T y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void Or(T x, ReadOnlySpan<T> y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void Xor(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void Xor(ReadOnlySpan<T> x, T y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void Xor(T x, ReadOnlySpan<T> y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void LeftShift(ReadOnlySpan<T> x, int y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void RightShift(ReadOnlySpan<T> x, int y, Span<T> destination) => throw new NotSupportedException();

        protected virtual void ElementwiseEquals(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination) => throw new NotSupportedException();

        protected virtual void ElementwiseEquals(ReadOnlySpan<T> x, T y, Span<bool> destination) => throw new NotSupportedException();

        protected virtual void ElementwiseNotEquals(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination) => throw new NotSupportedException();

        protected virtual void ElementwiseNotEquals(ReadOnlySpan<T> x, T y, Span<bool> destination) => throw new NotSupportedException();

        protected virtual void ElementwiseGreaterThanOrEqual(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination) => throw new NotSupportedException();

        protected virtual void ElementwiseGreaterThanOrEqual(ReadOnlySpan<T> x, T y, Span<bool> destination) => throw new NotSupportedException();

        protected virtual void ElementwiseLessThanOrEqual(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination) => throw new NotSupportedException();

        protected virtual void ElementwiseLessThanOrEqual(ReadOnlySpan<T> x, T y, Span<bool> destination) => throw new NotSupportedException();

        protected virtual void ElementwiseGreaterThan(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination) => throw new NotSupportedException();

        protected virtual void ElementwiseGreaterThan(ReadOnlySpan<T> x, T y, Span<bool> destination) => throw new NotSupportedException();

        protected virtual void ElementwiseLessThan(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination) => throw new NotSupportedException();

        protected virtual void ElementwiseLessThan(ReadOnlySpan<T> x, T y, Span<bool> destination) => throw new NotSupportedException();

        protected virtual T Divide(T x, T y) => throw new NotSupportedException();

        protected virtual T Modulo(T x, T y) => throw new NotSupportedException();
    }
}
