
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#if NET8_0_OR_GREATER

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
        #region Nested CompositeAritmetic class
        private class CompositeArithmetic<T>(INumericOperations<T> numericOps, IBitwiseOperations<T> bitwiseOps = null, IShiftOperations<T> shiftOps = null) : IArithmetic<T>
        where T : unmanaged
        {
            private readonly INumericOperations<T> _numericOps = numericOps;
            private readonly IBitwiseOperations<T> _bitwiseOps = bitwiseOps;
            private readonly IShiftOperations<T> _shiftOps = shiftOps;

            //Binary operations

            public void HandleOperation(BinaryOperation operation, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
            {
                if (_numericOps != null)
                {
                    switch (operation)
                    {
                        case BinaryOperation.Add:
                            _numericOps.Add(x, y, destination);
                            return;
                        case BinaryOperation.Subtract:
                            _numericOps.Subtract(x, y, destination);
                            return;
                        case BinaryOperation.Multiply:
                            _numericOps.Multiply(x, y, destination);
                            return;
                        case BinaryOperation.Divide:
                            _numericOps.Divide(x, y, destination);
                            return;
                        case BinaryOperation.Modulo:
                            _numericOps.Modulo(x, y, destination);
                            return;
                    }
                }

                if (_bitwiseOps != null)
                {
                    switch (operation)
                    {
                        case BinaryOperation.And:
                            _bitwiseOps.And(x, y, destination);
                            return;
                        case BinaryOperation.Or:
                            _bitwiseOps.Or(x, y, destination);
                            return;
                        case BinaryOperation.Xor:
                            _bitwiseOps.Xor(x, y, destination);
                            return;
                    }
                }

                throw new NotSupportedException();
            }

            public void HandleOperation(BinaryOperation operation, ReadOnlySpan<T> x, T y, Span<T> destination)
            {
                if (_numericOps != null)
                {
                    switch (operation)
                    {
                        case BinaryOperation.Add:
                            _numericOps.Add(x, y, destination);
                            return;
                        case BinaryOperation.Subtract:
                            _numericOps.Subtract(x, y, destination);
                            return;
                        case BinaryOperation.Multiply:
                            _numericOps.Multiply(x, y, destination);
                            return;
                        case BinaryOperation.Divide:
                            _numericOps.Divide(x, y, destination);
                            return;
                        case BinaryOperation.Modulo:
                            _numericOps.Modulo(x, y, destination);
                            return;
                    }
                }

                if (_bitwiseOps != null)
                {
                    switch (operation)
                    {
                        case BinaryOperation.And:
                            _bitwiseOps.And(x, y, destination);
                            return;
                        case BinaryOperation.Or:
                            _bitwiseOps.Or(x, y, destination);
                            return;
                        case BinaryOperation.Xor:
                            _bitwiseOps.Xor(x, y, destination);
                            return;
                    }
                }

                throw new NotSupportedException();
            }

            public void HandleOperation(BinaryOperation operation, T x, ReadOnlySpan<T> y, Span<T> destination)
            {
                if (_numericOps != null)
                {
                    switch (operation)
                    {
                        case BinaryOperation.Add:
                            _numericOps.Add(x, y, destination);
                            return;
                        case BinaryOperation.Subtract:
                            _numericOps.Subtract(x, y, destination);
                            return;
                        case BinaryOperation.Multiply:
                            _numericOps.Multiply(x, y, destination);
                            return;
                        case BinaryOperation.Divide:
                            _numericOps.Divide(x, y, destination);
                            return;
                        case BinaryOperation.Modulo:
                            _numericOps.Modulo(x, y, destination);
                            return;
                    }
                }

                if (_bitwiseOps != null)
                {
                    switch (operation)
                    {
                        case BinaryOperation.And:
                            _bitwiseOps.And(x, y, destination);
                            return;
                        case BinaryOperation.Or:
                            _bitwiseOps.Or(x, y, destination);
                            return;
                        case BinaryOperation.Xor:
                            _bitwiseOps.Xor(x, y, destination);
                            return;
                    }
                }

                throw new NotSupportedException();
            }

            public T HandleOperation(BinaryOperation operation, T x, T y)
            {
                if (_numericOps == null)
                    throw new NotSupportedException();

                if (operation == BinaryOperation.Divide)
                    return _numericOps.Divide(x, y);

                if (operation == BinaryOperation.Modulo)
                    return _numericOps.Modulo(x, y);

                throw new NotSupportedException();
            }


            //Binary Int operations

            public void HandleOperation(BinaryIntOperation operation, ReadOnlySpan<T> x, int y, Span<T> destination)
            {
                if (_shiftOps == null)
                    throw new NotSupportedException();

                switch (operation)
                {
                    case BinaryIntOperation.LeftShift:
                        _shiftOps.LeftShift(x, y, destination);
                        break;
                    case BinaryIntOperation.RightShift:
                        _shiftOps.RightShift(x, y, destination);
                        break;
                }
            }

            //Comparison operations

            public void HandleOperation(ComparisonOperation operation, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
            {
                if (_numericOps == null)
                    throw new NotSupportedException();

                switch (operation)
                {
                    case ComparisonOperation.ElementwiseEquals:
                        _numericOps.ElementwiseEquals(x, y, destination);
                        break;
                    case ComparisonOperation.ElementwiseNotEquals:
                        _numericOps.ElementwiseNotEquals(x, y, destination);
                        break;
                    case ComparisonOperation.ElementwiseGreaterThanOrEqual:
                        _numericOps.ElementwiseGreaterThanOrEqual(x, y, destination);
                        break;
                    case ComparisonOperation.ElementwiseLessThanOrEqual:
                        _numericOps.ElementwiseLessThanOrEqual(x, y, destination);
                        break;
                    case ComparisonOperation.ElementwiseGreaterThan:
                        _numericOps.ElementwiseGreaterThan(x, y, destination);
                        break;
                    case ComparisonOperation.ElementwiseLessThan:
                        _numericOps.ElementwiseLessThan(x, y, destination);
                        break;
                }
            }

            public void HandleOperation(ComparisonOperation operation, ReadOnlySpan<T> x, T y, Span<bool> destination)
            {
                if (_numericOps == null)
                    throw new NotSupportedException();

                switch (operation)
                {
                    case ComparisonOperation.ElementwiseEquals:
                        _numericOps.ElementwiseEquals(x, y, destination);
                        break;
                    case ComparisonOperation.ElementwiseNotEquals:
                        _numericOps.ElementwiseNotEquals(x, y, destination);
                        break;
                    case ComparisonOperation.ElementwiseGreaterThanOrEqual:
                        _numericOps.ElementwiseGreaterThanOrEqual(x, y, destination);
                        break;
                    case ComparisonOperation.ElementwiseLessThanOrEqual:
                        _numericOps.ElementwiseLessThanOrEqual(x, y, destination);
                        break;
                    case ComparisonOperation.ElementwiseGreaterThan:
                        _numericOps.ElementwiseGreaterThan(x, y, destination);
                        break;
                    case ComparisonOperation.ElementwiseLessThan:
                        _numericOps.ElementwiseLessThan(x, y, destination);
                        break;
                }
            }
        }
        #endregion

        public static IArithmetic<T> GetArithmetic<T>()
            where T : unmanaged
        {

            if (typeof(T) == typeof(double))
                return (IArithmetic<T>)new CompositeArithmetic<double>(new NumericOperations<double>(), new BitwiseOperations<double>());
            if (typeof(T) == typeof(float))
                return (IArithmetic<T>)new CompositeArithmetic<float>(new NumericOperations<float>(), new BitwiseOperations<float>());
            if (typeof(T) == typeof(int))
                return (IArithmetic<T>)new CompositeArithmetic<int>(new NumericOperations<int>(), new BitwiseOperations<int>(), new ShiftOperations<int>());
            if (typeof(T) == typeof(long))
                return (IArithmetic<T>)new CompositeArithmetic<long>(new NumericOperations<long>(), new BitwiseOperations<long>(), new ShiftOperations<long>());
            if (typeof(T) == typeof(sbyte))
                return (IArithmetic<T>)new CompositeArithmetic<sbyte>(new NumericOperations<sbyte>(), new BitwiseOperations<sbyte>(), new ShiftOperations<sbyte>());
            if (typeof(T) == typeof(short))
                return (IArithmetic<T>)new CompositeArithmetic<short>(new NumericOperations<short>(), new BitwiseOperations<short>(), new ShiftOperations<short>());
            if (typeof(T) == typeof(uint))
                return (IArithmetic<T>)new CompositeArithmetic<uint>(new NumericOperations<uint>(), new BitwiseOperations<uint>(), new ShiftOperations<uint>());
            if (typeof(T) == typeof(ulong))
                return (IArithmetic<T>)new CompositeArithmetic<ulong>(new NumericOperations<ulong>(), new BitwiseOperations<ulong>(), new ShiftOperations<ulong>());
            if (typeof(T) == typeof(ushort))
                return (IArithmetic<T>)new CompositeArithmetic<ushort>(new NumericOperations<ushort>(), new BitwiseOperations<ushort>(), new ShiftOperations<ushort>());
            if (typeof(T) == typeof(byte))
                return (IArithmetic<T>)new CompositeArithmetic<byte>(new NumericOperations<byte>(), new BitwiseOperations<byte>(), new ShiftOperations<byte>());
            if (typeof(T) == typeof(char))
                return (IArithmetic<T>)new CompositeArithmetic<char>(new NumericOperations<char>(), new BitwiseOperations<char>(), new ShiftOperations<char>());
            if (typeof(T) == typeof(decimal))
                return (IArithmetic<T>)new CompositeArithmetic<decimal>(new NumericOperations<decimal>());
            if (typeof(T) == typeof(DateTime))
                return (IArithmetic<T>)new DateTimeArithmetic();
            if (typeof(T) == typeof(bool))
                return (IArithmetic<T>)new BoolArithmetic();
            throw new NotSupportedException();
        }
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
#endif
