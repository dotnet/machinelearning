
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#if NET8_0_OR_GREATER

using System;
using System.Numerics;
using System.Numerics.Tensors;

namespace Microsoft.Data.Analysis
{
    internal static partial class Arithmetic
    {
        #region Nested classes for Operations
        private interface IBitwiseOperations<T>
            where T : unmanaged
        {
            static abstract void And(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination);
            static abstract void And(ReadOnlySpan<T> x, T y, Span<T> destination);
            static abstract void And(T x, ReadOnlySpan<T> y, Span<T> destination);
            static abstract void Or(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination);
            static abstract void Or(ReadOnlySpan<T> x, T y, Span<T> destination);
            static abstract void Or(T x, ReadOnlySpan<T> y, Span<T> destination);
            static abstract void Xor(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination);
            static abstract void Xor(ReadOnlySpan<T> x, T y, Span<T> destination);
            static abstract void Xor(T x, ReadOnlySpan<T> y, Span<T> destination);
        }

        private interface IShiftOperations<T>
            where T : unmanaged
        {
            static abstract void LeftShift(ReadOnlySpan<T> x, int shiftAmount, Span<T> destination);
            static abstract void RightShift(ReadOnlySpan<T> x, int shiftAmount, Span<T> destination);
        }

        private interface INumericOperations<T>
            where T : unmanaged
        {
            static abstract void Add(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination);
            static abstract void Add(ReadOnlySpan<T> x, T y, Span<T> destination);
            static abstract void Add(T x, ReadOnlySpan<T> y, Span<T> destination);
            static abstract void Subtract(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination);
            static abstract void Subtract(ReadOnlySpan<T> x, T y, Span<T> destination);
            static abstract void Subtract(T x, ReadOnlySpan<T> y, Span<T> destination);
            static abstract void Multiply(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination);
            static abstract void Multiply(ReadOnlySpan<T> x, T y, Span<T> destination);
            static abstract void Multiply(T x, ReadOnlySpan<T> y, Span<T> destination);
            static abstract void Divide(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination);
            static abstract void Divide(ReadOnlySpan<T> x, T y, Span<T> destination);
            static abstract void Divide(T x, ReadOnlySpan<T> y, Span<T> destination);
            static abstract T Divide(T x, T y);
            static abstract void Modulo(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination);
            static abstract void Modulo(ReadOnlySpan<T> x, T y, Span<T> destination);
            static abstract void Modulo(T x, ReadOnlySpan<T> y, Span<T> destination);
            static abstract T Modulo(T x, T y);

            static abstract void ElementwiseEquals(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination);
            static abstract void ElementwiseEquals(ReadOnlySpan<T> x, T y, Span<bool> destination);
            static abstract void ElementwiseNotEquals(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination);
            static abstract void ElementwiseNotEquals(ReadOnlySpan<T> x, T y, Span<bool> destination);
            static abstract void ElementwiseGreaterThanOrEqual(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination);
            static abstract void ElementwiseGreaterThanOrEqual(ReadOnlySpan<T> x, T y, Span<bool> destination);
            static abstract void ElementwiseLessThanOrEqual(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination);
            static abstract void ElementwiseLessThanOrEqual(ReadOnlySpan<T> x, T y, Span<bool> destination);
            static abstract void ElementwiseGreaterThan(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination);
            static abstract void ElementwiseGreaterThan(ReadOnlySpan<T> x, T y, Span<bool> destination);
            static abstract void ElementwiseLessThan(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination);
            static abstract void ElementwiseLessThan(ReadOnlySpan<T> x, T y, Span<bool> destination);
        }

        private readonly struct BitwiseOperations<T> : IBitwiseOperations<T>
        where T : unmanaged, IBitwiseOperators<T, T, T>
        {
            public static void And(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.BitwiseAnd(x, y, destination);
            public static void And(ReadOnlySpan<T> x, T y, Span<T> destination) => TensorPrimitives.BitwiseAnd(x, y, destination);
            public static void And(T x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.BitwiseAnd(y, x, destination);
            public static void Or(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.BitwiseOr(x, y, destination);
            public static void Or(ReadOnlySpan<T> x, T y, Span<T> destination) => TensorPrimitives.BitwiseOr(x, y, destination);
            public static void Or(T x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.BitwiseOr(y, x, destination);
            public static void Xor(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Xor(x, y, destination);
            public static void Xor(ReadOnlySpan<T> x, T y, Span<T> destination) => TensorPrimitives.Xor(x, y, destination);
            public static void Xor(T x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Xor(y, x, destination);
        }

        private readonly struct ShiftOperations<T> : IShiftOperations<T>
            where T : unmanaged, IShiftOperators<T, int, T>
        {
            public static void LeftShift(ReadOnlySpan<T> x, int shiftAmount, Span<T> destination) => TensorPrimitives.ShiftLeft(x, shiftAmount, destination);
            public static void RightShift(ReadOnlySpan<T> x, int shiftAmount, Span<T> destination) => TensorPrimitives.ShiftRightArithmetic(x, shiftAmount, destination);
        }

        private readonly struct NumericOperations<T> : INumericOperations<T>
            where T : unmanaged, INumber<T>
        {
            public static void Add(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Add(x, y, destination);
            public static void Add(ReadOnlySpan<T> x, T y, Span<T> destination) => TensorPrimitives.Add(x, y, destination);
            public static void Add(T x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Add(y, x, destination);
            public static void Subtract(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Subtract(x, y, destination);
            public static void Subtract(ReadOnlySpan<T> x, T y, Span<T> destination) => TensorPrimitives.Subtract(x, y, destination);
            public static void Subtract(T x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Subtract(x, y, destination);
            public static void Multiply(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Multiply(x, y, destination);
            public static void Multiply(ReadOnlySpan<T> x, T y, Span<T> destination) => TensorPrimitives.Multiply(x, y, destination);
            public static void Multiply(T x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Multiply(y, x, destination);
            public static void Divide(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Divide(x, y, destination);
            public static void Divide(ReadOnlySpan<T> x, T y, Span<T> destination) => TensorPrimitives.Divide(x, y, destination);
            public static void Divide(T x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Divide(x, y, destination);
            public static T Divide(T x, T y) => x / y;

            public static void Modulo(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = x[i] % y[i];
                }
            }

            public static void Modulo(ReadOnlySpan<T> x, T y, Span<T> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = x[i] % y;
                }
            }

            public static void Modulo(T x, ReadOnlySpan<T> y, Span<T> destination)
            {
                for (var i = 0; i < y.Length; i++)
                {
                    destination[i] = x % y[i];
                }
            }

            public static T Modulo(T x, T y) => x % y;

            public static void ElementwiseEquals(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y[i]);
                }
            }

            public static void ElementwiseEquals(ReadOnlySpan<T> x, T y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y);
                }
            }

            public static void ElementwiseNotEquals(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y[i]);
                }
            }

            public static void ElementwiseNotEquals(ReadOnlySpan<T> x, T y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y);
                }
            }

            public static void ElementwiseGreaterThanOrEqual(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y[i]);
                }
            }

            public static void ElementwiseGreaterThanOrEqual(ReadOnlySpan<T> x, T y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y);
                }
            }

            public static void ElementwiseLessThanOrEqual(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y[i]);
                }
            }

            public static void ElementwiseLessThanOrEqual(ReadOnlySpan<T> x, T y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y);
                }
            }

            public static void ElementwiseGreaterThan(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y[i]);
                }
            }

            public static void ElementwiseGreaterThan(ReadOnlySpan<T> x, T y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y);
                }
            }

            public static void ElementwiseLessThan(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y[i]);
                }
            }

            public static void ElementwiseLessThan(ReadOnlySpan<T> x, T y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y);
                }
            }
        }
        #endregion

        #region Nested classes for Arithmetics
        private class CompositeArithmetic<T, TNumericOperations> : IArithmetic<T>
          where T : unmanaged
          where TNumericOperations : struct, INumericOperations<T>
        {
            public virtual void HandleOperation(BinaryOperation operation, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
            {
                switch (operation)
                {
                    case BinaryOperation.Add:
                        TNumericOperations.Add(x, y, destination);
                        break;
                    case BinaryOperation.Subtract:
                        TNumericOperations.Subtract(x, y, destination);
                        break;
                    case BinaryOperation.Multiply:
                        TNumericOperations.Multiply(x, y, destination);
                        break;
                    case BinaryOperation.Divide:
                        TNumericOperations.Divide(x, y, destination);
                        break;
                    case BinaryOperation.Modulo:
                        TNumericOperations.Modulo(x, y, destination);
                        break;
                    default:
                        throw new NotSupportedException();
                }
            }

            public virtual void HandleOperation(BinaryOperation operation, ReadOnlySpan<T> x, T y, Span<T> destination)
            {
                switch (operation)
                {
                    case BinaryOperation.Add:
                        TNumericOperations.Add(x, y, destination);
                        break;
                    case BinaryOperation.Subtract:
                        TNumericOperations.Subtract(x, y, destination);
                        break;
                    case BinaryOperation.Multiply:
                        TNumericOperations.Multiply(x, y, destination);
                        break;
                    case BinaryOperation.Divide:
                        TNumericOperations.Divide(x, y, destination);
                        break;
                    case BinaryOperation.Modulo:
                        TNumericOperations.Modulo(x, y, destination);
                        break;
                    default:
                        throw new NotSupportedException();
                }
            }

            public virtual void HandleOperation(BinaryOperation operation, T x, ReadOnlySpan<T> y, Span<T> destination)
            {
                switch (operation)
                {
                    case BinaryOperation.Add:
                        TNumericOperations.Add(x, y, destination);
                        break;
                    case BinaryOperation.Subtract:
                        TNumericOperations.Subtract(x, y, destination);
                        break;
                    case BinaryOperation.Multiply:
                        TNumericOperations.Multiply(x, y, destination);
                        break;
                    case BinaryOperation.Divide:
                        TNumericOperations.Divide(x, y, destination);
                        break;
                    case BinaryOperation.Modulo:
                        TNumericOperations.Modulo(x, y, destination);
                        break;
                    default:
                        throw new NotSupportedException();
                }
            }

            public T HandleOperation(BinaryOperation operation, T x, T y)
            {
                if (operation == BinaryOperation.Divide)
                    return TNumericOperations.Divide(x, y);

                if (operation == BinaryOperation.Modulo)
                    return TNumericOperations.Modulo(x, y);

                throw new NotSupportedException();
            }

            public virtual void HandleOperation(BinaryIntOperation operation, ReadOnlySpan<T> x, int y, Span<T> destination)
            {
                throw new NotSupportedException();
            }

            public void HandleOperation(ComparisonOperation operation, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
            {
                switch (operation)
                {
                    case ComparisonOperation.ElementwiseEquals:
                        TNumericOperations.ElementwiseEquals(x, y, destination);
                        break;
                    case ComparisonOperation.ElementwiseNotEquals:
                        TNumericOperations.ElementwiseNotEquals(x, y, destination);
                        break;
                    case ComparisonOperation.ElementwiseGreaterThanOrEqual:
                        TNumericOperations.ElementwiseGreaterThanOrEqual(x, y, destination);
                        break;
                    case ComparisonOperation.ElementwiseLessThanOrEqual:
                        TNumericOperations.ElementwiseLessThanOrEqual(x, y, destination);
                        break;
                    case ComparisonOperation.ElementwiseGreaterThan:
                        TNumericOperations.ElementwiseGreaterThan(x, y, destination);
                        break;
                    case ComparisonOperation.ElementwiseLessThan:
                        TNumericOperations.ElementwiseLessThan(x, y, destination);
                        break;
                    default:
                        throw new NotSupportedException();
                }
            }

            public void HandleOperation(ComparisonOperation operation, ReadOnlySpan<T> x, T y, Span<bool> destination)
            {
                switch (operation)
                {
                    case ComparisonOperation.ElementwiseEquals:
                        TNumericOperations.ElementwiseEquals(x, y, destination);
                        break;
                    case ComparisonOperation.ElementwiseNotEquals:
                        TNumericOperations.ElementwiseNotEquals(x, y, destination);
                        break;
                    case ComparisonOperation.ElementwiseGreaterThanOrEqual:
                        TNumericOperations.ElementwiseGreaterThanOrEqual(x, y, destination);
                        break;
                    case ComparisonOperation.ElementwiseLessThanOrEqual:
                        TNumericOperations.ElementwiseLessThanOrEqual(x, y, destination);
                        break;
                    case ComparisonOperation.ElementwiseGreaterThan:
                        TNumericOperations.ElementwiseGreaterThan(x, y, destination);
                        break;
                    case ComparisonOperation.ElementwiseLessThan:
                        TNumericOperations.ElementwiseLessThan(x, y, destination);
                        break;
                    default:
                        throw new NotSupportedException();
                }
            }
        }

        private class CompositeArithmetic<T, TNumericOperations, TBitwiseOperations> : CompositeArithmetic<T, TNumericOperations>, IArithmetic<T>
            where T : unmanaged
            where TNumericOperations : struct, INumericOperations<T>
            where TBitwiseOperations : struct, IBitwiseOperations<T>
        {
            public override void HandleOperation(BinaryOperation operation, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
            {
                switch (operation)
                {
                    case BinaryOperation.And:
                        TBitwiseOperations.And(x, y, destination);
                        return;
                    case BinaryOperation.Or:
                        TBitwiseOperations.Or(x, y, destination);
                        return;
                    case BinaryOperation.Xor:
                        TBitwiseOperations.Xor(x, y, destination);
                        return;
                }

                base.HandleOperation(operation, x, y, destination);
            }

            public override void HandleOperation(BinaryOperation operation, ReadOnlySpan<T> x, T y, Span<T> destination)
            {
                switch (operation)
                {
                    case BinaryOperation.And:
                        TBitwiseOperations.And(x, y, destination);
                        return;
                    case BinaryOperation.Or:
                        TBitwiseOperations.Or(x, y, destination);
                        return;
                    case BinaryOperation.Xor:
                        TBitwiseOperations.Xor(x, y, destination);
                        return;
                }

                base.HandleOperation(operation, x, y, destination);
            }

            public override void HandleOperation(BinaryOperation operation, T x, ReadOnlySpan<T> y, Span<T> destination)
            {
                switch (operation)
                {
                    case BinaryOperation.And:
                        TBitwiseOperations.And(x, y, destination);
                        return;
                    case BinaryOperation.Or:
                        TBitwiseOperations.Or(x, y, destination);
                        return;
                    case BinaryOperation.Xor:
                        TBitwiseOperations.Xor(x, y, destination);
                        return;
                }

                base.HandleOperation(operation, x, y, destination);
            }
        }

        private class CompositeArithmetic<T, TNumericOperations, TBitwiseOperations, TShiftOperations> : CompositeArithmetic<T, TNumericOperations, TBitwiseOperations>, IArithmetic<T>
            where T : unmanaged
            where TNumericOperations : struct, INumericOperations<T>
            where TBitwiseOperations : struct, IBitwiseOperations<T>
            where TShiftOperations : struct, IShiftOperations<T>
        {

            public override void HandleOperation(BinaryIntOperation operation, ReadOnlySpan<T> x, int y, Span<T> destination)
            {
                switch (operation)
                {
                    case BinaryIntOperation.LeftShift:
                        TShiftOperations.LeftShift(x, y, destination);
                        break;
                    case BinaryIntOperation.RightShift:
                        TShiftOperations.RightShift(x, y, destination);
                        break;
                    default:
                        throw new NotSupportedException();
                }
            }
        }

        //Special case
        internal class DateTimeArithmetic : Arithmetic<DateTime>
        {
            protected override void ElementwiseEquals(ReadOnlySpan<DateTime> x, ReadOnlySpan<DateTime> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y[i]);
                }
            }

            protected override void ElementwiseEquals(ReadOnlySpan<DateTime> x, DateTime y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<DateTime> x, ReadOnlySpan<DateTime> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y[i]);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<DateTime> x, DateTime y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<DateTime> x, ReadOnlySpan<DateTime> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y[i]);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<DateTime> x, DateTime y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<DateTime> x, ReadOnlySpan<DateTime> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y[i]);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<DateTime> x, DateTime y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<DateTime> x, ReadOnlySpan<DateTime> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y[i]);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<DateTime> x, DateTime y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<DateTime> x, ReadOnlySpan<DateTime> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y[i]);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<DateTime> x, DateTime y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y);
                }
            }
        }

        internal class BoolArithmetic : Arithmetic<bool>
        {
            protected override void And(ReadOnlySpan<bool> x, ReadOnlySpan<bool> y, Span<bool> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (bool)(x[i] & y[i]);
                    i++;
                }
            }

            protected override void And(ReadOnlySpan<bool> x, bool y, Span<bool> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (bool)(x[i] & y);
                    i++;
                }
            }

            protected override void And(bool x, ReadOnlySpan<bool> y, Span<bool> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (bool)(x & y[i]);
                    i++;
                }
            }

            protected override void Or(ReadOnlySpan<bool> x, ReadOnlySpan<bool> y, Span<bool> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (bool)(x[i] | y[i]);
                    i++;
                }
            }

            protected override void Or(ReadOnlySpan<bool> x, bool y, Span<bool> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (bool)(x[i] | y);
                    i++;
                }
            }

            protected override void Or(bool x, ReadOnlySpan<bool> y, Span<bool> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (bool)(x | y[i]);
                    i++;
                }
            }

            protected override void Xor(ReadOnlySpan<bool> x, ReadOnlySpan<bool> y, Span<bool> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (bool)(x[i] ^ y[i]);
                    i++;
                }
            }

            protected override void Xor(ReadOnlySpan<bool> x, bool y, Span<bool> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (bool)(x[i] ^ y);
                    i++;
                }
            }

            protected override void Xor(bool x, ReadOnlySpan<bool> y, Span<bool> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (bool)(x ^ y[i]);
                    i++;
                }
            }

            protected override void ElementwiseEquals(ReadOnlySpan<bool> x, ReadOnlySpan<bool> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y[i]);
                }
            }

            protected override void ElementwiseEquals(ReadOnlySpan<bool> x, bool y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<bool> x, ReadOnlySpan<bool> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y[i]);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<bool> x, bool y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y);
                }
            }
        }
        #endregion

        internal static IArithmetic<T> GetArithmetic<T>()
            where T : unmanaged
        {
            if (typeof(T) == typeof(double))
                return (IArithmetic<T>)new CompositeArithmetic<double, NumericOperations<double>, BitwiseOperations<double>>();
            if (typeof(T) == typeof(float))
                return (IArithmetic<T>)new CompositeArithmetic<float, NumericOperations<float>, BitwiseOperations<float>>();
            if (typeof(T) == typeof(int))
                return (IArithmetic<T>)new CompositeArithmetic<int, NumericOperations<int>, BitwiseOperations<int>, ShiftOperations<int>>();
            if (typeof(T) == typeof(long))
                return (IArithmetic<T>)new CompositeArithmetic<long, NumericOperations<long>, BitwiseOperations<long>, ShiftOperations<long>>();
            if (typeof(T) == typeof(sbyte))
                return (IArithmetic<T>)new CompositeArithmetic<sbyte, NumericOperations<sbyte>, BitwiseOperations<sbyte>, ShiftOperations<sbyte>>();
            if (typeof(T) == typeof(short))
                return (IArithmetic<T>)new CompositeArithmetic<short, NumericOperations<short>, BitwiseOperations<short>, ShiftOperations<short>>();
            if (typeof(T) == typeof(uint))
                return (IArithmetic<T>)new CompositeArithmetic<uint, NumericOperations<uint>, BitwiseOperations<uint>, ShiftOperations<uint>>();
            if (typeof(T) == typeof(ulong))
                return (IArithmetic<T>)new CompositeArithmetic<ulong, NumericOperations<ulong>, BitwiseOperations<ulong>, ShiftOperations<ulong>>();
            if (typeof(T) == typeof(ushort))
                return (IArithmetic<T>)new CompositeArithmetic<ushort, NumericOperations<ushort>, BitwiseOperations<ushort>, ShiftOperations<ushort>>();
            if (typeof(T) == typeof(byte))
                return (IArithmetic<T>)new CompositeArithmetic<byte, NumericOperations<byte>, BitwiseOperations<byte>, ShiftOperations<byte>>();
            if (typeof(T) == typeof(char))
                return (IArithmetic<T>)new CompositeArithmetic<char, NumericOperations<char>, BitwiseOperations<char>, ShiftOperations<char>>();
            if (typeof(T) == typeof(decimal))
                return (IArithmetic<T>)new CompositeArithmetic<decimal, NumericOperations<decimal>>();
            if (typeof(T) == typeof(DateTime))
                return (IArithmetic<T>)new DateTimeArithmetic();
            if (typeof(T) == typeof(bool))
                return (IArithmetic<T>)new BoolArithmetic();
            throw new NotSupportedException();
        }
    }
}
#endif
