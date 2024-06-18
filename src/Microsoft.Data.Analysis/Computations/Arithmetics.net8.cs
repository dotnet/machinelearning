// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.


#if NET8_0_OR_GREATER

using System;
using System.Numerics;
using System.Numerics.Tensors;

namespace Microsoft.Data.Analysis
{

    internal interface IBitwiseOperations<T>
        where T : unmanaged
    {
        void And(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination);
        void And(ReadOnlySpan<T> x, T y, Span<T> destination);
        void And(T x, ReadOnlySpan<T> y, Span<T> destination);
        void Or(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination);
        void Or(ReadOnlySpan<T> x, T y, Span<T> destination);
        void Or(T x, ReadOnlySpan<T> y, Span<T> destination);
        void Xor(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination);
        void Xor(ReadOnlySpan<T> x, T y, Span<T> destination);
        void Xor(T x, ReadOnlySpan<T> y, Span<T> destination);
    }

    internal interface IShiftOperations<T>
        where T : unmanaged
    {
        void LeftShift(ReadOnlySpan<T> x, int shiftAmount, Span<T> destination);
        void RightShift(ReadOnlySpan<T> x, int shiftAmount, Span<T> destination);
    }

    internal interface INumericOperations<T>
        where T : unmanaged
    {
        void Add(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination);
        void Add(ReadOnlySpan<T> x, T y, Span<T> destination);
        void Add(T x, ReadOnlySpan<T> y, Span<T> destination);
        void Subtract(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination);
        void Subtract(ReadOnlySpan<T> x, T y, Span<T> destination);
        void Subtract(T x, ReadOnlySpan<T> y, Span<T> destination);
        void Multiply(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination);
        void Multiply(ReadOnlySpan<T> x, T y, Span<T> destination);
        void Multiply(T x, ReadOnlySpan<T> y, Span<T> destination);
        void Divide(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination);
        void Divide(ReadOnlySpan<T> x, T y, Span<T> destination);
        void Divide(T x, ReadOnlySpan<T> y, Span<T> destination);
        T Divide(T x, T y);
        void Modulo(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination);
        void Modulo(ReadOnlySpan<T> x, T y, Span<T> destination);
        void Modulo(T x, ReadOnlySpan<T> y, Span<T> destination);
        T Modulo(T x, T y);

        void ElementwiseEquals(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination);
        void ElementwiseEquals(ReadOnlySpan<T> x, T y, Span<bool> destination);
        void ElementwiseNotEquals(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination);
        void ElementwiseNotEquals(ReadOnlySpan<T> x, T y, Span<bool> destination);
        void ElementwiseGreaterThanOrEqual(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination);
        void ElementwiseGreaterThanOrEqual(ReadOnlySpan<T> x, T y, Span<bool> destination);
        void ElementwiseLessThanOrEqual(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination);
        void ElementwiseLessThanOrEqual(ReadOnlySpan<T> x, T y, Span<bool> destination);
        void ElementwiseGreaterThan(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination);
        void ElementwiseGreaterThan(ReadOnlySpan<T> x, T y, Span<bool> destination);
        void ElementwiseLessThan(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination);
        void ElementwiseLessThan(ReadOnlySpan<T> x, T y, Span<bool> destination);
    }

    internal class BitwiseOperations<T> : IBitwiseOperations<T>
        where T : unmanaged, IBitwiseOperators<T, T, T>
    {
        public void And(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.BitwiseAnd(x, y, destination);
        public void And(ReadOnlySpan<T> x, T y, Span<T> destination) => TensorPrimitives.BitwiseAnd(x, y, destination);
        public void And(T x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.BitwiseAnd(y, x, destination);
        public void Or(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.BitwiseOr(x, y, destination);
        public void Or(ReadOnlySpan<T> x, T y, Span<T> destination) => TensorPrimitives.BitwiseOr(x, y, destination);
        public void Or(T x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.BitwiseOr(y, x, destination);
        public void Xor(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Xor(x, y, destination);
        public void Xor(ReadOnlySpan<T> x, T y, Span<T> destination) => TensorPrimitives.Xor(x, y, destination);
        public void Xor(T x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Xor(y, x, destination);
    }

    internal class ShiftOperations<T> : IShiftOperations<T>
        where T : unmanaged, IShiftOperators<T, int, T>
    {
        public void LeftShift(ReadOnlySpan<T> x, int shiftAmount, Span<T> destination) => TensorPrimitives.ShiftLeft(x, shiftAmount, destination);
        public void RightShift(ReadOnlySpan<T> x, int shiftAmount, Span<T> destination) => TensorPrimitives.ShiftRightArithmetic(x, shiftAmount, destination);
    }

    internal class NumericOperations<T> : INumericOperations<T>
        where T : unmanaged, INumber<T>
    {
        public void Add(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Add(x, y, destination);
        public void Add(ReadOnlySpan<T> x, T y, Span<T> destination) => TensorPrimitives.Add(x, y, destination);
        public void Add(T x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Add(y, x, destination);
        public void Subtract(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Subtract(x, y, destination);
        public void Subtract(ReadOnlySpan<T> x, T y, Span<T> destination) => TensorPrimitives.Subtract(x, y, destination);
        public void Subtract(T x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Subtract(x, y, destination);
        public void Multiply(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Multiply(x, y, destination);
        public void Multiply(ReadOnlySpan<T> x, T y, Span<T> destination) => TensorPrimitives.Multiply(x, y, destination);
        public void Multiply(T x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Multiply(y, x, destination);
        public void Divide(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Divide(x, y, destination);
        public void Divide(ReadOnlySpan<T> x, T y, Span<T> destination) => TensorPrimitives.Divide(x, y, destination);
        public void Divide(T x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Divide(x, y, destination);
        public T Divide(T x, T y) => x / y;

        public void Modulo(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = x[i] % y[i];
            }
        }

        public void Modulo(ReadOnlySpan<T> x, T y, Span<T> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = x[i] % y;
            }
        }

        public void Modulo(T x, ReadOnlySpan<T> y, Span<T> destination)
        {
            for (var i = 0; i < y.Length; i++)
            {
                destination[i] = x % y[i];
            }
        }

        public T Modulo(T x, T y) => x % y;

        public void ElementwiseEquals(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = (x[i] == y[i]);
            }
        }

        public void ElementwiseEquals(ReadOnlySpan<T> x, T y, Span<bool> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = (x[i] == y);
            }
        }

        public void ElementwiseNotEquals(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = (x[i] != y[i]);
            }
        }

        public void ElementwiseNotEquals(ReadOnlySpan<T> x, T y, Span<bool> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = (x[i] != y);
            }
        }

        public void ElementwiseGreaterThanOrEqual(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = (x[i] >= y[i]);
            }
        }

        public void ElementwiseGreaterThanOrEqual(ReadOnlySpan<T> x, T y, Span<bool> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = (x[i] >= y);
            }
        }

        public void ElementwiseLessThanOrEqual(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = (x[i] <= y[i]);
            }
        }

        public void ElementwiseLessThanOrEqual(ReadOnlySpan<T> x, T y, Span<bool> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = (x[i] <= y);
            }
        }

        public void ElementwiseGreaterThan(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = (x[i] > y[i]);
            }
        }

        public void ElementwiseGreaterThan(ReadOnlySpan<T> x, T y, Span<bool> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = (x[i] > y);
            }
        }

        public void ElementwiseLessThan(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = (x[i] < y[i]);
            }
        }

        public void ElementwiseLessThan(ReadOnlySpan<T> x, T y, Span<bool> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = (x[i] < y);
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
}
#endif
