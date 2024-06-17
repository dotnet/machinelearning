// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.


#if NET8_0_OR_GREATER

using System;
using System.Numerics;
using System.Numerics.Tensors;

namespace Microsoft.Data.Analysis
{

    internal class NumericArithmetic<T> : Arithmetic<T>
        where T : unmanaged, INumber<T>, IBitwiseOperators<T, T, T> //, IShiftOperators<T, int, T>
    {
        protected override void LeftShift(ReadOnlySpan<T> x, int y, Span<T> destination) => throw new NotSupportedException();

        protected override void RightShift(ReadOnlySpan<T> x, int y, Span<T> destination) => throw new NotSupportedException();

        protected override void Add(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Add(x, y, destination);

        protected override void Add(ReadOnlySpan<T> x, T y, Span<T> destination) => TensorPrimitives.Add(x, y, destination);

        protected override void Add(T x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Add(y, x, destination);

        protected override void Subtract(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Subtract(x, y, destination);

        protected override void Subtract(ReadOnlySpan<T> x, T y, Span<T> destination) => TensorPrimitives.Subtract(x, y, destination);

        protected override void Subtract(T x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Subtract(x, y, destination);

        protected override void Multiply(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Multiply(x, y, destination);

        protected override void Multiply(ReadOnlySpan<T> x, T y, Span<T> destination) => TensorPrimitives.Multiply(x, y, destination);

        protected override void Multiply(T x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Multiply(y, x, destination);

        protected override void Divide(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Divide(x, y, destination);

        protected override void Divide(ReadOnlySpan<T> x, T y, Span<T> destination) => TensorPrimitives.Divide(x, y, destination);

        protected override void Divide(T x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Divide(x, y, destination);

        protected override T Divide(T x, T y) => x % y;

        protected override void Modulo(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
        {
            int i = 0;

            while (i < x.Length)
            {
                destination[i] = x[i] % y[i];
                i++;
            }
        }

        protected override void Modulo(ReadOnlySpan<T> x, T y, Span<T> destination)
        {
            int i = 0;

            while (i < x.Length)
            {
                destination[i] = x[i] % y;
                i++;
            }
        }

        protected override void Modulo(T x, ReadOnlySpan<T> y, Span<T> destination)
        {
            int i = 0;

            while (i < y.Length)
            {
                destination[i] = x % y[i];
                i++;
            }
        }

        protected override T Modulo(T x, T y) => x % y;

        protected override void And(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.BitwiseAnd(x, y, destination);

        protected override void And(ReadOnlySpan<T> x, T y, Span<T> destination) => TensorPrimitives.BitwiseAnd(x, y, destination);

        protected override void And(T x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.BitwiseAnd(y, x, destination);

        protected override void Or(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.BitwiseOr(x, y, destination);

        protected override void Or(ReadOnlySpan<T> x, T y, Span<T> destination) => TensorPrimitives.BitwiseOr(x, y, destination);

        protected override void Or(T x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.BitwiseOr(y, x, destination);

        protected override void Xor(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Xor(x, y, destination);

        protected override void Xor(ReadOnlySpan<T> x, T y, Span<T> destination) => TensorPrimitives.Xor(x, y, destination);

        protected override void Xor(T x, ReadOnlySpan<T> y, Span<T> destination) => TensorPrimitives.Xor(y, x, destination);

        protected override void ElementwiseEquals(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = (x[i] == y[i]);
            }
        }

        protected override void ElementwiseEquals(ReadOnlySpan<T> x, T y, Span<bool> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = (x[i] == y);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = (x[i] != y[i]);
            }
        }

        protected override void ElementwiseNotEquals(ReadOnlySpan<T> x, T y, Span<bool> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = (x[i] != y);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = (x[i] >= y[i]);
            }
        }

        protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<T> x, T y, Span<bool> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = (x[i] == y);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = (x[i] <= y[i]);
            }
        }

        protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<T> x, T y, Span<bool> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = (x[i] <= y);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = (x[i] > y[i]);
            }
        }

        protected override void ElementwiseGreaterThan(ReadOnlySpan<T> x, T y, Span<bool> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = (x[i] > y);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
        {
            for (var i = 0; i < x.Length; i++)
            {
                destination[i] = (x[i] < y[i]);
            }
        }

        protected override void ElementwiseLessThan(ReadOnlySpan<T> x, T y, Span<bool> destination)
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
}
#endif
