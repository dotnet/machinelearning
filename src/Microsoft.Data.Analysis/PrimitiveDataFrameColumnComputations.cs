

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from PrimitiveColumnComputations.tt. Do not modify directly

using System;
using System.Collections.Generic;
using System.Net.Http.Headers;

namespace Microsoft.Data.Analysis
{
    internal interface IPrimitiveColumnComputation<T>
        where T : unmanaged
    {
        void Abs(PrimitiveColumnContainer<T> column);
        void All(PrimitiveColumnContainer<T> column, out bool ret);
        void Any(PrimitiveColumnContainer<T> column, out bool ret);
        void CumulativeMax(PrimitiveColumnContainer<T> column);
        void CumulativeMax(PrimitiveColumnContainer<T> column, IEnumerable<long> rows);
        void CumulativeMin(PrimitiveColumnContainer<T> column);
        void CumulativeMin(PrimitiveColumnContainer<T> column, IEnumerable<long> rows);
        void CumulativeProduct(PrimitiveColumnContainer<T> column);
        void CumulativeProduct(PrimitiveColumnContainer<T> column, IEnumerable<long> rows);
        void CumulativeSum(PrimitiveColumnContainer<T> column);
        void CumulativeSum(PrimitiveColumnContainer<T> column, IEnumerable<long> rows);
        void Max(PrimitiveColumnContainer<T> column, out T? ret);
        void Max(PrimitiveColumnContainer<T> column, IEnumerable<long> rows, out T? ret);
        void Min(PrimitiveColumnContainer<T> column, out T? ret);
        void Min(PrimitiveColumnContainer<T> column, IEnumerable<long> rows, out T? ret);
        void Product(PrimitiveColumnContainer<T> column, out T? ret);
        void Product(PrimitiveColumnContainer<T> column, IEnumerable<long> rows, out T? ret);
        void Sum(PrimitiveColumnContainer<T> column, out T? ret);
        void Sum(PrimitiveColumnContainer<T> column, IEnumerable<long> rows, out T? ret);
        void Round(PrimitiveColumnContainer<T> column);
        PrimitiveColumnContainer<U> CreateTruncating<U>(PrimitiveColumnContainer<T> column) where U : unmanaged, INumber<U>;
    }

    internal static class PrimitiveColumnComputation<T>
        where T : unmanaged
    {
        public static IPrimitiveColumnComputation<T> Instance { get; } = PrimitiveColumnComputation.GetComputation<T>();
    }

    internal static class PrimitiveColumnComputation
    {

        public static IPrimitiveColumnComputation<T> GetComputation<T>()
            where T : unmanaged
        {
            if (typeof(T) == typeof(bool))
            {
                return (IPrimitiveColumnComputation<T>)new BoolComputation();
            }
            else if (typeof(T) == typeof(byte))
            {
                return (IPrimitiveColumnComputation<T>)new NumberMathComputation<byte>();
            }
            else if (typeof(T) == typeof(char))
            {
                return (IPrimitiveColumnComputation<T>)new NumberMathComputation<char>();
            }
            else if (typeof(T) == typeof(decimal))
            {
                return (IPrimitiveColumnComputation<T>)new DecimalMathComputation();
            }
            else if (typeof(T) == typeof(double))
            {
                return (IPrimitiveColumnComputation<T>)new FloatingPointMathComputation<double>();
            }
            else if (typeof(T) == typeof(float))
            {
                return (IPrimitiveColumnComputation<T>)new FloatingPointMathComputation<float>();
            }
            else if (typeof(T) == typeof(int))
            {
                return (IPrimitiveColumnComputation<T>)new NumberMathComputation<int>();
            }
            else if (typeof(T) == typeof(long))
            {
                return (IPrimitiveColumnComputation<T>)new NumberMathComputation<long>();
            }
            else if (typeof(T) == typeof(sbyte))
            {
                return (IPrimitiveColumnComputation<T>)new NumberMathComputation<sbyte>();
            }
            else if (typeof(T) == typeof(short))
            {
                return (IPrimitiveColumnComputation<T>)new NumberMathComputation<short>();
            }
            else if (typeof(T) == typeof(uint))
            {
                return (IPrimitiveColumnComputation<T>)new NumberMathComputation<uint>();
            }
            else if (typeof(T) == typeof(ulong))
            {
                return (IPrimitiveColumnComputation<T>)new NumberMathComputation<ulong>();
            }
            else if (typeof(T) == typeof(ushort))
            {
                return (IPrimitiveColumnComputation<T>)new NumberMathComputation<ushort>();
            }
            else if (typeof(T) == typeof(DateTime))
            {
                return (IPrimitiveColumnComputation<T>)new DateTimeComputation();
            }

            throw new NotSupportedException();
        }
    }

    internal class BoolComputation : IPrimitiveColumnComputation<bool>
    {
        public void Abs(PrimitiveColumnContainer<bool> column)
        {
            throw new NotSupportedException();
        }

        public void All(PrimitiveColumnContainer<bool> column, out bool ret)
        {
            ret = true;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var span = buffer.ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    if (span[i] == false)
                    {
                        ret = false;
                        return;
                    }
                }
            }
        }

        public void Any(PrimitiveColumnContainer<bool> column, out bool ret)
        {
            ret = false;
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var span = buffer.ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    if (span[i] == true)
                    {
                        ret = true;
                        return;
                    }
                }
            }
        }

        public void CumulativeMax(PrimitiveColumnContainer<bool> column)
        {
            throw new NotSupportedException();
        }

        public void CumulativeMax(PrimitiveColumnContainer<bool> column, IEnumerable<long> rows)
        {
            throw new NotSupportedException();
        }

        public void CumulativeMin(PrimitiveColumnContainer<bool> column)
        {
            throw new NotSupportedException();
        }

        public void CumulativeMin(PrimitiveColumnContainer<bool> column, IEnumerable<long> rows)
        {
            throw new NotSupportedException();
        }

        public void CumulativeProduct(PrimitiveColumnContainer<bool> column)
        {
            throw new NotSupportedException();
        }

        public void CumulativeProduct(PrimitiveColumnContainer<bool> column, IEnumerable<long> rows)
        {
            throw new NotSupportedException();
        }

        public void CumulativeSum(PrimitiveColumnContainer<bool> column)
        {
            throw new NotSupportedException();
        }

        public void CumulativeSum(PrimitiveColumnContainer<bool> column, IEnumerable<long> rows)
        {
            throw new NotSupportedException();
        }

        public void Max(PrimitiveColumnContainer<bool> column, out bool? ret)
        {
            throw new NotSupportedException();
        }

        public void Max(PrimitiveColumnContainer<bool> column, IEnumerable<long> rows, out bool? ret)
        {
            throw new NotSupportedException();
        }

        public void Min(PrimitiveColumnContainer<bool> column, out bool? ret)
        {
            throw new NotSupportedException();
        }

        public void Min(PrimitiveColumnContainer<bool> column, IEnumerable<long> rows, out bool? ret)
        {
            throw new NotSupportedException();
        }

        public void Product(PrimitiveColumnContainer<bool> column, out bool? ret)
        {
            throw new NotSupportedException();
        }

        public void Product(PrimitiveColumnContainer<bool> column, IEnumerable<long> rows, out bool? ret)
        {
            throw new NotSupportedException();
        }

        public void Sum(PrimitiveColumnContainer<bool> column, out bool? ret)
        {
            throw new NotSupportedException();
        }

        public void Sum(PrimitiveColumnContainer<bool> column, IEnumerable<long> rows, out bool? ret)
        {
            throw new NotSupportedException();
        }

        public void Round(PrimitiveColumnContainer<bool> column)
        {
            throw new NotSupportedException();
        }

        public PrimitiveColumnContainer<U> CreateTruncating<U>(PrimitiveColumnContainer<bool> column) where U : unmanaged, INumber<U>
        {
            throw new NotImplementedException();
        }
    }
}
