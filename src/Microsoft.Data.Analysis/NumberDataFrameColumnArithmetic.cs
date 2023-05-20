// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Microsoft.Data.Analysis
{
    internal class NumberDataFrameColumnArithmetic<T> : FloatingPointDataFrameColumnArithmetic<T>, IPrimitiveDataFrameColumnArithmetic<T>
        where T : unmanaged, INumber<T>, IShiftOperators<T, T>, IBitwiseOperators<T, T, T>
    {
        public NumberDataFrameColumnArithmetic() : base()
        {
        }
        public override void And(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (T)(span[i] & otherSpan[i]);
                }
            }
        }
        public override void And(PrimitiveColumnContainer<T> column, T scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (T)(span[i] & scalar);
                }
            }
        }
        public override void And(T scalar, PrimitiveColumnContainer<T> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (T)(scalar & span[i]);
                }
            }
        }
        public override void Or(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (T)(span[i] | otherSpan[i]);
                }
            }
        }
        public override void Or(PrimitiveColumnContainer<T> column, T scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (T)(span[i] | scalar);
                }
            }
        }
        public override void Or(T scalar, PrimitiveColumnContainer<T> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (T)(scalar | span[i]);
                }
            }
        }
        public override void Xor(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right)
        {
            for (int b = 0; b < left.Buffers.Count; b++)
            {
                var buffer = left.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                left.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[b].ReadOnlySpan;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (T)(span[i] ^ otherSpan[i]);
                }
            }
        }
        public override void Xor(PrimitiveColumnContainer<T> column, T scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (T)(span[i] ^ scalar);
                }
            }
        }
        public override void Xor(T scalar, PrimitiveColumnContainer<T> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (T)(scalar ^ span[i]);
                }
            }
        }
        public override void LeftShift(PrimitiveColumnContainer<T> column, int value)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (T)(span[i] << value);
                }
            }
        }
        public override void RightShift(PrimitiveColumnContainer<T> column, int value)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (T)(span[i] >> value);
                }
            }
        }
    }
}
