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
    internal class FloatingPointDataFrameColumnArithmetic<T> : IPrimitiveDataFrameColumnArithmetic<T>
        where T : unmanaged, INumber<T>
    {
        public FloatingPointDataFrameColumnArithmetic() : base()
        {
        }
        public void Add(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right)
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
                    span[i] = (T)(span[i] + otherSpan[i]);
                }
            }
        }
        public void Add(PrimitiveColumnContainer<T> column, T scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (T)(span[i] + scalar);
                }
            }
        }
        public void Add(T scalar, PrimitiveColumnContainer<T> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (T)(scalar + span[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right)
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
                    span[i] = (T)(span[i] - otherSpan[i]);
                }
            }
        }
        public void Subtract(PrimitiveColumnContainer<T> column, T scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (T)(span[i] - scalar);
                }
            }
        }
        public void Subtract(T scalar, PrimitiveColumnContainer<T> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (T)(scalar - span[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right)
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
                    span[i] = (T)(span[i] * otherSpan[i]);
                }
            }
        }
        public void Multiply(PrimitiveColumnContainer<T> column, T scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (T)(span[i] * scalar);
                }
            }
        }
        public void Multiply(T scalar, PrimitiveColumnContainer<T> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (T)(scalar * span[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right)
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
                    span[i] = (T)(span[i] / otherSpan[i]);
                }
            }
        }
        public void Divide(PrimitiveColumnContainer<T> column, T scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (T)(span[i] / scalar);
                }
            }
        }
        public void Divide(T scalar, PrimitiveColumnContainer<T> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (T)(scalar / span[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right)
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
                    span[i] = (T)(span[i] % otherSpan[i]);
                }
            }
        }
        public void Modulo(PrimitiveColumnContainer<T> column, T scalar)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (T)(span[i] % scalar);
                }
            }
        }
        public void Modulo(T scalar, PrimitiveColumnContainer<T> column)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    span[i] = (T)(scalar % span[i]);
                }
            }
        }
        public virtual void And(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right)
        {
            throw new NotSupportedException();
        }
        public virtual void And(PrimitiveColumnContainer<T> column, T scalar)
        {
            throw new NotSupportedException();
        }
        public virtual void And(T scalar, PrimitiveColumnContainer<T> column)
        {
            throw new NotSupportedException();
        }
        public virtual void Or(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right)
        {
            throw new NotSupportedException();
        }
        public virtual void Or(PrimitiveColumnContainer<T> column, T scalar)
        {
            throw new NotSupportedException();
        }
        public virtual void Or(T scalar, PrimitiveColumnContainer<T> column)
        {
            throw new NotSupportedException();
        }
        public virtual void Xor(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right)
        {
            throw new NotSupportedException();
        }
        public virtual void Xor(PrimitiveColumnContainer<T> column, T scalar)
        {
            throw new NotSupportedException();
        }
        public virtual void Xor(T scalar, PrimitiveColumnContainer<T> column)
        {
            throw new NotSupportedException();
        }
        public virtual void LeftShift(PrimitiveColumnContainer<T> column, int value)
        {
            throw new NotSupportedException();
        }
        public virtual void RightShift(PrimitiveColumnContainer<T> column, int value)
        {
            throw new NotSupportedException();
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right, PrimitiveColumnContainer<bool> ret)
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
                    ret[i] = (span[i] == otherSpan[i]);
                }
            }
        }
        public void ElementwiseEquals(PrimitiveColumnContainer<T> column, T scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] == scalar);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right, PrimitiveColumnContainer<bool> ret)
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
                    ret[i] = (span[i] != otherSpan[i]);
                }
            }
        }
        public void ElementwiseNotEquals(PrimitiveColumnContainer<T> column, T scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] != scalar);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right, PrimitiveColumnContainer<bool> ret)
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
                    ret[i] = (span[i] >= otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<T> column, T scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] >= scalar);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right, PrimitiveColumnContainer<bool> ret)
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
                    ret[i] = (span[i] <= otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThanOrEqual(PrimitiveColumnContainer<T> column, T scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] <= scalar);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right, PrimitiveColumnContainer<bool> ret)
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
                    ret[i] = (span[i] > otherSpan[i]);
                }
            }
        }
        public void ElementwiseGreaterThan(PrimitiveColumnContainer<T> column, T scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] > scalar);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<T> left, PrimitiveColumnContainer<T> right, PrimitiveColumnContainer<bool> ret)
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
                    ret[i] = (span[i] < otherSpan[i]);
                }
            }
        }
        public void ElementwiseLessThan(PrimitiveColumnContainer<T> column, T scalar, PrimitiveColumnContainer<bool> ret)
        {
            for (int b = 0; b < column.Buffers.Count; b++)
            {
                var buffer = column.Buffers[b];
                var mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(buffer);
                column.Buffers[b] = mutableBuffer;
                var span = mutableBuffer.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    ret[i] = (span[i] < scalar);
                }
            }
        }
    }
}
