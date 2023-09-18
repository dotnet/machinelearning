
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from PrimitiveColumnContainer.BinaryOperations.tt. Do not modify directly

namespace Microsoft.Data.Analysis
{
    internal partial class PrimitiveColumnContainer<T>
        where T : unmanaged
    {

        public PrimitiveColumnContainer<T> Add(PrimitiveColumnContainer<T> right)
        {
            for (int i = 0; i < this.Buffers.Count; i++)
            {
                //Calculate raw values
                var mutableBuffer = this.Buffers.GetOrCreateMutable(i);
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[i].ReadOnlySpan;

                PrimitiveDataFrameColumnArithmetic<T>.Instance.Add(span, otherSpan, span);

                //Calculate validity
                var validityBuffer = this.NullBitMapBuffers.GetOrCreateMutable(i);
                var otherValidityBuffer = right.NullBitMapBuffers[i];

                BitmapHelper.ElementwiseAnd(validityBuffer.ReadOnlySpan, otherValidityBuffer.ReadOnlySpan, validityBuffer.Span);

                //Calculate NullCount
                this.NullCount = 0;
            }

            return this;
        }

        public PrimitiveColumnContainer<T> Add(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Add(this, scalar);
            return this;
        }

        public PrimitiveColumnContainer<T> Subtract(PrimitiveColumnContainer<T> right)
        {
            for (int i = 0; i < this.Buffers.Count; i++)
            {
                //Calculate raw values
                var mutableBuffer = this.Buffers.GetOrCreateMutable(i);
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[i].ReadOnlySpan;

                PrimitiveDataFrameColumnArithmetic<T>.Instance.Subtract(span, otherSpan, span);

                //Calculate validity
                var validityBuffer = this.NullBitMapBuffers.GetOrCreateMutable(i);
                var otherValidityBuffer = right.NullBitMapBuffers[i];

                BitmapHelper.ElementwiseAnd(validityBuffer.ReadOnlySpan, otherValidityBuffer.ReadOnlySpan, validityBuffer.Span);

                //Calculate NullCount
                this.NullCount = 0;
            }

            return this;
        }

        public PrimitiveColumnContainer<T> Subtract(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Subtract(this, scalar);
            return this;
        }

        public PrimitiveColumnContainer<T> Multiply(PrimitiveColumnContainer<T> right)
        {
            for (int i = 0; i < this.Buffers.Count; i++)
            {
                //Calculate raw values
                var mutableBuffer = this.Buffers.GetOrCreateMutable(i);
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[i].ReadOnlySpan;

                PrimitiveDataFrameColumnArithmetic<T>.Instance.Multiply(span, otherSpan, span);

                //Calculate validity
                var validityBuffer = this.NullBitMapBuffers.GetOrCreateMutable(i);
                var otherValidityBuffer = right.NullBitMapBuffers[i];

                BitmapHelper.ElementwiseAnd(validityBuffer.ReadOnlySpan, otherValidityBuffer.ReadOnlySpan, validityBuffer.Span);

                //Calculate NullCount
                this.NullCount = 0;
            }

            return this;
        }

        public PrimitiveColumnContainer<T> Multiply(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Multiply(this, scalar);
            return this;
        }

        public PrimitiveColumnContainer<T> Divide(PrimitiveColumnContainer<T> right)
        {
            for (int i = 0; i < this.Buffers.Count; i++)
            {
                //Calculate raw values
                var mutableBuffer = this.Buffers.GetOrCreateMutable(i);
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[i].ReadOnlySpan;

                PrimitiveDataFrameColumnArithmetic<T>.Instance.Divide(span, otherSpan, span);

                //Calculate validity
                var validityBuffer = this.NullBitMapBuffers.GetOrCreateMutable(i);
                var otherValidityBuffer = right.NullBitMapBuffers[i];

                BitmapHelper.ElementwiseAnd(validityBuffer.ReadOnlySpan, otherValidityBuffer.ReadOnlySpan, validityBuffer.Span);

                //Calculate NullCount
                this.NullCount = 0;
            }

            return this;
        }

        public PrimitiveColumnContainer<T> Divide(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Divide(this, scalar);
            return this;
        }

        public PrimitiveColumnContainer<T> Modulo(PrimitiveColumnContainer<T> right)
        {
            for (int i = 0; i < this.Buffers.Count; i++)
            {
                //Calculate raw values
                var mutableBuffer = this.Buffers.GetOrCreateMutable(i);
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[i].ReadOnlySpan;

                PrimitiveDataFrameColumnArithmetic<T>.Instance.Modulo(span, otherSpan, span);

                //Calculate validity
                var validityBuffer = this.NullBitMapBuffers.GetOrCreateMutable(i);
                var otherValidityBuffer = right.NullBitMapBuffers[i];

                BitmapHelper.ElementwiseAnd(validityBuffer.ReadOnlySpan, otherValidityBuffer.ReadOnlySpan, validityBuffer.Span);

                //Calculate NullCount
                this.NullCount = 0;
            }

            return this;
        }

        public PrimitiveColumnContainer<T> Modulo(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Modulo(this, scalar);
            return this;
        }

        public PrimitiveColumnContainer<T> And(PrimitiveColumnContainer<T> right)
        {
            for (int i = 0; i < this.Buffers.Count; i++)
            {
                //Calculate raw values
                var mutableBuffer = this.Buffers.GetOrCreateMutable(i);
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[i].ReadOnlySpan;

                PrimitiveDataFrameColumnArithmetic<T>.Instance.And(span, otherSpan, span);

                //Calculate validity
                var validityBuffer = this.NullBitMapBuffers.GetOrCreateMutable(i);
                var otherValidityBuffer = right.NullBitMapBuffers[i];

                BitmapHelper.ElementwiseAnd(validityBuffer.ReadOnlySpan, otherValidityBuffer.ReadOnlySpan, validityBuffer.Span);

                //Calculate NullCount
                this.NullCount = 0;
            }

            return this;
        }

        public PrimitiveColumnContainer<T> And(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.And(this, scalar);
            return this;
        }

        public PrimitiveColumnContainer<T> Or(PrimitiveColumnContainer<T> right)
        {
            for (int i = 0; i < this.Buffers.Count; i++)
            {
                //Calculate raw values
                var mutableBuffer = this.Buffers.GetOrCreateMutable(i);
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[i].ReadOnlySpan;

                PrimitiveDataFrameColumnArithmetic<T>.Instance.Or(span, otherSpan, span);

                //Calculate validity
                var validityBuffer = this.NullBitMapBuffers.GetOrCreateMutable(i);
                var otherValidityBuffer = right.NullBitMapBuffers[i];

                BitmapHelper.ElementwiseAnd(validityBuffer.ReadOnlySpan, otherValidityBuffer.ReadOnlySpan, validityBuffer.Span);

                //Calculate NullCount
                this.NullCount = 0;
            }

            return this;
        }

        public PrimitiveColumnContainer<T> Or(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Or(this, scalar);
            return this;
        }

        public PrimitiveColumnContainer<T> Xor(PrimitiveColumnContainer<T> right)
        {
            for (int i = 0; i < this.Buffers.Count; i++)
            {
                //Calculate raw values
                var mutableBuffer = this.Buffers.GetOrCreateMutable(i);
                var span = mutableBuffer.Span;
                var otherSpan = right.Buffers[i].ReadOnlySpan;

                PrimitiveDataFrameColumnArithmetic<T>.Instance.Xor(span, otherSpan, span);

                //Calculate validity
                var validityBuffer = this.NullBitMapBuffers.GetOrCreateMutable(i);
                var otherValidityBuffer = right.NullBitMapBuffers[i];

                BitmapHelper.ElementwiseAnd(validityBuffer.ReadOnlySpan, otherValidityBuffer.ReadOnlySpan, validityBuffer.Span);

                //Calculate NullCount
                this.NullCount = 0;
            }

            return this;
        }

        public PrimitiveColumnContainer<T> Xor(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Xor(this, scalar);
            return this;
        }

        public PrimitiveColumnContainer<T> LeftShift(int value)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.LeftShift(this, value);
            return this;
        }

        public PrimitiveColumnContainer<T> RightShift(int value)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.RightShift(this, value);
            return this;
        }

        public PrimitiveColumnContainer<bool> ElementwiseEquals(PrimitiveColumnContainer<T> right)
        {
            return PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseEquals(this, right);
        }

        public PrimitiveColumnContainer<bool> ElementwiseEquals(T scalar)
        {
            return PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseEquals(this, scalar);
        }

        public PrimitiveColumnContainer<bool> ElementwiseNotEquals(PrimitiveColumnContainer<T> right)
        {
            return PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseNotEquals(this, right);
        }

        public PrimitiveColumnContainer<bool> ElementwiseNotEquals(T scalar)
        {
            return PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseNotEquals(this, scalar);
        }

        public PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<T> right)
        {
            return PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseGreaterThanOrEqual(this, right);
        }

        public PrimitiveColumnContainer<bool> ElementwiseGreaterThanOrEqual(T scalar)
        {
            return PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseGreaterThanOrEqual(this, scalar);
        }

        public PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<T> right)
        {
            return PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseLessThanOrEqual(this, right);
        }

        public PrimitiveColumnContainer<bool> ElementwiseLessThanOrEqual(T scalar)
        {
            return PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseLessThanOrEqual(this, scalar);
        }

        public PrimitiveColumnContainer<bool> ElementwiseGreaterThan(PrimitiveColumnContainer<T> right)
        {
            return PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseGreaterThan(this, right);
        }

        public PrimitiveColumnContainer<bool> ElementwiseGreaterThan(T scalar)
        {
            return PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseGreaterThan(this, scalar);
        }

        public PrimitiveColumnContainer<bool> ElementwiseLessThan(PrimitiveColumnContainer<T> right)
        {
            return PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseLessThan(this, right);
        }

        public PrimitiveColumnContainer<bool> ElementwiseLessThan(T scalar)
        {
            return PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseLessThan(this, scalar);
        }

        public PrimitiveColumnContainer<T> ReverseAdd(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Add(scalar, this);
            return this;
        }

        public PrimitiveColumnContainer<T> ReverseSubtract(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Subtract(scalar, this);
            return this;
        }

        public PrimitiveColumnContainer<T> ReverseMultiply(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Multiply(scalar, this);
            return this;
        }

        public PrimitiveColumnContainer<T> ReverseDivide(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Divide(scalar, this);
            return this;
        }

        public PrimitiveColumnContainer<T> ReverseModulo(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Modulo(scalar, this);
            return this;
        }

        public PrimitiveColumnContainer<T> ReverseAnd(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.And(scalar, this);
            return this;
        }

        public PrimitiveColumnContainer<T> ReverseOr(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Or(scalar, this);
            return this;
        }

        public PrimitiveColumnContainer<T> ReverseXor(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Xor(scalar, this);
            return this;
        }
    }
}