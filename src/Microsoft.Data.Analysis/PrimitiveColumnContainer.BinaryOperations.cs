
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from PrimitiveColumnContainer.BinaryOperations.tt. Do not modify directly

namespace Microsoft.Data.Analysis
{
    internal partial class PrimitiveColumnContainer<T>
        where T : unmanaged
    {
        public PrimitiveColumnContainer<T> HandleOperation(BinaryOperation operation, PrimitiveColumnContainer<T> right)
        {
            var arithmetic = PrimitiveDataFrameColumnArithmetic<T>.Instance;
            long nullCount = 0;
            for (int i = 0; i < this.Buffers.Count; i++)
            {
                var mutableBuffer = this.Buffers.GetOrCreateMutable(i);
                var leftSpan = mutableBuffer.Span;
                var rightSpan = right.Buffers[i].ReadOnlySpan;

                var leftValidity = this.NullBitMapBuffers.GetOrCreateMutable(i).Span;
                var rightValidity = right.NullBitMapBuffers[i].ReadOnlySpan;

                arithmetic.HandleOperation(operation, leftSpan, leftValidity, rightSpan, rightValidity);

                //Calculate NullCount
                nullCount += BitmapHelper.GetBitCount(leftValidity, mutableBuffer.Length);
            }

            NullCount = nullCount;
            return this;
        }

        public PrimitiveColumnContainer<T> HandleOperation(BinaryScalarOperation operation, T right)
        {
            var arithmetic = PrimitiveDataFrameColumnArithmetic<T>.Instance;
            for (int i = 0; i < this.Buffers.Count; i++)
            {
                var leftSpan = this.Buffers.GetOrCreateMutable(i).Span;

                arithmetic.HandleOperation(operation, leftSpan, right);
            }

            return this;
        }

        public PrimitiveColumnContainer<T> HandleReverseOperation(BinaryScalarOperation operation, T left)
        {
            var arithmetic = PrimitiveDataFrameColumnArithmetic<T>.Instance;
            for (int i = 0; i < this.Buffers.Count; i++)
            {
                var rightSpan = this.Buffers.GetOrCreateMutable(i).Span;
                var rightValidity = this.NullBitMapBuffers[i].ReadOnlySpan;

                arithmetic.HandleOperation(operation, left, rightSpan, rightValidity);
            }

            return this;
        }

        public PrimitiveColumnContainer<T> HandleOperation(BinaryIntOperation operation, int right)
        {
            var arithmetic = PrimitiveDataFrameColumnArithmetic<T>.Instance;
            for (int i = 0; i < this.Buffers.Count; i++)
            {
                var leftSpan = this.Buffers.GetOrCreateMutable(i).Span;

                arithmetic.HandleOperation(operation, leftSpan, right);
            }

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
    }
}