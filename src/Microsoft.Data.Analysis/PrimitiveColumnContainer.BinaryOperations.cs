// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

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
                nullCount += BitUtility.GetBitCount(leftValidity, mutableBuffer.Length);
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

        public PrimitiveColumnContainer<bool> HandleOperation(ComparisonOperation operation, PrimitiveColumnContainer<T> right)
        {
            var arithmetic = PrimitiveDataFrameColumnArithmetic<T>.Instance;

            var ret = new PrimitiveColumnContainer<bool>(Length);
            long offset = 0;
            for (int i = 0; i < this.Buffers.Count; i++)
            {
                var leftSpan = this.Buffers[i].ReadOnlySpan;
                var rightSpan = right.Buffers[i].ReadOnlySpan;

                arithmetic.HandleOperation(operation, leftSpan, rightSpan, ret, offset);
                offset += leftSpan.Length;
            }

            return ret;
        }

        public PrimitiveColumnContainer<bool> HandleOperation(ComparisonScalarOperation operation, T right)
        {
            var ret = new PrimitiveColumnContainer<bool>(Length);
            long offset = 0;
            var arithmetic = PrimitiveDataFrameColumnArithmetic<T>.Instance;
            for (int i = 0; i < this.Buffers.Count; i++)
            {
                var leftSpan = this.Buffers[i].ReadOnlySpan;

                arithmetic.HandleOperation(operation, leftSpan, right, ret, offset);
            }

            return ret;
        }
    }
}
