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
            var arithmetic = Arithmetic<T>.Instance;

            //Divisions are special cases
            var specialCase = (operation == BinaryOperation.Divide || operation == BinaryOperation.Modulo);

            long nullCount = specialCase ? NullCount : 0;
            for (int i = 0; i < this.Buffers.Count; i++)
            {
                var mutableBuffer = this.Buffers.GetOrCreateMutable(i);
                var leftSpan = mutableBuffer.Span;
                var rightSpan = right.Buffers[i].ReadOnlySpan;

                var leftValidity = this.NullBitMapBuffers.GetOrCreateMutable(i).Span;
                var rightValidity = right.NullBitMapBuffers[i].ReadOnlySpan;

                if (specialCase)
                {
                    for (var j = 0; j < leftSpan.Length; j++)
                    {
                        if (BitUtility.GetBit(rightValidity, j))
                            leftSpan[j] = arithmetic.HandleOperation(operation, leftSpan[j], rightSpan[j]);
                        else if (BitUtility.GetBit(leftValidity, j))
                        {
                            BitUtility.ClearBit(leftValidity, j);

                            //Increase NullCount
                            nullCount++;
                        }
                    }
                }
                else
                {
                    arithmetic.HandleOperation(operation, leftSpan, rightSpan, leftSpan);
                    ValidityElementwiseAnd(leftValidity, rightValidity, leftValidity);

                    //Calculate NullCount
                    nullCount += BitUtility.GetBitCount(leftValidity, mutableBuffer.Length);
                }
            }

            NullCount = nullCount;
            return this;
        }

        public PrimitiveColumnContainer<T> HandleOperation(BinaryOperation operation, T right)
        {
            var arithmetic = Arithmetic<T>.Instance;
            for (int i = 0; i < this.Buffers.Count; i++)
            {
                var leftSpan = this.Buffers.GetOrCreateMutable(i).Span;

                arithmetic.HandleOperation(operation, leftSpan, right, leftSpan);
            }

            return this;
        }

        public PrimitiveColumnContainer<T> HandleReverseOperation(BinaryOperation operation, T left)
        {
            var arithmetic = Arithmetic<T>.Instance;
            for (int i = 0; i < this.Buffers.Count; i++)
            {
                var rightSpan = this.Buffers.GetOrCreateMutable(i).Span;
                var rightValidity = this.NullBitMapBuffers[i].ReadOnlySpan;

                if (operation != BinaryOperation.Divide)
                    arithmetic.HandleOperation(operation, left, rightSpan, rightSpan);
                else
                {
                    //Division is a special case
                    for (var j = 0; j < rightSpan.Length; j++)
                    {
                        if (BitUtility.GetBit(rightValidity, j))
                            rightSpan[j] = arithmetic.HandleOperation(operation, left, rightSpan[j]);
                    }
                }
            }

            return this;
        }

        public PrimitiveColumnContainer<T> HandleOperation(BinaryIntOperation operation, int right)
        {
            var arithmetic = Arithmetic<T>.Instance;
            for (int i = 0; i < this.Buffers.Count; i++)
            {
                var leftSpan = this.Buffers.GetOrCreateMutable(i).Span;

                arithmetic.HandleOperation(operation, leftSpan, right, leftSpan);
            }

            return this;
        }

        public PrimitiveColumnContainer<bool> HandleOperation(ComparisonOperation operation, PrimitiveColumnContainer<T> right)
        {
            var arithmetic = Arithmetic<T>.Instance;

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

        public PrimitiveColumnContainer<bool> HandleOperation(ComparisonOperation operation, T right)
        {
            var ret = new PrimitiveColumnContainer<bool>(Length);
            long offset = 0;
            var arithmetic = Arithmetic<T>.Instance;
            for (int i = 0; i < this.Buffers.Count; i++)
            {
                var leftSpan = this.Buffers[i].ReadOnlySpan;

                arithmetic.HandleOperation(operation, leftSpan, right, ret, offset);
            }

            return ret;
        }

        private static void ValidityElementwiseAnd(ReadOnlySpan<byte> left, ReadOnlySpan<byte> right, Span<byte> destination)
        {
            for (var i = 0; i < left.Length; i++)
                destination[i] = (byte)(left[i] & right[i]);
        }
    }
}
