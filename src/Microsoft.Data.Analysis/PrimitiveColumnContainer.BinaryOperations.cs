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
                    nullCount += mutableBuffer.Length - BitUtility.GetBitCount(leftValidity, mutableBuffer.Length);
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

                if (operation == BinaryOperation.Divide || operation == BinaryOperation.Modulo)
                {
                    //Divisions are special cases
                    for (var j = 0; j < rightSpan.Length; j++)
                    {
                        if (BitUtility.GetBit(rightValidity, j))
                            rightSpan[j] = arithmetic.HandleOperation(operation, left, rightSpan[j]);
                    }
                }
                else
                    arithmetic.HandleOperation(operation, left, rightSpan, rightSpan);
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
            var ret = new PrimitiveColumnContainer<bool>(Length, false);
            var arithmetic = Arithmetic<T>.Instance;

            //Size of any buffer in PrimitiveColumnContainer<bool> is larger (or equal) than size of the buffers for other types
            long index = 0;
            for (int i = 0; i < this.Buffers.Count; i++)
            {
                var leftSpan = this.Buffers[i].ReadOnlySpan;
                var rightSpan = right.Buffers[i].ReadOnlySpan;

                // Get correct ret Span for storing results
                var retSpanIndex = ret.GetIndexOfBufferContainingRowIndex(index);
                var retSpan = ret.Buffers.GetOrCreateMutable(retSpanIndex).Span;

                //Get offset in the buffer to store new data
                var retOffset = (int)(index % DataFrameBuffer<bool>.MaxCapacity);

                //Check if there is enought space in the current ret buffer
                var availableInRetSpan = DataFrameBuffer<bool>.MaxCapacity - retOffset;
                if (availableInRetSpan < leftSpan.Length)
                {
                    //We are not able to place all results into remaining space into our ret buffer, have to split the results

                    //This will be simplified when the size of buffers of different types are done equal
                    //(not supported by classic .Net framework due to the 2 Gb limitation on array size)

                    arithmetic.HandleOperation(operation, leftSpan.Slice(0, availableInRetSpan), rightSpan.Slice(0, availableInRetSpan), retSpan.Slice(retOffset));

                    var nextRetSpan = ret.Buffers.GetOrCreateMutable(retSpanIndex + 1).Span;
                    arithmetic.HandleOperation(operation, leftSpan.Slice(availableInRetSpan), rightSpan.Slice(availableInRetSpan), nextRetSpan);
                }
                else
                    arithmetic.HandleOperation(operation, leftSpan, rightSpan, retSpan.Slice(retOffset));

                index += leftSpan.Length;
            }

            return ret;
        }

        public PrimitiveColumnContainer<bool> HandleOperation(ComparisonOperation operation, T right)
        {
            var ret = new PrimitiveColumnContainer<bool>(Length, false);
            var arithmetic = Arithmetic<T>.Instance;

            //Size of any buffer in PrimitiveColumnContainer<bool> is larger (or equal) than size of the buffers for other types
            long index = 0;
            for (int i = 0; i < this.Buffers.Count; i++)
            {
                var leftSpan = this.Buffers[i].ReadOnlySpan;

                //Get correct ret Span for storing results
                var retSpanIndex = ret.GetIndexOfBufferContainingRowIndex(index);
                var retSpan = ret.Buffers.GetOrCreateMutable(retSpanIndex).Span;

                //Get offset in the buffer to store new data
                var retOffset = (int)(index % DataFrameBuffer<bool>.MaxCapacity);

                //Check if there is enought space in the current ret buffer
                var availableInRetSpan = DataFrameBuffer<bool>.MaxCapacity - retOffset;

                if (availableInRetSpan < leftSpan.Length)
                {
                    //We are not able to place all results into remaining space into our ret buffer, have to split the results

                    //This will be simplified when the size of buffers of different types are done equal
                    //(not supported by classic .Net framework due to the 2 Gb limitation on array size)

                    arithmetic.HandleOperation(operation, leftSpan.Slice(0, availableInRetSpan), right, retSpan.Slice(retOffset));

                    var nextRetSpan = ret.Buffers.GetOrCreateMutable(retSpanIndex + 1).Span;
                    arithmetic.HandleOperation(operation, leftSpan.Slice(availableInRetSpan), right, nextRetSpan);
                }
                else
                    arithmetic.HandleOperation(operation, leftSpan, right, retSpan.Slice(retOffset));

                index += leftSpan.Length;
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
