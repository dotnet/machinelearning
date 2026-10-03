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
            //Null values are only stored in the validity bitmaps, so they need to be checked as well as the data buffers
            var hasNulls = this.NullCount > 0 || right.NullCount > 0;

            long index = 0;
            for (int i = 0; i < this.Buffers.Count; i++)
            {
                var leftSpan = this.Buffers[i].ReadOnlySpan;
                var rightSpan = right.Buffers[i].ReadOnlySpan;

                //Empty validity span means that all values are valid
                var leftValidity = this.NullCount > 0 ? this.NullBitMapBuffers[i].ReadOnlySpan : ReadOnlySpan<byte>.Empty;
                var rightValidity = right.NullCount > 0 ? right.NullBitMapBuffers[i].ReadOnlySpan : ReadOnlySpan<byte>.Empty;

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

                    var firstRetSpan = retSpan.Slice(retOffset, availableInRetSpan);
                    arithmetic.HandleOperation(operation, leftSpan.Slice(0, availableInRetSpan), rightSpan.Slice(0, availableInRetSpan), firstRetSpan);

                    var nextRetSpan = ret.Buffers.GetOrCreateMutable(retSpanIndex + 1).Span.Slice(0, leftSpan.Length - availableInRetSpan);
                    arithmetic.HandleOperation(operation, leftSpan.Slice(availableInRetSpan), rightSpan.Slice(availableInRetSpan), nextRetSpan);

                    if (hasNulls)
                    {
                        ApplyNullComparisonResults(operation, leftValidity, rightValidity, 0, firstRetSpan);
                        ApplyNullComparisonResults(operation, leftValidity, rightValidity, availableInRetSpan, nextRetSpan);
                    }
                }
                else
                {
                    var currentRetSpan = retSpan.Slice(retOffset, leftSpan.Length);
                    arithmetic.HandleOperation(operation, leftSpan, rightSpan, currentRetSpan);

                    if (hasNulls)
                        ApplyNullComparisonResults(operation, leftValidity, rightValidity, 0, currentRetSpan);
                }

                index += leftSpan.Length;
            }

            return ret;
        }

        public PrimitiveColumnContainer<bool> HandleOperation(ComparisonOperation operation, T right)
        {
            var ret = new PrimitiveColumnContainer<bool>(Length, false);
            var arithmetic = Arithmetic<T>.Instance;

            //Size of any buffer in PrimitiveColumnContainer<bool> is larger (or equal) than size of the buffers for other types
            //Null values are only stored in the validity bitmaps, so they need to be checked as well as the data buffers
            var hasNulls = this.NullCount > 0;

            long index = 0;
            for (int i = 0; i < this.Buffers.Count; i++)
            {
                var leftSpan = this.Buffers[i].ReadOnlySpan;
                var leftValidity = hasNulls ? this.NullBitMapBuffers[i].ReadOnlySpan : ReadOnlySpan<byte>.Empty;

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

                    var firstRetSpan = retSpan.Slice(retOffset, availableInRetSpan);
                    arithmetic.HandleOperation(operation, leftSpan.Slice(0, availableInRetSpan), right, firstRetSpan);

                    var nextRetSpan = ret.Buffers.GetOrCreateMutable(retSpanIndex + 1).Span.Slice(0, leftSpan.Length - availableInRetSpan);
                    arithmetic.HandleOperation(operation, leftSpan.Slice(availableInRetSpan), right, nextRetSpan);

                    if (hasNulls)
                    {
                        //The scalar is never null
                        ApplyNullComparisonResults(operation, leftValidity, ReadOnlySpan<byte>.Empty, 0, firstRetSpan);
                        ApplyNullComparisonResults(operation, leftValidity, ReadOnlySpan<byte>.Empty, availableInRetSpan, nextRetSpan);
                    }
                }
                else
                {
                    var currentRetSpan = retSpan.Slice(retOffset, leftSpan.Length);
                    arithmetic.HandleOperation(operation, leftSpan, right, currentRetSpan);

                    if (hasNulls)
                        ApplyNullComparisonResults(operation, leftValidity, ReadOnlySpan<byte>.Empty, 0, currentRetSpan);
                }

                index += leftSpan.Length;
            }

            return ret;
        }

        /// <summary>
        /// Overwrites the comparison results for every element where at least one side is null.
        /// The results produced from the data buffers are meaningless for these elements, as the
        /// data buffers hold an arbitrary value (usually default) in place of a null.
        /// </summary>
        /// <remarks>
        /// Two nulls are considered equal, so for null elements:
        ///  - null == null, null &lt;= null and null &gt;= null are true;
        ///  - null != value and value != null are true;
        ///  - every other comparison is false.
        /// </remarks>
        /// <param name="operation">The comparison operation that produced the results.</param>
        /// <param name="leftValidity">Validity bitmap of the left side. An empty span means that all left values are valid.</param>
        /// <param name="rightValidity">Validity bitmap of the right side. An empty span means that all right values are valid.</param>
        /// <param name="start">Index in the validity bitmaps that corresponds to the first element of <paramref name="destination"/>.</param>
        /// <param name="destination">Comparison results to correct.</param>
        private static void ApplyNullComparisonResults(ComparisonOperation operation, ReadOnlySpan<byte> leftValidity, ReadOnlySpan<byte> rightValidity, int start, Span<bool> destination)
        {
            var leftAllValid = leftValidity.IsEmpty;
            var rightAllValid = rightValidity.IsEmpty;
            if (leftAllValid && rightAllValid)
                return;

            var bothNullResult = operation == ComparisonOperation.ElementwiseEquals
                || operation == ComparisonOperation.ElementwiseLessThanOrEqual
                || operation == ComparisonOperation.ElementwiseGreaterThanOrEqual;
            var oneNullResult = operation == ComparisonOperation.ElementwiseNotEquals;

            var end = start + destination.Length;
            var i = start;
            while (i < end)
            {
                //Skip whole bitmap bytes where both sides are valid, as the comparison results are already correct
                if ((i & 7) == 0 && end - i >= 8)
                {
                    var leftByte = leftAllValid ? (byte)0xFF : leftValidity[i >> 3];
                    var rightByte = rightAllValid ? (byte)0xFF : rightValidity[i >> 3];
                    if ((leftByte & rightByte) == 0xFF)
                    {
                        i += 8;
                        continue;
                    }
                }

                var leftIsValid = leftAllValid || BitUtility.IsValid(leftValidity, i);
                var rightIsValid = rightAllValid || BitUtility.IsValid(rightValidity, i);
                if (!leftIsValid || !rightIsValid)
                    destination[i - start] = leftIsValid == rightIsValid ? bothNullResult : oneNullResult;

                i++;
            }
        }

        private static void ValidityElementwiseAnd(ReadOnlySpan<byte> left, ReadOnlySpan<byte> right, Span<byte> destination)
        {
            for (var i = 0; i < left.Length; i++)
                destination[i] = (byte)(left[i] & right[i]);
        }
    }
}
