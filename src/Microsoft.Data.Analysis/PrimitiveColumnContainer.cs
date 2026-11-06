// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace Microsoft.Data.Analysis
{
    /// <summary>
    /// PrimitiveColumnContainer is just a store for the column data. APIs that want to change the data must be defined in PrimitiveDataFrameColumn
    /// </summary>
    /// <typeparam name="T"></typeparam>
    internal partial class PrimitiveColumnContainer<T> : IEnumerable<T?>
        where T : unmanaged
    {
        public IList<ReadOnlyDataFrameBuffer<T>> Buffers = new List<ReadOnlyDataFrameBuffer<T>>();

        // To keep the mapping simple, each buffer is mapped 1v1 to a nullBitMapBuffer
        // A set bit implies a valid value. An unset bit => null value
        public IList<ReadOnlyDataFrameBuffer<byte>> NullBitMapBuffers = new List<ReadOnlyDataFrameBuffer<byte>>();

        public PrimitiveColumnContainer(IEnumerable<T> values)
        {
            values = values ?? throw new ArgumentNullException(nameof(values));
            foreach (T value in values)
            {
                Append(value);
            }
        }

        public PrimitiveColumnContainer(IEnumerable<T?> values)
        {
            values = values ?? throw new ArgumentNullException(nameof(values));
            foreach (T? value in values)
            {
                Append(value);
            }
        }

        public PrimitiveColumnContainer(ReadOnlyMemory<byte> buffer, ReadOnlyMemory<byte> nullBitMap, int length, int nullCount)
        {
            ReadOnlyDataFrameBuffer<T> dataBuffer;
            if (buffer.IsEmpty)
            {
                DataFrameBuffer<T> mutableBuffer = new DataFrameBuffer<T>(length);
                mutableBuffer.IncreaseSize(length);
                mutableBuffer.RawSpan.Fill(default(T));
                dataBuffer = mutableBuffer;
            }
            else
            {
                dataBuffer = new ReadOnlyDataFrameBuffer<T>(buffer, length);
            }
            Buffers.Add(dataBuffer);

            int bitMapBufferLength = (length + 7) / 8;
            ReadOnlyDataFrameBuffer<byte> nullDataFrameBuffer;
            if (nullBitMap.IsEmpty)
            {
                if (nullCount != 0)
                {
                    throw new ArgumentNullException(Strings.InconsistentNullBitMapAndNullCount, nameof(nullBitMap));
                }
                if (!buffer.IsEmpty)
                {
                    // Create a new bitMap with all the bits up to length set
                    var bitMap = new DataFrameBuffer<byte>(bitMapBufferLength);
                    bitMap.IncreaseSize(bitMapBufferLength);

                    var span = bitMap.Span;
                    span.Fill(255);
                    int lastByte = 1 << (length - (bitMapBufferLength - 1) * 8);
                    span[bitMapBufferLength - 1] = (byte)(lastByte - 1);

                    nullDataFrameBuffer = bitMap;
                }
                else
                {
                    nullDataFrameBuffer = new DataFrameBuffer<byte>();
                }
            }
            else
            {
                if (nullBitMap.Length < bitMapBufferLength)
                {
                    throw new ArgumentException(Strings.InconsistentNullBitMapAndLength, nameof(nullBitMap));
                }
                nullDataFrameBuffer = new ReadOnlyDataFrameBuffer<byte>(nullBitMap, bitMapBufferLength);
            }
            NullBitMapBuffers.Add(nullDataFrameBuffer);
            Length = length;
            NullCount = nullCount;
        }

        public PrimitiveColumnContainer(long length = 0, T? defaulValue = null)
        {
            AppendMany(defaulValue, length);
        }

        public void Resize(long length)
        {
            if (length < Length)
                throw new ArgumentException(Strings.CannotResizeDown, nameof(length));
            AppendMany(default, length - Length);
        }

        public void Append(T? value)
        {
            if (Buffers.Count == 0)
            {
                Buffers.Add(new DataFrameBuffer<T>());
                NullBitMapBuffers.Add(new DataFrameBuffer<byte>());
            }

            if (Buffers[Buffers.Count - 1].Length == ReadOnlyDataFrameBuffer<T>.MaxCapacity)
            {
                Buffers.Add(new DataFrameBuffer<T>());
                NullBitMapBuffers.Add(new DataFrameBuffer<byte>());
            }

            DataFrameBuffer<T> mutableLastBuffer = Buffers.GetOrCreateMutable(Buffers.Count - 1);
            mutableLastBuffer.Append(value ?? default);
            SetValidityBit(Length, value.HasValue);
            Length++;
        }

        public void AppendMany(T? value, long count)
        {
            if (!value.HasValue)
            {
                NullCount += count;
            }

            var remaining = count;
            while (remaining > 0)
            {
                if (Buffers.Count == 0)
                {
                    Buffers.Add(new DataFrameBuffer<T>());
                    NullBitMapBuffers.Add(new DataFrameBuffer<byte>());
                }

                if (Buffers[Buffers.Count - 1].Length == ReadOnlyDataFrameBuffer<T>.MaxCapacity)
                {
                    Buffers.Add(new DataFrameBuffer<T>());
                    NullBitMapBuffers.Add(new DataFrameBuffer<byte>());
                }

                DataFrameBuffer<T> mutableLastBuffer = Buffers.GetOrCreateMutable(Buffers.Count - 1);
                DataFrameBuffer<byte> lastNullBitMapBuffer = NullBitMapBuffers.GetOrCreateMutable(NullBitMapBuffers.Count - 1);

                //Calculate how many values we can additionaly allocate and not exceed the MaxCapacity
                int originalBufferLength = mutableLastBuffer.Length;
                int allocatable = (int)Math.Min(remaining, ReadOnlyDataFrameBuffer<T>.MaxCapacity - originalBufferLength);
                mutableLastBuffer.IncreaseSize(allocatable);

                //Calculate how many bytes we have additionaly allocate to store allocatable number of bits (need to take into account unused bits inside already allocated bytes)
                int nullBufferAllocatable = (originalBufferLength + allocatable + 7) / 8 - lastNullBitMapBuffer.Length;
                lastNullBitMapBuffer.IncreaseSize(nullBufferAllocatable);
                Length += allocatable;

                if (value.HasValue)
                {
                    mutableLastBuffer.RawSpan.Slice(mutableLastBuffer.Length - allocatable, allocatable).Fill(value.Value);
                    BitUtility.SetBits(lastNullBitMapBuffer.RawSpan, originalBufferLength, allocatable, true);
                }

                remaining -= allocatable;
            }
        }

        public void ApplyElementwise(Func<T?, long, T?> func)
        {
            long curIndex = 0;
            for (int b = 0; b < Buffers.Count; b++)
            {
                Span<T> mutableBuffer = Buffers.GetOrCreateMutable(b).Span;
                Span<byte> mutableNullBitMapBuffer = NullBitMapBuffers.GetOrCreateMutable(b).Span;

                for (int i = 0; i < mutableBuffer.Length; i++)
                {
                    bool isValid = BitUtility.IsValid(mutableNullBitMapBuffer, i);
                    T? value = func(isValid ? mutableBuffer[i] : null, curIndex);
                    mutableBuffer[i] = value.GetValueOrDefault();
                    SetValidityBit(mutableNullBitMapBuffer, i, value != null);
                    curIndex++;
                }
            }
        }

        public void Apply(Func<T, T> func)
        {
            for (int b = 0; b < Buffers.Count; b++)
            {
                var span = Buffers.GetOrCreateMutable(b).Span;
                var validitySpan = NullBitMapBuffers.GetOrCreateMutable(b).Span;

                for (int i = 0; i < span.Length; i++)
                {
                    if (NullCount == 0 || BitUtility.IsValid(validitySpan, i))
                    {
                        span[i] = func(span[i]);
                    }
                }
            }
        }

        [Obsolete]
        public void Apply<TResult>(Func<T?, TResult?> func, PrimitiveColumnContainer<TResult> resultContainer)
            where TResult : unmanaged
        {
            for (int b = 0; b < Buffers.Count; b++)
            {
                var sourceBuffer = Buffers[b];
                var sourceNullBitMap = NullBitMapBuffers[b].ReadOnlySpan;

                Span<TResult> mutableResultBuffer = resultContainer.Buffers.GetOrCreateMutable(b).Span;
                Span<byte> mutableResultNullBitMapBuffer = resultContainer.NullBitMapBuffers.GetOrCreateMutable(b).Span;

                for (int i = 0; i < sourceBuffer.Length; i++)
                {
                    bool isValid = BitUtility.IsValid(sourceNullBitMap, i);
                    TResult? value = func(isValid ? sourceBuffer[i] : null);
                    mutableResultBuffer[i] = value.GetValueOrDefault();
                    //Actually there is a bug in the previouse line. This code will not work correctly with containers having more than 1 buffers
                    //As buffer size for type T (sourceBuffer) is different from the size of buffer for type TResult (mutableResultBuffer) in case sizeof(T) not equal to sizeof(TResult)
                    //TODO fix (https://github.com/dotnet/machinelearning/issues/7122)
                    resultContainer.SetValidityBit(mutableResultNullBitMapBuffer, i, value != null);
                }
            }
        }

        public void FillNulls(T value)
        {

            for (int b = 0; b < Buffers.Count; b++)
            {
                var span = Buffers.GetOrCreateMutable(b).Span;
                var validitySpan = NullBitMapBuffers.GetOrCreateMutable(b).Span;

                for (int i = 0; i < span.Length; i++)
                {
                    if (BitUtility.IsValid(validitySpan, i))
                        continue;

                    span[i] = value;
                    BitUtility.SetBit(validitySpan, i, true);
                }
            }

            NullCount = 0;
        }

        public bool IsValid(long index) => NullCount == 0 || GetValidityBit(index);

        private byte SetBit(byte curBitMap, int index, bool value)
        {
            byte newBitMap;
            if (value)
            {
                newBitMap = (byte)(curBitMap | (byte)(1 << (index & 7))); //bit hack for index % 8
                if (BitUtility.IsBitClear(curBitMap, index) && index < Length && NullCount > 0)
                {
                    // Old value was null.
                    NullCount--;
                }
            }
            else
            {
                if (BitUtility.IsBitSet(curBitMap, index) && index < Length)
                {
                    // old value was NOT null and new value is null
                    NullCount++;
                }
                else if (index == Length)
                {
                    // New entry from an append
                    NullCount++;
                }
                newBitMap = (byte)(curBitMap & (byte)~(1 << (int)((uint)index & 7)));
            }
            return newBitMap;
        }

        // private function. Faster to use when we already have a span since it avoids indexing
        private void SetValidityBit(Span<byte> bitMapBufferSpan, int index, bool value)
        {
            int bitMapBufferIndex = (int)((uint)index / 8);
            Debug.Assert(bitMapBufferSpan.Length >= bitMapBufferIndex);
            byte curBitMap = bitMapBufferSpan[bitMapBufferIndex];
            byte newBitMap = SetBit(curBitMap, index, value);
            bitMapBufferSpan[bitMapBufferIndex] = newBitMap;
        }

        /// <summary>
        /// A null value has an unset bit
        /// A NON-null value has a set bit
        /// </summary>
        /// <param name="index"></param>
        /// <param name="value"></param>
        internal void SetValidityBit(long index, bool value)
        {
            if ((ulong)index > (ulong)Length)
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }
            // First find the right bitMapBuffer
            int bitMapIndex = (int)(index / ReadOnlyDataFrameBuffer<T>.MaxCapacity);
            Debug.Assert(NullBitMapBuffers.Count > bitMapIndex);
            DataFrameBuffer<byte> bitMapBuffer = (DataFrameBuffer<byte>)NullBitMapBuffers[bitMapIndex];

            // Set the bit
            index -= bitMapIndex * ReadOnlyDataFrameBuffer<T>.MaxCapacity;
            int bitMapBufferIndex = (int)((uint)index / 8);
            Debug.Assert(bitMapBuffer.Length >= bitMapBufferIndex);
            if (bitMapBuffer.Length == bitMapBufferIndex)
                bitMapBuffer.Append(0);
            SetValidityBit(bitMapBuffer.Span, (int)index, value);
        }

        private bool GetValidityBit(long index)
        {
            if ((uint)index >= Length)
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }
            // First find the right bitMapBuffer
            int bitMapIndex = (int)(index / ReadOnlyDataFrameBuffer<T>.MaxCapacity);
            Debug.Assert(NullBitMapBuffers.Count > bitMapIndex);
            ReadOnlyDataFrameBuffer<byte> bitMapBuffer = NullBitMapBuffers[bitMapIndex];

            // Get the bit
            index -= bitMapIndex * ReadOnlyDataFrameBuffer<T>.MaxCapacity;
            int bitMapBufferIndex = (int)((uint)index / 8);
            Debug.Assert(bitMapBuffer.Length > bitMapBufferIndex);
            byte curBitMap = bitMapBuffer[bitMapBufferIndex];
            return BitUtility.IsBitSet(curBitMap, (int)index);
        }

        public long Length { get; private set; }

        public long NullCount { get; private set; }

        public int GetIndexOfBufferContainingRowIndex(long rowIndex)
        {
            if (rowIndex >= Length)
            {
                throw new ArgumentOutOfRangeException(Strings.IndexIsGreaterThanColumnLength, nameof(rowIndex));
            }
            return (int)(rowIndex / ReadOnlyDataFrameBuffer<T>.MaxCapacity);
        }

        internal int MaxRecordBatchLength(long startIndex)
        {
            if (Length == 0)
                return 0;
            int bufferIndex = GetIndexOfBufferContainingRowIndex(startIndex);
            startIndex = startIndex - bufferIndex * ReadOnlyDataFrameBuffer<T>.MaxCapacity;
            return Buffers[bufferIndex].Length - (int)startIndex;
        }

        public IReadOnlyList<T?> this[long startIndex, int length]
        {
            get
            {
                var ret = new List<T?>(length);
                long endIndex = Math.Min(Length, startIndex + length);
                for (long i = startIndex; i < endIndex; i++)
                {
                    ret.Add(this[i]);
                }
                return ret;
            }
        }

        public T? this[long rowIndex]
        {
            get
            {
                if (!IsValid(rowIndex))
                {
                    return null;
                }
                int bufferIndex = GetIndexOfBufferContainingRowIndex(rowIndex);
                var bufferOffset = (int)(rowIndex % ReadOnlyDataFrameBuffer<T>.MaxCapacity);
                return Buffers[bufferIndex][bufferOffset];
            }
            set
            {
                int bufferIndex = GetIndexOfBufferContainingRowIndex(rowIndex);
                var bufferOffset = (int)(rowIndex % ReadOnlyDataFrameBuffer<T>.MaxCapacity);

                Buffers.GetOrCreateMutable(bufferIndex);
                NullBitMapBuffers.GetOrCreateMutable(bufferIndex);

                if (value.HasValue)
                {
                    Buffers[bufferIndex][bufferOffset] = value.Value;
                    SetValidityBit(rowIndex, true);
                }
                else
                {
                    Buffers[bufferIndex][bufferOffset] = default;
                    SetValidityBit(rowIndex, false);
                }
            }
        }

        public IEnumerator<T?> GetEnumerator()
        {
            for (long i = 0; i < Length; i++)
            {
                yield return this[i];
            }
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < Length; i++)
            {
                T? value = this[i];
                if (value.HasValue)
                {
                    sb.Append(this[i]).Append(" ");
                }
                else
                {
                    sb.Append("null").Append(" ");
                }
                // Can this run out of memory? Just being safe here
                if (sb.Length > 1000)
                {
                    sb.Append("...");
                    break;
                }
            }
            return sb.ToString();
        }

        private List<ReadOnlyDataFrameBuffer<byte>> CloneNullBitMapBuffers()
        {
            List<ReadOnlyDataFrameBuffer<byte>> ret = new List<ReadOnlyDataFrameBuffer<byte>>();
            foreach (ReadOnlyDataFrameBuffer<byte> buffer in NullBitMapBuffers)
            {
                DataFrameBuffer<byte> newBuffer = new DataFrameBuffer<byte>(buffer.ReadOnlyBuffer, buffer.Length);
                ret.Add(newBuffer);
            }
            return ret;
        }

        public PrimitiveColumnContainer<T> Clone<U>(PrimitiveColumnContainer<U> mapIndices, Type type, bool invertMapIndices = false)
            where U : unmanaged
        {
            PrimitiveColumnContainer<T> ret = new PrimitiveColumnContainer<T>(mapIndices.Length);

            if (Buffers.Count == 0)
            {
                return ret;
            }

            ReadOnlySpan<T> thisSpan = Buffers[0].ReadOnlySpan;
            ReadOnlySpan<byte> thisNullBitMapSpan = NullBitMapBuffers[0].ReadOnlySpan;
            long minRange = 0;
            long maxRange = DataFrameBuffer<T>.MaxCapacity;
            long maxCapacity = maxRange;
            for (int b = 0; b < mapIndices.Buffers.Count; b++)
            {
                int index = b;
                if (invertMapIndices)
                    index = mapIndices.Buffers.Count - 1 - b;

                ReadOnlyDataFrameBuffer<U> buffer = mapIndices.Buffers[index];
                ReadOnlySpan<byte> mapIndicesNullBitMapSpan = mapIndices.NullBitMapBuffers[index].ReadOnlySpan;
                ReadOnlySpan<U> mapIndicesSpan = buffer.ReadOnlySpan;
                ReadOnlySpan<long> mapIndicesLongSpan = default;
                ReadOnlySpan<int> mapIndicesIntSpan = default;
                DataFrameBuffer<T> mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(ret.Buffers[index]);
                ret.Buffers[index] = mutableBuffer;
                Span<T> retSpan = mutableBuffer.Span;
                DataFrameBuffer<byte> mutableNullBuffer = DataFrameBuffer<byte>.GetMutableBuffer(ret.NullBitMapBuffers[index]);
                ret.NullBitMapBuffers[index] = mutableNullBuffer;
                Span<byte> retNullBitMapSpan = mutableNullBuffer.Span;
                if (type == typeof(long))
                {
                    mapIndicesLongSpan = MemoryMarshal.Cast<U, long>(mapIndicesSpan);
                }
                if (type == typeof(int))
                {
                    mapIndicesIntSpan = MemoryMarshal.Cast<U, int>(mapIndicesSpan);
                }
                for (int i = 0; i < buffer.Length; i++)
                {
                    int spanIndex = i;
                    if (invertMapIndices)
                        spanIndex = buffer.Length - 1 - i;

                    long mapRowIndex = mapIndicesIntSpan.IsEmpty ? mapIndicesLongSpan[spanIndex] : mapIndicesIntSpan[spanIndex];
                    bool mapRowIndexIsValid = BitUtility.IsValid(mapIndicesNullBitMapSpan, spanIndex);
                    if (mapRowIndexIsValid && (mapRowIndex < minRange || mapRowIndex >= maxRange))
                    {
                        int bufferIndex = (int)(mapRowIndex / maxCapacity);
                        thisSpan = Buffers[bufferIndex].ReadOnlySpan;
                        thisNullBitMapSpan = NullBitMapBuffers[bufferIndex].ReadOnlySpan;
                        minRange = bufferIndex * maxCapacity;
                        maxRange = (bufferIndex + 1) * maxCapacity;
                    }
                    T value = default;
                    bool isValid = false;
                    if (mapRowIndexIsValid)
                    {
                        mapRowIndex -= minRange;
                        value = thisSpan[(int)mapRowIndex];
                        isValid = BitUtility.IsValid(thisNullBitMapSpan, (int)mapRowIndex);
                    }

                    retSpan[i] = isValid ? value : default;
                    ret.SetValidityBit(retNullBitMapSpan, i, isValid);
                }
            }
            return ret;
        }

        public PrimitiveColumnContainer<T> Clone()
        {
            var ret = new PrimitiveColumnContainer<T>();
            foreach (ReadOnlyDataFrameBuffer<T> buffer in Buffers)
            {
                DataFrameBuffer<T> newBuffer = new DataFrameBuffer<T>(buffer.ReadOnlyBuffer, buffer.Length);
                ret.Buffers.Add(newBuffer);
                ret.Length += buffer.Length;
            }
            ret.NullBitMapBuffers = CloneNullBitMapBuffers();
            ret.NullCount = NullCount;
            return ret;
        }

        // The values of the source buffers are redistributed so that every buffer of the result (except the last one)
        // holds exactly ReadOnlyDataFrameBuffer<U>.MaxCapacity values, as sizeof(U) may differ from sizeof(T)
        private PrimitiveColumnContainer<U> CloneAs<U>(Func<T, U> convert)
            where U : unmanaged
        {
            var ret = new PrimitiveColumnContainer<U>();
            int maxCapacity = ReadOnlyDataFrameBuffer<U>.MaxCapacity;
            long remaining = Length;
            DataFrameBuffer<U> newBuffer = null;
            DataFrameBuffer<byte> newNullBitMapBuffer = null;

            for (int b = 0; b < Buffers.Count; b++)
            {
                ReadOnlySpan<T> span = Buffers[b].ReadOnlySpan;
                ReadOnlySpan<byte> nullBitMapSpan = NullBitMapBuffers[b].ReadOnlySpan;
                int index = 0;
                while (index < span.Length)
                {
                    if (newBuffer == null || newBuffer.Length == maxCapacity)
                    {
                        int capacity = (int)Math.Min(remaining, maxCapacity);
                        newBuffer = new DataFrameBuffer<U>(capacity);
                        newNullBitMapBuffer = new DataFrameBuffer<byte>((capacity + 7) / 8);
                        newNullBitMapBuffer.IncreaseSize((capacity + 7) / 8);
                        ret.Buffers.Add(newBuffer);
                        ret.NullBitMapBuffers.Add(newNullBitMapBuffer);
                    }

                    int newIndex = newBuffer.Length;
                    int count = Math.Min(span.Length - index, maxCapacity - newIndex);
                    newBuffer.IncreaseSize(count);
                    Span<U> newSpan = newBuffer.Span;
                    Span<byte> newNullBitMapSpan = newNullBitMapBuffer.Span;
                    for (int i = 0; i < count; i++)
                    {
                        newSpan[newIndex + i] = convert(span[index + i]);
                    }

                    if (NullCount == 0)
                    {
                        BitUtility.SetBits(newNullBitMapSpan, newIndex, count, true);
                    }
                    else
                    {
                        for (int i = 0; i < count; i++)
                        {
                            BitUtility.SetBit(newNullBitMapSpan, newIndex + i, BitUtility.IsValid(nullBitMapSpan, index + i));
                        }
                    }

                    index += count;
                    remaining -= count;
                }
            }
            ret.Length = Length;
            ret.NullCount = NullCount;
            return ret;
        }

        internal PrimitiveColumnContainer<bool> CloneAsBoolContainer()
        {
            return CloneAs<bool>(value => typeof(T) == typeof(bool) && Unsafe.As<T, bool>(ref value));
        }

        internal PrimitiveColumnContainer<byte> CloneAsByteContainer()
        {
            return CloneAs<byte>(value => ByteConverter<T>.Instance.GetByte(value));
        }

        internal PrimitiveColumnContainer<sbyte> CloneAsSByteContainer()
        {
            return CloneAs<sbyte>(value => SByteConverter<T>.Instance.GetSByte(value));
        }

        internal PrimitiveColumnContainer<double> CloneAsDoubleContainer()
        {
            return CloneAs<double>(value => DoubleConverter<T>.Instance.GetDouble(value));
        }

        internal PrimitiveColumnContainer<decimal> CloneAsDecimalContainer()
        {
            return CloneAs<decimal>(value => DecimalConverter<T>.Instance.GetDecimal(value));
        }

        internal PrimitiveColumnContainer<short> CloneAsShortContainer()
        {
            return CloneAs<short>(value => Int16Converter<T>.Instance.GetInt16(value));
        }

        internal PrimitiveColumnContainer<ushort> CloneAsUShortContainer()
        {
            return CloneAs<ushort>(value => UInt16Converter<T>.Instance.GetUInt16(value));
        }

        internal PrimitiveColumnContainer<int> CloneAsIntContainer()
        {
            return CloneAs<int>(value => Int32Converter<T>.Instance.GetInt32(value));
        }

        internal PrimitiveColumnContainer<uint> CloneAsUIntContainer()
        {
            return CloneAs<uint>(value => UInt32Converter<T>.Instance.GetUInt32(value));
        }

        internal PrimitiveColumnContainer<long> CloneAsLongContainer()
        {
            return CloneAs<long>(value => Int64Converter<T>.Instance.GetInt64(value));
        }

        internal PrimitiveColumnContainer<ulong> CloneAsULongContainer()
        {
            return CloneAs<ulong>(value => UInt64Converter<T>.Instance.GetUInt64(value));
        }

        internal PrimitiveColumnContainer<float> CloneAsFloatContainer()
        {
            return CloneAs<float>(value => SingleConverter<T>.Instance.GetSingle(value));
        }
    }
}
