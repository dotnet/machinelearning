// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using Apache.Arrow;
using Apache.Arrow.Types;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Microsoft.Data.Analysis
{
    /// <summary>
    /// An immutable column to hold Arrow style strings
    /// </summary>
    public partial class ArrowStringDataFrameColumn : DataFrameColumn, IEnumerable<string>
    {
        private IList<ReadOnlyDataFrameBuffer<byte>> _dataBuffers;
        private IList<ReadOnlyDataFrameBuffer<int>> _offsetsBuffers;
        private IList<ReadOnlyDataFrameBuffer<byte>> _nullBitMapBuffers;

        /// <summary>
        /// Constructs an empty <see cref="ArrowStringDataFrameColumn"/> with the given <paramref name="name"/>.
        /// </summary>
        /// <param name="name">The name of the column.</param>
        public ArrowStringDataFrameColumn(string name) : base(name, 0, typeof(string))
        {
            _dataBuffers = new List<ReadOnlyDataFrameBuffer<byte>>();
            _offsetsBuffers = new List<ReadOnlyDataFrameBuffer<int>>();
            _nullBitMapBuffers = new List<ReadOnlyDataFrameBuffer<byte>>();
        }

        /// <summary>
        /// Constructs an <see cref="ArrowStringDataFrameColumn"/> with the given <paramref name="name"/>, <paramref name="length"/> and <paramref name="nullCount"/>. The <paramref name="values"/>, <paramref name="offsets"/> and <paramref name="nullBits"/> are the contents of the column in the Arrow format.
        /// </summary>
        /// <param name="name">The name of the column.</param>
        /// <param name="values">The Arrow formatted string values in this column.</param>
        /// <param name="offsets">The Arrow formatted offets in this column.</param>
        /// <param name="nullBits">The Arrow formatted null bits in this column.</param>
        /// <param name="length">The length of the column.</param>
        /// <param name="nullCount">The number of <see langword="null" /> values in this column.</param>
        public ArrowStringDataFrameColumn(string name, ReadOnlyMemory<byte> values, ReadOnlyMemory<byte> offsets, ReadOnlyMemory<byte> nullBits, int length, int nullCount) : base(name, length, typeof(string))
        {
            ReadOnlyDataFrameBuffer<byte> dataBuffer = new ReadOnlyDataFrameBuffer<byte>(values, values.Length);
            ReadOnlyDataFrameBuffer<int> offsetBuffer = new ReadOnlyDataFrameBuffer<int>(offsets, length + 1);
            ReadOnlyDataFrameBuffer<byte> nullBitMapBuffer = new ReadOnlyDataFrameBuffer<byte>(nullBits, nullBits.Length);

            if (length + 1 != offsetBuffer.Length)
                throw new ArgumentException(nameof(offsetBuffer));

            _dataBuffers = new List<ReadOnlyDataFrameBuffer<byte>>();
            _offsetsBuffers = new List<ReadOnlyDataFrameBuffer<int>>();
            _nullBitMapBuffers = new List<ReadOnlyDataFrameBuffer<byte>>();

            _dataBuffers.Add(dataBuffer);
            _offsetsBuffers.Add(offsetBuffer);
            _nullBitMapBuffers.Add(nullBitMapBuffer);

            _nullCount = nullCount;

        }

        private long _nullCount;

        /// <inheritdoc/>
        public override long NullCount => _nullCount;

        /// <summary>
        /// Indicates if the value at this <paramref name="index"/> is <see langword="null" />.
        /// </summary>
        /// <param name="index">The index to look up.</param>
        /// <returns>A boolean value indicating the validity at this <paramref name="index"/>.</returns>
        public bool IsValid(long index) => NullCount == 0 || GetValidityBit(index);

        private bool GetValidityBit(long index)
        {
            if ((ulong)index > (ulong)Length)
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }
            // First find the right bitMapBuffer
            int bitMapIndex = GetBufferIndexContainingRowIndex(index, out int indexInBuffer);
            Debug.Assert(_nullBitMapBuffers.Count > bitMapIndex);
            ReadOnlyDataFrameBuffer<byte> bitMapBuffer = _nullBitMapBuffers[bitMapIndex];
            int bitMapBufferIndex = (int)((uint)index / 8);
            Debug.Assert(bitMapBuffer.Length > bitMapBufferIndex);
            byte curBitMap = bitMapBuffer[bitMapBufferIndex];
            return ((curBitMap >> (indexInBuffer & 7)) & 1) != 0;
        }

        private void SetValidityBit(long index, bool value)
        {
            if ((ulong)index > (ulong)Length)
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }
            // First find the right bitMapBuffer
            int bitMapIndex = GetBufferIndexContainingRowIndex(index, out int indexInBuffer);
            Debug.Assert(_nullBitMapBuffers.Count > bitMapIndex);
            DataFrameBuffer<byte> bitMapBuffer = (DataFrameBuffer<byte>)_nullBitMapBuffers[bitMapIndex];

            // Set the bit
            int bitMapBufferIndex = (int)((uint)indexInBuffer / 8);
            Debug.Assert(bitMapBuffer.Length >= bitMapBufferIndex);
            if (bitMapBuffer.Length == bitMapBufferIndex)
                bitMapBuffer.Append(0);
            byte curBitMap = bitMapBuffer[bitMapBufferIndex];
            byte newBitMap;
            if (value)
            {
                newBitMap = (byte)(curBitMap | (byte)(1 << (indexInBuffer & 7))); //bit hack for index % 8
                if (((curBitMap >> (indexInBuffer & 7)) & 1) == 0 && indexInBuffer < Length - 1 && NullCount > 0)
                {
                    // Old value was null.
                    _nullCount--;
                }
            }
            else
            {
                if (((curBitMap >> (indexInBuffer & 7)) & 1) == 1 && indexInBuffer < Length)
                {
                    // old value was NOT null and new value is null
                    _nullCount++;
                }
                else if (indexInBuffer == Length - 1)
                {
                    // New entry from an append
                    _nullCount++;
                }
                newBitMap = (byte)(curBitMap & (byte)~(1 << (int)((uint)indexInBuffer & 7)));
            }
            bitMapBuffer[bitMapBufferIndex] = newBitMap;
        }

        /// <summary>
        /// Returns an enumeration of immutable buffers representing the underlying values in the Apache Arrow format
        /// </summary>
        /// <remarks><see langword="null" /> values are encoded in the buffers returned by GetReadOnlyNullBitmapBuffers in the Apache Arrow format</remarks>
        /// <remarks>The offsets buffers returned by GetReadOnlyOffsetBuffers can be used to delineate each value</remarks>
        /// <returns>An enumeration of <see cref="ReadOnlyMemory{Byte}"/> whose elements are the raw data buffers for the UTF8 string values.</returns>
        public IEnumerable<ReadOnlyMemory<byte>> GetReadOnlyDataBuffers()
        {
            for (int i = 0; i < _dataBuffers.Count; i++)
            {
                ReadOnlyDataFrameBuffer<byte> buffer = _dataBuffers[i];
                yield return buffer.RawReadOnlyMemory;
            }
        }

        /// <summary>
        /// Returns an enumeration of immutable <see cref="ReadOnlyMemory{Byte}"/> buffers representing <see langword="null" /> values in the Apache Arrow format
        /// </summary>
        /// <remarks>Each <see cref="ReadOnlyMemory{Byte}"/> encodes the indices of <see langword="null" /> values in its corresponding Data buffer</remarks>
        /// <returns>An enumeration of <see cref="ReadOnlyMemory{Byte}"/> objects whose elements encode the null bit maps for the column's values</returns>
        public IEnumerable<ReadOnlyMemory<byte>> GetReadOnlyNullBitMapBuffers()
        {
            for (int i = 0; i < _nullBitMapBuffers.Count; i++)
            {
                ReadOnlyDataFrameBuffer<byte> buffer = _nullBitMapBuffers[i];
                yield return buffer.RawReadOnlyMemory;
            }
        }

        /// <summary>
        /// Returns an enumeration of immutable <see cref="ReadOnlyMemory{Int32}"/> representing offsets into its corresponding Data buffer.
        /// The Apache Arrow format specifies how the offset buffer encodes the length of each value in the Data buffer
        /// </summary>
        /// <returns>An enumeration of <see cref="ReadOnlyMemory{Int32}"/> objects.</returns>
        public IEnumerable<ReadOnlyMemory<int>> GetReadOnlyOffsetsBuffers()
        {
            for (int i = 0; i < _offsetsBuffers.Count; i++)
            {
                ReadOnlyDataFrameBuffer<int> buffer = _offsetsBuffers[i];
                yield return buffer.ReadOnlyMemory;
            }
        }

        // This is an immutable column, however this method exists to support Clone(). Keep this method private
        // Appending a default string is equivalent to appending null. It increases the NullCount and sets a null bitmap bit
        // Appending an empty string is valid. It does NOT affect the NullCount. It instead adds a new offset entry
        private void Append(ReadOnlySpan<byte> value)
        {
            if (_dataBuffers.Count == 0)
            {
                _dataBuffers.Add(new DataFrameBuffer<byte>());
                _nullBitMapBuffers.Add(new DataFrameBuffer<byte>());
                _offsetsBuffers.Add(new DataFrameBuffer<int>());
            }
            DataFrameBuffer<int> mutableOffsetsBuffer = (DataFrameBuffer<int>)_offsetsBuffers[_offsetsBuffers.Count - 1];
            if (mutableOffsetsBuffer.Length == 0)
            {
                mutableOffsetsBuffer.Append(0);
            }
            Length++;
            if (value == default)
            {
                mutableOffsetsBuffer.Append(mutableOffsetsBuffer[mutableOffsetsBuffer.Length - 1]);
            }
            else
            {
                DataFrameBuffer<byte> mutableDataBuffer = (DataFrameBuffer<byte>)_dataBuffers[_dataBuffers.Count - 1];
                if (mutableDataBuffer.Length == ReadOnlyDataFrameBuffer<byte>.MaxCapacity)
                {
                    mutableDataBuffer = new DataFrameBuffer<byte>();
                    _dataBuffers.Add(mutableDataBuffer);
                    _nullBitMapBuffers.Add(new DataFrameBuffer<byte>());
                    mutableOffsetsBuffer = new DataFrameBuffer<int>();
                    _offsetsBuffers.Add(mutableOffsetsBuffer);
                    mutableOffsetsBuffer.Append(0);
                }
                mutableDataBuffer.EnsureCapacity(value.Length);
                value.CopyTo(mutableDataBuffer.RawSpan.Slice(mutableDataBuffer.Length));
                mutableDataBuffer.Length += value.Length;
                mutableOffsetsBuffer.Append(mutableOffsetsBuffer[mutableOffsetsBuffer.Length - 1] + value.Length);
            }
            SetValidityBit(Length - 1, value != default);

        }

        private int GetBufferIndexContainingRowIndex(long rowIndex, out int indexInBuffer)
        {
            if (rowIndex >= Length)
            {
                throw new ArgumentOutOfRangeException(Strings.ColumnIndexOutOfRange, nameof(rowIndex));
            }

            // Since the strings here could be of variable length, scan linearly
            int curArrayIndex = 0;
            int numBuffers = _offsetsBuffers.Count;
            while (curArrayIndex < numBuffers && rowIndex > _offsetsBuffers[curArrayIndex].Length - 1)
            {
                rowIndex -= _offsetsBuffers[curArrayIndex].Length - 1;
                curArrayIndex++;
            }
            indexInBuffer = (int)rowIndex;
            return curArrayIndex;
        }

        private ReadOnlySpan<byte> GetBytes(long index)
        {
            int offsetsBufferIndex = GetBufferIndexContainingRowIndex(index, out int indexInBuffer);
            ReadOnlySpan<int> offsetBufferSpan = _offsetsBuffers[offsetsBufferIndex].ReadOnlySpan;
            int currentOffset = offsetBufferSpan[indexInBuffer];
            int nextOffset = offsetBufferSpan[indexInBuffer + 1];
            int numberOfBytes = nextOffset - currentOffset;
            return _dataBuffers[offsetsBufferIndex].ReadOnlySpan.Slice(currentOffset, numberOfBytes);
        }

        /// <inheritdoc/>
        protected override object GetValue(long rowIndex) => GetValueImplementation(rowIndex);

        private string GetValueImplementation(long rowIndex)
        {
            if (!IsValid(rowIndex))
            {
                return null;
            }
            var bytes = GetBytes(rowIndex);
            unsafe
            {
                fixed (byte* data = &MemoryMarshal.GetReference(bytes))
                    return Encoding.UTF8.GetString(data, bytes.Length);
            }
        }

        /// <inheritdoc/>
        protected override IReadOnlyList<object> GetValues(long startIndex, int length)
        {
            var ret = new List<object>();
            while (ret.Count < length)
            {
                ret.Add(GetValueImplementation(startIndex++));
            }
            return ret;
        }

        /// <inheritdoc/>
        protected override void SetValue(long rowIndex, object value) => throw new NotSupportedException(Strings.ImmutableColumn);


        /// <summary>
        /// Indexer to get values. This is an immutable column
        /// </summary>
        /// <param name="rowIndex">Zero based row index</param>
        /// <returns>The value stored at this <paramref name="rowIndex"/></returns>
        public new string this[long rowIndex]
        {
            get => GetValueImplementation(rowIndex);
            set => throw new NotSupportedException(Strings.ImmutableColumn);
        }

        /// <summary>
        /// Returns <paramref name="length"/> number of values starting from <paramref name="startIndex"/>.
        /// </summary>
        /// <param name="startIndex">The index of the first value to return.</param>
        /// <param name="length">The number of values to return starting from <paramref name="startIndex"/></param>
        /// <returns>A new list of string values</returns>
        public new List<string> this[long startIndex, int length]
        {
            get
            {
                var ret = new List<string>();
                while (ret.Count < length)
                {
                    ret.Add(GetValueImplementation(startIndex++));
                }
                return ret;
            }
        }

        /// <summary>
        /// Returns an enumerator that iterates through the string values in this column.
        /// </summary>
        public IEnumerator<string> GetEnumerator()
        {
            for (long i = 0; i < Length; i++)
            {
                yield return this[i];
            }
        }

        /// <inheritdoc/>
        protected override IEnumerator GetEnumeratorCore() => GetEnumerator();

        /// <inheritdoc/>
        protected internal override Field GetArrowField() => new Field(Name, StringType.Default, NullCount != 0);

        /// <inheritdoc/>
        protected internal override int GetMaxRecordBatchLength(long startIndex)
        {
            if (Length == 0)
                return 0;
            int offsetsBufferIndex = GetBufferIndexContainingRowIndex(startIndex, out int indexInBuffer);
            Debug.Assert(indexInBuffer <= Int32.MaxValue);
            return _offsetsBuffers[offsetsBufferIndex].Length - indexInBuffer;
        }

        private int GetNullCount(long startIndex, int numberOfRows)
        {
            int nullCount = 0;
            for (long i = startIndex; i < numberOfRows; i++)
            {
                if (!IsValid(i))
                    nullCount++;
            }
            return nullCount;
        }

        /// <inheritdoc/>
        protected internal override Apache.Arrow.Array ToArrowArray(long startIndex, int numberOfRows)
        {
            if (numberOfRows == 0)
                return new StringArray(numberOfRows, ArrowBuffer.Empty, ArrowBuffer.Empty, ArrowBuffer.Empty);
            int offsetsBufferIndex = GetBufferIndexContainingRowIndex(startIndex, out int indexInBuffer);
            if (numberOfRows != 0 && numberOfRows > _offsetsBuffers[offsetsBufferIndex].Length - 1 - indexInBuffer)
            {
                throw new ArgumentException(Strings.SpansMultipleBuffers, nameof(numberOfRows));
            }
            ArrowBuffer dataBuffer = new ArrowBuffer(_dataBuffers[offsetsBufferIndex].ReadOnlyBuffer);
            ArrowBuffer offsetsBuffer = new ArrowBuffer(_offsetsBuffers[offsetsBufferIndex].ReadOnlyBuffer);
            ArrowBuffer nullBuffer = new ArrowBuffer(_nullBitMapBuffers[offsetsBufferIndex].ReadOnlyBuffer);
            int nullCount = GetNullCount(indexInBuffer, numberOfRows);
            return new StringArray(numberOfRows, offsetsBuffer, dataBuffer, nullBuffer, nullCount, indexInBuffer);
        }

        /// <inheritdoc/>
        public override DataFrameColumn Sort(bool ascending = true) => throw new NotSupportedException();

        /// <inheritdoc/>
        public override DataFrameColumn Clone(DataFrameColumn mapIndices = null, bool invertMapIndices = false, long numberOfNullsToAppend = 0)
        {
            ArrowStringDataFrameColumn clone;
            if (!(mapIndices is null))
            {
                Type dataType = mapIndices.DataType;
                if (dataType != typeof(long) && dataType != typeof(int) && dataType != typeof(bool))
                    throw new ArgumentException(String.Format(Strings.MultipleMismatchedValueType, typeof(long), typeof(int), typeof(bool)), nameof(mapIndices));
                if (mapIndices.DataType == typeof(long))
                    clone = Clone(mapIndices as PrimitiveDataFrameColumn<long>, invertMapIndices);
                else if (dataType == typeof(int))
                    clone = Clone(mapIndices as PrimitiveDataFrameColumn<int>, invertMapIndices);
                else
                    clone = Clone(mapIndices as PrimitiveDataFrameColumn<bool>);
            }
            else
            {
                clone = Clone();
            }
            for (long i = 0; i < numberOfNullsToAppend; i++)
            {
                clone.Append(default);
            }
            return clone;
        }

        private ArrowStringDataFrameColumn Clone(PrimitiveDataFrameColumn<bool> boolColumn)
        {
            if (boolColumn.Length > Length)
                throw new ArgumentException(Strings.MapIndicesExceedsColumnLenth, nameof(boolColumn));
            ArrowStringDataFrameColumn ret = new ArrowStringDataFrameColumn(Name);
            for (long i = 0; i < boolColumn.Length; i++)
            {
                bool? value = boolColumn[i];
                if (value == true)
                    ret.Append(IsValid(i) ? GetBytes(i) : default(ReadOnlySpan<byte>));
            }
            return ret;
        }

        private ArrowStringDataFrameColumn CloneImplementation<U>(PrimitiveDataFrameColumn<U> mapIndices, bool invertMapIndices = false)
            where U : unmanaged
        {
            ArrowStringDataFrameColumn ret = new ArrowStringDataFrameColumn(Name);
            mapIndices.ApplyElementwise((U? mapIndex, long rowIndex) =>
            {
                if (mapIndex == null)
                {
                    ret.Append(default);
                    return mapIndex;
                }
                if (invertMapIndices)
                {
                    long index = mapIndices.Length - 1 - rowIndex;
                    ret.Append(IsValid(index) ? GetBytes(index) : default(ReadOnlySpan<byte>));
                }
                else
                {
                    ret.Append(IsValid(rowIndex) ? GetBytes(rowIndex) : default(ReadOnlySpan<byte>));
                }
                return mapIndex;
            });
            return ret;
        }

        private ArrowStringDataFrameColumn Clone(PrimitiveDataFrameColumn<long> mapIndices = null, bool invertMapIndex = false)
        {
            if (mapIndices is null)
            {
                ArrowStringDataFrameColumn ret = new ArrowStringDataFrameColumn(Name);
                for (long i = 0; i < Length; i++)
                {
                    ret.Append(IsValid(i) ? GetBytes(i) : default(ReadOnlySpan<byte>));
                }
                return ret;
            }
            else
                return CloneImplementation(mapIndices, invertMapIndex);
        }

        private ArrowStringDataFrameColumn Clone(PrimitiveDataFrameColumn<int> mapIndices, bool invertMapIndex = false)
        {
            return CloneImplementation(mapIndices, invertMapIndex);
        }

        /// <inheritdoc/>
        public override DataFrame ValueCounts()
        {
            Dictionary<string, ICollection<long>> groupedValues = GroupColumnValues<string>();
            return StringDataFrameColumn.ValueCountsImplementation(groupedValues);
        }

        /// <inheritdoc/>
        public override GroupBy GroupBy(int columnIndex, DataFrame parent)
        {
            Dictionary<string, ICollection<long>> dictionary = GroupColumnValues<string>();
            return new GroupBy<string>(parent, columnIndex, dictionary);
        }

        /// <inheritdoc/>
        public override Dictionary<TKey, ICollection<long>> GroupColumnValues<TKey>()
        {
            if (typeof(TKey) == typeof(string))
            {
                Dictionary<string, ICollection<long>> multimap = new Dictionary<string, ICollection<long>>(EqualityComparer<string>.Default);
                for (long i = 0; i < Length; i++)
                {
                    string str = this[i] ?? "__null__";
                    bool containsKey = multimap.TryGetValue(str, out ICollection<long> values);
                    if (containsKey)
                    {
                        values.Add(i);
                    }
                    else
                    {
                        multimap.Add(str, new List<long>() { i });
                    }
                }
                return multimap as Dictionary<TKey, ICollection<long>>;
            }
            else
            {
                throw new NotSupportedException(nameof(TKey));
            }
        }

        /// <inheritdoc/>
        public ArrowStringDataFrameColumn FillNulls(string value, bool inPlace = false) 
        {
            if (value == null)
            {
                throw new ArgumentException(nameof(value));
            }
            if (inPlace)
            {
                /* For now throw an exception if inPlace = true. Need to investigate if Apache Arrow
                 * format supports filling nulls for variable length arrays
                 */
                throw new NotSupportedException();
            }

            ArrowStringDataFrameColumn ret = new ArrowStringDataFrameColumn(Name);
            for (long i = 0; i < Length; i++)
            {
                ret.Append(IsValid(i) ? GetBytes(i) : Encoding.UTF8.GetBytes(value));
            }
            return ret;
        }

        protected override DataFrameColumn FillNullsImplementation(object value, bool inPlace)
        {
            if (value is string valueString)
            {
                return FillNulls(valueString, inPlace);
            }
            else
            {
                throw new ArgumentException(String.Format(Strings.MismatchedValueType, typeof(string)), nameof(value));
            }
        }

        public override DataFrameColumn Clamp<U>(U min, U max, bool inPlace = false) => throw new NotSupportedException();

        public override DataFrameColumn Filter<U>(U min, U max) => throw new NotSupportedException();

        /// <inheritdoc/>
        protected internal override void AddDataViewColumn(DataViewSchema.Builder builder)
        {
            builder.AddColumn(Name, TextDataViewType.Instance);
        }

        /// <inheritdoc/>
        protected internal override Delegate GetDataViewGetter(DataViewRowCursor cursor)
        {
            return CreateValueGetterDelegate(cursor);
        }

        private ValueGetter<ReadOnlyMemory<char>> CreateValueGetterDelegate(DataViewRowCursor cursor) =>
            (ref ReadOnlyMemory<char> value) => value = this[cursor.Position].AsMemory();

        /// <summary>
        /// Returns a boolean column that is the result of an elementwise equality comparison of each value in the column with <paramref name="value"/>
        /// </summary>
        public PrimitiveDataFrameColumn<bool> ElementwiseEquals(string value)
        {
            ReadOnlySpan<byte> bytes = value != null ? Encoding.UTF8.GetBytes(value) : default(ReadOnlySpan<byte>);
            PrimitiveDataFrameColumn<bool> ret = new PrimitiveDataFrameColumn<bool>(Name, Length);
            if (value == null)
            {
                for (long i = 0; i < Length; i++)
                {
                    ret[i] = !IsValid(i);
                }
            }
            else
            {
                for (long i = 0; i < Length; i++)
                {
                    var strBytes = GetBytes(i);
                    ret[i] = strBytes.SequenceEqual(bytes);
                }
            }
            return ret;
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ElementwiseEquals<T>(T value)
        {
            if (value is DataFrameColumn column)
            {
                return ElementwiseEquals(column);
            }
            return ElementwiseEquals(value.ToString());
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ElementwiseEquals(DataFrameColumn column)
        {
            return StringDataFrameColumn.ElementwiseEqualsImplementation(this, column);
        }

        /// <summary>
        /// Returns a boolean column that is the result of an elementwise not-equal comparison of each value in the column with <paramref name="value"/>
        /// </summary>
        public PrimitiveDataFrameColumn<bool> ElementwiseNotEquals(string value)
        {
            ReadOnlySpan<byte> bytes = value != null ? Encoding.UTF8.GetBytes(value) : default(ReadOnlySpan<byte>);
            PrimitiveDataFrameColumn<bool> ret = new PrimitiveDataFrameColumn<bool>(Name, Length);
            if (value == null)
            {
                for (long i = 0; i < Length; i++)
                {
                    ret[i] = IsValid(i);
                }
            }
            else
            {
                for (long i = 0; i < Length; i++)
                {
                    var strBytes = GetBytes(i);
                    ret[i] = !strBytes.SequenceEqual(bytes);
                }
            }
            return ret;
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ElementwiseNotEquals<T>(T value)
        {
            if (value is DataFrameColumn column)
            {
                return ElementwiseNotEquals(column);
            }
            return ElementwiseNotEquals(value.ToString());
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ElementwiseNotEquals(DataFrameColumn column)
        {
            return StringDataFrameColumn.ElementwiseNotEqualsImplementation(this, column);
        }

        /// <summary>
        /// Applies a function to all the values
        /// </summary>
        /// <param name="func">The function to apply</param>
        /// <returns>A <see cref="ArrowStringDataFrameColumn"/> containing the new string values</returns>
        /// <remarks>This function converts from UTF-8 to UTF-16 strings</remarks>
        public ArrowStringDataFrameColumn Apply(Func<string, string> func)
        {
            ArrowStringDataFrameColumn ret = new ArrowStringDataFrameColumn(Name);
            Encoding encoding = Encoding.UTF8;
            for (long i = 0; i < Length; i++)
            {
                string cur = this[i];
                string funcResult = func(cur);
                ret.Append(funcResult != null ? encoding.GetBytes(funcResult) : default(ReadOnlySpan<byte>));
            }
            return ret;
        }
    }
}
