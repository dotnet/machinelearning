﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Microsoft.Data.Analysis
{
    /// <summary>
    /// A mutable column to hold strings
    /// </summary>
    /// <remarks> Is NOT Arrow compatible </remarks>
    public partial class StringDataFrameColumn : DataFrameColumn, IEnumerable<string>
    {
        public static int MaxCapacity = ArrayUtility.ArrayMaxSize / Unsafe.SizeOf<IntPtr>(); // Max Size in bytes / size of pointer (8 bytes on x64)

        private readonly List<List<string>> _stringBuffers = new List<List<string>>(); // To store more than intMax number of strings

        public StringDataFrameColumn(string name, long length = 0) : base(name, length, typeof(string))
        {
            int numberOfBuffersRequired = (int)(length / MaxCapacity + 1);
            for (int i = 0; i < numberOfBuffersRequired; i++)
            {
                long bufferLen = length - _stringBuffers.Count * MaxCapacity;
                List<string> buffer = new List<string>((int)Math.Min(MaxCapacity, bufferLen));
                _stringBuffers.Add(buffer);
                for (int j = 0; j < bufferLen; j++)
                {
                    buffer.Add(default);
                }
            }
            _nullCount = length;
        }

        public StringDataFrameColumn(string name, IEnumerable<string> values) : base(name, 0, typeof(string))
        {
            values = values ?? throw new ArgumentNullException(nameof(values));
            if (_stringBuffers.Count == 0)
            {
                _stringBuffers.Add(new List<string>());
            }
            foreach (var value in values)
            {
                Append(value);
            }
        }

        private long _nullCount;
        public override long NullCount => _nullCount;

        protected internal override void Resize(long length)
        {
            if (length < Length)
                throw new ArgumentException(Strings.CannotResizeDown, nameof(length));

            for (long i = Length; i < length; i++)
            {
                Append(null);
            }
        }

        public void Append(string value)
        {
            List<string> lastBuffer = _stringBuffers[_stringBuffers.Count - 1];
            if (lastBuffer.Count == MaxCapacity)
            {
                lastBuffer = new List<string>();
                _stringBuffers.Add(lastBuffer);
            }
            lastBuffer.Add(value);
            if (value == null)
                _nullCount++;
            Length++;
        }

        /// <summary>
        /// Applies a function to all values in the column, that are not null.
        /// </summary>
        /// <param name="func">The function to apply.</param>
        /// /// <param name="inPlace">A boolean flag to indicate if the operation should be in place.</param>
        /// <returns>A new <see cref="PrimitiveDataFrameColumn{T}"/> if <paramref name="inPlace"/> is not set. Returns this column otherwise.</returns>
        public StringDataFrameColumn Apply(Func<string, string> func, bool inPlace = false)
        {
            var column = inPlace ? this : Clone();

            for (long i = 0; i < column.Length; i++)
            {
                var value = column[i];

                if (value != null)
                    column[i] = func(value);
            }

            return column;
        }

        private int GetBufferIndexContainingRowIndex(long rowIndex)
        {
            if (rowIndex >= Length)
            {
                throw new ArgumentOutOfRangeException(Strings.IndexIsGreaterThanColumnLength, nameof(rowIndex));
            }
            return (int)(rowIndex / MaxCapacity);
        }

        protected override object GetValue(long rowIndex)
        {
            int bufferIndex = GetBufferIndexContainingRowIndex(rowIndex);
            return _stringBuffers[bufferIndex][(int)(rowIndex % MaxCapacity)];
        }

        protected override IReadOnlyList<object> GetValues(long startIndex, int length)
        {
            var ret = new List<object>();
            int bufferIndex = GetBufferIndexContainingRowIndex(startIndex);
            int bufferOffset = (int)(startIndex % MaxCapacity);
            while (ret.Count < length && bufferIndex < _stringBuffers.Count)
            {
                for (int i = bufferOffset; ret.Count < length && i < _stringBuffers[bufferIndex].Count; i++)
                {
                    ret.Add(_stringBuffers[bufferIndex][i]);
                }
                bufferIndex++;
                bufferOffset = 0;
            }
            return ret;
        }

        protected override void SetValue(long rowIndex, object value)
        {
            if (value == null || value is string)
            {
                int bufferIndex = GetBufferIndexContainingRowIndex(rowIndex);
                int bufferOffset = (int)(rowIndex % MaxCapacity);
                var oldValue = this[rowIndex];
                _stringBuffers[bufferIndex][bufferOffset] = (string)value;
                if (oldValue != (string)value)
                {
                    if (value == null)
                        _nullCount++;
                    if (oldValue == null && _nullCount > 0)
                        _nullCount--;
                }
            }
            else
            {
                throw new ArgumentException(string.Format(Strings.MismatchedValueType, typeof(string)), nameof(value));
            }
        }

        public new string this[long rowIndex]
        {
            get => (string)GetValue(rowIndex);
            set => SetValue(rowIndex, value);
        }

        public new List<string> this[long startIndex, int length]
        {
            get
            {
                var ret = new List<string>();
                int bufferIndex = GetBufferIndexContainingRowIndex(startIndex);
                int bufferOffset = (int)(startIndex % MaxCapacity);
                while (ret.Count < length && bufferIndex < _stringBuffers.Count)
                {
                    for (int i = bufferOffset; ret.Count < length && i < _stringBuffers[bufferIndex].Count; i++)
                    {
                        ret.Add(_stringBuffers[bufferIndex][i]);
                    }
                    bufferIndex++;
                    bufferOffset = 0;
                }
                return ret;
            }
        }

        public IEnumerator<string> GetEnumerator()
        {
            foreach (List<string> buffer in _stringBuffers)
            {
                foreach (string value in buffer)
                {
                    yield return value;
                }
            }
        }

        protected override IEnumerator GetEnumeratorCore() => GetEnumerator();

        public override DataFrameColumn Clamp<U>(U min, U max, bool inPlace = false) => throw new NotSupportedException();

        public override DataFrameColumn Filter<U>(U min, U max) => throw new NotSupportedException();

        public new StringDataFrameColumn Sort(bool ascending = true, bool putNullValuesLast = true)
        {
            return (StringDataFrameColumn)base.Sort(ascending, putNullValuesLast);
        }

        protected internal override PrimitiveDataFrameColumn<long> GetSortIndices(bool ascending, bool putNullValuesLast)
        {
            var comparer = Comparer<string>.Default;

            List<int[]> bufferSortIndices = new List<int[]>(_stringBuffers.Count);
            var columnNullIndices = new Int64DataFrameColumn("NullIndices", NullCount);
            long nullIndicesSlot = 0;
            foreach (List<string> buffer in _stringBuffers)
            {
                var sortIndices = new int[buffer.Count];
                for (int i = 0; i < buffer.Count; i++)
                {
                    sortIndices[i] = i;
                    if (buffer[i] == null)
                    {
                        columnNullIndices[nullIndicesSlot] = i + bufferSortIndices.Count * MaxCapacity;
                        nullIndicesSlot++;
                    }
                }
                // TODO: Refactor the sort routine to also work with IList?
                string[] array = buffer.ToArray();
                IntrospectiveSort(array, array.Length, sortIndices, comparer);
                bufferSortIndices.Add(sortIndices);
            }
            // Simple merge sort to build the full column's sort indices
            ValueTuple<string, int> GetFirstNonNullValueStartingAtIndex(int stringBufferIndex, int startIndex)
            {
                string value = _stringBuffers[stringBufferIndex][bufferSortIndices[stringBufferIndex][startIndex]];
                while (value == null && ++startIndex < bufferSortIndices[stringBufferIndex].Length)
                {
                    value = _stringBuffers[stringBufferIndex][bufferSortIndices[stringBufferIndex][startIndex]];
                }
                return (value, startIndex);
            }

            SortedDictionary<string, List<ValueTuple<int, int>>> heapOfValueAndListOfTupleOfSortAndBufferIndex = new SortedDictionary<string, List<ValueTuple<int, int>>>(comparer);
            List<List<string>> buffers = _stringBuffers;
            for (int i = 0; i < buffers.Count; i++)
            {
                List<string> buffer = buffers[i];
                ValueTuple<string, int> valueAndBufferSortIndex = GetFirstNonNullValueStartingAtIndex(i, 0);
                if (valueAndBufferSortIndex.Item1 == null)
                {
                    // All nulls
                    continue;
                }
                if (heapOfValueAndListOfTupleOfSortAndBufferIndex.ContainsKey(valueAndBufferSortIndex.Item1))
                {
                    heapOfValueAndListOfTupleOfSortAndBufferIndex[valueAndBufferSortIndex.Item1].Add((valueAndBufferSortIndex.Item2, i));
                }
                else
                {
                    heapOfValueAndListOfTupleOfSortAndBufferIndex.Add(valueAndBufferSortIndex.Item1, new List<ValueTuple<int, int>>() { (valueAndBufferSortIndex.Item2, i) });
                }
            }
            var columnSortIndices = new Int64DataFrameColumn("SortIndices", Length);

            GetBufferSortIndex getBufferSortIndex = new GetBufferSortIndex((int bufferIndex, int sortIndex) => (bufferSortIndices[bufferIndex][sortIndex]) + bufferIndex * bufferSortIndices[0].Length);
            GetValueAndBufferSortIndexAtBuffer<string> getValueAtBuffer = new GetValueAndBufferSortIndexAtBuffer<string>((int bufferIndex, int sortIndex) => GetFirstNonNullValueStartingAtIndex(bufferIndex, sortIndex));
            GetBufferLengthAtIndex getBufferLengthAtIndex = new GetBufferLengthAtIndex((int bufferIndex) => bufferSortIndices[bufferIndex].Length);

            PopulateColumnSortIndicesWithHeap(heapOfValueAndListOfTupleOfSortAndBufferIndex,
                columnSortIndices,
                columnNullIndices,
                ascending,
                putNullValuesLast,
                getBufferSortIndex,
                getValueAtBuffer,
                getBufferLengthAtIndex);

            return columnSortIndices;
        }

        public new StringDataFrameColumn Clone(DataFrameColumn mapIndices, bool invertMapIndices, long numberOfNullsToAppend)
        {
            return (StringDataFrameColumn)CloneImplementation(mapIndices, invertMapIndices, numberOfNullsToAppend);
        }

        public new StringDataFrameColumn Clone(long numberOfNullsToAppend = 0)
        {
            return (StringDataFrameColumn)CloneImplementation(numberOfNullsToAppend);
        }

        protected override DataFrameColumn CloneImplementation(long numberOfNullsToAppend)
        {
            StringDataFrameColumn ret = new StringDataFrameColumn(Name, Length);
            for (long i = 0; i < Length; i++)
                ret[i] = this[i];

            for (long i = 0; i < numberOfNullsToAppend; i++)
                ret.Append(null);

            return ret;
        }

        protected override DataFrameColumn CloneImplementation(DataFrameColumn mapIndices, bool invertMapIndices = false, long numberOfNullsToAppend = 0)
        {
            StringDataFrameColumn clone;
            if (!(mapIndices is null))
            {
                Type dataType = mapIndices.DataType;
                if (dataType != typeof(long) && dataType != typeof(int) && dataType != typeof(bool))
                    throw new ArgumentException(String.Format(Strings.MultipleMismatchedValueType, typeof(long), typeof(int), typeof(bool)), nameof(mapIndices));
                if (mapIndices.DataType == typeof(long))
                    clone = CloneImplementation(mapIndices as PrimitiveDataFrameColumn<long>, invertMapIndices);
                else if (dataType == typeof(int))
                    clone = CloneImplementation(mapIndices as PrimitiveDataFrameColumn<int>, invertMapIndices);
                else
                    clone = CloneImplementation(mapIndices as PrimitiveDataFrameColumn<bool>);

                for (long i = 0; i < numberOfNullsToAppend; i++)
                    clone.Append(null);
            }
            else
            {
                clone = Clone(numberOfNullsToAppend);
            }

            return clone;
        }

        private StringDataFrameColumn CloneImplementation(PrimitiveDataFrameColumn<bool> boolColumn)
        {
            if (boolColumn.Length > Length)
                throw new ArgumentException(Strings.MapIndicesExceedsColumnLength, nameof(boolColumn));
            StringDataFrameColumn ret = new StringDataFrameColumn(Name, 0);
            for (long i = 0; i < boolColumn.Length; i++)
            {
                bool? value = boolColumn[i];
                if (value.HasValue && value.Value)
                    ret.Append(this[i]);
            }
            return ret;
        }

        private StringDataFrameColumn CloneImplementation(PrimitiveDataFrameColumn<int> mapIndices, bool invertMapIndices = false)
        {
            mapIndices = mapIndices ?? throw new ArgumentNullException(nameof(mapIndices));
            var ret = new StringDataFrameColumn(Name, mapIndices.Length);

            long rowIndex = 0;
            for (int b = 0; b < mapIndices.ColumnContainer.Buffers.Count; b++)
            {
                var span = mapIndices.ColumnContainer.Buffers[b].ReadOnlySpan;
                var validitySpan = mapIndices.ColumnContainer.NullBitMapBuffers[b].ReadOnlySpan;

                for (int i = 0; i < span.Length; i++)
                {
                    long index = invertMapIndices ? mapIndices.Length - 1 - rowIndex : rowIndex;
                    ret[index] = BitUtility.IsValid(validitySpan, i) ? this[span[i]] : null;
                    rowIndex++;
                }
            }

            return ret;
        }

        private StringDataFrameColumn CloneImplementation(PrimitiveDataFrameColumn<long> mapIndices, bool invertMapIndices = false)
        {
            mapIndices = mapIndices ?? throw new ArgumentNullException(nameof(mapIndices));
            var ret = new StringDataFrameColumn(Name, mapIndices.Length);

            long rowIndex = 0;
            for (int b = 0; b < mapIndices.ColumnContainer.Buffers.Count; b++)
            {
                var span = mapIndices.ColumnContainer.Buffers[b].ReadOnlySpan;
                var validitySpan = mapIndices.ColumnContainer.NullBitMapBuffers[b].ReadOnlySpan;

                for (int i = 0; i < span.Length; i++)
                {
                    long index = invertMapIndices ? mapIndices.Length - 1 - rowIndex : rowIndex;
                    ret[index] = BitUtility.IsValid(validitySpan, i) ? this[span[i]] : null;
                    rowIndex++;
                }
            }

            return ret;
        }

        internal static DataFrame ValueCountsImplementation(Dictionary<string, ICollection<long>> groupedValues)
        {
            StringDataFrameColumn keys = new StringDataFrameColumn("Values", 0);
            PrimitiveDataFrameColumn<long> counts = new PrimitiveDataFrameColumn<long>("Counts");
            foreach (KeyValuePair<string, ICollection<long>> keyValuePair in groupedValues)
            {
                keys.Append(keyValuePair.Key);
                counts.Append(keyValuePair.Value.Count);
            }
            return new DataFrame(new List<DataFrameColumn> { keys, counts });
        }

        public override DataFrame ValueCounts()
        {
            Dictionary<string, ICollection<long>> groupedValues = GroupColumnValues<string>(out HashSet<long> _);
            return ValueCountsImplementation(groupedValues);
        }

        public override GroupBy GroupBy(int columnIndex, DataFrame parent)
        {
            Dictionary<string, ICollection<long>> dictionary = GroupColumnValues<string>(out HashSet<long> _);
            return new GroupBy<string>(parent, columnIndex, dictionary);
        }

        public override Dictionary<TKey, ICollection<long>> GroupColumnValues<TKey>(out HashSet<long> nullIndices)
        {
            if (typeof(TKey) == typeof(string))
            {
                Dictionary<string, ICollection<long>> multimap = new Dictionary<string, ICollection<long>>(EqualityComparer<string>.Default);
                nullIndices = new HashSet<long>();
                for (long i = 0; i < Length; i++)
                {
                    string str = this[i];
                    if (str != null)
                    {
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
                    else
                    {
                        nullIndices.Add(i);
                    }
                }
                return multimap as Dictionary<TKey, ICollection<long>>;
            }
            else
            {
                throw new NotImplementedException(nameof(TKey));
            }
        }

        /// <summary>
        /// Returns a new column with <see langword="null" /> elements replaced by <paramref name="value"/>.
        /// </summary>
        /// <remarks>Tries to convert value to the column's DataType</remarks>
        /// <param name="value"></param>
        /// <param name="inPlace">Indicates if the operation should be performed in place</param>
        public StringDataFrameColumn FillNulls(string value, bool inPlace = false)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            StringDataFrameColumn column = inPlace ? this : Clone();

            for (long i = 0; i < column.Length; i++)
            {
                if (column[i] == null)
                    column[i] = value;
            }
            return column;
        }

        protected override DataFrameColumn FillNullsImplementation(object value, bool inPlace)
        {
            if (value is string valueString)
                return FillNulls(valueString, inPlace);
            else
                throw new ArgumentException(String.Format(Strings.MismatchedValueType, typeof(string)), nameof(value));
        }

        /// <inheritdoc/>
        public new StringDataFrameColumn DropNulls()
        {
            return (StringDataFrameColumn)DropNullsImplementation();
        }

        protected override DataFrameColumn DropNullsImplementation()
        {
            var ret = new StringDataFrameColumn(Name, Length - NullCount);

            long j = 0;
            for (long i = 0; i < Length; i++)
            {
                var value = this[i];

                if (value != null)
                {
                    ret[j++] = value;
                }
            }

            return ret;
        }

        protected internal override void AddDataViewColumn(DataViewSchema.Builder builder)
        {
            builder.AddColumn(Name, TextDataViewType.Instance);
        }

        protected internal override Delegate GetDataViewGetter(DataViewRowCursor cursor)
        {
            return CreateValueGetterDelegate(cursor);
        }

        private ValueGetter<ReadOnlyMemory<char>> CreateValueGetterDelegate(DataViewRowCursor cursor) =>
            (ref ReadOnlyMemory<char> value) => value = this[cursor.Position].AsMemory();

        protected internal override void AddValueUsingCursor(DataViewRowCursor cursor, Delegate getter)
        {
            long row = cursor.Position;
            ReadOnlyMemory<char> value = default;
            Debug.Assert(getter != null, "Excepted getter to be valid");

            (getter as ValueGetter<ReadOnlyMemory<char>>)(ref value);

            if (Length > row)
            {
                this[row] = value.ToString();
            }
            else if (Length == row)
            {
                Append(value.ToString());
            }
            else
            {
                throw new IndexOutOfRangeException(nameof(row));
            }
        }

        protected internal override Delegate GetValueGetterUsingCursor(DataViewRowCursor cursor, DataViewSchema.Column schemaColumn)
        {
            return cursor.GetGetter<ReadOnlyMemory<char>>(schemaColumn);
        }

        public override Dictionary<long, ICollection<long>> GetGroupedOccurrences(DataFrameColumn other, out HashSet<long> otherColumnNullIndices)
        {
            return GetGroupedOccurrences<string>(other, out otherColumnNullIndices);
        }
    }
}
