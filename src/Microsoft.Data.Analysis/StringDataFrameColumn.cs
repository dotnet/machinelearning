// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
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
        private List<List<string>> _stringBuffers = new List<List<string>>(); // To store more than intMax number of strings

        public StringDataFrameColumn(string name, long length) : base(name, length, typeof(string))
        {
            int numberOfBuffersRequired = Math.Max((int)(length / int.MaxValue), 1);
            for (int i = 0; i < numberOfBuffersRequired; i++)
            {
                long bufferLen = length - _stringBuffers.Count * int.MaxValue;
                List<string> buffer = new List<string>((int)Math.Min(int.MaxValue, bufferLen));
                _stringBuffers.Add(buffer);
                for (int j = 0; j < bufferLen; j++)
                {
                    buffer.Add(default);
                }
            }
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
            if (lastBuffer.Count == int.MaxValue)
            {
                lastBuffer = new List<string>();
                _stringBuffers.Add(lastBuffer);
            }
            lastBuffer.Add(value);
            if (value == null)
                _nullCount++;
            Length++;
        }

        private int GetBufferIndexContainingRowIndex(ref long rowIndex)
        {
            if (rowIndex > Length)
            {
                throw new ArgumentOutOfRangeException(Strings.ColumnIndexOutOfRange, nameof(rowIndex));
            }
            return (int)(rowIndex / int.MaxValue);
        }

        protected override object GetValue(long rowIndex)
        {
            int bufferIndex = GetBufferIndexContainingRowIndex(ref rowIndex);
            return _stringBuffers[bufferIndex][(int)rowIndex];
        }

        protected override IReadOnlyList<object> GetValues(long startIndex, int length)
        {
            var ret = new List<object>();
            int bufferIndex = GetBufferIndexContainingRowIndex(ref startIndex);
            while (ret.Count < length && bufferIndex < _stringBuffers.Count)
            {
                for (int i = (int)startIndex; ret.Count < length && i < _stringBuffers[bufferIndex].Count; i++)
                {
                    ret.Add(_stringBuffers[bufferIndex][i]);
                }
                bufferIndex++;
                startIndex = 0;
            }
            return ret;
        }

        protected override void SetValue(long rowIndex, object value)
        {
            if (value == null || value is string)
            {
                int bufferIndex = GetBufferIndexContainingRowIndex(ref rowIndex);
                var oldValue = this[rowIndex];
                _stringBuffers[bufferIndex][(int)rowIndex] = (string)value;
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
                int bufferIndex = GetBufferIndexContainingRowIndex(ref startIndex);
                while (ret.Count < length && bufferIndex < _stringBuffers.Count)
                {
                    for (int i = (int)startIndex; ret.Count < length && i < _stringBuffers[bufferIndex].Count; i++)
                    {
                        ret.Add(_stringBuffers[bufferIndex][i]);
                    }
                    bufferIndex++;
                    startIndex = 0;
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

        public override DataFrameColumn Clip<U>(U lower, U upper, bool inPlace = false) => throw new NotSupportedException();

        public override DataFrameColumn Filter<U>(U lower, U upper) => throw new NotSupportedException();

        public new StringDataFrameColumn Sort(bool ascending = true)
        {
            PrimitiveDataFrameColumn<long> columnSortIndices = GetAscendingSortIndices();
            return Clone(columnSortIndices, !ascending, NullCount);
        }

        internal override PrimitiveDataFrameColumn<long> GetAscendingSortIndices()
        {
            GetSortIndices(Comparer<string>.Default, out PrimitiveDataFrameColumn<long> columnSortIndices);
            return columnSortIndices;
        }

        private void GetSortIndices(Comparer<string> comparer, out PrimitiveDataFrameColumn<long> columnSortIndices)
        {
            List<int[]> bufferSortIndices = new List<int[]>(_stringBuffers.Count);
            foreach (List<string> buffer in _stringBuffers)
            {
                var sortIndices = new int[buffer.Count];
                for (int i = 0; i < buffer.Count; i++)
                {
                    sortIndices[i] = i;
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
            columnSortIndices = new PrimitiveDataFrameColumn<long>("SortIndices");
            GetBufferSortIndex getBufferSortIndex = new GetBufferSortIndex((int bufferIndex, int sortIndex) => (bufferSortIndices[bufferIndex][sortIndex]) + bufferIndex * bufferSortIndices[0].Length);
            GetValueAndBufferSortIndexAtBuffer<string> getValueAtBuffer = new GetValueAndBufferSortIndexAtBuffer<string>((int bufferIndex, int sortIndex) => GetFirstNonNullValueStartingAtIndex(bufferIndex, sortIndex));
            GetBufferLengthAtIndex getBufferLengthAtIndex = new GetBufferLengthAtIndex((int bufferIndex) => bufferSortIndices[bufferIndex].Length);
            PopulateColumnSortIndicesWithHeap(heapOfValueAndListOfTupleOfSortAndBufferIndex, columnSortIndices, getBufferSortIndex, getValueAtBuffer, getBufferLengthAtIndex);
        }

        public new StringDataFrameColumn Clone(DataFrameColumn mapIndices, bool invertMapIndices, long numberOfNullsToAppend)
        {
            StringDataFrameColumn clone;
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
                clone.Append(null);
            }
            return clone;
        }

        protected override DataFrameColumn CloneImplementation(DataFrameColumn mapIndices = null, bool invertMapIndices = false, long numberOfNullsToAppend = 0)
        {
            return Clone(mapIndices, invertMapIndices, numberOfNullsToAppend);
        }

        private StringDataFrameColumn Clone(PrimitiveDataFrameColumn<bool> boolColumn)
        {
            if (boolColumn.Length > Length)
                throw new ArgumentException(Strings.MapIndicesExceedsColumnLenth, nameof(boolColumn));
            StringDataFrameColumn ret = new StringDataFrameColumn(Name, 0);
            for (long i = 0; i < boolColumn.Length; i++)
            {
                bool? value = boolColumn[i];
                if (value.HasValue && value.Value == true)
                    ret.Append(this[i]);
            }
            return ret;
        }

        private StringDataFrameColumn CloneImplementation<U>(PrimitiveDataFrameColumn<U> mapIndices, bool invertMapIndices = false)
            where U : unmanaged
        {
            mapIndices = mapIndices ?? throw new ArgumentNullException(nameof(mapIndices));
            StringDataFrameColumn ret = new StringDataFrameColumn(Name, mapIndices.Length);

            List<string> setBuffer = ret._stringBuffers[0];
            long setBufferMinRange = 0;
            long setBufferMaxRange = int.MaxValue;
            List<string> getBuffer = _stringBuffers[0];
            long getBufferMinRange = 0;
            long getBufferMaxRange = int.MaxValue;
            long maxCapacity = int.MaxValue;
            if (mapIndices.DataType == typeof(long))
            {
                PrimitiveDataFrameColumn<long> longMapIndices = mapIndices as PrimitiveDataFrameColumn<long>;
                longMapIndices.ApplyElementwise((long? mapIndex, long rowIndex) =>
                {
                    long index = rowIndex;
                    if (invertMapIndices)
                        index = longMapIndices.Length - 1 - index;
                    if (index < setBufferMinRange || index >= setBufferMaxRange)
                    {
                        int bufferIndex = (int)(index / maxCapacity);
                        setBuffer = ret._stringBuffers[bufferIndex];
                        setBufferMinRange = bufferIndex * maxCapacity;
                        setBufferMaxRange = (bufferIndex + 1) * maxCapacity;
                    }
                    index -= setBufferMinRange;
                    if (mapIndex == null)
                    {
                        setBuffer[(int)index] = null;
                        ret._nullCount++;
                        return mapIndex;
                    }

                    if (mapIndex.Value < getBufferMinRange || mapIndex.Value >= getBufferMaxRange)
                    {
                        int bufferIndex = (int)(mapIndex.Value / maxCapacity);
                        getBuffer = _stringBuffers[bufferIndex];
                        getBufferMinRange = bufferIndex * maxCapacity;
                        getBufferMaxRange = (bufferIndex + 1) * maxCapacity;
                    }
                    int bufferLocalMapIndex = (int)(mapIndex - getBufferMinRange);
                    string value = getBuffer[bufferLocalMapIndex];
                    setBuffer[(int)index] = value;
                    if (value == null)
                        ret._nullCount++;

                    return mapIndex;
                });
            }
            else if (mapIndices.DataType == typeof(int))
            {
                PrimitiveDataFrameColumn<int> intMapIndices = mapIndices as PrimitiveDataFrameColumn<int>;
                intMapIndices.ApplyElementwise((int? mapIndex, long rowIndex) =>
                {
                    long index = rowIndex;
                    if (invertMapIndices)
                        index = intMapIndices.Length - 1 - index;

                    if (mapIndex == null)
                    {
                        setBuffer[(int)index] = null;
                        ret._nullCount++;
                        return mapIndex;
                    }
                    string value = getBuffer[mapIndex.Value];
                    setBuffer[(int)index] = value;
                    if (value == null)
                        ret._nullCount++;

                    return mapIndex;
                });
            }
            else
            {
                Debug.Assert(false, nameof(mapIndices.DataType));
            }

            return ret;
        }

        private StringDataFrameColumn Clone(PrimitiveDataFrameColumn<long> mapIndices = null, bool invertMapIndex = false)
        {
            if (mapIndices is null)
            {
                StringDataFrameColumn ret = new StringDataFrameColumn(Name, mapIndices is null ? Length : mapIndices.Length);
                for (long i = 0; i < Length; i++)
                {
                    ret[i] = this[i];
                }
                return ret;
            }
            else
            {
                return CloneImplementation(mapIndices, invertMapIndex);
            }
        }

        private StringDataFrameColumn Clone(PrimitiveDataFrameColumn<int> mapIndices, bool invertMapIndex = false)
        {
            return CloneImplementation(mapIndices, invertMapIndex);
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
            Dictionary<string, ICollection<long>> groupedValues = GroupColumnValues<string>();
            return ValueCountsImplementation(groupedValues);
        }

        public override GroupBy GroupBy(int columnIndex, DataFrame parent)
        {
            Dictionary<string, ICollection<long>> dictionary = GroupColumnValues<string>();
            return new GroupBy<string>(parent, columnIndex, dictionary);
        }

        public override Dictionary<TKey, ICollection<long>> GroupColumnValues<TKey>()
        {
            if (typeof(TKey) == typeof(string))
            {
                Dictionary<string, ICollection<long>> multimap = new Dictionary<string, ICollection<long>>(EqualityComparer<string>.Default);
                for (long i = 0; i < Length; i++)
                {
                    bool containsKey = multimap.TryGetValue(this[i] ?? default, out ICollection<long> values);
                    if (containsKey)
                    {
                        values.Add(i);
                    }
                    else
                    {
                        multimap.Add(this[i] ?? default, new List<long>() { i });
                    }
                }
                return multimap as Dictionary<TKey, ICollection<long>>;
            }
            else
            {
                throw new NotImplementedException(nameof(TKey));
            }
        }

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
    }
}
