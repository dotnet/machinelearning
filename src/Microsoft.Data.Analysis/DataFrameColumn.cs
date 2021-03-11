// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Apache.Arrow;
using Microsoft.ML;

namespace Microsoft.Data.Analysis
{
    /// <summary>
    /// The base column type. All APIs should be defined here first
    /// </summary>
    public abstract partial class DataFrameColumn : IEnumerable
    {
        /// <summary>
        /// The base <see cref="DataFrameColumn"/> constructor.
        /// </summary>
        /// <param name="name">The name of this column.</param>
        /// <param name="length">The length of this column.</param>
        /// <param name="type">The type of data this column holds.</param>
        protected DataFrameColumn(string name, long length, Type type)
        {
            Length = length;
            _name = name;
            DataType = type;
        }

        /// <summary>
        /// A static factory method to create a <see cref="PrimitiveDataFrameColumn{T}"/>. 
        /// It allows you to take advantage of type inference based on the type of the values supplied.
        /// </summary>
        /// <typeparam name="T">The type of the column to create.</typeparam>
        /// <param name="name">The name of the column.</param>
        /// <param name="values">The initial values to populate in the column.</param>
        /// <returns>A <see cref="PrimitiveDataFrameColumn{T}"/> populated with the provided data.</returns>
        public static PrimitiveDataFrameColumn<T> Create<T>(string name, IEnumerable<T?> values) where T : unmanaged
        {
            return new PrimitiveDataFrameColumn<T>(name, values);
        }

        /// <summary>
        /// A static factory method to create a <see cref="PrimitiveDataFrameColumn{T}"/>. 
        /// It allows you to take advantage of type inference based on the type of the values supplied.
        /// </summary>
        /// <typeparam name="T">The type of the column to create.</typeparam>
        /// <param name="name">The name of the column.</param>
        /// <param name="values">The initial values to populate in the column.</param>
        /// <returns>A <see cref="PrimitiveDataFrameColumn{T}"/> populated with the provided data.</returns>
        public static PrimitiveDataFrameColumn<T> Create<T>(string name, IEnumerable<T> values) where T : unmanaged
        {
            return new PrimitiveDataFrameColumn<T>(name, values);
        }

        /// <summary>
        /// A static factory method to create a <see cref="StringDataFrameColumn"/>. 
        /// It allows you to take advantage of type inference based on the type of the values supplied.
        /// </summary>
        /// <param name="name">The name of the column.</param>
        /// <param name="values">The initial values to populate in the column.</param>
        /// <returns>A <see cref="StringDataFrameColumn"/> populated with the provided data.</returns>
        public static StringDataFrameColumn Create(string name, IEnumerable<string> values)
        {
            return new StringDataFrameColumn(name, values);
        }

        private long _length;

        /// <summary>
        /// The length of this column
        /// </summary>
        public long Length
        {
            get => _length;
            protected set
            {
                if (value < 0)
                    throw new ArgumentOutOfRangeException();
                _length = value;
            }
        }

        /// <summary>
        /// The number of <see langword="null" /> values in this column.
        /// </summary>
        public abstract long NullCount
        {
            get;
        }

        private string _name;

        /// <summary>
        /// The name of this column.
        /// </summary>
        public string Name => _name;

        /// <summary>
        /// Updates the name of this column.
        /// </summary>
        /// <param name="newName">The new name.</param>
        /// <param name="dataFrame">If passed in, update the column name in <see cref="DataFrame.Columns"/></param>
        public void SetName(string newName, DataFrame dataFrame = null)
        {
            if (!(dataFrame is null))
            {
                dataFrame.Columns.SetColumnName(this, newName);
            }
            _name = newName;
        }

        /// <summary>
        /// The type of data this column holds.
        /// </summary>
        public Type DataType { get; }

        /// <summary>
        /// Indexer to get/set values at <paramref name="rowIndex"/>
        /// </summary>
        /// <param name="rowIndex">The index to look up</param>
        /// <returns>The value at <paramref name="rowIndex"/></returns>
        public object this[long rowIndex]
        {
            get => GetValue(rowIndex);
            set => SetValue(rowIndex, value);
        }

        /// <summary>
        /// Returns the value at <paramref name="rowIndex"/>.
        /// </summary>
        /// <param name="rowIndex"></param>
        /// <returns>The value at <paramref name="rowIndex"/>.</returns>
        protected abstract object GetValue(long rowIndex);

        /// <summary>
        /// Returns <paramref name="length"/> number of values starting from <paramref name="startIndex"/>.
        /// </summary>
        /// <param name="startIndex">The first index to return values from.</param>
        /// <param name="length">The number of values to return.</param>
        /// <returns>A read only list of values</returns>
        protected abstract IReadOnlyList<object> GetValues(long startIndex, int length);

        /// <summary>
        /// Sets the value at <paramref name="rowIndex"/> with <paramref name="value"/>
        /// </summary>
        /// <param name="rowIndex">The row index</param>
        /// <param name="value">The new value.</param>
        protected abstract void SetValue(long rowIndex, object value);

        /// <summary>
        /// Returns <paramref name="length"/> number of values starting from <paramref name="startIndex"/>.
        /// </summary>
        /// <param name="startIndex">The first index to return values from.</param>
        /// <param name="length">The number of values to return.</param>
        /// <returns>A read only list of values</returns>
        public IReadOnlyList<object> this[long startIndex, int length]
        {
            get => GetValues(startIndex, length);
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumeratorCore();

        /// <summary>
        /// Returns an enumerator that iterates this column.
        /// </summary>
        protected abstract IEnumerator GetEnumeratorCore();

        /// <summary>
        /// Called internally from Append, Merge and GroupBy. Resizes the column to the specified length to allow setting values by indexing
        /// </summary>
        /// <param name="length">The new length of the column</param>
        protected internal virtual void Resize(long length) => throw new NotImplementedException();

        /// <summary>
        /// Clone column to produce a copy potentially changing the order of values by supplying mapIndices and an invert flag
        /// </summary>
        /// <param name="mapIndices"></param>
        /// <param name="invertMapIndices"></param>
        /// <param name="numberOfNullsToAppend"></param>
        /// <returns>A new <see cref="DataFrameColumn"/></returns>
        public virtual DataFrameColumn Clone(DataFrameColumn mapIndices = null, bool invertMapIndices = false, long numberOfNullsToAppend = 0) => CloneImplementation(mapIndices, invertMapIndices, numberOfNullsToAppend);

        /// <summary>
        /// Clone column to produce a copy potentially changing the order of values by supplying mapIndices and an invert flag
        /// </summary>
        /// <param name="mapIndices"></param>
        /// <param name="invertMapIndices"></param>
        /// <param name="numberOfNullsToAppend"></param>
        /// <returns>A new <see cref="DataFrameColumn"/></returns>
        protected virtual DataFrameColumn CloneImplementation(DataFrameColumn mapIndices, bool invertMapIndices, long numberOfNullsToAppend) => throw new NotImplementedException();

        /// <summary>
        /// Returns a copy of this column sorted by its values
        /// </summary>
        /// <param name="ascending"></param>
        public virtual DataFrameColumn Sort(bool ascending = true)
        {
            PrimitiveDataFrameColumn<long> sortIndices = GetAscendingSortIndices();
            return Clone(sortIndices, !ascending, NullCount);
        }

        public virtual Dictionary<TKey, ICollection<long>> GroupColumnValues<TKey>() => throw new NotImplementedException();

        /// <summary>
        /// Returns a DataFrame containing counts of unique values
        /// </summary>
        public virtual DataFrame ValueCounts() => throw new NotImplementedException();

        public virtual GroupBy GroupBy(int columnIndex, DataFrame parent) => throw new NotImplementedException();

        /// <summary>
        /// Returns a new column with <see langword="null" /> elements replaced by <paramref name="value"/>.
        /// </summary>
        /// <remarks>Tries to convert value to the column's DataType</remarks>
        /// <param name="value"></param>
        /// <param name="inPlace">Indicates if the operation should be performed in place</param>
        public virtual DataFrameColumn FillNulls(object value, bool inPlace = false) => FillNullsImplementation(value, inPlace);

        protected virtual DataFrameColumn FillNullsImplementation(object value, bool inPlace) => throw new NotImplementedException();

        // Arrow related APIs
        protected internal virtual Field GetArrowField() => throw new NotImplementedException();
        /// <summary>
        /// Returns the max number of values that are contiguous in memory
        /// </summary>
        protected internal virtual int GetMaxRecordBatchLength(long startIndex) => 0;
        protected internal virtual Apache.Arrow.Array ToArrowArray(long startIndex, int numberOfRows) => throw new NotImplementedException();

        /// <summary>
        /// Creates a <see cref="ValueGetter{TValue}"/> that will return the value of the column for the row
        /// the cursor is referencing.
        /// </summary>
        /// <param name="cursor">
        /// The row cursor which has the current position.
        /// </param>
        protected internal virtual Delegate GetDataViewGetter(DataViewRowCursor cursor) => throw new NotImplementedException();

        /// <summary>
        /// Adds a new <see cref="DataViewSchema.Column"/> to the specified builder for the current column.
        /// </summary>
        /// <param name="builder">
        /// The builder to which to add the schema column.
        /// </param>
        protected internal virtual void AddDataViewColumn(DataViewSchema.Builder builder) => throw new NotImplementedException();

        /// <summary>
        /// Clamps values beyond the specified thresholds
        /// </summary>
        /// <typeparam name="U"></typeparam>
        /// <param name="min">Minimum value. All values below this threshold will be set to it</param>
        /// <param name="max">Maximum value. All values above this threshold will be set to it</param>
        /// <param name="inPlace">Indicates if the operation should be performed in place</param>
        public virtual DataFrameColumn Clamp<U>(U min, U max, bool inPlace = false) => ClampImplementation(min, max, inPlace);

        /// <summary>
        /// Clamps values beyond the specified thresholds
        /// </summary>
        /// <typeparam name="U"></typeparam>
        /// <param name="min">Minimum value. All values below this threshold will be set to it</param>
        /// <param name="max">Maximum value. All values above this threshold will be set to it</param>
        /// <param name="inPlace">Indicates if the operation should be performed in place</param>
        protected virtual DataFrameColumn ClampImplementation<U>(U min, U max, bool inPlace) => throw new NotImplementedException();

        /// <summary>
        /// Returns a new column filtered by the lower and upper bounds
        /// </summary>
        /// <typeparam name="U"></typeparam>
        /// <param name="min">The minimum value in the resulting column</param>
        /// <param name="max">The maximum value in the resulting column</param>
        public virtual DataFrameColumn Filter<U>(U min, U max) => FilterImplementation(min, max);

        /// <summary>
        /// Returns a new column filtered by the lower and upper bounds
        /// </summary>
        /// <typeparam name="U"></typeparam>
        /// <param name="min"></param>
        /// <param name="max"></param>
        protected virtual DataFrameColumn FilterImplementation<U>(U min, U max) => throw new NotImplementedException();

        /// <summary>
        /// Determines if the column is of a numeric type
        /// </summary>
        public virtual bool IsNumericColumn() => false;

        /// <summary>
        /// Returns the mean of the values in the column. Throws if this is not a numeric column
        /// </summary>
        public virtual double Mean() => throw new NotImplementedException();

        /// <summary>
        /// Returns the median of the values in the column. Throws if this is not a numeric column
        /// </summary>
        public virtual double Median() => throw new NotImplementedException();

        /// <summary>
        /// Used to exclude columns from the Description method
        /// </summary>
        public virtual bool HasDescription() => false;

        /// <summary>
        /// Returns a <seealso cref="StringDataFrameColumn"/> containing the DataType and Length of this column
        /// </summary>
        public virtual StringDataFrameColumn Info()
        {
            StringDataFrameColumn dataColumn = new StringDataFrameColumn(Name, 2);
            dataColumn[0] = DataType.ToString();
            dataColumn[1] = (Length - NullCount).ToString();
            return dataColumn;
        }

        /// <summary>
        /// Returns a <see cref= "DataFrameColumn"/> with statistics that describe the column
        /// </summary>
        public virtual DataFrameColumn Description() => throw new NotImplementedException();

        internal virtual PrimitiveDataFrameColumn<long> GetAscendingSortIndices() => throw new NotImplementedException();

        internal delegate long GetBufferSortIndex(int bufferIndex, int sortIndex);
        internal delegate ValueTuple<T, int> GetValueAndBufferSortIndexAtBuffer<T>(int bufferIndex, int valueIndex);
        internal delegate int GetBufferLengthAtIndex(int bufferIndex);
        internal void PopulateColumnSortIndicesWithHeap<T>(SortedDictionary<T, List<ValueTuple<int, int>>> heapOfValueAndListOfTupleOfSortAndBufferIndex,
                                                            PrimitiveDataFrameColumn<long> columnSortIndices,
                                                            GetBufferSortIndex getBufferSortIndex,
                                                            GetValueAndBufferSortIndexAtBuffer<T> getValueAndBufferSortIndexAtBuffer,
                                                            GetBufferLengthAtIndex getBufferLengthAtIndex)
        {
            while (heapOfValueAndListOfTupleOfSortAndBufferIndex.Count > 0)
            {
                KeyValuePair<T, List<ValueTuple<int, int>>> minElement = heapOfValueAndListOfTupleOfSortAndBufferIndex.ElementAt(0);
                List<ValueTuple<int, int>> tuplesOfSortAndBufferIndex = minElement.Value;
                (int sortIndex, int bufferIndex) sortAndBufferIndex;
                if (tuplesOfSortAndBufferIndex.Count == 1)
                {
                    heapOfValueAndListOfTupleOfSortAndBufferIndex.Remove(minElement.Key);
                    sortAndBufferIndex = tuplesOfSortAndBufferIndex[0];
                }
                else
                {
                    sortAndBufferIndex = tuplesOfSortAndBufferIndex[tuplesOfSortAndBufferIndex.Count - 1];
                    tuplesOfSortAndBufferIndex.RemoveAt(tuplesOfSortAndBufferIndex.Count - 1);
                }
                int sortIndex = sortAndBufferIndex.sortIndex;
                int bufferIndex = sortAndBufferIndex.bufferIndex;
                long bufferSortIndex = getBufferSortIndex(bufferIndex, sortIndex);
                columnSortIndices.Append(bufferSortIndex);
                if (sortIndex + 1 < getBufferLengthAtIndex(bufferIndex))
                {
                    int nextSortIndex = sortIndex + 1;
                    (T value, int bufferSortIndex) nextValueAndBufferSortIndex = getValueAndBufferSortIndexAtBuffer(bufferIndex, nextSortIndex);
                    T nextValue = nextValueAndBufferSortIndex.value;
                    if (nextValue != null)
                    {
                        heapOfValueAndListOfTupleOfSortAndBufferIndex.Add(nextValue, new List<ValueTuple<int, int>>() { (nextValueAndBufferSortIndex.bufferSortIndex, bufferIndex) });
                    }
                }
            }

        }
        internal static int FloorLog2PlusOne(int n)
        {
            Debug.Assert(n >= 2);
            int result = 2;
            n >>= 2;
            while (n > 0)
            {
                ++result;
                n >>= 1;
            }
            return result;
        }

        internal static void IntrospectiveSort<T>(
            ReadOnlySpan<T> span,
            int length,
            Span<int> sortIndices,
            IComparer<T> comparer)
        {
            var depthLimit = 2 * FloorLog2PlusOne(length);
            IntroSortRecursive(span, 0, length - 1, depthLimit, sortIndices, comparer);
        }

        internal static void IntroSortRecursive<T>(
            ReadOnlySpan<T> span,
            int lo, int hi, int depthLimit,
            Span<int> sortIndices,
            IComparer<T> comparer)
        {
            Debug.Assert(comparer != null);
            Debug.Assert(lo >= 0);

            while (hi > lo)
            {
                int partitionSize = hi - lo + 1;
                if (partitionSize <= 16)
                {
                    if (partitionSize == 1)
                    {
                        return;
                    }
                    if (partitionSize == 2)
                    {
                        Sort2(span, lo, hi, sortIndices, comparer);
                        return;
                    }
                    if (partitionSize == 3)
                    {
                        Sort3(span, lo, hi - 1, hi, sortIndices, comparer);
                        return;
                    }

                    InsertionSort(span, lo, hi, sortIndices, comparer);
                    return;
                }

                if (depthLimit == 0)
                {
                    HeapSort(span, lo, hi, sortIndices, comparer);
                    return;
                }
                depthLimit--;

                // We should never reach here, unless > 3 elements due to partition size
                int p = PickPivotAndPartition(span, lo, hi, sortIndices, comparer);
                // Note we've already partitioned around the pivot and do not have to move the pivot again.
                IntroSortRecursive(span, p + 1, hi, depthLimit, sortIndices, comparer);
                hi = p - 1;
            }
        }

        private static int PickPivotAndPartition<TKey, TComparer>(
            ReadOnlySpan<TKey> span, int lo, int hi,
            Span<int> sortIndices,
            TComparer comparer)
            where TComparer : IComparer<TKey>
        {
            Debug.Assert(comparer != null);
            Debug.Assert(lo >= 0);
            Debug.Assert(hi > lo);

            // median-of-three
            int middle = (int)(((uint)hi + (uint)lo) >> 1);

            // Sort lo, mid and hi appropriately, then pick mid as the pivot.
            Sort3(span, lo, middle, hi, sortIndices, comparer);

            TKey pivot = span[sortIndices[middle]];

            int left = lo;
            int right = hi - 1;
            // We already partitioned lo and hi and put the pivot in hi - 1.  
            Swap(ref sortIndices[middle], ref sortIndices[right]);

            while (left < right)
            {
                while (left < (hi - 1) && comparer.Compare(span[sortIndices[++left]], pivot) < 0)
                    ;
                // Check if bad comparable/comparer
                if (left == (hi - 1) && comparer.Compare(span[sortIndices[left]], pivot) < 0)
                    throw new ArgumentException("Bad comparer");

                while (right > lo && comparer.Compare(pivot, span[sortIndices[--right]]) < 0)
                    ;
                // Check if bad comparable/comparer
                if (right == lo && comparer.Compare(pivot, span[sortIndices[right]]) < 0)
                    throw new ArgumentException("Bad comparer");

                if (left >= right)
                    break;

                Swap(ref sortIndices[left], ref sortIndices[right]);
            }
            // Put pivot in the right location.
            right = hi - 1;
            if (left != right)
            {
                Swap(ref sortIndices[left], ref sortIndices[right]);
            }
            return left;
        }

        internal static void Swap<TKey>(ref TKey a, ref TKey b)
        {
            TKey temp = a;
            a = b;
            b = temp;
        }

        private static void HeapSort<TKey, TComparer>(
            ReadOnlySpan<TKey> span, int lo, int hi,
            Span<int> sortIndices,
            TComparer comparer)
            where TComparer : IComparer<TKey>
        {
            Debug.Assert(comparer != null);
            Debug.Assert(lo >= 0);
            Debug.Assert(hi > lo);

            int n = hi - lo + 1;
            for (int i = n / 2; i >= 1; --i)
            {
                DownHeap(span, i, n, lo, sortIndices, comparer);
            }
            for (int i = n; i > 1; --i)
            {
                Swap(ref sortIndices[lo], ref sortIndices[lo + i - 1]);
                DownHeap(span, 1, i - 1, lo, sortIndices, comparer);
            }
        }

        private static void DownHeap<TKey, TComparer>(
            ReadOnlySpan<TKey> span, int i, int n, int lo,
            Span<int> sortIndices,
            TComparer comparer)
            where TComparer : IComparer<TKey>
        {
            // Max Heap
            Debug.Assert(comparer != null);
            Debug.Assert(lo >= 0);

            int di = sortIndices[lo - 1 + i];
            TKey d = span[di];
            var nHalf = n / 2;
            while (i <= nHalf)
            {
                int child = i << 1;

                if (child < n &&
                    comparer.Compare(span[sortIndices[lo + child - 1]], span[sortIndices[lo + child]]) < 0)
                {
                    ++child;
                }

                if (!(comparer.Compare(d, span[sortIndices[lo + child - 1]]) < 0))
                    break;

                sortIndices[lo + i - 1] = sortIndices[lo + child - 1];

                i = child;
            }
            sortIndices[lo + i - 1] = di;
        }

        private static void InsertionSort<TKey, TComparer>(
            ReadOnlySpan<TKey> span, int lo, int hi,
            Span<int> sortIndices,
            TComparer comparer)
            where TComparer : IComparer<TKey>
        {
            Debug.Assert(lo >= 0);
            Debug.Assert(hi >= lo);

            for (int i = lo; i < hi; ++i)
            {
                int j = i;
                var t = span[sortIndices[j + 1]];
                var ti = sortIndices[j + 1];
                if (j >= lo && comparer.Compare(t, span[sortIndices[j]]) < 0)
                {
                    do
                    {
                        sortIndices[j + 1] = sortIndices[j];
                        --j;
                    }
                    while (j >= lo && comparer.Compare(t, span[sortIndices[j]]) < 0);

                    sortIndices[j + 1] = ti;
                }
            }
        }

        private static void Sort3<TKey, TComparer>(
            ReadOnlySpan<TKey> span, int i, int j, int k,
            Span<int> sortIndices,
            TComparer comparer)
            where TComparer : IComparer<TKey>
        {
            Sort2(span, i, j, sortIndices, comparer);
            Sort2(span, i, k, sortIndices, comparer);
            Sort2(span, j, k, sortIndices, comparer);
        }

        private static void Sort2<TKey>(
            ReadOnlySpan<TKey> span, int i, int j,
            Span<int> sortIndices,
            IComparer<TKey> comparer)
        {
            Debug.Assert(i != j);
            if (comparer.Compare(span[sortIndices[i]], span[sortIndices[j]]) > 0)
            {
                int temp = sortIndices[i];
                sortIndices[i] = sortIndices[j];
                sortIndices[j] = temp;
            }
        }
    }
}
