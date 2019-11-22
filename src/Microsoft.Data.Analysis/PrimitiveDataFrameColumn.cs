// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using Apache.Arrow;
using Apache.Arrow.Types;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Microsoft.Data.Analysis
{
    /// <summary>
    /// A column to hold primitive types such as int, float etc.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public partial class PrimitiveDataFrameColumn<T> : DataFrameColumn, IEnumerable<T?>
        where T : unmanaged
    {
        private PrimitiveColumnContainer<T> _columnContainer;

        internal PrimitiveDataFrameColumn(string name, PrimitiveColumnContainer<T> column) : base(name, column.Length, typeof(T))
        {
            _columnContainer = column;
        }

        public PrimitiveDataFrameColumn(string name, IEnumerable<T?> values) : base(name, 0, typeof(T))
        {
            _columnContainer = new PrimitiveColumnContainer<T>(values);
            Length = _columnContainer.Length;
        }

        public PrimitiveDataFrameColumn(string name, IEnumerable<T> values) : base(name, 0, typeof(T))
        {
            _columnContainer = new PrimitiveColumnContainer<T>(values);
            Length = _columnContainer.Length;
        }

        public PrimitiveDataFrameColumn(string name, long length = 0) : base(name, length, typeof(T))
        {
            _columnContainer = new PrimitiveColumnContainer<T>(length);
        }

        public PrimitiveDataFrameColumn(string name, ReadOnlyMemory<byte> buffer, ReadOnlyMemory<byte> nullBitMap, int length = 0, int nullCount = 0) : base(name, length, typeof(T))
        {
            _columnContainer = new PrimitiveColumnContainer<T>(buffer, nullBitMap, length, nullCount);
        }

        /// <summary>
        /// Returns an enumerable of immutable memory buffers representing the underlying values
        /// </summary>
        /// <remarks>Null values are encoded in the buffers returned by GetReadOnlyNullBitmapBuffers in the Apache Arrow format</remarks>
        /// <returns>IEnumerable<ReadOnlyMemory<typeparamref name="T"/>></returns>
        public IEnumerable<ReadOnlyMemory<T>> GetReadOnlyDataBuffers()
        {
            for (int i = 0; i < _columnContainer.Buffers.Count; i++)
            {
                ReadOnlyDataFrameBuffer<T> buffer = _columnContainer.Buffers[i];
                yield return buffer.ReadOnlyMemory;
            }
        }

        /// <summary>
        /// Returns an enumerable of immutable ReadOnlyMemory<byte> buffers representing null values in the Apache Arrow format
        /// </summary>
        /// <remarks>Each ReadOnlyMemory<byte> encodes the null values for its corresponding Data buffer</remarks>
        /// <returns>IEnumerable<ReadOnlyMemory<byte>></returns>
        public IEnumerable<ReadOnlyMemory<byte>> GetReadOnlyNullBitMapBuffers()
        {
            for (int i = 0; i < _columnContainer.NullBitMapBuffers.Count; i++)
            {
                ReadOnlyDataFrameBuffer<byte> buffer = _columnContainer.NullBitMapBuffers[i];
                yield return buffer.RawReadOnlyMemory;
            }
        }

        private IArrowType GetArrowType()
        {
            if (typeof(T) == typeof(bool))
                return BooleanType.Default;
            else if (typeof(T) == typeof(double))
                return DoubleType.Default;
            else if (typeof(T) == typeof(float))
                return FloatType.Default;
            else if (typeof(T) == typeof(sbyte))
                return Int8Type.Default;
            else if (typeof(T) == typeof(int))
                return Int32Type.Default;
            else if (typeof(T) == typeof(long))
                return Int64Type.Default;
            else if (typeof(T) == typeof(short))
                return Int16Type.Default;
            else if (typeof(T) == typeof(byte))
                return UInt8Type.Default;
            else if (typeof(T) == typeof(uint))
                return UInt32Type.Default;
            else if (typeof(T) == typeof(ulong))
                return UInt64Type.Default;
            else if (typeof(T) == typeof(ushort))
                return UInt16Type.Default;
            else
                throw new NotImplementedException(nameof(T));
        }

        protected internal override Field GetArrowField() => new Field(Name, GetArrowType(), NullCount != 0);

        protected internal override int GetMaxRecordBatchLength(long startIndex) => _columnContainer.MaxRecordBatchLength(startIndex);

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

        protected internal override Apache.Arrow.Array ToArrowArray(long startIndex, int numberOfRows)
        {
            int arrayIndex = numberOfRows == 0 ? 0 : _columnContainer.GetArrayContainingRowIndex(startIndex);
            int offset = (int)(startIndex - arrayIndex * ReadOnlyDataFrameBuffer<T>.MaxCapacity);
            if (numberOfRows != 0 && numberOfRows > _columnContainer.Buffers[arrayIndex].Length - offset)
            {
                throw new ArgumentException(Strings.SpansMultipleBuffers, nameof(numberOfRows));
            }
            ArrowBuffer valueBuffer = numberOfRows == 0 ? ArrowBuffer.Empty : new ArrowBuffer(_columnContainer.GetValueBuffer(startIndex));
            ArrowBuffer nullBuffer = numberOfRows == 0 ? ArrowBuffer.Empty : new ArrowBuffer(_columnContainer.GetNullBuffer(startIndex));
            int nullCount = GetNullCount(startIndex, numberOfRows);
            Type type = this.DataType;
            if (type == typeof(bool))
                return new BooleanArray(valueBuffer, nullBuffer, numberOfRows, nullCount, offset);
            else if (type == typeof(double))
                return new DoubleArray(valueBuffer, nullBuffer, numberOfRows, nullCount, offset);
            else if (type == typeof(float))
                return new FloatArray(valueBuffer, nullBuffer, numberOfRows, nullCount, offset);
            else if (type == typeof(int))
                return new Int32Array(valueBuffer, nullBuffer, numberOfRows, nullCount, offset);
            else if (type == typeof(long))
                return new Int64Array(valueBuffer, nullBuffer, numberOfRows, nullCount, offset);
            else if (type == typeof(sbyte))
                return new Int8Array(valueBuffer, nullBuffer, numberOfRows, nullCount, offset);
            else if (type == typeof(short))
                return new Int16Array(valueBuffer, nullBuffer, numberOfRows, nullCount, offset);
            else if (type == typeof(uint))
                return new UInt32Array(valueBuffer, nullBuffer, numberOfRows, nullCount, offset);
            else if (type == typeof(ulong))
                return new UInt64Array(valueBuffer, nullBuffer, numberOfRows, nullCount, offset);
            else if (type == typeof(ushort))
                return new UInt16Array(valueBuffer, nullBuffer, numberOfRows, nullCount, offset);
            else if (type == typeof(byte))
                return new UInt8Array(valueBuffer, nullBuffer, numberOfRows, nullCount, offset);
            else
                throw new NotImplementedException(type.ToString());
        }

        public new IReadOnlyList<T?> this[long startIndex, int length]
        {
            get
            {
                if (startIndex > Length)
                {
                    throw new ArgumentOutOfRangeException(nameof(startIndex));
                }
                return _columnContainer[startIndex, length];
            }
        }

        protected override IReadOnlyList<object> GetValues(long startIndex, int length)
        {
            if (startIndex > Length)
            {
                throw new ArgumentOutOfRangeException(nameof(startIndex));
            }

            var ret = new List<object>(length);
            long endIndex = Math.Min(Length, startIndex + length);
            for (long i = startIndex; i < endIndex; i++)
            {
                ret.Add(this[i]);
            }
            return ret;
        }

        internal T? GetTypedValue(long rowIndex) => _columnContainer[rowIndex];

        protected override object GetValue(long rowIndex) => GetTypedValue(rowIndex);

        protected override void SetValue(long rowIndex, object value)
        {
            if (value == null || value.GetType() == typeof(T))
            {
                _columnContainer[rowIndex] = (T?)value;
            }
            else
            {
                throw new ArgumentException(string.Format(Strings.MismatchedValueType, DataType), nameof(value));
            }
        }

        public new T? this[long rowIndex]
        {
            get => GetTypedValue(rowIndex);
            set
            {
                if (value == null || value.GetType() == typeof(T))
                {
                    _columnContainer[rowIndex] = value;
                }
                else
                {
                    throw new ArgumentException(string.Format(Strings.MismatchedValueType, DataType), nameof(value));
                }
            }
        }

        public override double Median()
        {
            // Not the most efficient implementation. Using a selection algorithm here would be O(n) instead of O(nLogn)
            if (Length == 0)
                return 0;
            PrimitiveDataFrameColumn<long> sortIndices = GetAscendingSortIndices();
            long middle = sortIndices.Length / 2;
            double middleValue = (double)Convert.ChangeType(this[sortIndices[middle].Value].Value, typeof(double));
            if (Length % 2 == 0)
            {
                double otherMiddleValue = (double)Convert.ChangeType(this[sortIndices[middle - 1].Value].Value, typeof(double));
                return (middleValue + otherMiddleValue) / 2;
            }
            else
            {
                return middleValue;
            }
        }

        public override double Mean()
        {
            if (Length == 0)
                return 0;
            return (double)Convert.ChangeType((T)Sum(), typeof(double)) / Length;
        }

        protected internal override void Resize(long length)
        {
            _columnContainer.Resize(length);
            Length = _columnContainer.Length;
        }

        public void Append(T? value)
        {
            _columnContainer.Append(value);
            Length++;
        }

        public void AppendMany(T? value, long count)
        {
            _columnContainer.AppendMany(value, count);
            Length += count;
        }

        public override long NullCount
        {
            get
            {
                Debug.Assert(_columnContainer.NullCount >= 0);
                return _columnContainer.NullCount;
            }
        }

        public bool IsValid(long index) => _columnContainer.IsValid(index);

        public IEnumerator<T?> GetEnumerator() => _columnContainer.GetEnumerator();

        protected override IEnumerator GetEnumeratorCore() => GetEnumerator();

        public override bool IsNumericColumn()
        {
            bool ret = true;
            if (typeof(T) == typeof(char) || typeof(T) == typeof(bool))
                ret = false;
            return ret;
        }

        /// <summary>
        /// Returns a new column with nulls replaced by value
        /// </summary>
        /// <param name="value"></param>
        public PrimitiveDataFrameColumn<T> FillNulls(T value, bool inPlace = false)
        {
            PrimitiveDataFrameColumn<T> column = inPlace ? this : Clone();
            column.ApplyElementwise((T? columnValue, long index) =>
            {
                if (columnValue.HasValue == false)
                    return value;
                else
                    return columnValue.Value;
            });
            return column;
        }

        protected override DataFrameColumn FillNullsImplementation(object value, bool inPlace)
        {
            T convertedValue = (T)Convert.ChangeType(value, typeof(T));
            return FillNulls(convertedValue, inPlace);
        }

        public override DataFrame ValueCounts()
        {
            Dictionary<T, ICollection<long>> groupedValues = GroupColumnValues<T>();
            PrimitiveDataFrameColumn<T> keys = new PrimitiveDataFrameColumn<T>("Values");
            PrimitiveDataFrameColumn<long> counts = new PrimitiveDataFrameColumn<long>("Counts");
            foreach (KeyValuePair<T, ICollection<long>> keyValuePair in groupedValues)
            {
                keys.Append(keyValuePair.Key);
                counts.Append(keyValuePair.Value.Count);
            }
            return new DataFrame(new List<DataFrameColumn> { keys, counts });
        }

        public override bool HasDescription() => IsNumericColumn();

        public override string ToString()
        {
            return $"{Name}: {_columnContainer.ToString()}";
        }

        public new PrimitiveDataFrameColumn<T> Clone(DataFrameColumn mapIndices, bool invertMapIndices, long numberOfNullsToAppend)
        {
            PrimitiveDataFrameColumn<T> clone;
            if (!(mapIndices is null))
            {
                Type dataType = mapIndices.DataType;
                if (dataType != typeof(long) && dataType != typeof(int) && dataType != typeof(bool))
                    throw new ArgumentException(String.Format(Strings.MultipleMismatchedValueType, typeof(long), typeof(int), typeof(bool)), nameof(mapIndices));
                if (dataType == typeof(long))
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
            Debug.Assert(!ReferenceEquals(clone, null));
            clone.AppendMany(null, numberOfNullsToAppend);
            return clone;
        }

        protected override DataFrameColumn CloneImplementation(DataFrameColumn mapIndices, bool invertMapIndices, long numberOfNullsToAppend)
        {
            return Clone(mapIndices, invertMapIndices, numberOfNullsToAppend);
        }

        private PrimitiveDataFrameColumn<T> Clone(PrimitiveDataFrameColumn<bool> boolColumn)
        {
            if (boolColumn.Length > Length)
                throw new ArgumentException(Strings.MapIndicesExceedsColumnLenth, nameof(boolColumn));
            PrimitiveDataFrameColumn<T> ret = new PrimitiveDataFrameColumn<T>(Name);
            for (long i = 0; i < boolColumn.Length; i++)
            {
                bool? value = boolColumn[i];
                if (value.HasValue && value.Value == true)
                    ret.Append(this[i]);
            }
            return ret;
        }

        private PrimitiveDataFrameColumn<T> CloneImplementation<U>(PrimitiveDataFrameColumn<U> mapIndices, bool invertMapIndices = false)
            where U : unmanaged
        {
            if (!mapIndices.IsNumericColumn())
                throw new ArgumentException(String.Format(Strings.MismatchedValueType, Strings.NumericColumnType), nameof(mapIndices));

            PrimitiveColumnContainer<T> retContainer;
            if (mapIndices.DataType == typeof(long))
            {
                retContainer = _columnContainer.Clone(mapIndices._columnContainer, typeof(long), invertMapIndices);
            }
            else if (mapIndices.DataType == typeof(int))
            {
                retContainer = _columnContainer.Clone(mapIndices._columnContainer, typeof(int), invertMapIndices);
            }
            else
                throw new NotImplementedException();
            PrimitiveDataFrameColumn<T> ret = new PrimitiveDataFrameColumn<T>(Name, retContainer);
            return ret;
        }

        public PrimitiveDataFrameColumn<T> Clone(PrimitiveDataFrameColumn<long> mapIndices = null, bool invertMapIndices = false)
        {
            if (mapIndices is null)
            {
                PrimitiveColumnContainer<T> newColumnContainer = _columnContainer.Clone();
                return new PrimitiveDataFrameColumn<T>(Name, newColumnContainer);
            }
            else
            {
                return CloneImplementation(mapIndices, invertMapIndices);
            }
        }

        public PrimitiveDataFrameColumn<T> Clone(PrimitiveDataFrameColumn<int> mapIndices, bool invertMapIndices = false)
        {
            return CloneImplementation(mapIndices, invertMapIndices);
        }

        public PrimitiveDataFrameColumn<T> Clone(IEnumerable<long> mapIndices)
        {
            IEnumerator<long> rows = mapIndices.GetEnumerator();
            PrimitiveDataFrameColumn<T> ret = new PrimitiveDataFrameColumn<T>(Name);
            ret._columnContainer._modifyNullCountWhileIndexing = false;
            long numberOfRows = 0;
            while (rows.MoveNext() && numberOfRows < Length)
            {
                numberOfRows++;
                long curRow = rows.Current;
                T? value = _columnContainer[curRow];
                ret[curRow] = value;
                if (!value.HasValue)
                    ret._columnContainer.NullCount++;
            }
            ret._columnContainer._modifyNullCountWhileIndexing = true;
            return ret;
        }

        internal PrimitiveDataFrameColumn<bool> CloneAsBoolColumn()
        {
            PrimitiveColumnContainer<bool> newColumnContainer = _columnContainer.CloneAsBoolContainer();
            return new PrimitiveDataFrameColumn<bool>(Name, newColumnContainer);
        }

        internal PrimitiveDataFrameColumn<double> CloneAsDoubleColumn()
        {
            PrimitiveColumnContainer<double> newColumnContainer = _columnContainer.CloneAsDoubleContainer();
            return new PrimitiveDataFrameColumn<double>(Name, newColumnContainer);
        }

        internal PrimitiveDataFrameColumn<decimal> CloneAsDecimalColumn()
        {
            PrimitiveColumnContainer<decimal> newColumnContainer = _columnContainer.CloneAsDecimalContainer();
            return new PrimitiveDataFrameColumn<decimal>(Name, newColumnContainer);
        }

        public override GroupBy GroupBy(int columnIndex, DataFrame parent)
        {
            Dictionary<T, ICollection<long>> dictionary = GroupColumnValues<T>();
            return new GroupBy<T>(parent, columnIndex, dictionary);
        }

        public override Dictionary<TKey, ICollection<long>> GroupColumnValues<TKey>()
        {
            if (typeof(TKey) == typeof(T))
            {
                Dictionary<T, ICollection<long>> multimap = new Dictionary<T, ICollection<long>>(EqualityComparer<T>.Default);
                for (int b = 0; b < _columnContainer.Buffers.Count; b++)
                {
                    ReadOnlyDataFrameBuffer<T> buffer = _columnContainer.Buffers[b];
                    ReadOnlySpan<T> readOnlySpan = buffer.ReadOnlySpan;
                    long previousLength = b * ReadOnlyDataFrameBuffer<T>.MaxCapacity;
                    for (int i = 0; i < readOnlySpan.Length; i++)
                    {
                        long currentLength = i + previousLength;
                        bool containsKey = multimap.TryGetValue(readOnlySpan[i], out ICollection<long> values);
                        if (containsKey)
                        {
                            values.Add(currentLength);
                        }
                        else
                        {
                            multimap.Add(readOnlySpan[i], new List<long>() { currentLength });
                        }
                    }
                }
                return multimap as Dictionary<TKey, ICollection<long>>;
            }
            else
            {
                throw new NotImplementedException(nameof(TKey));
            }
        }

        public void ApplyElementwise(Func<T?, long, T?> func) => _columnContainer.ApplyElementwise(func);

        /// <summary>
        /// Clips values beyond the specified thresholds
        /// </summary>
        /// <param name="lower">Minimum value. All values below this threshold will be set to it</param>
        /// <param name="upper">Maximum value. All values above this threshold will be set to it</param>
        public PrimitiveDataFrameColumn<T> Clip(T lower, T upper, bool inPlace = false)
        {
            PrimitiveDataFrameColumn<T> ret = inPlace ? this : Clone();

            Comparer<T> comparer = Comparer<T>.Default;
            for (long i = 0; i < ret.Length; i++)
            {
                T? value = ret[i];
                if (value == null)
                    continue;

                if (comparer.Compare(value.Value, lower) < 0)
                    ret[i] = lower;

                if (comparer.Compare(value.Value, upper) > 0)
                    ret[i] = upper;
            }
            return ret;
        }

        protected override DataFrameColumn ClipImplementation<U>(U lower, U upper, bool inPlace)
        {
            object convertedLower = Convert.ChangeType(lower, typeof(T));
            if (typeof(T) == typeof(U) || convertedLower != null)
                return Clip((T)convertedLower, (T)Convert.ChangeType(upper, typeof(T)), inPlace);
            else
                throw new ArgumentException(string.Format(Strings.MismatchedValueType, typeof(T)), nameof(U));
        }

        /// <summary>
        /// Returns a new column filtered by the lower and upper bounds
        /// </summary>
        /// <param name="lower"></param>
        /// <param name="upper"></param>
        public PrimitiveDataFrameColumn<T> Filter(T lower, T upper)
        {
            PrimitiveDataFrameColumn<T> ret = new PrimitiveDataFrameColumn<T>(Name);
            Comparer<T> comparer = Comparer<T>.Default;
            for (long i = 0; i < Length; i++)
            {
                T? value = this[i];
                if (value == null)
                    continue;

                if (comparer.Compare(value.Value, lower) >= 0 && comparer.Compare(value.Value, upper) <= 0)
                    ret.Append(value);
            }
            return ret;
        }

        protected override DataFrameColumn FilterImplementation<U>(U lower, U upper)
        {
            object convertedLower = Convert.ChangeType(lower, typeof(T));
            if (typeof(T) == typeof(U) || convertedLower != null)
                return Filter((T)convertedLower, (T)Convert.ChangeType(upper, typeof(T)));
            else
                throw new ArgumentException(string.Format(Strings.MismatchedValueType, typeof(T)), nameof(U));
        }

        public override DataFrameColumn Description()
        {
            float? max;
            float? min;
            float? mean;
            try
            {
                max = (float)Convert.ChangeType(Max(), typeof(float));
            }
            catch (Exception)
            {
                max = null;
            }
            try
            {
                min = (float)Convert.ChangeType(Min(), typeof(float));
            }
            catch (Exception)
            {
                min = null;
            }
            try
            {
                mean = (float)Convert.ChangeType(Sum(), typeof(float)) / Length;
            }
            catch (Exception)
            {
                mean = null;
            }
            PrimitiveDataFrameColumn<float> column = new PrimitiveDataFrameColumn<float>(Name);
            column.Append(Length - NullCount);
            column.Append(max);
            column.Append(min);
            column.Append(mean);
            return column;
        }

        protected internal override void AddDataViewColumn(DataViewSchema.Builder builder)
        {
            builder.AddColumn(Name, GetDataViewType());
        }

        private static DataViewType GetDataViewType()
        {
            if (typeof(T) == typeof(bool))
            {
                return BooleanDataViewType.Instance;
            }
            else if (typeof(T) == typeof(byte))
            {
                return NumberDataViewType.Byte;
            }
            else if (typeof(T) == typeof(double))
            {
                return NumberDataViewType.Double;
            }
            else if (typeof(T) == typeof(float))
            {
                return NumberDataViewType.Single;
            }
            else if (typeof(T) == typeof(int))
            {
                return NumberDataViewType.Int32;
            }
            else if (typeof(T) == typeof(long))
            {
                return NumberDataViewType.Int64;
            }
            else if (typeof(T) == typeof(sbyte))
            {
                return NumberDataViewType.SByte;
            }
            else if (typeof(T) == typeof(short))
            {
                return NumberDataViewType.Int16;
            }
            else if (typeof(T) == typeof(uint))
            {
                return NumberDataViewType.UInt32;
            }
            else if (typeof(T) == typeof(ulong))
            {
                return NumberDataViewType.UInt64;
            }
            else if (typeof(T) == typeof(ushort))
            {
                return NumberDataViewType.UInt16;
            }
            // The following 2 implementations are not ideal, but IDataView doesn't support
            // these types
            else if (typeof(T) == typeof(char))
            {
                return NumberDataViewType.UInt16;
            }
            else if (typeof(T) == typeof(decimal))
            {
                return NumberDataViewType.Double;
            }

            throw new NotSupportedException();
        }

        protected internal override Delegate GetDataViewGetter(DataViewRowCursor cursor)
        {
            // special cases for types that have NA values
            if (typeof(T) == typeof(float))
            {
                return CreateSingleValueGetterDelegate(cursor, (PrimitiveDataFrameColumn<float>)(object)this);
            }
            else if (typeof(T) == typeof(double))
            {
                return CreateDoubleValueGetterDelegate(cursor, (PrimitiveDataFrameColumn<double>)(object)this);
            }
            // special cases for types not supported
            else if (typeof(T) == typeof(char))
            {
                return CreateCharValueGetterDelegate(cursor, (PrimitiveDataFrameColumn<char>)(object)this);
            }
            else if (typeof(T) == typeof(decimal))
            {
                return CreateDecimalValueGetterDelegate(cursor, (PrimitiveDataFrameColumn<decimal>)(object)this);
            }
            return CreateValueGetterDelegate(cursor);
        }

        private ValueGetter<T> CreateValueGetterDelegate(DataViewRowCursor cursor) =>
            (ref T value) => value = this[cursor.Position].GetValueOrDefault();

        private static ValueGetter<float> CreateSingleValueGetterDelegate(DataViewRowCursor cursor, PrimitiveDataFrameColumn<float> column) =>
            (ref float value) => value = column[cursor.Position] ?? float.NaN;

        private static ValueGetter<double> CreateDoubleValueGetterDelegate(DataViewRowCursor cursor, PrimitiveDataFrameColumn<double> column) =>
            (ref double value) => value = column[cursor.Position] ?? double.NaN;

        private static ValueGetter<ushort> CreateCharValueGetterDelegate(DataViewRowCursor cursor, PrimitiveDataFrameColumn<char> column) =>
            (ref ushort value) => value = column[cursor.Position].GetValueOrDefault();

        private static ValueGetter<double> CreateDecimalValueGetterDelegate(DataViewRowCursor cursor, PrimitiveDataFrameColumn<decimal> column) =>
            (ref double value) => value = (double?)column[cursor.Position] ?? double.NaN;
    }
}
