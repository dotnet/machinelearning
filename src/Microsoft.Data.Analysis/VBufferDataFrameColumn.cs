// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Apache.Arrow;
using Apache.Arrow.Types;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Microsoft.Data.Analysis
{
    /// <summary>
    /// Column to hold VBuffer
    /// </summary>
    public partial class VBufferDataFrameColumn<T> : DataFrameColumn, IEnumerable<VBuffer<T>>
    {
        private readonly List<List<VBuffer<T>>> _vBuffers = new List<List<VBuffer<T>>>(); // To store more than intMax number of vbuffers

        /// <summary>
        /// Constructs an empty VBufferDataFrameColumn with the given <paramref name="name"/>.
        /// </summary>
        /// <param name="name">The name of the column.</param>
        /// <param name="length">Length of values</param>
        public VBufferDataFrameColumn(string name, long length = 0) : base(name, length, typeof(VBuffer<T>))
        {
            int numberOfBuffersRequired = Math.Max((int)(length / int.MaxValue), 1);
            for (int i = 0; i < numberOfBuffersRequired; i++)
            {
                long bufferLen = length - _vBuffers.Count * int.MaxValue;
                List<VBuffer<T>> buffer = new List<VBuffer<T>>((int)Math.Min(int.MaxValue, bufferLen));
                _vBuffers.Add(buffer);
                for (int j = 0; j < bufferLen; j++)
                {
                    buffer.Add(default);
                }
            }
        }

        public VBufferDataFrameColumn(string name, IEnumerable<VBuffer<T>> values) : base(name, 0, typeof(VBuffer<T>))
        {
            values = values ?? throw new ArgumentNullException(nameof(values));
            if (_vBuffers.Count == 0)
            {
                _vBuffers.Add(new List<VBuffer<T>>());
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
                Append(default);
            }
        }

        public void Append(VBuffer<T> value)
        {
            List<VBuffer<T>> lastBuffer = _vBuffers[_vBuffers.Count - 1];
            if (lastBuffer.Count == int.MaxValue)
            {
                lastBuffer = new List<VBuffer<T>>();
                _vBuffers.Add(lastBuffer);
            }
            lastBuffer.Add(value);
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
            return _vBuffers[bufferIndex][(int)rowIndex];
        }

        protected override IReadOnlyList<object> GetValues(long startIndex, int length)
        {
            var ret = new List<object>();
            int bufferIndex = GetBufferIndexContainingRowIndex(ref startIndex);
            while (ret.Count < length && bufferIndex < _vBuffers.Count)
            {
                for (int i = (int)startIndex; ret.Count < length && i < _vBuffers[bufferIndex].Count; i++)
                {
                    ret.Add(_vBuffers[bufferIndex][i]);
                }
                bufferIndex++;
                startIndex = 0;
            }
            return ret;
        }

        protected override void SetValue(long rowIndex, object value)
        {
            if (value == null || value is VBuffer<T>)
            {
                int bufferIndex = GetBufferIndexContainingRowIndex(ref rowIndex);
                var oldValue = this[rowIndex];
                _vBuffers[bufferIndex][(int)rowIndex] = (VBuffer<T>)value;
                if (!oldValue.Equals((VBuffer<T>)value))
                {
                    if (value == null)
                        _nullCount++;
                    if (oldValue.Length == 0 && _nullCount > 0)
                        _nullCount--;
                }
            }
            else
            {
                throw new ArgumentException(string.Format(Strings.MismatchedValueType, typeof(VBuffer<T>)), nameof(value));
            }
        }

        public new VBuffer<T> this[long rowIndex]
        {
            get => (VBuffer<T>)GetValue(rowIndex);
            set => SetValue(rowIndex, value);
        }

        /// <summary>
        /// Returns an enumerator that iterates through the VBuffer values in this column.
        /// </summary>
        public IEnumerator<VBuffer<T>> GetEnumerator()
        {
            foreach (List<VBuffer<T>> buffer in _vBuffers)
            {
                foreach (VBuffer<T> value in buffer)
                {
                    yield return value;
                }
            }
        }

        /// <inheritdoc/>
        protected override IEnumerator GetEnumeratorCore() => GetEnumerator();

        /// <inheritdoc/>
        protected internal override void AddDataViewColumn(DataViewSchema.Builder builder)
        {
            builder.AddColumn(Name, GetDataViewType());
        }

        /// <inheritdoc/>
        protected internal override Delegate GetDataViewGetter(DataViewRowCursor cursor)
        {
            return CreateValueGetterDelegate(cursor);
        }

        private ValueGetter<VBuffer<T>> CreateValueGetterDelegate(DataViewRowCursor cursor) =>
            (ref VBuffer<T> value) => value = this[cursor.Position];

        public override Dictionary<long, ICollection<long>> GetGroupedOccurrences(DataFrameColumn other, out HashSet<long> otherColumnNullIndices)
        {
            return GetGroupedOccurrences<string>(other, out otherColumnNullIndices);
        }

        protected internal override Delegate GetValueGetterUsingCursor(DataViewRowCursor cursor, DataViewSchema.Column schemaColumn)
        {
            return cursor.GetGetter<VBuffer<T>>(schemaColumn);
        }

        protected internal override void AddValueUsingCursor(DataViewRowCursor cursor, Delegate getter)
        {
            long row = cursor.Position;
            VBuffer<T> value = default;
            Debug.Assert(getter != null, "Excepted getter to be valid");

            (getter as ValueGetter<VBuffer<T>>)(ref value);

            if (Length > row)
            {
                this[row] = value;
            }
            else if (Length == row)
            {
                Append(value);
            }
            else
            {
                throw new IndexOutOfRangeException(nameof(row));
            }
        }

        private VBufferDataFrameColumn<T> Clone(PrimitiveDataFrameColumn<bool> boolColumn)
        {
            if (boolColumn.Length > Length)
                throw new ArgumentException(Strings.MapIndicesExceedsColumnLenth, nameof(boolColumn));
            VBufferDataFrameColumn<T> ret = new VBufferDataFrameColumn<T>(Name, 0);
            for (long i = 0; i < boolColumn.Length; i++)
            {
                bool? value = boolColumn[i];
                if (value.HasValue && value.Value == true)
                    ret.Append(this[i]);
            }
            return ret;
        }

        private VBufferDataFrameColumn<T> Clone(PrimitiveDataFrameColumn<long> mapIndices = null, bool invertMapIndex = false)
        {
            if (mapIndices is null)
            {
                VBufferDataFrameColumn<T> ret = new VBufferDataFrameColumn<T>(Name, Length);
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

        private VBufferDataFrameColumn<T> Clone(PrimitiveDataFrameColumn<int> mapIndices, bool invertMapIndex = false)
        {
            return CloneImplementation(mapIndices, invertMapIndex);
        }

        private VBufferDataFrameColumn<T> CloneImplementation<U>(PrimitiveDataFrameColumn<U> mapIndices, bool invertMapIndices = false, long numberOfNullsToAppend = 0)
           where U : unmanaged
        {
            mapIndices = mapIndices ?? throw new ArgumentNullException(nameof(mapIndices));
            VBufferDataFrameColumn<T> ret = new VBufferDataFrameColumn<T>(Name, mapIndices.Length);

            List<VBuffer<T>> setBuffer = ret._vBuffers[0];
            long setBufferMinRange = 0;
            long setBufferMaxRange = int.MaxValue;
            List<VBuffer<T>> getBuffer = _vBuffers[0];
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
                        setBuffer = ret._vBuffers[bufferIndex];
                        setBufferMinRange = bufferIndex * maxCapacity;
                        setBufferMaxRange = (bufferIndex + 1) * maxCapacity;
                    }
                    index -= setBufferMinRange;

                    if (mapIndex.Value < getBufferMinRange || mapIndex.Value >= getBufferMaxRange)
                    {
                        int bufferIndex = (int)(mapIndex.Value / maxCapacity);
                        getBuffer = _vBuffers[bufferIndex];
                        getBufferMinRange = bufferIndex * maxCapacity;
                        getBufferMaxRange = (bufferIndex + 1) * maxCapacity;
                    }
                    int bufferLocalMapIndex = (int)(mapIndex - getBufferMinRange);
                    VBuffer<T> value = getBuffer[bufferLocalMapIndex];
                    setBuffer[(int)index] = value;

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

                    VBuffer<T> value = getBuffer[mapIndex.Value];
                    setBuffer[(int)index] = value;

                    return mapIndex;
                });
            }
            else
            {
                Debug.Assert(false, nameof(mapIndices.DataType));
            }

            return ret;
        }

        public new VBufferDataFrameColumn<T> Clone(DataFrameColumn mapIndices, bool invertMapIndices, long numberOfNullsToAppend)
        {
            VBufferDataFrameColumn<T> clone;
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

            return clone;
        }

        protected override DataFrameColumn CloneImplementation(DataFrameColumn mapIndices = null, bool invertMapIndices = false, long numberOfNullsToAppend = 0)
        {
            return Clone(mapIndices, invertMapIndices, numberOfNullsToAppend);
        }

        private static VectorDataViewType GetDataViewType()
        {
            if (typeof(T) == typeof(bool))
            {
                return new VectorDataViewType(BooleanDataViewType.Instance);
            }
            else if (typeof(T) == typeof(byte))
            {
                return new VectorDataViewType(NumberDataViewType.Byte);
            }
            else if (typeof(T) == typeof(double))
            {
                return new VectorDataViewType(NumberDataViewType.Double);
            }
            else if (typeof(T) == typeof(float))
            {
                return new VectorDataViewType(NumberDataViewType.Single);
            }
            else if (typeof(T) == typeof(int))
            {
                return new VectorDataViewType(NumberDataViewType.Int32);
            }
            else if (typeof(T) == typeof(long))
            {
                return new VectorDataViewType(NumberDataViewType.Int64);
            }
            else if (typeof(T) == typeof(sbyte))
            {
                return new VectorDataViewType(NumberDataViewType.SByte);
            }
            else if (typeof(T) == typeof(short))
            {
                return new VectorDataViewType(NumberDataViewType.Int16);
            }
            else if (typeof(T) == typeof(uint))
            {
                return new VectorDataViewType(NumberDataViewType.UInt32);
            }
            else if (typeof(T) == typeof(ulong))
            {
                return new VectorDataViewType(NumberDataViewType.UInt64);
            }
            else if (typeof(T) == typeof(ushort))
            {
                return new VectorDataViewType(NumberDataViewType.UInt16);
            }
            else if (typeof(T) == typeof(char))
            {
                return new VectorDataViewType(NumberDataViewType.UInt16);
            }
            else if (typeof(T) == typeof(decimal))
            {
                return new VectorDataViewType(NumberDataViewType.Double);
            }
            else if (typeof(T) == typeof(ReadOnlyMemory<char>))
            {
                return new VectorDataViewType(TextDataViewType.Instance);
            }

            throw new NotSupportedException();
        }
    }
}
