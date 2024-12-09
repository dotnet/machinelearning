// Licensed to the .NET Foundation under one or more agreements.
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
    /// Column to hold VBuffer
    /// </summary>
    public partial class VBufferDataFrameColumn<T> : DataFrameColumn, IEnumerable<VBuffer<T>>
    {
        public static int MaxCapacity = ArrayUtility.ArrayMaxSize / Unsafe.SizeOf<VBuffer<T>>();

        private readonly List<List<VBuffer<T>>> _vBuffers = new List<List<VBuffer<T>>>(); // To store more than intMax number of vbuffers

        /// <summary>
        /// Constructs an empty VBufferDataFrameColumn with the given <paramref name="name"/>.
        /// </summary>
        /// <param name="name">The name of the column.</param>
        /// <param name="length">Length of values</param>
        public VBufferDataFrameColumn(string name, long length = 0) : base(name, length, typeof(VBuffer<T>))
        {
            int numberOfBuffersRequired = (int)(length / MaxCapacity + 1);
            for (int i = 0; i < numberOfBuffersRequired; i++)
            {
                int bufferLen = (int)Math.Min(MaxCapacity, length - _vBuffers.Count * MaxCapacity);
                List<VBuffer<T>> buffer = new List<VBuffer<T>>(bufferLen);
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

        public override long NullCount => 0;

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
            if (lastBuffer.Count == MaxCapacity)
            {
                lastBuffer = new List<VBuffer<T>>();
                _vBuffers.Add(lastBuffer);
            }
            lastBuffer.Add(value);
            Length++;
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
            return GetTypedValue(rowIndex);
        }

        protected VBuffer<T> GetTypedValue(long rowIndex)
        {
            int bufferIndex = GetBufferIndexContainingRowIndex(rowIndex);
            return _vBuffers[bufferIndex][(int)(rowIndex % MaxCapacity)];
        }

        protected override IReadOnlyList<object> GetValues(long startIndex, int length)
        {
            var ret = new List<object>();
            int bufferIndex = GetBufferIndexContainingRowIndex(startIndex);
            int bufferOffset = (int)(startIndex % MaxCapacity);
            while (ret.Count < length && bufferIndex < _vBuffers.Count)
            {
                for (int i = bufferOffset; ret.Count < length && i < _vBuffers[bufferIndex].Count; i++)
                {
                    ret.Add(_vBuffers[bufferIndex][i]);
                }
                bufferIndex++;
                bufferOffset = 0;
            }
            return ret;
        }

        protected override void SetValue(long rowIndex, object value)
        {
            if (value == null)
            {
                throw new NotSupportedException("Null values are not supported by VBufferDataFrameColumn");
            }
            else if (value is VBuffer<T> vbuffer)
            {
                SetTypedValue(rowIndex, vbuffer);
            }
            else
            {
                throw new ArgumentException(string.Format(Strings.MismatchedValueType, typeof(VBuffer<T>)), nameof(value));
            }
        }

        protected void SetTypedValue(long rowIndex, VBuffer<T> value)
        {
            int bufferIndex = GetBufferIndexContainingRowIndex(rowIndex);
            _vBuffers[bufferIndex][(int)(rowIndex % MaxCapacity)] = value;
        }

        public new VBuffer<T> this[long rowIndex]
        {
            get => GetTypedValue(rowIndex);
            set => SetTypedValue(rowIndex, value);
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

        private VBufferDataFrameColumn<T> CloneImplementation(PrimitiveDataFrameColumn<bool> boolColumn)
        {
            if (boolColumn.Length > Length)
                throw new ArgumentException(Strings.MapIndicesExceedsColumnLength, nameof(boolColumn));
            VBufferDataFrameColumn<T> ret = new VBufferDataFrameColumn<T>(Name, 0);
            for (long i = 0; i < boolColumn.Length; i++)
            {
                bool? value = boolColumn[i];
                if (value.HasValue && value.Value == true)
                    ret.Append(this[i]);
            }
            return ret;
        }

        private VBufferDataFrameColumn<T> CloneImplementation(PrimitiveDataFrameColumn<long> mapIndices, bool invertMapIndices = false)
        {
            mapIndices = mapIndices ?? throw new ArgumentNullException(nameof(mapIndices));
            var ret = new VBufferDataFrameColumn<T>(Name, mapIndices.Length);

            long rowIndex = 0;
            for (int b = 0; b < mapIndices.ColumnContainer.Buffers.Count; b++)
            {
                var span = mapIndices.ColumnContainer.Buffers[b].ReadOnlySpan;
                var validitySpan = mapIndices.ColumnContainer.NullBitMapBuffers[b].ReadOnlySpan;

                for (int i = 0; i < span.Length; i++)
                {
                    long index = invertMapIndices ? mapIndices.Length - 1 - rowIndex : rowIndex;
                    ret[index] = BitUtility.IsValid(validitySpan, i) ? this[span[i]] : default;
                    rowIndex++;
                }
            }

            return ret;
        }

        private VBufferDataFrameColumn<T> CloneImplementation(PrimitiveDataFrameColumn<int> mapIndices, bool invertMapIndices = false)
        {
            mapIndices = mapIndices ?? throw new ArgumentNullException(nameof(mapIndices));
            var ret = new VBufferDataFrameColumn<T>(Name, mapIndices.Length);

            long rowIndex = 0;
            for (int b = 0; b < mapIndices.ColumnContainer.Buffers.Count; b++)
            {
                var span = mapIndices.ColumnContainer.Buffers[b].ReadOnlySpan;
                var validitySpan = mapIndices.ColumnContainer.NullBitMapBuffers[b].ReadOnlySpan;

                for (int i = 0; i < span.Length; i++)
                {
                    long index = invertMapIndices ? mapIndices.Length - 1 - rowIndex : rowIndex;
                    ret[index] = BitUtility.IsValid(validitySpan, i) ? this[span[i]] : default;
                    rowIndex++;
                }
            }

            return ret;
        }

        public new VBufferDataFrameColumn<T> Clone(DataFrameColumn mapIndices, bool invertMapIndices, long numberOfNullsToAppend)
        {
            return (VBufferDataFrameColumn<T>)CloneImplementation(mapIndices, invertMapIndices, numberOfNullsToAppend);
        }

        public new VBufferDataFrameColumn<T> Clone(long numberOfNullsToAppend = 0)
        {
            return (VBufferDataFrameColumn<T>)CloneImplementation(numberOfNullsToAppend);
        }

        protected override DataFrameColumn CloneImplementation(DataFrameColumn mapIndices, bool invertMapIndices = false, long numberOfNullsToAppend = 0)
        {
            VBufferDataFrameColumn<T> clone;
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
            }
            else
            {
                clone = Clone();
            }

            return clone;
        }

        protected override DataFrameColumn CloneImplementation(long numberOfNullsToAppend)
        {
            var ret = new VBufferDataFrameColumn<T>(Name, Length);

            for (long i = 0; i < Length; i++)
                ret[i] = this[i];

            return ret;
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

        protected override DataFrameColumn FillNullsImplementation(object value, bool inPlace)
        {
            //Do nothing as VBufferColumn doesn't have null values
            return inPlace ? this : Clone();
        }

        protected override DataFrameColumn DropNullsImplementation()
        {
            //Do nothing as VBufferColumn doesn't have null values
            return Clone();
        }

        protected internal override PrimitiveDataFrameColumn<long> GetSortIndices(bool ascending, bool putNullValuesLast) => throw new NotImplementedException();
    }
}
