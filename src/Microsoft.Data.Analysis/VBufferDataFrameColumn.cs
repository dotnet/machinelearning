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
        // private readonly IList<List<VBuffer<T>>> _dataBuffers;

        private readonly List<VBuffer<T>> _vBuffers = new List<VBuffer<T>>();

        /// <summary>
        /// Constructs an empty <see cref="ArrowStringDataFrameColumn"/> with the given <paramref name="name"/>.
        /// </summary>
        /// <param name="name">The name of the column.</param>
        public VBufferDataFrameColumn(string name) : base(name, 0, typeof(VBuffer<T>))
        {
            _vBuffers = new List<VBuffer<T>>();
            _nullCount = 0;
        }

        public VBufferDataFrameColumn(string name, IEnumerable<VBuffer<T>> values) : base(name, 0, typeof(VBuffer<T>))
        {
            _vBuffers = new List<VBuffer<T>>();

            values = values ?? throw new ArgumentNullException(nameof(values));
            if (_vBuffers.Count == 0)
            {
                _vBuffers.Add(new VBuffer<T>());
            }
            foreach (var value in values)
            {
                Append(value);
            }
        }

        private readonly long _nullCount;

        /// <inheritdoc/>
        public override long NullCount => _nullCount;

        /// <summary>
        /// Indicates if the value at this <paramref name="index"/> is <see langword="null" />.
        /// </summary>
        /// <param name="index">The index to look up.</param>
        /// <returns>A boolean value indicating the validity at this <paramref name="index"/>.</returns>
        public bool IsValid(long index) => NullCount == 0;

        private void Append(VBuffer<T> value)
        {
            Length++;
            _vBuffers.Add(value);
        }

        /// <inheritdoc/>
        protected override object GetValue(long rowIndex) => GetValueImplementation(rowIndex);

        private VBuffer<T> GetValueImplementation(long rowIndex)
        {
            if (!IsValid(rowIndex))
            {
                throw new ArgumentOutOfRangeException(nameof(rowIndex));
            }
            return _vBuffers.ElementAt((int)rowIndex);
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
        public new VBuffer<T> this[long rowIndex]
        {
            get => GetValueImplementation(rowIndex);
            set => throw new NotSupportedException(Strings.ImmutableColumn);
        }

        /// <summary>
        /// Returns an enumerator that iterates through the string values in this column.
        /// </summary>
        public IEnumerator<VBuffer<T>> GetEnumerator()
        {
            for (long i = 0; i < Length; i++)
            {
                yield return this[i];
            }
        }

        /// <inheritdoc/>
        protected override IEnumerator GetEnumeratorCore() => GetEnumerator();

        /// <inheritdoc/>
        public override DataFrameColumn Sort(bool ascending = true) => throw new NotSupportedException();

        /// <inheritdoc/>
        public override DataFrameColumn Clone(DataFrameColumn mapIndices = null, bool invertMapIndices = false, long numberOfNullsToAppend = 0)
        {
            throw new NotImplementedException();
        }

        /// <inheritdoc/>
        public VBufferDataFrameColumn<T> FillNulls(VBuffer<T> value, bool inPlace = false)
        {
            if (inPlace)
            {
                // For now throw an exception if inPlace = true.
                throw new NotSupportedException();
            }

            VBufferDataFrameColumn<T> ret = new VBufferDataFrameColumn<T>(Name);
            for (long i = 0; i < Length; i++)
            {
                ret.Append(value);
            }
            return ret;
        }

        protected override DataFrameColumn FillNullsImplementation(object value, bool inPlace)
        {
            if (value is VBuffer<T> valueBuffer)
            {
                return FillNulls(valueBuffer, inPlace);
            }
            else
            {
                throw new ArgumentException(String.Format(Strings.MismatchedValueType, typeof(VBuffer<T>)), nameof(value));
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


        private ValueGetter<VBuffer<T>> CreateValueGetterDelegate(DataViewRowCursor cursor) =>
            (ref VBuffer<T> value) => value = this[cursor.Position];

        /// <summary>
        /// Returns a boolean column that is the result of an elementwise equality comparison of each value in the column with <paramref name="value"/>
        /// </summary>
        public PrimitiveDataFrameColumn<bool> ElementwiseEquals(VBuffer<T> value)
        {
            throw new NotImplementedException();
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ElementwiseEquals<U>(U value)
        {
            if (value is DataFrameColumn column)
            {
                return ElementwiseEquals(column);
            }
            return ElementwiseEquals(value.ToString());
        }

        public override Dictionary<long, ICollection<long>> GetGroupedOccurrences(DataFrameColumn other, out HashSet<long> otherColumnNullIndices)
        {
            return GetGroupedOccurrences<string>(other, out otherColumnNullIndices);
        }

        protected internal override Delegate GetValueGetterUsingCursor(DataViewRowCursor cursor, DataViewSchema.Column schemaColumn)
        {
            return cursor.GetGetter<VBuffer<T>>(schemaColumn);
        }

        IEnumerator<VBuffer<T>> IEnumerable<VBuffer<T>>.GetEnumerator()
        {
            throw new NotImplementedException();
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
    }
}
