// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// A data view that is a reservoir sample of a passed-in, parent data view.
    /// All of the sampled data is held IN MEMORY, so be careful not to make the
    /// sample size too large.
    /// </summary>
    internal class ReservoirSampledDataView : IDataView
    {
        private readonly IDataView _originalData;
        private readonly Random _random;
        private readonly object[] _sampledColumns;
        private readonly int _sampleSize;

        private bool _initialized = false;
        private long _originalDataRowCount;

        public ReservoirSampledDataView(IDataView data, long sampleSize)
        {
            _originalData = data;
            _random = new Random();
            _sampledColumns = new object[Schema.Count];
            _sampleSize = sampleSize;
        }

        public bool CanShuffle => false;

        public DataViewSchema Schema => _originalData.Schema;

        public long? GetRowCount()
        {
            if (!_initialized)
            {
                return null;
            }
            return Math.Min(_originalDataRowCount, _sampleSize);
        }

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            if (!_initialized)
            {
                Initialize();
            }

            return new Cursor(Schema, _sampledColumns, GetRowCount().Value);
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            if (!_initialized)
            {
                Initialize();
            }

            // note about parallelism
            var cursors = new DataViewRowCursor[n];
            for (var i = 0; i < n; i++)
            {
                cursors[i] = new Cursor(Schema, _sampledColumns, GetRowCount().Value);
            }
            return cursors;
        }

        /// <summary>
        /// Initializes the data view.
        /// Iterates through every record in the data view to build the reservoir sample.
        /// </summary>
        private void Initialize()
        {
            var cursor = _originalData.GetRowCursor(Schema);

            // Init data structures for each column
            var columnSampleBuilders = new ColumnSampleBuilder[Schema.Count];
            for (var i = 0; i < Schema.Count; i++)
            {
                // Ignore inactive columns
                if (!cursor.IsColumnActive(Schema[i]))
                {
                    continue;
                }

                var colType = Schema[i].Type;

                // Init SampledColumn
                Type sampledColumnType = colType.IsVector() ?
                    typeof(VectorSampledColumn<>).MakeGenericType(colType.GetItemType().RawType) :
                    typeof(SingleSlotSampledColumn<>).MakeGenericType(colType.RawType);
                var constructor = sampledColumnType.GetConstructor(new[] { typeof(int) });
                _sampledColumns[i] = constructor.Invoke(new object[] { _sampleSize });

                // Init ColumnSampleBuilder
                var columnSampleBuilderType = typeof(ColumnSampleBuilder<>).MakeGenericType(colType.RawType);
                var originalColumnGetter = MLNetUtils.MarshalInvoke(GetGetter<int>, colType.RawType, cursor, i);
                constructor = columnSampleBuilderType.GetConstructor(new[] { sampledColumnType, originalColumnGetter.GetType() });
                columnSampleBuilders[i] = constructor.Invoke(new object[] { _sampledColumns[i], originalColumnGetter }) as ColumnSampleBuilder;
            }

            var sampleBuilder = new SampleBuilder(columnSampleBuilders);

            // Reservoir sample over the entire dataset
            long rowIdx = 0;
            while (cursor.MoveNext())
            {
                // Cache first _sampleSize rows
                if (rowIdx < _sampleSize)
                {
                    sampleBuilder.StoreNextRow(rowIdx);
                }
                else
                {
                    // Reservoir sample row
                    var idx = GetRandomLong(rowIdx + 1);
                    if (idx < _sampleSize)
                    {
                        sampleBuilder.StoreNextRow(idx);
                    }
                    else
                    {
                        sampleBuilder.DiscardNextRow();
                    }
                }

                rowIdx++;
            }

            _originalDataRowCount = rowIdx;
            _initialized = true;
        }

        private Delegate GetGetter<T>(DataViewRowCursor cursor, int i)
        {
            return cursor.GetGetter<T>(Schema[i]);
        }

        private long GetRandomLong(long max)
        {
            var bytes = new byte[8];
            _random.NextBytes(bytes);
            var longRand = BitConverter.ToInt64(bytes, 0);
            return Math.Abs(longRand % max);
        }

        /// <summary>
        /// Cursor that returns data from reservoir sample.
        /// </summary>
        private class Cursor : DataViewRowCursor
        {
            public override long Position => _position;

            public override long Batch => throw new NotImplementedException();

            public override DataViewSchema Schema => _schema;

            private readonly long _rowCount;
            private readonly object[] _sampledColumns;
            private readonly DataViewSchema _schema;

            private long _position = -1;

            public Cursor(DataViewSchema schema, object[] sampledColumns, long rowCount)
            {
                _schema = schema;
                _sampledColumns = sampledColumns;
                _rowCount = rowCount;
            }

            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                var sampledColumn = _sampledColumns[column.Index] as SampledColumn<TValue>;
                ValueGetter<TValue> getter =
                    (ref TValue value) =>
                    {
                        sampledColumn.GetValue(_position, ref value);
                    };
                return getter;
            }

            public override ValueGetter<DataViewRowId> GetIdGetter()
            {
                throw new NotImplementedException();
            }
            
            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                return _sampledColumns[column.Index] != null;
            }

            public override bool MoveNext()
            {
                if (++_position == _rowCount)
                {
                    return false;
                }
                return true;
            }
        }

        /// <summary>
        /// Class that wraps the raw values for each column of the data view.
        /// </summary>
        private abstract class SampledColumn<T>
        {
            private readonly T[] Values;

            public SampledColumn(long size)
            {
                Values = new T[size];
            }

            public void SetValue(long idx, in T value)
            {
                Copy(value, ref Values[idx]);
            }

            public void GetValue(long idx, ref T value)
            {
                Copy(Values[idx], ref value);
            }

            protected abstract void Copy(in T src, ref T dst);
        }

        /// <summary>
        /// Sampled column for non-vector columns.
        /// </summary>
        private class SingleSlotSampledColumn<T> : SampledColumn<T>
        {
            public SingleSlotSampledColumn(int size) : base(size)
            {
            }

            protected override void Copy(in T src, ref T dst)
            {
                dst = src;
            }
        }

        /// <summary>
        /// Sampled column for vector columns.
        /// </summary>
        private class VectorSampledColumn<T> : SampledColumn<VBuffer<T>>
        {
            public VectorSampledColumn(int size) : base(size)
            {
            }

            protected override void Copy(in VBuffer<T> src, ref VBuffer<T> dst)
            {
                src.CopyTo(ref dst);
            }
        }

        /// <summary>
        /// This class pulls from an original / parent data view column
        /// to populate its corresponding sampled column.
        /// </summary>
        private class ColumnSampleBuilder<T> : ColumnSampleBuilder
        {
            private SampledColumn<T> _sampledColumn;
            private ValueGetter<T> _originalColumnValueGetter;

            public ColumnSampleBuilder(SampledColumn<T> sampledColumn, ValueGetter<T> originalColumnValueGetter)
            {
                _sampledColumn = sampledColumn;
                _originalColumnValueGetter = originalColumnValueGetter;
            }

            public override void DiscardNext()
            {
                var value = default(T);
                _originalColumnValueGetter.Invoke(ref value);
            }

            public override void StoreNext(long idx)
            {
                var value = default(T);
                _originalColumnValueGetter.Invoke(ref value);
                _sampledColumn.SetValue(idx, value);
            }
        }

        private abstract class ColumnSampleBuilder
        {
            public abstract void DiscardNext();
            public abstract void StoreNext(long idx);
        }
        
        /// <summary>
        /// Wraps all column sample builders to perform row-wise
        /// sample building operations.
        /// </summary>
        private class SampleBuilder
        {
            private readonly IEnumerable<ColumnSampleBuilder> _columnSampleBuilders;

            public SampleBuilder(IEnumerable<ColumnSampleBuilder> columnSampleBuilders)
            {
                _columnSampleBuilders = columnSampleBuilders;
            }

            public void DiscardNextRow()
            {
                foreach (var columnSampleBuilder in _columnSampleBuilders)
                {
                    columnSampleBuilder.DiscardNext();
                }
            }

            public void StoreNextRow(long idx)
            {
                foreach (var columnSampleBuilder in _columnSampleBuilders)
                {
                    columnSampleBuilder.StoreNext(idx);
                }
            }
        }
    }
}
