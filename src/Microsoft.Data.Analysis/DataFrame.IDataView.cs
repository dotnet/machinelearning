// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Microsoft.Data.Analysis
{
    public partial class DataFrame : IDataView
    {        
        // TODO: support shuffling
        bool IDataView.CanShuffle => false;

        private DataViewSchema _schema;
        private DataViewSchema DataViewSchema
        {
            get
            {
                if (_schema != null)
                {
                    return _schema;
                }

                var schemaBuilder = new DataViewSchema.Builder();
                for (int i = 0; i < Columns.Count; i++)
                {
                    DataFrameColumn baseColumn = Columns[i];
                    baseColumn.AddDataViewColumn(schemaBuilder);
                }
                _schema = schemaBuilder.ToSchema();
                return _schema;
            }
        }

        DataViewSchema IDataView.Schema => DataViewSchema;

        long? IDataView.GetRowCount() => Rows.Count;

        private DataViewRowCursor GetRowCursorCore(IEnumerable<DataViewSchema.Column> columnsNeeded)
        {
            var activeColumns = new bool[DataViewSchema.Count];
            foreach (DataViewSchema.Column column in columnsNeeded)
            {
                if (column.Index < activeColumns.Length)
                {
                    activeColumns[column.Index] = true;
                }
            }

            return new RowCursor(this, activeColumns);
        }
        DataViewRowCursor IDataView.GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand)
        {
            return GetRowCursorCore(columnsNeeded);
        }

        DataViewRowCursor[] IDataView.GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand)
        {
            // TODO: change to support parallel cursors
            return new DataViewRowCursor[] { GetRowCursorCore(columnsNeeded) };
        }

        private sealed class RowCursor : DataViewRowCursor
        {
            private bool _disposed;
            private long _position;
            private readonly DataFrame _dataFrame;
            private readonly Delegate[] _getters;

            public RowCursor(DataFrame dataFrame, bool[] activeColumns)
            {
                Debug.Assert(dataFrame != null);
                Debug.Assert(activeColumns != null);

                _position = -1;
                _dataFrame = dataFrame;
                _getters = new Delegate[Schema.Count];
                for (int i = 0; i < _getters.Length; i++)
                {
                    if (!activeColumns[i])
                        continue;
                    _getters[i] = CreateGetterDelegate(i);
                    Debug.Assert(_getters[i] != null);
                }
            }

            public override long Position => _position;
            public override long Batch => 0;
            public override DataViewSchema Schema => _dataFrame.DataViewSchema;

            protected override void Dispose(bool disposing)
            {
                if (_disposed)
                    return;
                if (disposing)
                {
                    _position = -1;
                }
                _disposed = true;
                base.Dispose(disposing);
            }

            private Delegate CreateGetterDelegate(int col)
            {
                DataFrameColumn column = _dataFrame.Columns[col];
                return column.GetDataViewGetter(this);
            }

            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                if (!IsColumnActive(column))
                    throw new ArgumentOutOfRangeException(nameof(column));

                return (ValueGetter<TValue>)_getters[column.Index];
            }

            public override ValueGetter<DataViewRowId> GetIdGetter()
            {
                return (ref DataViewRowId value) => value = new DataViewRowId((ulong)_position, 0);
            }

            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                return _getters[column.Index] != null;
            }

            public override bool MoveNext()
            {
                if (_disposed)
                    return false;
                _position++;
                return _position < _dataFrame.Rows.Count;
            }
        }
    }
}
