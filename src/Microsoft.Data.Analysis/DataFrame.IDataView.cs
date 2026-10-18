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
        bool IDataView.CanShuffle => true;

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

        private static int[] GetRandomPermutation(Random rand, long count)
        {
            int n = (int)count;
            int[] permutation = new int[n];
            for (int i = 0; i < n; i++)
            {
                permutation[i] = i;
            }
            for (int i = n - 1; i > 0; i--)
            {
                int j = rand.Next(i + 1);
                int temp = permutation[i];
                permutation[i] = permutation[j];
                permutation[j] = temp;
            }
            return permutation;
        }

        private DataViewRowCursor GetRowCursorCore(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand)
        {
            var activeColumns = new bool[DataViewSchema.Count];
            foreach (DataViewSchema.Column column in columnsNeeded)
            {
                if (column.Index < activeColumns.Length)
                {
                    activeColumns[column.Index] = true;
                }
            }

            int[] permutation = rand != null ? GetRandomPermutation(rand, Rows.Count) : null;
            return new RowCursor(this, activeColumns, permutation, 0, Rows.Count, 0);
        }

        DataViewRowCursor IDataView.GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand)
        {
            return GetRowCursorCore(columnsNeeded, rand);
        }

        DataViewRowCursor[] IDataView.GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand)
        {
            if (n <= 1 || Rows.Count == 0)
            {
                return new DataViewRowCursor[] { GetRowCursorCore(columnsNeeded, rand) };
            }

            var activeColumns = new bool[DataViewSchema.Count];
            foreach (DataViewSchema.Column column in columnsNeeded)
            {
                if (column.Index < activeColumns.Length)
                {
                    activeColumns[column.Index] = true;
                }
            }

            int[] permutation = rand != null ? GetRandomPermutation(rand, Rows.Count) : null;
            int numCursors = Math.Min(n, (int)Math.Min(Rows.Count, int.MaxValue));
            long rowsPerCursor = Rows.Count / numCursors;
            long remainder = Rows.Count % numCursors;

            var cursors = new DataViewRowCursor[numCursors];
            long currentStart = 0;

            for (int i = 0; i < numCursors; i++)
            {
                long cursorRows = rowsPerCursor + (i < remainder ? 1 : 0);
                long currentEnd = currentStart + cursorRows;
                cursors[i] = new RowCursor(this, activeColumns, permutation, currentStart, currentEnd, i);
                currentStart = currentEnd;
            }

            return cursors;
        }

        private sealed class RowCursor : DataViewRowCursor
        {
            private bool _disposed;
            private long _cursorStep;
            private readonly long _startRow;
            private readonly long _endRow;
            private readonly long _batch;
            private readonly int[] _permutation;
            private readonly DataFrame _dataFrame;
            private readonly Delegate[] _getters;

            public RowCursor(DataFrame dataFrame, bool[] activeColumns, int[] permutation, long startRow, long endRow, long batch)
            {
                Debug.Assert(dataFrame != null);
                Debug.Assert(activeColumns != null);

                _cursorStep = -1;
                _dataFrame = dataFrame;
                _permutation = permutation;
                _startRow = startRow;
                _endRow = endRow;
                _batch = batch;

                _getters = new Delegate[Schema.Count];
                for (int i = 0; i < _getters.Length; i++)
                {
                    if (!activeColumns[i])
                        continue;
                    _getters[i] = CreateGetterDelegate(i);
                    Debug.Assert(_getters[i] != null);
                }
            }

            private long CurrentPhysicalRow => _startRow + _cursorStep;
            private long TargetDataFrameRowIndex => _permutation != null ? _permutation[CurrentPhysicalRow] : CurrentPhysicalRow;

            public override long Position => TargetDataFrameRowIndex;
            public override long Batch => _batch;
            public override DataViewSchema Schema => _dataFrame.DataViewSchema;

            protected override void Dispose(bool disposing)
            {
                if (_disposed)
                    return;
                if (disposing)
                {
                    _cursorStep = -1;
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
                return (ref DataViewRowId value) => value = new DataViewRowId((ulong)TargetDataFrameRowIndex, 0);
            }

            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                return _getters[column.Index] != null;
            }

            public override bool MoveNext()
            {
                if (_disposed)
                    return false;
                _cursorStep++;
                return (_startRow + _cursorStep) < _endRow;
            }
        }
    }
}
