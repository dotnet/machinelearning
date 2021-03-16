// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace Microsoft.Data.Analysis
{
    public partial class DataFrame : IDataView
    {
        // TODO: support shuffling
        bool IDataView.CanShuffle => false;

        private DataViewSchema _schema;
        internal DataViewSchema DataViewSchema
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
    }
}
