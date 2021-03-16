// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.Data.Analysis;
using Microsoft.ML.Data;

namespace Microsoft.ML
{
    public static class IDataViewExtensions
    {
        private const int defaultMaxRows = 100;

        public static DataFrame ToDataFrame(this IDataView dataView, long maxRows = defaultMaxRows)
        {
            return ToDataFrame(dataView, maxRows, null);
        }

        public static DataFrame ToDataFrame(this IDataView dataView, params string[] selectColumns)
        {
            return ToDataFrame(dataView, defaultMaxRows, selectColumns);
        }

        public static DataFrame ToDataFrame(this IDataView dataView, long maxRows, params string[] selectColumns)
        {
            DataViewSchema schema = dataView.Schema;
            List<DataFrameColumn> columns = new List<DataFrameColumn>(schema.Count);

            HashSet<string> selectColumnsSet = null;
            if (selectColumns != null && selectColumns.Length > 0)
            {
                selectColumnsSet = new HashSet<string>(selectColumns);
            }

            List<DataViewSchema.Column> activeColumns = new List<DataViewSchema.Column>();
            foreach (DataViewSchema.Column column in schema)
            {
                if (column.IsHidden || (selectColumnsSet != null && !selectColumnsSet.Contains(column.Name)))
                {
                    continue;
                }

                activeColumns.Add(column);
                DataViewType type = column.Type;
                if (type == BooleanDataViewType.Instance)
                {
                    columns.Add(new BooleanDataFrameColumn(column.Name));
                }
                else if (type == NumberDataViewType.Byte)
                {
                    columns.Add(new ByteDataFrameColumn(column.Name));
                }
                else if (type == NumberDataViewType.Double)
                {
                    columns.Add(new DoubleDataFrameColumn(column.Name));
                }
                else if (type == NumberDataViewType.Single)
                {
                    columns.Add(new SingleDataFrameColumn(column.Name));
                }
                else if (type == NumberDataViewType.Int32)
                {
                    columns.Add(new Int32DataFrameColumn(column.Name));
                }
                else if (type == NumberDataViewType.Int64)
                {
                    columns.Add(new Int64DataFrameColumn(column.Name));
                }
                else if (type == NumberDataViewType.SByte)
                {
                    columns.Add(new SByteDataFrameColumn(column.Name));
                }
                else if (type == NumberDataViewType.Int16)
                {
                    columns.Add(new Int16DataFrameColumn(column.Name));
                }
                else if (type == NumberDataViewType.UInt32)
                {
                    columns.Add(new UInt32DataFrameColumn(column.Name));
                }
                else if (type == NumberDataViewType.UInt64)
                {
                    columns.Add(new UInt64DataFrameColumn(column.Name));
                }
                else if (type == NumberDataViewType.UInt16)
                {
                    columns.Add(new UInt16DataFrameColumn(column.Name));
                }
                else if (type == TextDataViewType.Instance)
                {
                    columns.Add(new StringDataFrameColumn(column.Name));
                }
                else
                {
                    throw new NotSupportedException(String.Format(Microsoft.Data.Strings.NotSupportedColumnType, type.RawType.Name));
                }
            }

            List<Delegate> activeColumnDelegates = new List<Delegate>();

            DataViewRowCursor cursor = dataView.GetRowCursor(activeColumns);
            int columnIndex = 0;
            foreach (DataViewSchema.Column column in activeColumns)
            {
                Delegate valueGetter = columns[columnIndex].GetValueGetterUsingCursor(cursor, column);
                activeColumnDelegates.Add(valueGetter);
                columnIndex++;
            }
            while (cursor.MoveNext() && cursor.Position < maxRows)
            {
                columnIndex = 0;
                foreach (DataViewSchema.Column column in activeColumns)
                {
                    columns[columnIndex].AddValueUsingCursor(cursor, column, activeColumnDelegates[columnIndex]);
                    columnIndex++;
                }
            }

            return new DataFrame(columns);
        }
    }

}
