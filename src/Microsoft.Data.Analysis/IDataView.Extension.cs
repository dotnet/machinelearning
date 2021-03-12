﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Microsoft.Data.Analysis
{
    public static class IDataViewExtensions
    {
        public static DataFrame ToDataFrame(this IDataView dataView, long maxRows = -1, params string[] selectColumns)
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
                long length = maxRows >= 0 ? maxRows : long.MaxValue;
                length = Math.Min(length, dataView.GetRowCount() ?? 0);
                if (column.IsHidden || (selectColumnsSet != null && !selectColumnsSet.Contains(column.Name)))
                {
                    continue;
                }

                activeColumns.Add(column);
                DataViewType type = column.Type;
                if (type == BooleanDataViewType.Instance)
                {
                    columns.Add(new BooleanDataFrameColumn(column.Name, length));
                }
                else if (type == NumberDataViewType.Byte)
                {
                    columns.Add(new ByteDataFrameColumn(column.Name, length));
                }
                else if (type == NumberDataViewType.Double)
                {
                    columns.Add(new DoubleDataFrameColumn(column.Name, length));
                }
                else if (type == NumberDataViewType.Single)
                {
                    columns.Add(new SingleDataFrameColumn(column.Name, length));
                }
                else if (type == NumberDataViewType.Int32)
                {
                    columns.Add(new Int32DataFrameColumn(column.Name, length));
                }
                else if (type == NumberDataViewType.Int64)
                {
                    columns.Add(new Int64DataFrameColumn(column.Name, length));
                }
                else if (type == NumberDataViewType.SByte)
                {
                    columns.Add(new SByteDataFrameColumn(column.Name, length));
                }
                else if (type == NumberDataViewType.Int16)
                {
                    columns.Add(new Int16DataFrameColumn(column.Name, length));
                }
                else if (type == NumberDataViewType.UInt32)
                {
                    columns.Add(new UInt32DataFrameColumn(column.Name, length));
                }
                else if (type == NumberDataViewType.UInt64)
                {
                    columns.Add(new UInt64DataFrameColumn(column.Name, length));
                }
                else if (type == NumberDataViewType.UInt16)
                {
                    columns.Add(new UInt16DataFrameColumn(column.Name, length));
                }
                else if (type == TextDataViewType.Instance)
                {
                    columns.Add(new StringDataFrameColumn(column.Name, length));
                }
                else
                {
                    throw new NotSupportedException(nameof(type));
                }
            }

            DataFrame ret = new DataFrame(columns);
            DataViewRowCursor cursor = dataView.GetRowCursor(activeColumns);
            while (cursor.MoveNext())
            {
                foreach (var column in activeColumns)
                {
                    columns[column.Index].AddValueUsingCursor(cursor, column);
                }
            }

            return ret;
        }
    }

}
