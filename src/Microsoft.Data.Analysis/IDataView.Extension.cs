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

        /// <summary>
        /// Returns a <see cref="Microsoft.Data.Analysis.DataFrame"/> from this <paramref name="dataView"/>.
        /// </summary>
        /// <param name="dataView">The current <see cref="IDataView"/>.</param>
        /// <param name="maxRows">The max number or rows in the <see cref="Microsoft.Data.Analysis.DataFrame"/>. Defaults to 100. Use -1 to construct a DataFrame using all the rows in <paramref name="dataView"/>.</param>
        /// <returns>A <see cref="Microsoft.Data.Analysis.DataFrame"/> with <paramref name="maxRows"/>.</returns>
        public static DataFrame ToDataFrame(this IDataView dataView, long maxRows = defaultMaxRows)
        {
            return ToDataFrame(dataView, maxRows, null);
        }

        /// <summary>
        /// Returns a <see cref="Microsoft.Data.Analysis.DataFrame"/> with the first 100 rows of this <paramref name="dataView"/>.
        /// </summary>
        /// <param name="dataView">The current <see cref="IDataView"/>.</param>
        /// <param name="selectColumns">The columns selected for the resultant DataFrame</param>
        /// <returns>A <see cref="Microsoft.Data.Analysis.DataFrame"/> with the selected columns and 100 rows.</returns>
        public static DataFrame ToDataFrame(this IDataView dataView, params string[] selectColumns)
        {
            return ToDataFrame(dataView, defaultMaxRows, selectColumns);
        }

        /// <summary>
        /// Returns a <see cref="Microsoft.Data.Analysis.DataFrame"/> with the first <paramref name="maxRows"/> of this <paramref name="dataView"/>.
        /// </summary>
        /// <param name="dataView">The current <see cref="IDataView"/>.</param>
        /// <param name="maxRows">The max number or rows in the <see cref="Microsoft.Data.Analysis.DataFrame"/>. Use -1 to construct a DataFrame using all the rows in <paramref name="dataView"/>.</param>
        /// <param name="selectColumns">The columns selected for the resultant DataFrame</param>
        /// <returns>A <see cref="Microsoft.Data.Analysis.DataFrame"/> with the selected columns and <paramref name="maxRows"/> rows.</returns>
        public static DataFrame ToDataFrame(this IDataView dataView, long maxRows, params string[] selectColumns)
        {
            DataViewSchema schema = dataView.Schema;
            List<DataFrameColumn> dataFrameColumns = new List<DataFrameColumn>(schema.Count);
            maxRows = maxRows == -1 ? long.MaxValue : maxRows;

            HashSet<string> selectColumnsSet = null;
            if (selectColumns != null && selectColumns.Length > 0)
            {
                selectColumnsSet = new HashSet<string>(selectColumns);
            }

            List<DataViewSchema.Column> activeDataViewColumns = new List<DataViewSchema.Column>();
            foreach (DataViewSchema.Column dataViewColumn in schema)
            {
                if (dataViewColumn.IsHidden || (selectColumnsSet != null && !selectColumnsSet.Contains(dataViewColumn.Name)))
                {
                    continue;
                }

                activeDataViewColumns.Add(dataViewColumn);
                DataViewType type = dataViewColumn.Type;
                if (type == BooleanDataViewType.Instance)
                {
                    dataFrameColumns.Add(new BooleanDataFrameColumn(dataViewColumn.Name));
                }
                else if (type == NumberDataViewType.Byte)
                {
                    dataFrameColumns.Add(new ByteDataFrameColumn(dataViewColumn.Name));
                }
                else if (type == NumberDataViewType.Double)
                {
                    dataFrameColumns.Add(new DoubleDataFrameColumn(dataViewColumn.Name));
                }
                else if (type == NumberDataViewType.Single)
                {
                    dataFrameColumns.Add(new SingleDataFrameColumn(dataViewColumn.Name));
                }
                else if (type == NumberDataViewType.Int32)
                {
                    dataFrameColumns.Add(new Int32DataFrameColumn(dataViewColumn.Name));
                }
                else if (type == NumberDataViewType.Int64)
                {
                    dataFrameColumns.Add(new Int64DataFrameColumn(dataViewColumn.Name));
                }
                else if (type == NumberDataViewType.SByte)
                {
                    dataFrameColumns.Add(new SByteDataFrameColumn(dataViewColumn.Name));
                }
                else if (type == NumberDataViewType.Int16)
                {
                    dataFrameColumns.Add(new Int16DataFrameColumn(dataViewColumn.Name));
                }
                else if (type == NumberDataViewType.UInt32)
                {
                    dataFrameColumns.Add(new UInt32DataFrameColumn(dataViewColumn.Name));
                }
                else if (type == NumberDataViewType.UInt64)
                {
                    dataFrameColumns.Add(new UInt64DataFrameColumn(dataViewColumn.Name));
                }
                else if (type == NumberDataViewType.UInt16)
                {
                    dataFrameColumns.Add(new UInt16DataFrameColumn(dataViewColumn.Name));
                }
                else if (type == TextDataViewType.Instance)
                {
                    dataFrameColumns.Add(new StringDataFrameColumn(dataViewColumn.Name));
                }
                else
                {
                    throw new NotSupportedException(String.Format(Microsoft.Data.Strings.NotSupportedColumnType, type.RawType.Name));
                }
            }

            using (DataViewRowCursor cursor = dataView.GetRowCursor(activeDataViewColumns))
            {
                Delegate[] activeColumnDelegates = new Delegate[activeDataViewColumns.Count];
                int columnIndex = 0;
                foreach (DataViewSchema.Column activeDataViewColumn in activeDataViewColumns)
                {
                    Delegate valueGetter = dataFrameColumns[columnIndex].GetValueGetterUsingCursor(cursor, activeDataViewColumn);
                    activeColumnDelegates[columnIndex] = valueGetter;
                    columnIndex++;
                }
                while (cursor.MoveNext() && cursor.Position < maxRows)
                {
                    for (int i = 0; i < activeColumnDelegates.Length; i++)
                    {
                        dataFrameColumns[i].AddValueUsingCursor(cursor, activeColumnDelegates[i]);
                    }
                }
            }

            return new DataFrame(dataFrameColumns);
        }
    }

}
