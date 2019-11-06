// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.Data.Analysis
{
    /// <summary>
    /// A GroupBy class that is typically the result of a DataFrame.GroupBy call.
    /// It holds information to perform typical aggregation ops on it.
    /// </summary>
    public abstract class GroupBy
    {
        /// <summary>
        /// Compute the number of non-null values in each group 
        /// </summary>
        /// <returns></returns>
        public abstract DataFrame Count(params string[] columnNames);

        /// <summary>
        /// Return the first value in each group
        /// </summary>
        /// <returns></returns>
        public abstract DataFrame First(params string[] columnNames);

        /// <summary>
        /// Returns the first numberOfRows rows of each group
        /// </summary>
        /// <param name="numberOfRows"></param>
        /// <returns></returns>
        public abstract DataFrame Head(int numberOfRows);

        /// <summary>
        /// Returns the last numberOfRows rows of each group
        /// </summary>
        /// <param name="numberOfRowsInEachGroup"></param>
        /// <returns></returns>
        public abstract DataFrame Tail(int numberOfRows);

        /// <summary>
        /// Compute the max of group values
        /// </summary>
        /// <param name="columnNames">The columns to find the max of. A default value finds the max of all columns</param>
        public abstract DataFrame Max(params string[] columnNames);

        /// <summary>
        /// Compute the min of group values
        /// </summary>
        /// <param name="columnNames">The columns to find the min of. A default value finds the min of all columns</param>
        public abstract DataFrame Min(params string[] columnNames);

        /// <summary>
        /// Compute the product of group values
        /// </summary>
        /// <param name="columnNames">The columns to find the product of. A default value finds the product of all columns</param>
        public abstract DataFrame Product(params string[] columnNames);

        /// <summary>
        /// Compute the sum of group values
        /// </summary>
        /// <param name="columnNames">The columns to sum. A Default value sums up all columns</param>
        public abstract DataFrame Sum(params string[] columnNames);

        /// <summary>
        /// Compute the mean of group values
        /// </summary>
        /// <param name="columnNames">The columns to find the mean of. A Default value finds the mean of all columns</param>
        public abstract DataFrame Mean(params string[] columnNames);
    }

    public class GroupBy<TKey> : GroupBy
    {
        private int _groupByColumnIndex;
        private IDictionary<TKey, ICollection<long>> _keyToRowIndicesMap;
        private DataFrame _dataFrame;

        public GroupBy(DataFrame dataFrame, int groupByColumnIndex, IDictionary<TKey, ICollection<long>> keyToRowIndices)
        {
            if (dataFrame.Columns.Count < groupByColumnIndex || groupByColumnIndex < 0)
                throw new ArgumentException(nameof(groupByColumnIndex));
            _groupByColumnIndex = groupByColumnIndex;
            _keyToRowIndicesMap = keyToRowIndices ?? throw new ArgumentException(nameof(keyToRowIndices));
            _dataFrame = dataFrame ?? throw new ArgumentException(nameof(dataFrame));
        }

        private delegate void ColumnDelegate(int columnIndex, long rowIndex, ICollection<long> rows, TKey key, bool firstGroup);
        private delegate void GroupByColumnDelegate(long rowNumber, TKey key);
        private void EnumerateColumnsWithRows(GroupByColumnDelegate groupByColumnDelegate, ColumnDelegate columnDelegate, params string[] columnNames)
        {
            long rowNumber = 0;
            bool firstGroup = true;
            foreach (KeyValuePair<TKey, ICollection<long>> pairs in _keyToRowIndicesMap)
            {
                groupByColumnDelegate(rowNumber, pairs.Key);
                ICollection<long> rows = pairs.Value;
                IEnumerable<string> columns = columnNames;
                if (columnNames == null || columnNames.Length == 0)
                    columns = _dataFrame.GetColumnNames();
                // Assuming that the dataframe has not been modified after the groupby call
                foreach (string columnName in columns)
                {
                    int columnIndex = _dataFrame.Columns.IndexOf(columnName);
                    columnDelegate(columnIndex, rowNumber, rows, pairs.Key, firstGroup);
                }
                firstGroup = false;
                rowNumber++;
            }

        }

        public override DataFrame Count(params string[] columnNames)
        {
            DataFrame ret = new DataFrame();
            PrimitiveDataFrameColumn<long> empty = new PrimitiveDataFrameColumn<long>("Empty");
            DataFrameColumn firstColumn = _dataFrame.Columns[_groupByColumnIndex].Clone(empty);
            ret.Columns.Insert(ret.Columns.Count, firstColumn);
            GroupByColumnDelegate groupByColumnDelegate = new GroupByColumnDelegate((long rowIndex, TKey key) =>
            {
                firstColumn.Resize(rowIndex + 1);
                firstColumn[rowIndex] = key;
            });
            ColumnDelegate columnDelegate = new ColumnDelegate((int columnIndex, long rowIndex, ICollection<long> rowEnumerable, TKey key, bool firstGroup) =>
            {
                if (columnIndex == _groupByColumnIndex)
                    return;
                DataFrameColumn column = _dataFrame.Columns[columnIndex];
                long count = 0;
                foreach (long row in rowEnumerable)
                {
                    if (column[row] != null)
                        count++;
                }
                DataFrameColumn retColumn;
                if (firstGroup)
                {
                    retColumn = new PrimitiveDataFrameColumn<long>(column.Name);
                    ret.Columns.Insert(ret.Columns.Count, retColumn);
                }
                else
                {
                    // Assuming non duplicate column names
                    retColumn = ret[column.Name];
                }
                retColumn.Resize(rowIndex + 1);
                retColumn[rowIndex] = count;
            });

            EnumerateColumnsWithRows(groupByColumnDelegate, columnDelegate, columnNames);
            ret.SetTableRowCount(firstColumn.Length);

            return ret;
        }

        public override DataFrame First(params string[] columnNames)
        {
            DataFrame ret = new DataFrame();
            PrimitiveDataFrameColumn<long> empty = new PrimitiveDataFrameColumn<long>("Empty");
            DataFrameColumn firstColumn = _dataFrame.Columns[_groupByColumnIndex].Clone(empty);
            ret.Columns.Insert(ret.Columns.Count, firstColumn);

            GroupByColumnDelegate groupByColumnDelegate = new GroupByColumnDelegate((long rowIndex, TKey key) =>
            {
                firstColumn.Resize(rowIndex + 1);
                firstColumn[rowIndex] = key;
            });

            ColumnDelegate columnDelegate = new ColumnDelegate((int columnIndex, long rowIndex, ICollection<long> rowEnumerable, TKey key, bool firstGroup) =>
            {
                if (columnIndex == _groupByColumnIndex)
                    return;
                DataFrameColumn column = _dataFrame.Columns[columnIndex];
                foreach (long row in rowEnumerable)
                {
                    DataFrameColumn retColumn;
                    if (firstGroup)
                    {
                        retColumn = column.Clone(empty);
                        ret.Columns.Insert(ret.Columns.Count, retColumn);
                    }
                    else
                    {
                        // Assuming non duplicate column names
                        retColumn = ret[column.Name];
                    }
                    retColumn.Resize(rowIndex + 1);
                    retColumn[rowIndex] = column[row];
                    break;
                }
            });

            EnumerateColumnsWithRows(groupByColumnDelegate, columnDelegate, columnNames);
            ret.SetTableRowCount(firstColumn.Length);
            return ret;
        }

        public override DataFrame Head(int numberOfRows)
        {
            DataFrame ret = new DataFrame();
            PrimitiveDataFrameColumn<long> empty = new PrimitiveDataFrameColumn<long>("Empty");
            DataFrameColumn firstColumn = _dataFrame.Columns[_groupByColumnIndex].Clone(empty);
            ret.Columns.Insert(ret.Columns.Count, firstColumn);

            GroupByColumnDelegate groupByColumnDelegate = new GroupByColumnDelegate((long rowIndex, TKey key) =>
            {
            });

            ColumnDelegate columnDelegate = new ColumnDelegate((int columnIndex, long rowIndex, ICollection<long> rowEnumerable, TKey key, bool firstGroup) =>
            {
                if (columnIndex == _groupByColumnIndex)
                    return;
                DataFrameColumn column = _dataFrame.Columns[columnIndex];
                long count = 0;
                bool firstRow = true;
                foreach (long row in rowEnumerable)
                {
                    if (count < numberOfRows)
                    {
                        DataFrameColumn retColumn;
                        if (firstGroup && firstRow)
                        {
                            firstRow = false;
                            retColumn = column.Clone(empty);
                            ret.Columns.Insert(ret.Columns.Count, retColumn);
                        }
                        else
                        {
                            // Assuming non duplicate column names
                            retColumn = ret[column.Name];
                        }
                        long retColumnLength = retColumn.Length;
                        retColumn.Resize(retColumnLength + 1);
                        retColumn[retColumnLength] = column[row];
                        if (firstColumn.Length <= retColumnLength)
                        {
                            firstColumn.Resize(retColumnLength + 1);
                        }
                        firstColumn[retColumnLength] = key;
                        count++;
                    }
                    if (count == numberOfRows)
                        break;
                }
            });

            EnumerateColumnsWithRows(groupByColumnDelegate, columnDelegate);
            ret.SetTableRowCount(firstColumn.Length);
            return ret;
        }

        public override DataFrame Tail(int numberOfRows)
        {
            DataFrame ret = new DataFrame();
            PrimitiveDataFrameColumn<long> empty = new PrimitiveDataFrameColumn<long>("Empty");
            DataFrameColumn firstColumn = _dataFrame.Columns[_groupByColumnIndex].Clone(empty);
            ret.Columns.Insert(ret.Columns.Count, firstColumn);

            GroupByColumnDelegate groupByColumnDelegate = new GroupByColumnDelegate((long rowIndex, TKey key) =>
            {
            });

            ColumnDelegate columnDelegate = new ColumnDelegate((int columnIndex, long rowIndex, ICollection<long> rowEnumerable, TKey key, bool firstGroup) =>
            {
                if (columnIndex == _groupByColumnIndex)
                    return;
                DataFrameColumn column = _dataFrame.Columns[columnIndex];
                long count = 0;
                bool firstRow = true;
                ICollection<long> values = _keyToRowIndicesMap[key];
                int numberOfValues = values.Count;
                foreach (long row in rowEnumerable)
                {
                    if (count >= numberOfValues - numberOfRows)
                    {
                        DataFrameColumn retColumn;
                        if (firstGroup && firstRow)
                        {
                            firstRow = false;
                            retColumn = column.Clone(empty);
                            ret.Columns.Insert(ret.Columns.Count, retColumn);
                        }
                        else
                        {
                            // Assuming non duplicate column names
                            retColumn = ret[column.Name];
                        }
                        long retColumnLength = retColumn.Length;
                        if (firstColumn.Length <= retColumnLength)
                        {
                            firstColumn.Resize(retColumnLength + 1);
                            firstColumn[retColumnLength] = key;
                        }
                        retColumn.Resize(retColumnLength + 1);
                        retColumn[retColumnLength] = column[row];
                    }
                    count++;
                }
            });

            EnumerateColumnsWithRows(groupByColumnDelegate, columnDelegate);
            ret.SetTableRowCount(firstColumn.Length);
            return ret;
        }

        private DataFrameColumn ResizeAndInsertColumn(int columnIndex, long rowIndex, bool firstGroup, DataFrame ret, PrimitiveDataFrameColumn<long> empty, Func<string, DataFrameColumn> getColumn = null)
        {
            if (columnIndex == _groupByColumnIndex)
                return null;
            DataFrameColumn column = _dataFrame.Columns[columnIndex];
            DataFrameColumn retColumn;
            if (firstGroup)
            {
                retColumn = getColumn == null ? column.Clone(empty) : getColumn(column.Name);
                ret.Columns.Insert(ret.Columns.Count, retColumn);
            }
            else
            {
                // Assuming unique column names
                retColumn = ret[column.Name];
            }
            retColumn.Resize(rowIndex + 1);
            return retColumn;
        }

        public override DataFrame Max(params string[] columnNames)
        {
            DataFrame ret = new DataFrame();
            PrimitiveDataFrameColumn<long> empty = new PrimitiveDataFrameColumn<long>("Empty");
            DataFrameColumn firstColumn = _dataFrame.Columns[_groupByColumnIndex].Clone(empty);
            ret.Columns.Insert(ret.Columns.Count, firstColumn);
            GroupByColumnDelegate groupByColumnDelegate = new GroupByColumnDelegate((long rowIndex, TKey key) =>
            {
                firstColumn.Resize(rowIndex + 1);
                firstColumn[rowIndex] = key;
            });

            ColumnDelegate columnDelegate = new ColumnDelegate((int columnIndex, long rowIndex, ICollection<long> rowEnumerable, TKey key, bool firstGroup) =>
            {
                DataFrameColumn retColumn = ResizeAndInsertColumn(columnIndex, rowIndex, firstGroup, ret, empty);

                if (!(retColumn is null))
                {
                    retColumn[rowIndex] = _dataFrame.Columns[columnIndex].Max(rowEnumerable);
                }
            });

            EnumerateColumnsWithRows(groupByColumnDelegate, columnDelegate, columnNames);
            ret.SetTableRowCount(firstColumn.Length);

            return ret;
        }

        public override DataFrame Min(params string[] columnNames)
        {
            DataFrame ret = new DataFrame();
            PrimitiveDataFrameColumn<long> empty = new PrimitiveDataFrameColumn<long>("Empty");
            DataFrameColumn firstColumn = _dataFrame.Columns[_groupByColumnIndex].Clone(empty);
            ret.Columns.Insert(ret.Columns.Count, firstColumn);
            GroupByColumnDelegate groupByColumnDelegate = new GroupByColumnDelegate((long rowIndex, TKey key) =>
            {
                firstColumn.Resize(rowIndex + 1);
                firstColumn[rowIndex] = key;
            });

            ColumnDelegate columnDelegate = new ColumnDelegate((int columnIndex, long rowIndex, ICollection<long> rowEnumerable, TKey key, bool firstGroup) =>
            {
                DataFrameColumn retColumn = ResizeAndInsertColumn(columnIndex, rowIndex, firstGroup, ret, empty);

                if (!(retColumn is null))
                {
                    retColumn[rowIndex] = _dataFrame.Columns[columnIndex].Min(rowEnumerable);
                }
            });

            EnumerateColumnsWithRows(groupByColumnDelegate, columnDelegate, columnNames);
            ret.SetTableRowCount(firstColumn.Length);

            return ret;
        }

        public override DataFrame Product(params string[] columnNames)
        {
            DataFrame ret = new DataFrame();
            PrimitiveDataFrameColumn<long> empty = new PrimitiveDataFrameColumn<long>("Empty");
            DataFrameColumn firstColumn = _dataFrame.Columns[_groupByColumnIndex].Clone(empty);
            ret.Columns.Insert(ret.Columns.Count, firstColumn);
            GroupByColumnDelegate groupByColumnDelegate = new GroupByColumnDelegate((long rowIndex, TKey key) =>
            {
                firstColumn.Resize(rowIndex + 1);
                firstColumn[rowIndex] = key;
            });

            ColumnDelegate columnDelegate = new ColumnDelegate((int columnIndex, long rowIndex, ICollection<long> rowEnumerable, TKey key, bool firstGroup) =>
            {
                DataFrameColumn retColumn = ResizeAndInsertColumn(columnIndex, rowIndex, firstGroup, ret, empty);

                if (!(retColumn is null))
                {
                    retColumn[rowIndex] = _dataFrame.Columns[columnIndex].Product(rowEnumerable);
                }
            });

            EnumerateColumnsWithRows(groupByColumnDelegate, columnDelegate, columnNames);
            ret.SetTableRowCount(firstColumn.Length);

            return ret;
        }

        public override DataFrame Sum(params string[] columnNames)
        {
            DataFrame ret = new DataFrame();
            PrimitiveDataFrameColumn<long> empty = new PrimitiveDataFrameColumn<long>("Empty");
            DataFrameColumn firstColumn = _dataFrame.Columns[_groupByColumnIndex].Clone(empty);
            ret.Columns.Insert(ret.Columns.Count, firstColumn);
            GroupByColumnDelegate groupByColumnDelegate = new GroupByColumnDelegate((long rowIndex, TKey key) =>
            {
                firstColumn.Resize(rowIndex + 1);
                firstColumn[rowIndex] = key;
            });

            ColumnDelegate columnDelegate = new ColumnDelegate((int columnIndex, long rowIndex, ICollection<long> rowEnumerable, TKey key, bool firstGroup) =>
            {
                DataFrameColumn retColumn = ResizeAndInsertColumn(columnIndex, rowIndex, firstGroup, ret, empty);

                if (!(retColumn is null))
                {
                    retColumn[rowIndex] = _dataFrame.Columns[columnIndex].Sum(rowEnumerable);
                }
            });

            EnumerateColumnsWithRows(groupByColumnDelegate, columnDelegate, columnNames);
            ret.SetTableRowCount(firstColumn.Length);

            return ret;
        }

        public override DataFrame Mean(params string[] columnNames)
        {
            DataFrame ret = new DataFrame();
            PrimitiveDataFrameColumn<long> empty = new PrimitiveDataFrameColumn<long>("Empty");
            DataFrameColumn firstColumn = _dataFrame.Columns[_groupByColumnIndex].Clone(empty);
            ret.Columns.Insert(ret.Columns.Count, firstColumn);
            GroupByColumnDelegate groupByColumnDelegate = new GroupByColumnDelegate((long rowIndex, TKey key) =>
            {
                firstColumn.Resize(rowIndex + 1);
                firstColumn[rowIndex] = key;
            });


            ColumnDelegate columnDelegate = new ColumnDelegate((int columnIndex, long rowIndex, ICollection<long> rowEnumerable, TKey key, bool firstGroup) =>
            {
                DataFrameColumn retColumn = ResizeAndInsertColumn(columnIndex, rowIndex, firstGroup, ret, empty, (name) => new PrimitiveDataFrameColumn<double>(name));

                if (!(retColumn is null))
                {
                    retColumn[rowIndex] = (double)Convert.ChangeType(_dataFrame.Columns[columnIndex].Sum(rowEnumerable), typeof(double)) / rowEnumerable.Count;
                }
            });

            EnumerateColumnsWithRows(groupByColumnDelegate, columnDelegate, columnNames);
            ret.SetTableRowCount(firstColumn.Length);

            return ret;
        }

    }
}
