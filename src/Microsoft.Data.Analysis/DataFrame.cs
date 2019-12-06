// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace Microsoft.Data.Analysis
{

    /// <summary>
    /// Options for DropNull(). 
    /// </summary>
    public enum DropNullOptions
    {
        /// <summary>
        /// "Any" drops a row if any of the row values are null. 
        /// </summary>
        Any,
        /// <summary>
        /// "All" drops a row when all of the row values are null.
        /// </summary>
        All
    }

    /// <summary>
    /// A DataFrame to support indexing, binary operations, sorting, selection and other APIs. This will eventually also expose an IDataView for ML.NET
    /// </summary>
    public partial class DataFrame
    {
        private readonly DataFrameColumnCollection _columnCollection;
        private readonly DataFrameRowCollection _rowCollection;

        public DataFrame(IEnumerable<DataFrameColumn> columns)
        {
            _columnCollection = new DataFrameColumnCollection(columns, OnColumnsChanged);
            _rowCollection = new DataFrameRowCollection(this);
        }

        public DataFrame(params DataFrameColumn[] columns)
        {
            _columnCollection = new DataFrameColumnCollection(columns, OnColumnsChanged);
            _rowCollection = new DataFrameRowCollection(this);
        }

        public DataFrameColumnCollection Columns => _columnCollection;

        /// <summary>
        /// Returns a <see cref="DataFrameRowCollection"/> that contains a view of the rows in this <see cref="DataFrame"/>
        /// </summary>
        public DataFrameRowCollection Rows => _rowCollection;

        internal IReadOnlyList<string> GetColumnNames() => _columnCollection.GetColumnNames();

        #region Operators
        public object this[long rowIndex, int columnIndex]
        {
            get => _columnCollection[columnIndex][rowIndex];
            set => _columnCollection[columnIndex][rowIndex] = value;
        }

        /// <summary>
        /// Returns a new DataFrame using the boolean values in <paramref name="filter"/>
        /// </summary>
        /// <param name="filter">A column of booleans</param>
        public DataFrame Filter(PrimitiveDataFrameColumn<bool> filter) => Clone(filter);

        /// <summary>
        /// Returns a new DataFrame using the row indices in <paramref name="rowIndices"/>
        /// </summary>
        /// <param name="rowIndices">A column of row indices</param>
        public DataFrame Filter(PrimitiveDataFrameColumn<int> rowIndices) => Clone(rowIndices);

        /// <summary>
        /// Returns a new DataFrame using the row indices in <paramref name="rowIndices"/>
        /// </summary>
        /// <param name="rowIndices">A column of row indices</param>
        public DataFrame Filter(PrimitiveDataFrameColumn<long> rowIndices) => Clone(rowIndices);

        /// <summary>
        /// Returns a new DataFrame using the boolean values in filter
        /// </summary>
        /// <param name="filter">A column of booleans</param>
        public DataFrame this[PrimitiveDataFrameColumn<bool> filter] => Filter(filter);

        /// <summary>
        /// Returns a new DataFrame using the row indices in <paramref name="rowIndices"/>
        /// </summary>
        /// <param name="rowIndices">A column of row indices</param>
        public DataFrame this[PrimitiveDataFrameColumn<int> rowIndices] => Filter(rowIndices);

        /// <summary>
        /// Returns a new DataFrame using the row indices in <paramref name="rowIndices"/>
        /// </summary>
        /// <param name="rowIndices">A column of row indices</param>
        public DataFrame this[PrimitiveDataFrameColumn<long> rowIndices] => Filter(rowIndices);

        /// <summary>
        /// Returns a new DataFrame using the row indices in <paramref name="rowIndices"/>
        /// </summary>
        public DataFrame this[IEnumerable<int> rowIndices]
        {
            get
            {
                PrimitiveDataFrameColumn<int> filterColumn = new PrimitiveDataFrameColumn<int>("Filter", rowIndices);
                return Clone(filterColumn);
            }
        }

        /// <summary>
        /// Returns a new DataFrame using the row indices in <paramref name="rowIndices"/>
        /// </summary>
        public DataFrame this[IEnumerable<long> rowIndices]
        {
            get
            {
                PrimitiveDataFrameColumn<long> filterColumn = new PrimitiveDataFrameColumn<long>("Filter", rowIndices);
                return Clone(filterColumn);
            }
        }

        /// <summary>
        /// Returns a new DataFrame using the boolean values in <paramref name="boolFilter"/>
        /// </summary>
        public DataFrame this[IEnumerable<bool> boolFilter]
        {
            get
            {
                PrimitiveDataFrameColumn<bool> filterColumn = new PrimitiveDataFrameColumn<bool>("Filter", boolFilter);
                return Clone(filterColumn);
            }
        }

        public DataFrameColumn this[string columnName]
        {
            get
            {
                int columnIndex = _columnCollection.IndexOf(columnName);
                if (columnIndex == -1)
                    throw new ArgumentException(Strings.InvalidColumnName, nameof(columnName));
                return _columnCollection[columnIndex];
            }
            set
            {
                int columnIndex = _columnCollection.IndexOf(columnName);
                DataFrameColumn newColumn = value;
                newColumn.SetName(columnName);
                if (columnIndex == -1)
                {
                    _columnCollection.Insert(Columns.Count, newColumn);
                }
                else
                {
                    _columnCollection[columnIndex] = newColumn;
                }
            }
        }

        /// <summary>
        /// Returns the first <paramref name="numberOfRows"/> rows
        /// </summary>
        /// <param name="numberOfRows"></param>
        public DataFrame Head(int numberOfRows)
        {
            return Clone(new PrimitiveDataFrameColumn<int>("Filter", Enumerable.Range(0, numberOfRows)));
        }

        /// <summary>
        /// Returns the last <paramref name="numberOfRows"/> rows
        /// </summary>
        /// <param name="numberOfRows"></param>
        public DataFrame Tail(int numberOfRows)
        {
            PrimitiveDataFrameColumn<long> filter = new PrimitiveDataFrameColumn<long>("Filter", numberOfRows);
            for (long i = Rows.Count - numberOfRows; i < Rows.Count; i++)
            {
                filter[i - (Rows.Count - numberOfRows)] = i;
            }
            return Clone(filter);
        }
        // TODO: Add strongly typed versions of these APIs
        #endregion

        /// <summary>
        /// Returns a full copy
        /// </summary>
        public DataFrame Clone()
        {
            return Clone(mapIndices: null, invertMapIndices: false);
        }

        private DataFrame Clone(DataFrameColumn mapIndices = null, bool invertMapIndices = false)
        {
            List<DataFrameColumn> newColumns = new List<DataFrameColumn>(Columns.Count);
            for (int i = 0; i < Columns.Count; i++)
            {
                newColumns.Add(Columns[i].Clone(mapIndices, invertMapIndices));
            }
            return new DataFrame(newColumns);
        }

        /// <summary>
        /// Generates a concise summary of each column in the DataFrame
        /// </summary>
        public DataFrame Info()
        {
            DataFrame ret = new DataFrame();

            bool firstColumn = true;
            foreach (DataFrameColumn column in Columns)
            {
                if (firstColumn)
                {
                    firstColumn = false;
                    StringDataFrameColumn strColumn = new StringDataFrameColumn("Info", 2);
                    strColumn[0] = Strings.DataType;
                    strColumn[1] = Strings.DescriptionMethodLength;
                    ret.Columns.Add(strColumn);
                }
                ret.Columns.Add(column.Info());
            }
            return ret;
        }

        /// <summary>
        /// Generates descriptive statistics that summarize each numeric column
        /// </summary>
        public DataFrame Description()
        {
            DataFrame ret = new DataFrame();

            bool firstDescriptionColumn = true;
            foreach (DataFrameColumn column in Columns)
            {
                if (!column.HasDescription())
                {
                    continue;
                }
                if (firstDescriptionColumn)
                {
                    firstDescriptionColumn = false;
                    StringDataFrameColumn stringColumn = new StringDataFrameColumn("Description", 0);
                    stringColumn.Append(Strings.DescriptionMethodLength);
                    stringColumn.Append("Max");
                    stringColumn.Append("Min");
                    stringColumn.Append("Mean");
                    ret.Columns.Add(stringColumn);
                }
                ret.Columns.Add(column.Description());
            }
            return ret;
        }

        public DataFrame Sort(string columnName, bool ascending = true)
        {
            DataFrameColumn column = this[columnName];
            DataFrameColumn sortIndices = column.GetAscendingSortIndices();
            List<DataFrameColumn> newColumns = new List<DataFrameColumn>(Columns.Count);
            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn oldColumn = Columns[i];
                DataFrameColumn newColumn = oldColumn.Clone(sortIndices, !ascending, oldColumn.NullCount);
                Debug.Assert(newColumn.NullCount == oldColumn.NullCount);
                newColumns.Add(newColumn);
            }
            return new DataFrame(newColumns);
        }

        /// <summary>
        /// Clips values beyond the specified thresholds on numeric columns
        /// </summary>
        /// <typeparam name="U"></typeparam>
        /// <param name="lower">Minimum value. All values below this threshold will be set to it</param>
        /// <param name="upper">Maximum value. All values above this threshold will be set to it</param>
        public DataFrame Clip<U>(U lower, U upper, bool inPlace = false)
        {
            DataFrame ret = inPlace ? this : Clone();

            for (int i = 0; i < ret.Columns.Count; i++)
            {
                DataFrameColumn column = ret.Columns[i];
                if (column.IsNumericColumn())
                    column.Clip(lower, upper, inPlace: true);
            }
            return ret;
        }

        /// <summary>
        /// Adds a prefix to the column names
        /// </summary>
        public DataFrame AddPrefix(string prefix, bool inPlace = false)
        {
            DataFrame df = inPlace ? this : Clone();
            for (int i = 0; i < df.Columns.Count; i++)
            {
                DataFrameColumn column = df.Columns[i];
                df._columnCollection.SetColumnName(column, prefix + column.Name);
                df.OnColumnsChanged();
            }
            return df;
        }

        /// <summary>
        /// Adds a suffix to the column names
        /// </summary>
        public DataFrame AddSuffix(string suffix, bool inPlace = false)
        {
            DataFrame df = inPlace ? this : Clone();
            for (int i = 0; i < df.Columns.Count; i++)
            {
                DataFrameColumn column = df.Columns[i];
                df._columnCollection.SetColumnName(column, column.Name + suffix);
                df.OnColumnsChanged();
            }
            return df;
        }

        /// <summary>
        /// Returns a random sample of rows
        /// </summary>
        /// <param name="numberOfRows">Number of rows in the returned DataFrame</param>
        public DataFrame Sample(int numberOfRows)
        {
            Random rand = new Random();
            PrimitiveDataFrameColumn<long> indices = new PrimitiveDataFrameColumn<long>("Indices", numberOfRows);
            int randMaxValue = (int)Math.Min(Int32.MaxValue, Rows.Count);
            for (long i = 0; i < numberOfRows; i++)
            {
                indices[i] = rand.Next(randMaxValue);
            }

            return Clone(indices);
        }

        public GroupBy GroupBy(string columnName)
        {
            int columnIndex = _columnCollection.IndexOf(columnName);
            if (columnIndex == -1)
                throw new ArgumentException(Strings.InvalidColumnName, nameof(columnName));

            DataFrameColumn column = _columnCollection[columnIndex];
            return column.GroupBy(columnIndex, this);
        }

        // In GroupBy and ReadCsv calls, columns get resized. We need to set the RowCount to reflect the true Length of the DataFrame. This does internal validation
        internal void SetTableRowCount(long rowCount)
        {
            // Even if current RowCount == rowCount, do the validation
            for (int i = 0; i < Columns.Count; i++)
            {
                if (Columns[i].Length != rowCount)
                    throw new ArgumentException(String.Format("{0} {1}", Strings.MismatchedRowCount, Columns[i].Name));
            }
            _columnCollection.RowCount = rowCount;
        }

        /// <summary>
        /// Returns a DataFrame with no missing values
        /// </summary>
        /// <param name="options"></param>
        public DataFrame DropNulls(DropNullOptions options = DropNullOptions.Any)
        {
            DataFrame ret = new DataFrame();
            PrimitiveDataFrameColumn<bool> filter = new PrimitiveDataFrameColumn<bool>("Filter");
            if (options == DropNullOptions.Any)
            {
                filter.AppendMany(true, Rows.Count);

                for (int i = 0; i < Columns.Count; i++)
                {
                    DataFrameColumn column = Columns[i];
                    filter.ApplyElementwise((bool? value, long index) =>
                    {
                        return value.Value && (column[index] == null ? false : true);
                    });
                }
            }
            else
            {
                filter.AppendMany(false, Rows.Count);
                for (int i = 0; i < Columns.Count; i++)
                {
                    DataFrameColumn column = Columns[i];
                    filter.ApplyElementwise((bool? value, long index) =>
                    {
                        return value.Value || (column[index] == null ? false : true);
                    });
                }
            }
            return this[filter];
        }

        public DataFrame FillNulls(object value, bool inPlace = false)
        {
            DataFrame ret = inPlace ? this : Clone();
            for (int i = 0; i < ret.Columns.Count; i++)
            {
                ret.Columns[i].FillNulls(value, inPlace: true);
            }
            return ret;
        }

        public DataFrame FillNulls(IList<object> values, bool inPlace = false)
        {
            if (values.Count != Columns.Count)
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(values));

            DataFrame ret = inPlace ? this : Clone();
            for (int i = 0; i < ret.Columns.Count; i++)
            {
                Columns[i].FillNulls(values[i], inPlace: true);
            }
            return ret;
        }

        private void ResizeByOneAndAppend(DataFrameColumn column, object value)
        {
            long length = column.Length;
            column.Resize(length + 1);
            column[length] = value;
        }

        /// <summary> 
        /// Appends a row inplace to the DataFrame 
        /// </summary> 
        /// <remarks>If a column's value doesn't match its column's data type, a conversion will be attempted</remarks> 
        /// <remarks>If <paramref name="row"/> is null, a null value is appended to each column</remarks>
        /// <param name="row"></param> 
        public void Append(IEnumerable<object> row = null)
        {
            IEnumerator<DataFrameColumn> columnEnumerator = Columns.GetEnumerator();
            bool columnMoveNext = columnEnumerator.MoveNext();
            if (row != null)
            {
                // Go through row first to make sure there are no data type incompatibilities
                IEnumerator<object> rowEnumerator = row.GetEnumerator();
                bool rowMoveNext = rowEnumerator.MoveNext();
                List<object> cachedObjectConversions = new List<object>();
                while (columnMoveNext && rowMoveNext)
                {
                    DataFrameColumn column = columnEnumerator.Current;
                    object value = rowEnumerator.Current;
                    // StringDataFrameColumn can accept empty strings. The other columns interpret empty values as nulls
                    if (value is string stringValue && string.IsNullOrEmpty(stringValue) && column.DataType != typeof(string))
                    {
                        value = null;
                    }
                    if (value != null)
                    {
                        value = Convert.ChangeType(value, column.DataType);
                        if (value is null)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedValueType, column.DataType), value.GetType().ToString());
                        }
                    }
                    cachedObjectConversions.Add(value);
                    columnMoveNext = columnEnumerator.MoveNext();
                    rowMoveNext = rowEnumerator.MoveNext();
                }
                if (rowMoveNext)
                {
                    throw new ArgumentException(string.Format(Strings.ExceedsNumberOfColumns, Columns.Count), nameof(row));
                }
                columnEnumerator.Reset();
                columnMoveNext = columnEnumerator.MoveNext();
                rowEnumerator.Reset();
                rowMoveNext = rowEnumerator.MoveNext();
                int cacheIndex = 0;
                while (columnMoveNext && rowMoveNext)
                {
                    DataFrameColumn column = columnEnumerator.Current;
                    object value = cachedObjectConversions[cacheIndex];
                    ResizeByOneAndAppend(column, value);
                    columnMoveNext = columnEnumerator.MoveNext();
                    rowMoveNext = rowEnumerator.MoveNext();
                    cacheIndex++;
                }
            }
            while (columnMoveNext)
            {
                // Fill the remaining columns with null
                DataFrameColumn column = columnEnumerator.Current;
                ResizeByOneAndAppend(column, null);
                columnMoveNext = columnEnumerator.MoveNext();
            }
            Columns.RowCount++;
        }

        /// <summary> 
        /// Appends a row inplace by enumerating column names and values from <paramref name="row"/> 
        /// </summary> 
        /// <remarks>If a column's value doesn't match its column's data type, a conversion will be attempted</remarks> 
        /// <param name="row"></param> 
        public void Append(IEnumerable<KeyValuePair<string, object>> row)
        {
            if (row == null)
            {
                throw new ArgumentNullException(nameof(row));
            }

            List<object> cachedObjectConversions = new List<object>();
            foreach (KeyValuePair<string, object> columnAndValue in row)
            {
                string columnName = columnAndValue.Key;
                int index = Columns.IndexOf(columnName);
                if (index == -1)
                {
                    throw new ArgumentException(Strings.InvalidColumnName, nameof(columnName));
                }

                DataFrameColumn column = Columns[index];
                object value = columnAndValue.Value;
                if (value != null)
                {
                    value = Convert.ChangeType(value, column.DataType);
                    if (value is null)
                    {
                        throw new ArgumentException(string.Format(Strings.MismatchedValueType, column.DataType), value.GetType().ToString());
                    }
                }
                cachedObjectConversions.Add(value);
            }

            int cacheIndex = 0;
            foreach (KeyValuePair<string, object> columnAndValue in row)
            {
                string columnName = columnAndValue.Key;
                int index = Columns.IndexOf(columnName);

                DataFrameColumn column = Columns[index];
                object value = cachedObjectConversions[cacheIndex];
                ResizeByOneAndAppend(column, value);
                cacheIndex++;
            }

            foreach (DataFrameColumn column in Columns)
            {
                if (column.Length == Rows.Count)
                {
                    ResizeByOneAndAppend(column, null);
                }
            }
            Columns.RowCount++;
        }

        /// <summary>
        /// Invalidates any cached data after a column has changed.
        /// </summary>
        private void OnColumnsChanged()
        {
            _schema = null;
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            int longestColumnName = 0;
            for (int i = 0; i < Columns.Count; i++)
            {
                longestColumnName = Math.Max(longestColumnName, Columns[i].Name.Length);
            }
            for (int i = 0; i < Columns.Count; i++)
            {
                // Left align by 10
                sb.Append(string.Format(Columns[i].Name.PadRight(longestColumnName)));
            }
            sb.AppendLine();
            long numberOfRows = Math.Min(Rows.Count, 25);
            for (int i = 0; i < numberOfRows; i++)
            {
                foreach (object obj in Rows[i])
                {
                    sb.Append((obj ?? "null").ToString().PadRight(longestColumnName));
                }
                sb.AppendLine();
            }
            return sb.ToString();
        }
    }
}
