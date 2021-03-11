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

        /// <summary>
        /// Constructs a <see cref="DataFrame"/> with <paramref name="columns"/>.
        /// </summary>
        /// <param name="columns">The columns of this <see cref="DataFrame"/>.</param>
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

        /// <summary>
        /// Returns the columns contained in the <see cref="DataFrame"/> as a <see cref="DataFrameColumnCollection"/>
        /// </summary>
        public DataFrameColumnCollection Columns => _columnCollection;

        /// <summary>
        /// Returns a <see cref="DataFrameRowCollection"/> that contains a view of the rows in this <see cref="DataFrame"/>
        /// </summary>
        public DataFrameRowCollection Rows => _rowCollection;

        internal IReadOnlyList<string> GetColumnNames() => _columnCollection.GetColumnNames();

        #region Operators
        /// <summary>
        /// An Indexer to get or set values.
        /// </summary>
        /// <param name="rowIndex">Zero based row index</param>
        /// <param name="columnIndex">Zero based column index</param>
        /// <returns>The value stored at the intersection of <paramref name="rowIndex"/> and <paramref name="columnIndex"/></returns>
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
        /// <param name="rowFilter">A column of booleans</param>
        public DataFrame this[PrimitiveDataFrameColumn<bool> rowFilter] => Filter(rowFilter);

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
        /// Returns a new DataFrame using the boolean values in <paramref name="rowFilter"/>
        /// </summary>
        public DataFrame this[IEnumerable<bool> rowFilter]
        {
            get
            {
                PrimitiveDataFrameColumn<bool> filterColumn = new PrimitiveDataFrameColumn<bool>("Filter", rowFilter);
                return Clone(filterColumn);
            }
        }

        /// <summary>
        /// An indexer based on <see cref="DataFrameColumn.Name"/>
        /// </summary>
        /// <param name="columnName">The name of a <see cref="DataFrameColumn"/></param>
        /// <returns>A <see cref="DataFrameColumn"/> if it exists.</returns>
        /// <exception cref="ArgumentException">Throws if <paramref name="columnName"/> is not present in this <see cref="DataFrame"/></exception>
        public DataFrameColumn this[string columnName]
        {
            get => Columns[columnName];
            set => Columns[columnName] = value;
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

        /// <summary>
        /// Orders the data frame by a specified column.
        /// </summary>
        /// <param name="columnName">The column name to order by</param>
        public DataFrame OrderBy(string columnName)
        {
            return Sort(columnName, isAscending: true);
        }

        /// <summary>
        /// Orders the data frame by a specified column in descending order.
        /// </summary>
        /// <param name="columnName">The column name to order by</param>
        public DataFrame OrderByDescending(string columnName)
        {
            return Sort(columnName, isAscending: false);
        }

        /// <summary>
        /// Clamps values beyond the specified thresholds on numeric columns
        /// </summary>
        /// <typeparam name="U"></typeparam>
        /// <param name="min">Minimum value. All values below this threshold will be set to it</param>
        /// <param name="max">Maximum value. All values above this threshold will be set to it</param>
        /// <param name="inPlace">Indicates if the operation should be performed in place</param>
        public DataFrame Clamp<U>(U min, U max, bool inPlace = false)
        {
            DataFrame ret = inPlace ? this : Clone();

            for (int i = 0; i < ret.Columns.Count; i++)
            {
                DataFrameColumn column = ret.Columns[i];
                if (column.IsNumericColumn())
                    column.Clamp(min, max, inPlace: true);
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
            if (numberOfRows > Rows.Count)
            {
                throw new ArgumentException(string.Format(Strings.ExceedsNumberOfRows, Rows.Count), nameof(numberOfRows));
            }

            int shuffleLowerLimit = 0;
            int shuffleUpperLimit = (int)Math.Min(Int32.MaxValue, Rows.Count);
            
            int[] shuffleArray = Enumerable.Range(0, shuffleUpperLimit).ToArray();
            Random rand = new Random();
            while (shuffleLowerLimit < numberOfRows)
            {
                int randomIndex = rand.Next(shuffleLowerLimit, shuffleUpperLimit);
                int temp = shuffleArray[shuffleLowerLimit];
                shuffleArray[shuffleLowerLimit] = shuffleArray[randomIndex];
                shuffleArray[randomIndex] = temp;
                shuffleLowerLimit++;
            }
            ArraySegment<int> segment = new ArraySegment<int>(shuffleArray, 0, shuffleLowerLimit);

            PrimitiveDataFrameColumn<int> indices = new PrimitiveDataFrameColumn<int>("indices", segment);
            
            return Clone(indices);
        }

        /// <summary>
        /// Groups the rows of the <see cref="DataFrame"/> by unique values in the <paramref name="columnName"/> column.
        /// </summary>
        /// <param name="columnName">The column used to group unique values</param>
        /// <returns>A GroupBy object that stores the group information.</returns>
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

        /// <summary>
        /// Fills <see langword="null" /> values with <paramref name="value"/>.
        /// </summary>
        /// <param name="value">The value to replace <see langword="null" /> with.</param>
        /// <param name="inPlace">A boolean flag to indicate if the operation should be in place</param>
        /// <returns>A new <see cref="DataFrame"/> if <paramref name="inPlace"/> is not set. Returns this <see cref="DataFrame"/> otherwise.</returns>
        public DataFrame FillNulls(object value, bool inPlace = false)
        {
            DataFrame ret = inPlace ? this : Clone();
            for (int i = 0; i < ret.Columns.Count; i++)
            {
                ret.Columns[i].FillNulls(value, inPlace: true);
            }
            return ret;
        }

        /// <summary>
        /// Fills <see langword="null" /> values in each column with values from <paramref name="values"/>.
        /// </summary>
        /// <param name="values">The values to replace <see langword="null" /> with, one value per column. Should be equal to the number of columns in this <see cref="DataFrame"/>. </param>
        /// <param name="inPlace">A boolean flag to indicate if the operation should be in place</param>
        /// <returns>A new <see cref="DataFrame"/> if <paramref name="inPlace"/> is not set. Returns this <see cref="DataFrame"/> otherwise.</returns>
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
        /// Appends rows to the DataFrame 
        /// </summary> 
        /// <remarks>If an input column's value doesn't match a DataFrameColumn's data type, a conversion will be attempted</remarks> 
        /// <remarks>If a <seealso cref="DataFrameRow"/> in <paramref name="rows"/> is null, a null value is appended to each column</remarks>
        /// <param name="rows">The rows to be appended to this DataFrame </param> 
        /// <param name="inPlace">If set, appends <paramref name="rows"/> in place. Otherwise, a new DataFrame is returned with the <paramref name="rows"/> appended</param>
        public DataFrame Append(IEnumerable<DataFrameRow> rows, bool inPlace = false)
        {
            DataFrame ret = inPlace ? this : Clone();
            foreach (DataFrameRow row in rows)
            {
                ret.Append(row, inPlace: true);
            }
            return ret;
        }

        /// <summary> 
        /// Appends a row to the DataFrame 
        /// </summary> 
        /// <remarks>If a column's value doesn't match its column's data type, a conversion will be attempted</remarks> 
        /// <remarks>If <paramref name="row"/> is null, a null value is appended to each column</remarks>
        /// <param name="row"></param> 
        /// <param name="inPlace">If set, appends a <paramref name="row"/> in place. Otherwise, a new DataFrame is returned with an appended <paramref name="row"/> </param>
        public DataFrame Append(IEnumerable<object> row = null, bool inPlace = false)
        {
            DataFrame ret = inPlace ? this : Clone();
            IEnumerator<DataFrameColumn> columnEnumerator = ret.Columns.GetEnumerator();
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
                    if (value is string stringValue)
                    {
                        if (stringValue.Length == 0 && column.DataType != typeof(string))
                        {
                            value = null;
                        }
                        else if (stringValue.Equals("null", StringComparison.OrdinalIgnoreCase))
                        {
                            value = null;
                        }
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
                // Reset the enumerators
                columnEnumerator = ret.Columns.GetEnumerator();
                columnMoveNext = columnEnumerator.MoveNext();
                rowEnumerator = row.GetEnumerator();
                rowMoveNext = rowEnumerator.MoveNext();
                int cacheIndex = 0;
                while (columnMoveNext && rowMoveNext)
                {
                    DataFrameColumn column = columnEnumerator.Current;
                    object value = cachedObjectConversions[cacheIndex];
                    ret.ResizeByOneAndAppend(column, value);
                    columnMoveNext = columnEnumerator.MoveNext();
                    rowMoveNext = rowEnumerator.MoveNext();
                    cacheIndex++;
                }
            }
            while (columnMoveNext)
            {
                // Fill the remaining columns with null
                DataFrameColumn column = columnEnumerator.Current;
                ret.ResizeByOneAndAppend(column, null);
                columnMoveNext = columnEnumerator.MoveNext();
            }
            ret.Columns.RowCount++;
            return ret;
        }

        /// <summary> 
        /// Appends a row by enumerating column names and values from <paramref name="row"/> 
        /// </summary> 
        /// <remarks>If a column's value doesn't match its column's data type, a conversion will be attempted</remarks> 
        /// <param name="row">An enumeration of column name and value to be appended</param> 
        /// <param name="inPlace">If set, appends <paramref name="row"/> in place. Otherwise, a new DataFrame is returned with an appended <paramref name="row"/> </param>
        public DataFrame Append(IEnumerable<KeyValuePair<string, object>> row, bool inPlace = false)
        {
            DataFrame ret = inPlace ? this : Clone();
            if (row == null)
            {
                throw new ArgumentNullException(nameof(row));
            }

            List<object> cachedObjectConversions = new List<object>();
            foreach (KeyValuePair<string, object> columnAndValue in row)
            {
                string columnName = columnAndValue.Key;
                int index = ret.Columns.IndexOf(columnName);
                if (index == -1)
                {
                    throw new ArgumentException(Strings.InvalidColumnName, nameof(columnName));
                }

                DataFrameColumn column = ret.Columns[index];
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
                int index = ret.Columns.IndexOf(columnName);

                DataFrameColumn column = ret.Columns[index];
                object value = cachedObjectConversions[cacheIndex];
                ret.ResizeByOneAndAppend(column, value);
                cacheIndex++;
            }

            foreach (DataFrameColumn column in ret.Columns)
            {
                if (column.Length == Rows.Count)
                {
                    ret.ResizeByOneAndAppend(column, null);
                }
            }
            ret.Columns.RowCount++;
            return ret;
        }

        /// <summary>
        /// Invalidates any cached data after a column has changed.
        /// </summary>
        private void OnColumnsChanged()
        {
            _schema = null;
        }

        private DataFrame Sort(string columnName, bool isAscending)
        {
            DataFrameColumn column = Columns[columnName];
            DataFrameColumn sortIndices = column.GetAscendingSortIndices();
            List<DataFrameColumn> newColumns = new List<DataFrameColumn>(Columns.Count);
            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn oldColumn = Columns[i];
                DataFrameColumn newColumn = oldColumn.Clone(sortIndices, !isAscending, oldColumn.NullCount);
                Debug.Assert(newColumn.NullCount == oldColumn.NullCount);
                newColumns.Add(newColumn);
            }
            return new DataFrame(newColumns);
        }

        /// <summary>
        /// A preview of the contents of this <see cref="DataFrame"/> as a string.
        /// </summary>
        /// <returns>A preview of the contents of this <see cref="DataFrame"/>.</returns>
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
