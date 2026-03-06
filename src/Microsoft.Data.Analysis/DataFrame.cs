// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
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
        internal const int DefaultMaxRowsToShowInPreview = 25;

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
            return Clone(mapIndices: null);
        }

        private DataFrame Clone(DataFrameColumn mapIndices = null)
        {
            List<DataFrameColumn> newColumns = new List<DataFrameColumn>(Columns.Count);
            for (int i = 0; i < Columns.Count; i++)
            {
                newColumns.Add(Columns[i].Clone(mapIndices));
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
        /// <param name="columnName">The column name to order by.</param>
        /// <param name="ascending">Sorting order.</param>
        /// <param name="putNullValuesLast">If true, null values are always put at the end.</param>
        public DataFrame OrderBy(string columnName, bool ascending = true, bool putNullValuesLast = true)
        {
            return Sort(columnName, ascending, putNullValuesLast);
        }

        /// <summary>
        /// Orders the data frame by a specified column in descending order.
        /// </summary>
        /// <param name="columnName">The column name to order by.</param>
        /// <param name="putNullValuesLast">If true, null values are always put at the end.</param>
        public DataFrame OrderByDescending(string columnName, bool putNullValuesLast = true)
        {
            return Sort(columnName, false, putNullValuesLast);
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
                column.SetName(prefix + column.Name);
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
                column.SetName(column.Name + suffix);
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
                throw new ArgumentException(String.Format(Strings.InvalidColumnName, columnName), nameof(columnName));

            DataFrameColumn column = _columnCollection[columnIndex];
            return column.GroupBy(columnIndex, this);
        }

        /// <summary>
        /// Groups the rows of the <see cref="DataFrame"/> by unique values in the <paramref name="columnName"/> column.
        /// </summary>
        /// <typeparam name="TKey">Type of column used for grouping</typeparam>
        /// <param name="columnName">The column used to group unique values</param>
        /// <returns>A GroupBy object that stores the group information.</returns>
        public GroupBy<TKey> GroupBy<TKey>(string columnName)
        {
            GroupBy<TKey> group = GroupBy(columnName) as GroupBy<TKey>;

            if (group == null)
            {
                DataFrameColumn column = this[columnName];
                throw new InvalidCastException(String.Format(Strings.BadColumnCastDuringGrouping, columnName, column.DataType, typeof(TKey)));
            }

            return group;
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
            var filter = new BooleanDataFrameColumn("Filter");

            if (options == DropNullOptions.Any)
            {
                filter.AppendMany(true, Rows.Count);
                var buffers = filter.ColumnContainer.Buffers;

                foreach (var column in Columns)
                {
                    long index = 0;
                    for (int b = 0; b < buffers.Count; b++)
                    {
                        var span = buffers.GetOrCreateMutable(b).Span;

                        for (int i = 0; i < span.Length; i++)
                        {
                            span[i] = span[i] && column.IsValid(index);
                            index++;
                        }
                    }
                }
            }
            else
            {
                filter.AppendMany(false, Rows.Count);
                var buffers = filter.ColumnContainer.Buffers;

                foreach (var column in Columns)
                {
                    long index = 0;
                    for (int b = 0; b < buffers.Count; b++)
                    {
                        var span = buffers.GetOrCreateMutable(b).Span;

                        for (int i = 0; i < span.Length; i++)
                        {
                            span[i] = span[i] || column.IsValid(index);
                            index++;
                        }
                    }
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
        /// <remarks> Values are appended based on the column names</remarks>
        /// <param name="rows">The rows to be appended to this DataFrame </param> 
        /// <param name="inPlace">If set, appends <paramref name="rows"/> in place. Otherwise, a new DataFrame is returned with the <paramref name="rows"/> appended</param>
        /// <param name="cultureInfo">culture info for formatting values</param>
        public DataFrame Append(IEnumerable<DataFrameRow> rows, bool inPlace = false, CultureInfo cultureInfo = null)
        {
            DataFrame ret = inPlace ? this : Clone();
            foreach (DataFrameRow row in rows)
            {
                ret.Append(row.GetValues(), inPlace: true, cultureInfo: cultureInfo);
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
        /// <param name="cultureInfo">Culture info for formatting values</param>
        public DataFrame Append(IEnumerable<object> row = null, bool inPlace = false, CultureInfo cultureInfo = null)
        {
            if (cultureInfo == null)
            {
                cultureInfo = CultureInfo.CurrentCulture;
            }

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
                        value = Convert.ChangeType(value, column.DataType, cultureInfo);

                        if (value is null)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedValueType, column.DataType), column.Name);
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
        /// <param name="cultureInfo">Culture info for formatting values</param>
        public DataFrame Append(IEnumerable<KeyValuePair<string, object>> row, bool inPlace = false, CultureInfo cultureInfo = null)
        {
            if (cultureInfo == null)
            {
                cultureInfo = CultureInfo.CurrentCulture;
            }

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
                    throw new ArgumentException(String.Format(Strings.InvalidColumnName, columnName), nameof(columnName));
                }

                DataFrameColumn column = ret.Columns[index];
                object value = columnAndValue.Value;
                if (value != null)
                {
                    value = Convert.ChangeType(value, column.DataType, cultureInfo);
                    if (value is null)
                    {
                        throw new ArgumentException(string.Format(Strings.MismatchedValueType, column.DataType), column.Name);
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
        /// Transforms the DataFrame from wide format to long format by unpivoting specified columns.
        /// This operation takes multiple value columns and "melts" them into two columns: one containing
        /// the original column names (variable) and one containing the values.
        /// </summary>
        /// <param name="idColumns">
        /// Column names to use as identifier variables. These columns will be repeated in the output
        /// for each value column. Must contain at least one column name.
        /// </param>
        /// <param name="valueColumns">
        /// Column names to unpivot into the variable and value columns. If null, all columns not
        /// specified in <paramref name="idColumns"/> will be used as value columns.
        /// </param>
        /// <param name="variableName">
        /// Name for the new column that will contain the original value column names. Defaults to "variable".
        /// </param>
        /// <param name="valueName">
        /// Name for the new column that will contain the values from the unpivoted columns. Defaults to "value".
        /// If value columns contain different types, this column will be of type string; otherwise, it will
        /// match the type of the first value column.
        /// </param>
        /// <param name="dropNulls">
        /// If true, rows where the value is null or empty string will be excluded from the result.
        /// Defaults to false.
        /// </param>
        /// <returns>
        /// A new DataFrame in long format with columns for each ID column, plus the variable and value columns.
        /// The number of rows will be approximately (number of original rows × number of value columns),
        /// or fewer if <paramref name="dropNulls"/> is true.
        /// </returns>
        /// <exception cref="ArgumentException">
        /// Thrown when <paramref name="idColumns"/> is empty, when <paramref name="valueColumns"/> is specified
        /// but empty, or when any column appears in both <paramref name="idColumns"/> and <paramref name="valueColumns"/>.
        /// </exception>
        /// <exception cref="InvalidOperationException">
        /// Thrown when <paramref name="valueColumns"/> is null and there are no columns available to use as
        /// value columns after excluding the ID columns.
        /// </exception>
        /// <example>
        /// <code>
        /// // Original DataFrame:
        /// // | ID | Name  | 2020 | 2021 | 2022 |
        /// // |----|-------|------|------|------|
        /// // | 1  | Alice | 100  | 110  | 120  |
        /// // | 2  | Bob   | 200  | 210  | 220  |
        /// 
        /// var melted = df.Melt(
        ///     idColumns: new[] { "ID", "Name" },
        ///     valueColumns: new[] { "2020", "2021", "2022" },
        ///     variableName: "Year",
        ///     valueName: "Sales"
        /// );
        /// 
        /// // Result:
        /// // | ID | Name  | Year | Sales |
        /// // |----|-------|------|-------|
        /// // | 1  | Alice | 2020 | 100   |
        /// // | 1  | Alice | 2021 | 110   |
        /// // | 1  | Alice | 2022 | 120   |
        /// // | 2  | Bob   | 2020 | 200   |
        /// // | 2  | Bob   | 2021 | 210   |
        /// // | 2  | Bob   | 2022 | 220   |
        /// </code>
        /// </example>
        public DataFrame Melt(IEnumerable<string> idColumns, IEnumerable<string> valueColumns = null, string variableName = "variable", string valueName = "value", bool dropNulls = false)
        {
            if (string.IsNullOrWhiteSpace(variableName))
            {
                throw new ArgumentException("Parameter must not be null, empty, or whitespace", nameof(variableName));
            }

            if (string.IsNullOrWhiteSpace(valueName))
            {
                throw new ArgumentException("Parameter must not be null, empty, or whitespace", nameof(valueName));
            }

            var idColumnList = idColumns?.ToList() ?? new List<string>();

            HashSet<string> idColumnSet = null;

            if (valueColumns is null)
            {
                idColumnSet = [.. idColumnList];
            }

            var valueColumnList = valueColumns?.ToList()
                ?? _columnCollection
                    .Where(c => !idColumnSet.Contains(c.Name))
                    .Select(c => c.Name)
                    .ToList();

            if (idColumnList.Count == 0)
            {
                throw new ArgumentException("Must provide at least 1 ID column", nameof(idColumns));
            }

            if (valueColumns != null && valueColumnList.Count == 0)
            {
                throw new ArgumentException("Must provide at least 1 value column when specifying value columns manually", nameof(valueColumns));
            }

            if (valueColumns != null && valueColumnList.Any(v => idColumnList.Contains(v)))
            {
                throw new ArgumentException("Columns cannot exist in both idColumns and valueColumns", nameof(valueColumns));
            }

            if (valueColumns == null && valueColumnList.Count == 0)
            {
                throw new InvalidOperationException("There are no columns in the DataFrame to use as value columns after excluding the ID columns");
            }

            IEnumerable<string> existingColumnNames = _columnCollection.Select(c => c.Name);

            if (existingColumnNames.Contains(variableName))
            {
                throw new ArgumentException($"Variable name '{variableName}' matches an existing column name", nameof(variableName));
            }

            if (existingColumnNames.Contains(valueName))
            {
                throw new ArgumentException($"Value name '{valueName}' matches an existing column name", nameof(valueName));
            }

            long totalOutputRows = CalculateTotalOutputRows(valueColumnList, dropNulls);

            var outputCols = InitializeIdColumns(idColumnList, totalOutputRows);
            var variableColumn = new StringDataFrameColumn(variableName, totalOutputRows);
            var valueColumn = CreateValueColumn(valueColumnList, valueName, totalOutputRows);

            FillMeltedData(idColumnList, valueColumnList, outputCols, variableColumn, valueColumn, dropNulls);

            outputCols.Add(variableColumn);
            outputCols.Add(valueColumn);

            return new DataFrame(outputCols);
        }

        private long CalculateTotalOutputRows(List<string> valueColumnList, bool dropNulls)
        {
            if (!dropNulls)
            {
                return _rowCollection.Count * valueColumnList.Count;
            }

            long total = 0;

            foreach (var columnName in valueColumnList)
            {
                var column = _columnCollection[columnName];

                foreach (var item in column)
                {
                    if (item is not null and not "")
                    {
                        total++;
                    }
                }
            }

            return total;
        }

        private List<DataFrameColumn> InitializeIdColumns(List<string> idColumnList, long size)
        {
            PrimitiveDataFrameColumn<long> empty = new PrimitiveDataFrameColumn<long>("Empty");
            var outputCols = new List<DataFrameColumn>(idColumnList.Count);

            foreach (var idColumnName in idColumnList)
            {
                var sourceColumn = _columnCollection[idColumnName];
                var newColumn = sourceColumn.Clone(empty);
                newColumn.Resize(size);
                outputCols.Add(newColumn);
            }

            return outputCols;
        }

        private DataFrameColumn CreateValueColumn(List<string> valueColumnList, string valueName, long size)
        {
            var valueTypes = valueColumnList
                .Select(name => _columnCollection[name].DataType)
                .Distinct()
                .Count();

            DataFrameColumn valueColumn;

            if (valueTypes > 1)
            {
                valueColumn = new StringDataFrameColumn(valueName, size);
            }
            else
            {
                PrimitiveDataFrameColumn<long> empty = new PrimitiveDataFrameColumn<long>("Empty");
                valueColumn = _columnCollection[valueColumnList[0]].Clone(empty);
                valueColumn.SetName(valueName);
                valueColumn.Resize(size);
            }

            return valueColumn;
        }

        private void FillMeltedData(List<string> idColumnList, List<string> valueColumnList, List<DataFrameColumn> outputIdCols, StringDataFrameColumn variableColumn, DataFrameColumn valueColumn, bool dropNulls)
        {
            bool mixedTypes = valueColumn is StringDataFrameColumn;
            long currentRow = 0;
            long rowCount = _rowCollection.Count;
            int idColumnCount = idColumnList.Count;

            var idColumns = new DataFrameColumn[idColumnCount];
            for (int i = 0; i < idColumnCount; i++)
            {
                idColumns[i] = _columnCollection[idColumnList[i]];
            }

            foreach (var valueColumnName in valueColumnList)
            {
                var sourceValueColumn = _columnCollection[valueColumnName];

                for (long sourceRow = 0; sourceRow < rowCount; sourceRow++)
                {
                    var value = sourceValueColumn[sourceRow];

                    if (dropNulls && (value is null or ""))
                    {
                        continue;
                    }

                    for (int i = 0; i < idColumnCount; i++)
                    {
                        outputIdCols[i][currentRow] = idColumns[i][sourceRow];
                    }

                    variableColumn[currentRow] = valueColumnName;
                    valueColumn[currentRow] = mixedTypes ? value?.ToString() : value;
                    currentRow++;
                }
            }
        }

        /// <summary>
        /// Invalidates any cached data after a column has changed.
        /// </summary>
        private void OnColumnsChanged()
        {
            _schema = null;
        }

        private DataFrame Sort(string columnName, bool ascending, bool putNullValuesLast)
        {
            DataFrameColumn column = Columns[columnName];
            PrimitiveDataFrameColumn<long> sortIndices = column.GetSortIndices(ascending, putNullValuesLast);

            List<DataFrameColumn> newColumns = new List<DataFrameColumn>(Columns.Count);
            for (int i = 0; i < Columns.Count; i++)
            {
                DataFrameColumn oldColumn = Columns[i];
                DataFrameColumn newColumn = oldColumn.Clone(sortIndices);
                Debug.Assert(newColumn.NullCount == oldColumn.NullCount);
                newColumns.Add(newColumn);
            }
            return new DataFrame(newColumns);
        }

        /// <summary>
        /// A preview of the contents of this <see cref="DataFrame"/> as a string.
        /// </summary>
        /// <returns>A preview of the contents of this <see cref="DataFrame"/>.</returns>
        public override string ToString() => ToString(DefaultMaxRowsToShowInPreview);

        /// <summary>
        /// A preview of the contents of this <see cref="DataFrame"/> as a string.
        /// </summary>
        /// <param name="rowsToShow">Max amount of rows to show in preview.</param>
        /// <returns></returns>
        public string ToString(long rowsToShow)
        {
            StringBuilder sb = new StringBuilder();
            int longestColumnName = 0;
            for (int i = 0; i < Columns.Count; i++)
            {
                longestColumnName = Math.Max(longestColumnName, Columns[i].Name.Length);
            }

            int padding = Math.Max(10, longestColumnName + 1);
            for (int i = 0; i < Columns.Count; i++)
            {
                // Left align by 10 or more (in case of longer column names)
                sb.Append(string.Format(Columns[i].Name.PadRight(padding)));
            }
            sb.AppendLine();
            long numberOfRows = Math.Min(Rows.Count, rowsToShow);
            for (long i = 0; i < numberOfRows; i++)
            {
                foreach (object obj in Rows[i])
                {
                    sb.Append((obj ?? "null").ToString().PadRight(padding));
                }
                sb.AppendLine();
            }

            if (numberOfRows < Rows.Count)
            {
                sb.Append(String.Format(Strings.AmountOfRowsShown, rowsToShow, Rows.Count));
                sb.AppendLine();
            }

            return sb.ToString();
        }
    }
}
