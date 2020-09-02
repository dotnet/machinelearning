// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace Microsoft.Data.Analysis
{
    /// <summary>
    /// A DataFrameColumnCollection is just a container that holds a number of DataFrameColumn instances. 
    /// </summary>
    public class DataFrameColumnCollection : Collection<DataFrameColumn>
    {
        private Action ColumnsChanged;

        private List<string> _columnNames = new List<string>();

        private Dictionary<string, int> _columnNameToIndexDictionary = new Dictionary<string, int>(StringComparer.Ordinal);

        internal long RowCount { get; set; }

        internal DataFrameColumnCollection(IEnumerable<DataFrameColumn> columns, Action columnsChanged) : base()
        {
            columns = columns ?? throw new ArgumentNullException(nameof(columns));
            ColumnsChanged = columnsChanged;
            foreach (DataFrameColumn column in columns)
            {
                Add(column);
            }
        }

        internal IReadOnlyList<string> GetColumnNames()
        {
            var ret = new List<string>(Count);
            for (int i = 0; i < Count; i++)
            {
                ret.Add(this[i].Name);
            }
            return ret;
        }

        public void SetColumnName(DataFrameColumn column, string newName)
        {
            string currentName = column.Name;
            int currentIndex = _columnNameToIndexDictionary[currentName];
            column.SetName(newName);
            _columnNames[currentIndex] = newName;
            _columnNameToIndexDictionary.Remove(currentName);
            _columnNameToIndexDictionary.Add(newName, currentIndex);
            ColumnsChanged?.Invoke();
        }

        public void Insert<T>(int columnIndex, IEnumerable<T> column, string columnName)
            where T : unmanaged
        {
            DataFrameColumn newColumn = new PrimitiveDataFrameColumn<T>(columnName, column);
            Insert(columnIndex, newColumn); // calls InsertItem internally
        }

        protected override void InsertItem(int columnIndex, DataFrameColumn column)
        {
            column = column ?? throw new ArgumentNullException(nameof(column));
            if (RowCount > 0 && column.Length != RowCount)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }

            if (Count >= 1 && RowCount == 0 && column.Length != RowCount)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }

            if (_columnNameToIndexDictionary.ContainsKey(column.Name))
            {
                throw new ArgumentException(string.Format(Strings.DuplicateColumnName, column.Name), nameof(column));
            }
            RowCount = column.Length;
            _columnNames.Insert(columnIndex, column.Name);
            _columnNameToIndexDictionary[column.Name] = columnIndex;
            for (int i = columnIndex + 1; i < Count; i++)
            {
                _columnNameToIndexDictionary[_columnNames[i]]++;
            }
            base.InsertItem(columnIndex, column);
            ColumnsChanged?.Invoke();
        }

        protected override void SetItem(int columnIndex, DataFrameColumn column)
        {
            column = column ?? throw new ArgumentNullException(nameof(column));
            if (RowCount > 0 && column.Length != RowCount)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            bool existingColumn = _columnNameToIndexDictionary.TryGetValue(column.Name, out int existingColumnIndex);
            if (existingColumn && existingColumnIndex != columnIndex)
            {
                throw new ArgumentException(string.Format(Strings.DuplicateColumnName, column.Name), nameof(column));
            }
            _columnNameToIndexDictionary.Remove(_columnNames[columnIndex]);
            _columnNames[columnIndex] = column.Name;
            _columnNameToIndexDictionary[column.Name] = columnIndex;
            base.SetItem(columnIndex, column);
            ColumnsChanged?.Invoke();
        }

        protected override void RemoveItem(int columnIndex)
        {
            _columnNameToIndexDictionary.Remove(_columnNames[columnIndex]);
            for (int i = columnIndex + 1; i < Count; i++)
            {
                _columnNameToIndexDictionary[_columnNames[i]]--;
            }
            _columnNames.RemoveAt(columnIndex);
            base.RemoveItem(columnIndex);
            ColumnsChanged?.Invoke();
        }

        public void Remove(string columnName)
        {
            int columnIndex = IndexOf(columnName);
            if (columnIndex != -1)
            {
                RemoveAt(columnIndex); // calls RemoveItem internally
            }
        }

        /// <summary>
        /// Searches for a <see cref="DataFrameColumn"/> with the specified <paramref name="columnName"/> and returns the zero-based index of the first occurrence if found. Returns -1 otherwise
        /// </summary>
        /// <param name="columnName"></param>
        public int IndexOf(string columnName)
        {
            if (columnName != null && _columnNameToIndexDictionary.TryGetValue(columnName, out int columnIndex))
            {
                return columnIndex;
            }
            return -1;
        }

        protected override void ClearItems()
        {
            base.ClearItems();
            ColumnsChanged?.Invoke();
            _columnNames.Clear();
            _columnNameToIndexDictionary.Clear();
        }

        /// <summary>
        /// An indexer based on <see cref="DataFrameColumn.Name"/>
        /// </summary>
        /// <param name="columnName">The name of a <see cref="DataFrameColumn"/></param>
        /// <returns>A <see cref="DataFrameColumn"/> if it exists.</returns>
        /// <exception cref="ArgumentException">Throws if <paramref name="columnName"/> is not present in this <see cref="DataFrame"/></exception>
        public DataFrameColumn this[string columnName]
        {
            get
            {
                int columnIndex = IndexOf(columnName);
                if (columnIndex == -1)
                {
                    throw new ArgumentException(Strings.InvalidColumnName, nameof(columnName));
                }
                return this[columnIndex];
            }
            set
            {
                int columnIndex = IndexOf(columnName);
                DataFrameColumn newColumn = value;
                newColumn.SetName(columnName);
                if (columnIndex == -1)
                {
                    Insert(Count, newColumn);
                }
                else
                {
                    this[columnIndex] = newColumn;
                }
            }
        }

        /// <summary>
        /// Gets the <see cref="PrimitiveDataFrameColumn{T}"/> with the specified <paramref name="name"/>.
        /// </summary>
        /// <param name="name">The name of the column</param>
        /// <returns><see cref="PrimitiveDataFrameColumn{T}"/>.</returns>
        /// <exception cref="ArgumentException">A column named <paramref name="name"/> cannot be found, or if the column's type doesn't match.</exception>
        public PrimitiveDataFrameColumn<T> GetPrimitiveColumn<T>(string name)
            where T : unmanaged
        {
            DataFrameColumn column = this[name];
            if (column is PrimitiveDataFrameColumn<T> ret)
            {
                return ret;
            }

            throw new ArgumentException(string.Format(Strings.BadColumnCast, column.DataType, typeof(T)), nameof(T));
        }

        /// <summary>
        /// Gets the <see cref="ArrowStringDataFrameColumn"/> with the specified <paramref name="name"/>.
        /// </summary>
        /// <param name="name">The name of the column</param>
        /// <returns><see cref="ArrowStringDataFrameColumn"/>.</returns>
        /// <exception cref="ArgumentException">A column named <paramref name="name"/> cannot be found, or if the column's type doesn't match.</exception>
        public ArrowStringDataFrameColumn GetArrowStringColumn(string name)
        {
            DataFrameColumn column = this[name];
            if (column is ArrowStringDataFrameColumn ret)
            {
                return ret;
            }

            throw new ArgumentException(string.Format(Strings.BadColumnCast, column.DataType, typeof(string)));
        }

        /// <summary>
        /// Gets the <see cref="StringDataFrameColumn"/> with the specified <paramref name="name"/>.
        /// </summary>
        /// <param name="name">The name of the column</param>
        /// <returns><see cref="StringDataFrameColumn"/>.</returns>
        /// <exception cref="ArgumentException">A column named <paramref name="name"/> cannot be found, or if the column's type doesn't match.</exception>
        public StringDataFrameColumn GetStringColumn(string name)
        {
            DataFrameColumn column = this[name];
            if (column is StringDataFrameColumn ret)
            {
                return ret;
            }

            throw new ArgumentException(string.Format(Strings.BadColumnCast, column.DataType, typeof(string)));
        }

        /// <summary>
        /// Gets the <see cref="BooleanDataFrameColumn"/> with the specified <paramref name="name"/>.
        /// </summary>
        /// <param name="name">The name of the column</param>
        /// <returns><see cref="BooleanDataFrameColumn"/>.</returns>
        /// <exception cref="ArgumentException">A column named <paramref name="name"/> cannot be found, or if the column's type doesn't match.</exception>
        public BooleanDataFrameColumn GetBooleanColumn(string name)
        {
            DataFrameColumn column = this[name];
            if (column is BooleanDataFrameColumn ret)
            {
                return ret;
            }

            throw new ArgumentException(string.Format(Strings.BadColumnCast, column.DataType, typeof(Boolean)));
        }

        /// <summary>
        /// Gets the <see cref="ByteDataFrameColumn"/> with the specified <paramref name="name"/> and attempts to return it as an <see cref="ByteDataFrameColumn"/>. If <see cref="DataFrameColumn.DataType"/> is not of type <see cref="Byte"/>, an exception is thrown.
        /// </summary>
        /// <param name="name">The name of the column</param>
        /// <returns><see cref="ByteDataFrameColumn"/>.</returns>
        /// <exception cref="ArgumentException">A column named <paramref name="name"/> cannot be found, or if the column's type doesn't match.</exception>
        public ByteDataFrameColumn GetByteColumn(string name)
        {
            DataFrameColumn column = this[name];
            if (column is ByteDataFrameColumn ret)
            {
                return ret;
            }

            throw new ArgumentException(string.Format(Strings.BadColumnCast, column.DataType, typeof(Byte)));
        }

        /// <summary>
        /// Gets the <see cref="CharDataFrameColumn"/> with the specified <paramref name="name"/>.
        /// </summary>
        /// <param name="name">The name of the column</param>
        /// <returns><see cref="CharDataFrameColumn"/>.</returns>
        /// <exception cref="ArgumentException">A column named <paramref name="name"/> cannot be found, or if the column's type doesn't match.</exception>
        public CharDataFrameColumn GetCharColumn(string name)
        {
            DataFrameColumn column = this[name];
            if (column is CharDataFrameColumn ret)
            {
                return ret;
            }

            throw new ArgumentException(string.Format(Strings.BadColumnCast, column.DataType, typeof(Char)));
        }

        /// <summary>
        /// Gets the <see cref="DoubleDataFrameColumn"/> with the specified <paramref name="name"/>.
        /// </summary>
        /// <param name="name">The name of the column</param>
        /// <returns><see cref="DoubleDataFrameColumn"/>.</returns>
        /// <exception cref="ArgumentException">A column named <paramref name="name"/> cannot be found, or if the column's type doesn't match.</exception>
        public DoubleDataFrameColumn GetDoubleColumn(string name)
        {
            DataFrameColumn column = this[name];
            if (column is DoubleDataFrameColumn ret)
            {
                return ret;
            }

            throw new ArgumentException(string.Format(Strings.BadColumnCast, column.DataType, typeof(Double)));
        }

        /// <summary>
        /// Gets the <see cref="DecimalDataFrameColumn"/> with the specified <paramref name="name"/>.
        /// </summary>
        /// <param name="name">The name of the column</param>
        /// <returns><see cref="DecimalDataFrameColumn"/>.</returns>
        /// <exception cref="ArgumentException">A column named <paramref name="name"/> cannot be found, or if the column's type doesn't match.</exception>
        public DecimalDataFrameColumn GetDecimalColumn(string name)
        {
            DataFrameColumn column = this[name];
            if (column is DecimalDataFrameColumn ret)
            {
                return ret;
            }

            throw new ArgumentException(string.Format(Strings.BadColumnCast, column.DataType, typeof(Decimal)));
        }

        /// <summary>
        /// Gets the <see cref="SingleDataFrameColumn"/> with the specified <paramref name="name"/>.
        /// </summary>
        /// <param name="name">The name of the column</param>
        /// <returns><see cref="SingleDataFrameColumn"/>.</returns>
        /// <exception cref="ArgumentException">A column named <paramref name="name"/> cannot be found, or if the column's type doesn't match.</exception>
        public SingleDataFrameColumn GetSingleColumn(string name)
        {
            DataFrameColumn column = this[name];
            if (column is SingleDataFrameColumn ret)
            {
                return ret;
            }

            throw new ArgumentException(string.Format(Strings.BadColumnCast, column.DataType, typeof(Single)));
        }

        /// <summary>
        /// Gets the <see cref="Int32DataFrameColumn"/> with the specified <paramref name="name"/>.
        /// </summary>
        /// <param name="name">The name of the column</param>
        /// <returns><see cref="Int32DataFrameColumn"/>.</returns>
        /// <exception cref="ArgumentException">A column named <paramref name="name"/> cannot be found, or if the column's type doesn't match.</exception>
        public Int32DataFrameColumn GetInt32Column(string name)
        {
            DataFrameColumn column = this[name];
            if (column is Int32DataFrameColumn ret)
            {
                return ret;
            }

            throw new ArgumentException(string.Format(Strings.BadColumnCast, column.DataType, typeof(Int32)));
        }

        /// <summary>
        /// Gets the <see cref="Int64DataFrameColumn"/> with the specified <paramref name="name"/>.
        /// </summary>
        /// <param name="name">The name of the column</param>
        /// <returns><see cref="Int64DataFrameColumn"/>.</returns>
        /// <exception cref="ArgumentException">A column named <paramref name="name"/> cannot be found, or if the column's type doesn't match.</exception>
        public Int64DataFrameColumn GetInt64Column(string name)
        {
            DataFrameColumn column = this[name];
            if (column is Int64DataFrameColumn ret)
            {
                return ret;
            }

            throw new ArgumentException(string.Format(Strings.BadColumnCast, column.DataType, typeof(Int64)));
        }

        /// <summary>
        /// Gets the <see cref="SByteDataFrameColumn"/> with the specified <paramref name="name"/>.
        /// </summary>
        /// <param name="name">The name of the column</param>
        /// <returns><see cref="SByteDataFrameColumn"/>.</returns>
        /// <exception cref="ArgumentException">A column named <paramref name="name"/> cannot be found, or if the column's type doesn't match.</exception>
        public SByteDataFrameColumn GetSByteColumn(string name)
        {
            DataFrameColumn column = this[name];
            if (column is SByteDataFrameColumn ret)
            {
                return ret;
            }

            throw new ArgumentException(string.Format(Strings.BadColumnCast, column.DataType, typeof(SByte)));
        }

        /// <summary>
        /// Gets the <see cref="Int16DataFrameColumn"/> with the specified <paramref name="name"/>.
        /// </summary>
        /// <param name="name">The name of the column</param>
        /// <returns><see cref="Int16DataFrameColumn"/>.</returns>
        /// <exception cref="ArgumentException">A column named <paramref name="name"/> cannot be found, or if the column's type doesn't match.</exception>
        public Int16DataFrameColumn GetInt16Column(string name)
        {
            DataFrameColumn column = this[name];
            if (column is Int16DataFrameColumn ret)
            {
                return ret;
            }

            throw new ArgumentException(string.Format(Strings.BadColumnCast, column.DataType, typeof(Int16)));
        }

        /// <summary>
        /// Gets the <see cref="UInt32DataFrameColumn"/> with the specified <paramref name="name"/>.
        /// </summary>
        /// <param name="name">The name of the column</param>
        /// <returns><see cref="UInt32DataFrameColumn"/>.</returns>
        /// <exception cref="ArgumentException">A column named <paramref name="name"/> cannot be found, or if the column's type doesn't match.</exception>
        public UInt32DataFrameColumn GetUInt32Column(string name)
        {
            DataFrameColumn column = this[name];
            if (column is UInt32DataFrameColumn ret)
            {
                return ret;
            }

            throw new ArgumentException(string.Format(Strings.BadColumnCast, column.DataType, typeof(string)));
        }

        /// <summary>
        /// Gets the <see cref="UInt64DataFrameColumn"/> with the specified <paramref name="name"/>.
        /// </summary>
        /// <param name="name">The name of the column</param>
        /// <returns><see cref="UInt64DataFrameColumn"/>.</returns>
        /// <exception cref="ArgumentException">A column named <paramref name="name"/> cannot be found, or if the column's type doesn't match.</exception>
        public UInt64DataFrameColumn GetUInt64Column(string name)
        {
            DataFrameColumn column = this[name];
            if (column is UInt64DataFrameColumn ret)
            {
                return ret;
            }

            throw new ArgumentException(string.Format(Strings.BadColumnCast, column.DataType, typeof(UInt64)));
        }

        /// <summary>
        /// Gets the <see cref="UInt16DataFrameColumn"/> with the specified <paramref name="name"/>.
        /// </summary>
        /// <param name="name">The name of the column</param>
        /// <returns><see cref="UInt16DataFrameColumn"/>.</returns>
        /// <exception cref="ArgumentException">A column named <paramref name="name"/> cannot be found, or if the column's type doesn't match.</exception>
        public UInt16DataFrameColumn GetUInt16Column(string name)
        {
            DataFrameColumn column = this[name];
            if (column is UInt16DataFrameColumn ret)
            {
                return ret;
            }

            throw new ArgumentException(string.Format(Strings.BadColumnCast, column.DataType, typeof(UInt16)));
        }

    }
}
