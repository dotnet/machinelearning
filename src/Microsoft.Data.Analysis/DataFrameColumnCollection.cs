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
            if (_columnNameToIndexDictionary.TryGetValue(columnName, out int columnIndex))
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
    }
}
