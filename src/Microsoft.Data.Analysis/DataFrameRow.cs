// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Microsoft.Data.Analysis
{
    /// <summary>
    /// A DataFrameRow is a collection of values that represent a row in a <see cref="DataFrame"/>.
    /// </summary>
    public class DataFrameRow : IEnumerable<object>
    {
        private readonly DataFrame _dataFrame;
        private readonly long _rowIndex;
        internal DataFrameRow(DataFrame df, long rowIndex)
        {
            Debug.Assert(rowIndex < df.Columns.RowCount);
            _dataFrame = df;
            _rowIndex = rowIndex;
        }

        /// <summary>
        /// Returns an enumerator of the values in this row.
        /// </summary>
        public IEnumerator<object> GetEnumerator()
        {
            foreach (DataFrameColumn column in _dataFrame.Columns)
            {
                yield return column[_rowIndex];
            }
        }

        /// <summary>
        /// An indexer to return the value at <paramref name="index"/>.
        /// </summary>
        /// <param name="index">The index of the value to return</param>
        /// <returns>The value at this <paramref name="index"/>.</returns>
        public object this[int index]
        {
            get
            {
                return _dataFrame.Columns[index][_rowIndex];
            }
            set
            {
                _dataFrame.Columns[index][_rowIndex] = value;
            }
        }

        /// <summary>
        /// A simple string representation of the values in this row
        /// </summary>
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            foreach (object value in this)
            {
                sb.Append(value?.ToString() ?? "null").Append(" ");
            }
            return sb.ToString();
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
