// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;

namespace Microsoft.Data.Analysis
{
    /// <summary>
    /// Represents the rows of a <see cref="DataFrame"/>
    /// </summary>
    public class DataFrameRowCollection : IEnumerable<DataFrameRow>
    {
        private readonly DataFrame _dataFrame;

        /// <summary>
        /// Initializes a <see cref="DataFrameRowCollection"/>.
        /// </summary>
        internal DataFrameRowCollection(DataFrame dataFrame)
        {
            _dataFrame = dataFrame ?? throw new ArgumentNullException(nameof(dataFrame));
        }

        /// <summary>
        /// An indexer to return the <see cref="DataFrameRow"/> at <paramref name="index"/>
        /// </summary>
        /// <param name="index">The row index</param>
        public DataFrameRow this[long index]
        {
            get
            {
                return new DataFrameRow(_dataFrame, index);
            }
        }

        /// <summary>
        /// Returns an enumerator of <see cref="DataFrameRow"/> objects
        /// </summary>
        public IEnumerator<DataFrameRow> GetEnumerator()
        {
            for (long i = 0; i < Count; i++)
            {
                yield return new DataFrameRow(_dataFrame, i);
            }
        }

        /// <summary>
        /// The number of rows in this <see cref="DataFrame"/>.
        /// </summary>
        public long Count => _dataFrame.Columns.RowCount;

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
