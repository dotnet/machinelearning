// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Data
{
    public static class DataViewExtensions
    {
        /// <summary>
        /// Get a row cursor. The active column indices are those for which needCol(col) returns true.
        /// The schema of the returned cursor will be the same as the schema of the IDataView, but getting
        /// a getter for an inactive columns will throw. The <paramref name="namesOfcolsNeeded"/> predicate must be
        /// non-null.If set to <langword>null</langword> no column is active.
        /// </summary>
        public static RowCursor GetRowCursor(this IDataView dv, params string[] namesOfcolsNeeded)
            => dv.GetRowCursor(dv.Schema.GetColumns(namesOfcolsNeeded));

        /// <summary>
        /// Get a row cursor. The active column indices are those for which needCol(col) returns true.
        /// The schema of the returned cursor will be the same as the schema of the IDataView, but getting
        /// a getter for an inactive columns will throw. The <paramref name="indicesOfcolsNeeded"/> predicate must be
        /// non-null.If set to <langword>null</langword> no column is active.
        /// </summary>
        public static RowCursor GetRowCursor(this IDataView dv, params int[] indicesOfcolsNeeded)
            => dv.GetRowCursor(dv.Schema.GetColumns(indicesOfcolsNeeded));

        /// <summary>
        /// Get a row cursor. The active column indices are those for which needCol(col) returns true.
        /// The schema of the returned cursor will be the same as the schema of the IDataView, but getting
        /// a getter for an inactive columns will throw. The <paramref name="colNeeded"/> predicate must be
        /// non-null.If set to <langword>null</langword> no column is active.
        /// </summary>
        public static RowCursor GetRowCursor(this IDataView dv, string colNeeded)
            => dv.GetRowCursor(new[] { dv.Schema[colNeeded] });

        /// <summary>
        /// Get a row cursor. The active column indices are those for which needCol(col) returns true.
        /// The schema of the returned cursor will be the same as the schema of the IDataView, but getting
        /// a getter for an inactive columns will throw. The <paramref name="colNeeded"/> predicate must be
        /// non-null.If set to <langword>null</langword> no column is active.
        /// </summary>
        public static RowCursor GetRowCursor(this IDataView dv, int colNeeded)
            => dv.GetRowCursor(new[] { dv.Schema[colNeeded]});
    }
}
