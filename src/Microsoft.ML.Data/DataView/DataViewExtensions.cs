// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Data
{
    [BestFriend]
    internal static class DataViewExtensions
    {
        /// <summary>
        /// Get a row cursor. The <paramref name="columnsNeeded"/> are the active columns.
        /// The schema of the returned cursor will be the same as the schema of the IDataView, but getting
        /// a getter for an inactive columns will throw.
        /// </summary>
        /// <param name="columnsNeeded">The columns requested by this <see cref="RowCursor"/>, or as otherwise called, the active columns.
        /// An empty collection indicates that no column is needed.</param>
        /// <param name="dv">The <see cref="IDataView"/> containing the <paramref name="columnsNeeded"/>.</param>
        public static RowCursor GetRowCursor(this IDataView dv, params Schema.Column[] columnsNeeded)
        {
            Contracts.AssertValue(columnsNeeded, $"The {nameof(columnsNeeded)} cannot be null. Pass an empty array, to indicate that no columns is needed.");

            foreach (var col in columnsNeeded)
                Contracts.Assert(dv.Schema[col.Index].Equals(col), $"The requested column named: {col.Name}, with index: {col.Index} and type: {col.Type}" +
                    $" is not present in the {nameof(IDataView)} where the {nameof(RowCursor)} is being requested.");

            return dv.GetRowCursor(columnsNeeded);
        }

        /// <summary>
        /// Get a row cursor. The <paramref name="columnNeeded"/> is the active column.
        /// The schema of the returned cursor will be the same as the schema of the IDataView, but getting
        /// a getter for the other, inactive columns will throw.
        /// </summary>
        /// <param name="columnNeeded">The column requested by this <see cref="RowCursor"/>, or as otherwise called, the active column.</param>
        /// <param name="dv">The <see cref="IDataView"/> containing the <paramref name="columnNeeded"/>.</param>
        public static RowCursor GetRowCursor(this IDataView dv, Schema.Column columnNeeded)
        {
            Contracts.Assert(dv.Schema[columnNeeded.Index].Equals(columnNeeded), $"The requested column named: {columnNeeded.Name}, with index: {columnNeeded.Index} and type: {columnNeeded.Type}" +
                   $" is not present in the {nameof(IDataView)} where the {nameof(RowCursor)} is being requested.");

            return dv.GetRowCursor(Enumerable.Repeat(columnNeeded,1));
        }

        /// <summary>
        /// Get a row cursor. No colums are needed by this <see cref="RowCursor"/>.
        /// </summary>
        public static RowCursor GetRowCursor(this IDataView dv) =>  dv.GetRowCursor(Enumerable.Empty<Schema.Column>());

        /// <summary>
        /// Get a row cursor including all the columns of the <see cref="IDataView"/> it is called upon..
        /// </summary>
        public static RowCursor GetRowCursorForAllColumns(this IDataView dv) => dv.GetRowCursor(dv.Schema);
    }
}
