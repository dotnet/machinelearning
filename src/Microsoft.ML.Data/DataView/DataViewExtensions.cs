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
        /// Get a row cursor. The <paramref name="colsNeeded"/> are the active columns.
        /// The schema of the returned cursor will be the same as the schema of the IDataView, but getting
        /// a getter for an inactive columns will throw.
        /// </summary>
        /// <param name="colsNeeded">The columns requested by this <see cref="RowCursor"/>, or as otherwise called, the active columns.
        /// An empty collection indicates that no column is needed.</param>
        /// <param name="dv">The <see cref="IDataView"/> containing the <paramref name="colsNeeded"/>.</param>
        [BestFriend]
        internal static RowCursor GetRowCursor(this IDataView dv, params Schema.Column[] colsNeeded)
        {
            Contracts.AssertValue(colsNeeded, $"The {nameof(colsNeeded)} cannot be null. Pass an empty array, to indicate that no columns is needed.");

            foreach (var col in colsNeeded)
                Contracts.Assert(dv.Schema[col.Index].Equals(col), $"The requested column named: {col.Name}, with index: {col.Index} and type: {col.Type}" +
                    $" is not present in the {nameof(IDataView)} where the {nameof(RowCursor)} is being requested.");

            return dv.GetRowCursor(colsNeeded);
        }

        /// <summary>
        /// Get a row cursor. The <paramref name="colNeeded"/> is the active column.
        /// The schema of the returned cursor will be the same as the schema of the IDataView, but getting
        /// a getter for the other, inactive columns will throw.
        /// </summary>
        /// <param name="colNeeded">The column requested by this <see cref="RowCursor"/>, or as otherwise called, the active column.</param>
        /// <param name="dv">The <see cref="IDataView"/> containing the <paramref name="colNeeded"/>.</param>
        [BestFriend]
        internal static RowCursor GetRowCursor(this IDataView dv, Schema.Column colNeeded)
        {
            Contracts.Assert(dv.Schema[colNeeded.Index].Equals(colNeeded), $"The requested column named: {colNeeded.Name}, with index: {colNeeded.Index} and type: {colNeeded.Type}" +
                   $" is not present in the {nameof(IDataView)} where the {nameof(RowCursor)} is being requested.");

            var list = new List<Schema.Column>();
            list.Add(colNeeded);

            return dv.GetRowCursor(list);
        }

        /// <summary>
        /// Get a row cursor. No colums are needed by this <see cref="RowCursor"/>.
        /// </summary>
        [BestFriend]
        internal static RowCursor GetRowCursor(this IDataView dv) =>  dv.GetRowCursor(new List<Schema.Column>());
    }
}
