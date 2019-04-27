// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.AutoML
{
    internal static class RowCursorUtils
    {
        /// <summary>
        /// Given a collection of <see cref="DataViewSchema.Column"/>, that is a subset of the Schema of the data, create a predicate,
        /// that when passed a column index, will return <langword>true</langword> or <langword>false</langword>, based on whether
        /// the column with the given <see cref="DataViewSchema.Column.Index"/> is part of the <paramref name="columnsNeeded"/>.
        /// </summary>
        /// <param name="columnsNeeded">The subset of columns from the <see cref="DataViewSchema"/> that are needed from this <see cref="DataViewRowCursor"/>.</param>
        /// <param name="sourceSchema">The <see cref="DataViewSchema"/> from where the columnsNeeded originate.</param>
        internal static Func<int, bool> FromColumnsToPredicate(IEnumerable<DataViewSchema.Column> columnsNeeded, DataViewSchema sourceSchema)
        {
            Contracts.CheckValue(columnsNeeded, nameof(columnsNeeded));
            Contracts.CheckValue(sourceSchema, nameof(sourceSchema));

            bool[] indicesRequested = new bool[sourceSchema.Count];

            foreach (var col in columnsNeeded)
            {
                if (col.Index >= indicesRequested.Length)
                    throw Contracts.Except($"The requested column: {col} is not part of the {nameof(sourceSchema)}");

                indicesRequested[col.Index] = true;
            }

            return c => indicesRequested[c];
        }

        internal const string FetchValueStateError = "Values cannot be fetched at this time. This method was called either before the first call to "
            + nameof(DataViewRowCursor.MoveNext) + ", or at any point after that method returned false.";
    }
}
