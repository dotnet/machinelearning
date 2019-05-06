// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Data
{
    // REVIEW: Would it be a better apporach to add something akin to CanSeek,
    // as we have a CanShuffle? The idea is trying to make IRowSeekable propagate along certain transforms.
    /// <summary>
    /// Represents a data view that supports random access to a specific row.
    /// </summary>
    [BestFriend]
    internal interface IRowSeekable
    {
        RowSeeker GetSeeker(Func<int, bool> predicate);

        DataViewSchema Schema { get; }
    }

    /// <summary>
    /// Represents a row seeker with random access that can retrieve a specific row by the row index.
    /// For <see cref="RowSeeker"/>, when the state is valid (that is when <see cref="MoveTo(long)"/>
    /// returns <see langword="true"/>), it returns the current row index. Otherwise it's -1.
    /// </summary>
    [BestFriend]
    internal abstract class RowSeeker : DataViewRow
    {
        /// <summary>
        /// Moves the seeker to a row at a specific row index.
        /// If the row index specified is out of range (less than zero or not less than the
        /// row count), it returns false and sets its Position property to -1.
        /// </summary>
        /// <param name="rowIndex">The row index to move to.</param>
        /// <returns>True if a row with specified index is found; false otherwise.</returns>
        public abstract bool MoveTo(long rowIndex);
    }
}
