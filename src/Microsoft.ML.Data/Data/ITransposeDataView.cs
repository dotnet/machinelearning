// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Data
{
    // REVIEW: There are a couple problems. Firstly, what to do about cases where
    // the number of rows exceeds int.MaxValue? Right now we just fail. Practically this makes
    // little difference; if we suppose the major user of this is FastTree, it can't handle
    // nearly that many examples anyway, but other applications may arise. (One possible solution:
    // do not return a slot type, but return a row count, and have the getter accessor be able
    // to define a slot range. Or a paged dataview, but that seems more involved.)

    /// <summary>
    /// A view of data where columns can optionally be accessed slot by slot, as opposed to row
    /// by row in a typical dataview. A slot-accessible column can be accessed with a slot-by-slot
    /// cursor via an <see cref="SlotCursor"/> returned by <see cref="GetSlotCursor(int)"/>
    /// (naturally, as opposed to row-by-row through an <see cref="DataViewRowCursor"/>). This interface
    /// is intended to be implemented by classes that want to provide an option for an alternate
    /// way of accessing the data stored in a <see cref="IDataView"/>.
    ///
    /// The interface only advertises that columns may be accessible in slot-wise fashion. The i-th column
    /// is accessible in this fashion iff <see cref="GetSlotType"/> with col=i doesn't return <see langword="null"/>.
    /// </summary>
    [BestFriend]
    internal interface ITransposeDataView : IDataView
    {
        /// <summary>
        /// Presents a cursor over the slots of a transposable column, or throws if the column
        /// is not transposable.
        /// </summary>
        SlotCursor GetSlotCursor(int col);

        /// <summary>
        /// <see cref="GetSlotType"/> (input argument is named col) specifies the type of all values at the col-th column of
        /// <see cref="IDataView"/>.  For example, if <see cref="IDataView.Schema"/>[i] is a scalar float column, then
        /// <see cref="GetSlotType"/> with col=i may return a <see cref="VectorType"/> whose <see cref="VectorType.ItemType"/>
        /// field is <see cref="NumberDataViewType.Single"/>. If the i-th column can't be iterated column-wisely, this function may
        /// return <see langword="null"/>.
        /// </summary>
        VectorType GetSlotType(int col);
    }
}
