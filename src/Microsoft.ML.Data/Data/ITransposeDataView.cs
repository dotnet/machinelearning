// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Runtime.Data
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
    /// cursor via an <see cref="ISlotCursor"/> (naturally, as opposed to row-by-row through an
    /// <see cref="IRowCursor"/>). This interface is intended to be implemented by classes that
    /// want to provide an option for an alternate way of accessing the data stored in a
    /// <see cref="IDataView"/>.
    /// 
    /// The interface only advertises that columns may be accessible in slot-wise fashion. A column
    /// is accessible in this fashion iff <see cref="TransposeSchema"/>'s
    /// <see cref="ITransposeSchema.GetSlotType"/> returns a non-null value.
    /// </summary>
    public interface ITransposeDataView : IDataView
    {
        /// <summary>
        /// An enhanced schema, containing information on the transposition properties, if any,
        /// of each column. Note that there is no contract or suggestion that this property
        /// should be equal to <see cref="ISchematized.Schema"/>.
        /// </summary>
        ITransposeSchema TransposeSchema { get; }

        /// <summary>
        /// Presents a cursor over the slots of a transposable column, or throws if the column
        /// is not transposable.
        /// </summary>
        ISlotCursor GetSlotCursor(int col);
    }

    /// <summary>
    /// A cursor that allows slot-by-slot access of data.
    /// </summary>
    public interface ISlotCursor : ICursor
    {
        /// <summary>
        /// The slot type for this cursor. Note that this should equal the
        /// <see cref="ITransposeSchema.GetSlotType"/> for the column from which this slot cursor
        /// was created.
        /// </summary>
        VectorType GetSlotType();

        /// <summary>
        /// A getter delegate for the slot values. The type <typeparamref name="TValue"/> must correspond
        /// to the item type from <see cref="ITransposeSchema.GetSlotType"/>.
        /// </summary>
        ValueGetter<VBuffer<TValue>> GetGetter<TValue>();
    }

    /// <summary>
    /// The transpose schema returns the schema information of the view we have transposed.
    /// </summary>
    public interface ITransposeSchema : ISchema
    {
        /// <summary>
        /// Analogous to <see cref="ISchema.GetColumnType"/>, except instead of returning the type of value
        /// accessible through the <see cref="IRowCursor"/>, returns the item type of value accessible
        /// through the <see cref="ISlotCursor"/>. This will return <c>null</c> iff this particular
        /// column is not transposable, that is, it cannot be viewed in a slotwise fashion. Observe from
        /// the return type that this will always be a vector type. This vector type should be of fixed
        /// size and one dimension.
        /// </summary>
        VectorType GetSlotType(int col);
    }
}
