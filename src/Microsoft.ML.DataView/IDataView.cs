// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML
{
    /// <summary>
    /// The input and output of Query Operators (Transforms). This is the fundamental data pipeline
    /// type, comparable to <see cref="IEnumerable{T}"/> for LINQ.
    /// </summary>
    public interface IDataView
    {
        /// <summary>
        /// Whether this IDataView supports shuffling of rows, to any degree.
        /// </summary>
        bool CanShuffle { get; }

        /// <summary>
        /// Returns the number of rows if known. Returning null means that the row count is unknown but
        /// it might return a non-null value on a subsequent call. This indicates, that the transform does
        /// not YET know the number of rows, but may in the future. Its implementation's computation
        /// complexity should be O(1).
        ///
        /// Most implementation will return the same answer every time. Some, like a cache, might
        /// return null until the cache is fully populated.
        /// </summary>
        long? GetRowCount();

        /// <summary>
        /// Get a row cursor. The active column indices are those for which needCol(col) returns true.
        /// The schema of the returned cursor will be the same as the schema of the IDataView, but getting
        /// a getter for inactive columns will throw. The <paramref name="columnsNeeded"/> indicate the columns that are needed
        /// to iterate over.If set to an empty <see cref="IEnumerable"/> no column is requested.
        /// </summary>
        DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null);

        /// <summary>
        /// This constructs a set of parallel batch cursors. The value <paramref name="n"/> is a recommended limit on
        /// cardinality. If <paramref name="n"/> is non-positive, this indicates that the caller has no recommendation,
        /// and the implementation should have some default behavior to cover this case. Note that this is strictly a
        /// recommendation: it is entirely possible that an implementation can return a different number of cursors.
        ///
        /// The cursors should return the same data as returned through
        /// <see cref="GetRowCursor(IEnumerable{DataViewSchema.Column}, Random)"/>, except partitioned: no two cursors should return the
        /// "same" row as would have been returned through the regular serial cursor, but all rows should be returned by
        /// exactly one of the cursors returned from this cursor. The cursors can have their values reconciled
        /// downstream through the use of the <see cref="DataViewRow.Batch"/> property.
        ///
        /// The typical usage pattern is that a set of cursors is requested, each of them is then given to a set of
        /// working threads that consume from them independently while, ultimately, the results are finally collated in
        /// the end by exploiting the ordering of the <see cref="DataViewRow.Batch"/> property described above. More typical
        /// scenarios will be content with pulling from the single serial cursor of
        /// <see cref="GetRowCursor(IEnumerable{DataViewSchema.Column}, Random)"/>.
        /// </summary>
        /// <param name="columnsNeeded">The active columns needed. If passed an empty <see cref="IEnumerable"/> no column is requested.</param>
        /// <param name="n">The suggested degree of parallelism.</param>
        /// <param name="rand">An instance of <see cref="Random"/> to seed randomizing the access.</param>
        /// <returns></returns>
        DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null);

        /// <summary>
        /// Gets an instance of Schema.
        /// </summary>
        DataViewSchema Schema { get; }
    }

    /// <summary>
    /// Delegate type to get a value. This can be used for efficient access to data in a <see cref="DataViewRow"/>
    /// or <see cref="DataViewRowCursor"/>.
    /// </summary>
    public delegate void ValueGetter<TValue>(ref TValue value);

    /// <summary>
    /// A logical row. May be a row of an <see cref="IDataView"/> or a stand-alone row. If/when its contents
    /// change, its <see cref="Position"/> value is changed.
    /// </summary>
    public abstract class DataViewRow : IDisposable
    {
        /// <summary>
        /// This is incremented when the underlying contents changes, giving clients a way to detect change. It should be
        /// -1 when the object is in a state where values cannot be fetched. In particular, for an <see cref="DataViewRowCursor"/>,
        /// this will be before <see cref="DataViewRowCursor.MoveNext"/> if ever called for the first time, or after the first time
        /// <see cref="DataViewRowCursor.MoveNext"/> is called and returns <see langword="false"/>.
        ///
        /// Note that this position is not position within the underlying data, but position of this cursor only. If
        /// one, for example, opened a set of parallel streaming cursors, or a shuffled cursor, each such cursor's first
        /// valid entry would always have position 0.
        /// </summary>
        public abstract long Position { get; }

        /// <summary>
        /// This provides a means for reconciling multiple rows that have been produced generally from
        /// <see cref="IDataView.GetRowCursorSet(IEnumerable{DataViewSchema.Column}, int, Random)"/>. When getting a set, there is a need
        /// to, while allowing parallel processing to proceed, always have an aim that the original order should be
        /// reconverable. Note, whether or not a user cares about that original order in ones specific application is
        /// another story altogether (most callers of this as a practical matter do not, otherwise they would not call
        /// it), but at least in principle it should be possible to reconstruct the original order one would get from an
        /// identically configured <see cref="IDataView.GetRowCursor(IEnumerable{DataViewSchema.Column}, Random)"/>. So: for any cursor
        /// implementation, batch numbers should be non-decreasing. Furthermore, any given batch number should only
        /// appear in one of the cursors as returned by
        /// <see cref="IDataView.GetRowCursorSet(IEnumerable{DataViewSchema.Column}, int, Random)"/>. In this way, order is determined by
        /// batch number. An operation that reconciles these cursors to produce a consistent single cursoring, could do
        /// so by drawing from the single cursor, among all cursors in the set, that has the smallest batch number
        /// available.
        ///
        /// Note that there is no suggestion that the batches for a particular entry will be consistent from cursoring
        /// to cursoring, except for the consistency in resulting in the same overall ordering. The same entry could
        /// have different batch numbers from one cursoring to another. There is also no requirement that any given
        /// batch number must appear, at all. It is merely a mechanism for recovering ordering from a possibly arbitrary
        /// partitioning of the data. It also follows from this, of course, that considering the batch to be a property
        /// of the data is completely invalid.
        /// </summary>
        public abstract long Batch { get; }

        /// <summary>
        /// A getter for a 128-bit ID value. It is common for objects to serve multiple <see cref="DataViewRow"/>
        /// instances to iterate over what is supposed to be the same data, for example, in a <see cref="IDataView"/>
        /// a cursor set will produce the same data as a serial cursor, just partitioned, and a shuffled cursor will
        /// produce the same data as a serial cursor or any other shuffled cursor, only shuffled. The ID exists for
        /// applications that need to reconcile which entry is actually which. Ideally this ID should be unique, but for
        /// practical reasons, it suffices if collisions are simply extremely improbable.
        ///
        /// Note that this ID, while it must be consistent for multiple streams according to the semantics above, is not
        /// considered part of the data per se. So, to take the example of a data view specifically, a single data view
        /// must render consistent IDs across all cursorings, but there is no suggestion at all that if the "same" data
        /// were presented in a different data view (as by, say, being transformed, cached, saved, or whatever), that
        /// the IDs between the two different data views would have any discernable relationship.</summary>
        public abstract ValueGetter<DataViewRowId> GetIdGetter();

        /// <summary>
        /// Returns whether the given column is active in this row.
        /// </summary>
        public abstract bool IsColumnActive(DataViewSchema.Column column);

        /// <summary>
        /// Returns a value getter delegate to fetch the value of the given <paramref name="column"/>, from the row.
        /// This throws if the column is not active in this row, or if the type
        /// <typeparamref name="TValue"/> differs from this column's type.
        /// </summary>
        /// <typeparam name="TValue"> is the column's content type.</typeparam>
        /// <param name="column"> is the output column whose getter should be returned.</param>
        public abstract ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column);

        /// <summary>
        /// Gets a <see cref="Schema"/>, which provides name and type information for variables
        /// (i.e., columns in ML.NET's type system) stored in this row.
        /// </summary>
        public abstract DataViewSchema Schema { get; }

        /// <summary>
        /// Implementation of dispose. Calls <see cref="Dispose(bool)"/> with <see langword="true"/>.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// The disposable method for the disposable pattern. This default implementation does nothing.
        /// </summary>
        /// <param name="disposing">Whether this was called from <see cref="IDisposable.Dispose"/>.
        /// Subclasses that implement <see cref="object.Finalize"/> should call this method with
        /// <see langword="false"/>, but I hasten to add that implementing finalizers should be
        /// avoided if at all possible.</param>.
        protected virtual void Dispose(bool disposing)
        {
        }
    }

    /// <summary>
    /// The basic cursor base class to cursor through rows of an <see cref="IDataView"/>. Note that
    /// this is also an <see cref="DataViewRow"/>. The <see cref="DataViewRow.Position"/> is incremented by <see cref="MoveNext"/>.
    /// Prior to the first call to <see cref="MoveNext"/>, or after the first call to <see cref="MoveNext"/> that
    /// returns <see langword="false"/>, <see cref="DataViewRow.Position"/> is <c>-1</c>. Otherwise, in a situation where the
    /// last call to <see cref="MoveNext"/> returned <see langword="true"/>, <see cref="DataViewRow.Position"/> >= 0.
    /// </summary>
    public abstract class DataViewRowCursor : DataViewRow
    {
        /// <summary>
        /// Advance to the next row. When the cursor is first created, this method should be called to
        /// move to the first row. Returns <see langword="false"/> if there are no more rows.
        /// </summary>
        public abstract bool MoveNext();
    }
}