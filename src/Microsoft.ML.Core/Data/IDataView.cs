// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using System;
using System.Collections.Generic;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Legacy interface for schema information.
    /// Please avoid implementing this interface, use <see cref="Schema"/>.
    /// </summary>
    public interface ISchema
    {
        /// <summary>
        /// Number of columns.
        /// </summary>
        int ColumnCount { get; }

        /// <summary>
        /// If there is a column with the given name, set col to its index and return true.
        /// Otherwise, return false. The expectation is that if there are multiple columns
        /// with the same name, the greatest index is returned.
        /// </summary>
        bool TryGetColumnIndex(string name, out int col);

        /// <summary>
        /// Get the name of the given column index. Column names must be non-empty and non-null,
        /// but multiple columns may have the same name.
        /// </summary>
        string GetColumnName(int col);

        /// <summary>
        /// Get the type of the given column index. This must be non-null.
        /// </summary>
        ColumnType GetColumnType(int col);

        /// <summary>
        /// Produces the metadata kinds and associated types supported by the given column.
        /// If there is no metadata the returned enumerable should be non-null, but empty.
        /// The string key values are unique, non-empty, non-null strings. The type should
        /// be non-null.
        /// </summary>
        IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col);

        /// <summary>
        /// If the given column has metadata of the indicated kind, this returns the type of the metadata.
        /// Otherwise, it returns null.
        /// </summary>
        ColumnType GetMetadataTypeOrNull(string kind, int col);

        /// <summary>
        /// Fetches the indicated metadata for the indicated column.
        /// This should only be called if a corresponding call to GetMetadataTypeOrNull
        /// returned non-null. And the TValue type should be compatible with the type
        /// returned by that call. Otherwise, this should throw an exception.
        /// </summary>
        void GetMetadata<TValue>(string kind, int col, ref TValue value);
    }

    /// <summary>
    /// The input and output of Query Operators (Transforms). This is the fundamental data pipeline
    /// type, comparable to IEnumerable for LINQ.
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
        /// a getter for an inactive columns will throw. The <paramref name="needCol"/> predicate must be
        /// non-null. To activate all columns, pass "col => true".
        /// </summary>
        RowCursor GetRowCursor(Func<int, bool> needCol, Random rand = null);

        /// <summary>
        /// This constructs a set of parallel batch cursors. The value n is a recommended limit
        /// on cardinality. If <paramref name="n"/> is non-positive, this indicates that the caller
        /// has no recommendation, and the implementation should have some default behavior to cover
        /// this case. Note that this is strictly a recommendation: it is entirely possible that
        /// an implementation can return a different number of cursors.
        ///
        /// The cursors should return the same data as returned through
        /// <see cref="GetRowCursor(Func{int, bool}, Random)"/>, except partitioned: no two cursors
        /// should return the "same" row as would have been returned through the regular serial cursor,
        /// but all rows should be returned by exactly one of the cursors returned from this cursor.
        /// The cursors can have their values reconciled downstream through the use of the
        /// <see cref="Row.Batch"/> property.
        /// </summary>
        /// <param name="consolidator">This is an object that can be used to reconcile the
        /// returned array of cursors. When the array of cursors is of length 1, it is legal,
        /// indeed expected, that this parameter should be null.</param>
        /// <param name="needCol">The predicate, where a column is active if this returns true.</param>
        /// <param name="n">The suggested degree of parallelism.</param>
        /// <param name="rand">An instance </param>
        /// <returns></returns>
        RowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator,
            Func<int, bool> needCol, int n, Random rand = null);

        /// <summary>
        /// Gets an instance of Schema.
        /// </summary>
        Schema Schema { get; }
    }

    /// <summary>
    /// This is used to consolidate parallel cursors into a single cursor. The object that determines
    /// the number of cursors and splits the row "stream" provides the consolidator object.
    /// </summary>
    public interface IRowCursorConsolidator
    {
        /// <summary>
        /// Create a consolidated cursor from the given parallel cursor set.
        /// </summary>
        RowCursor CreateCursor(IChannelProvider provider, RowCursor[] inputs);
    }

    /// <summary>
    /// Delegate type to get a value. This can used for efficient access to data in an IRow
    /// or IRowCursor.
    /// </summary>
    public delegate void ValueGetter<TValue>(ref TValue value);

    /// <summary>
    /// A logical row. May be a row of an <see cref="IDataView"/> or a stand-alone row. If/when its contents
    /// change, its <see cref="Position"/> value is changed.
    /// </summary>
    public abstract class Row : IDisposable
    {
        /// <summary>
        /// This is incremented when the underlying contents changes, giving clients a way to detect change.
        /// Generally it's -1 when the object is in an invalid state. In particular, for an <see cref="RowCursor"/>, this is -1
        /// when the <see cref="RowCursor.State"/> is <see cref="CursorState.NotStarted"/> or <see cref="CursorState.Done"/>.
        ///
        /// Note that this position is not position within the underlying data, but position of this cursor only.
        /// If one, for example, opened a set of parallel streaming cursors, or a shuffled cursor, each such cursor's
        /// first valid entry would always have position 0.
        /// </summary>
        public abstract long Position { get; }

        /// <summary>
        /// This provides a means for reconciling multiple streams of counted things. Generally, in each stream,
        /// batch numbers should be non-decreasing. Furthermore, any given batch number should only appear in one
        /// of the streams. Order is determined by batch number. The reconciler ensures that each stream (that is
        /// still active) has at least one item available, then takes the item with the smallest batch number.
        ///
        /// Note that there is no suggestion that the batches for a particular entry will be consistent from
        /// cursoring to cursoring, except for the consistency in resulting in the same overall ordering. The same
        /// entry could have different batch numbers from one cursoring to another. There is also no requirement
        /// that any given batch number must appear, at all.
        /// </summary>
        public abstract long Batch { get; }

        /// <summary>
        /// A getter for a 128-bit ID value. It is common for objects to serve multiple <see cref="Row"/>
        /// instances to iterate over what is supposed to be the same data, for example, in a <see cref="IDataView"/>
        /// a cursor set will produce the same data as a serial cursor, just partitioned, and a shuffled cursor
        /// will produce the same data as a serial cursor or any other shuffled cursor, only shuffled. The ID
        /// exists for applications that need to reconcile which entry is actually which. Ideally this ID should
        /// be unique, but for practical reasons, it suffices if collisions are simply extremely improbable.
        ///
        /// Note that this ID, while it must be consistent for multiple streams according to the semantics
        /// above, is not considered part of the data per se. So, to take the example of a data view specifically,
        /// a single data view must render consistent IDs across all cursorings, but there is no suggestion at
        /// all that if the "same" data were presented in a different data view (as by, say, being transformed,
        /// cached, saved, or whatever), that the IDs between the two different data views would have any
        /// discernable relationship.</summary>
        public abstract ValueGetter<RowId> GetIdGetter();

        /// <summary>
        /// Returns whether the given column is active in this row.
        /// </summary>
        public abstract bool IsColumnActive(int col);

        /// <summary>
        /// Returns a value getter delegate to fetch the given column value from the row.
        /// This throws if the column is not active in this row, or if the type
        /// <typeparamref name="TValue"/> differs from this column's type.
        /// </summary>
        public abstract ValueGetter<TValue> GetGetter<TValue>(int col);

        /// <summary>
        /// Gets a <see cref="Schema"/>, which provides name and type information for variables
        /// (i.e., columns in ML.NET's type system) stored in this row.
        /// </summary>
        public abstract Schema Schema { get; }

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
    /// Defines the possible states of a cursor.
    /// </summary>
    public enum CursorState
    {
        NotStarted,
        Good,
        Done
    }

    /// <summary>
    /// The basic cursor base class to cursor through rows of an <see cref="IDataView"/>. Note that
    /// this is also an <see cref="Row"/>. The <see cref="Row.Position"/> is incremented by <see cref="MoveNext"/>
    /// and <see cref="MoveMany"/>. When the cursor state is <see cref="CursorState.NotStarted"/> or
    /// <see cref="CursorState.Done"/>, <see cref="Row.Position"/> is <c>-1</c>. Otherwise,
    /// <see cref="Row.Position"/> >= 0.
    /// </summary>
    public abstract class RowCursor : Row
    {
        /// <summary>
        /// Returns the state of the cursor. Before the first call to <see cref="MoveNext"/> or
        /// <see cref="MoveMany(long)"/> this should be <see cref="CursorState.NotStarted"/>. After
        /// any call those move functions that returns <see langword="true"/>, this should return
        /// <see cref="CursorState.Good"/>,
        /// </summary>
        public abstract CursorState State { get; }

        /// <summary>
        /// Advance to the next row. When the cursor is first created, this method should be called to
        /// move to the first row. Returns <c>false</c> if there are no more rows.
        /// </summary>
        public abstract bool MoveNext();

        /// <summary>
        /// Logically equivalent to calling <see cref="MoveNext"/> the given number of times. The
        /// <paramref name="count"/> parameter must be positive. Note that cursor implementations may be
        /// able to optimize this.
        /// </summary>
        public abstract bool MoveMany(long count);

        /// <summary>
        /// Returns a cursor that can be used for invoking <see cref="Row.Position"/>, <see cref="State"/>,
        /// <see cref="MoveNext"/>, and <see cref="MoveMany"/>, with results identical to calling those
        /// on this cursor. Generally, if the root cursor is not the same as this cursor, using the
        /// root cursor will be faster. As an aside, note that this is not necessarily the case of
        /// values from <see cref="Row.GetIdGetter"/>.
        /// </summary>
        public abstract RowCursor GetRootCursor();
    }
}