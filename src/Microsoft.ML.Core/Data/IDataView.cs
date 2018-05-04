// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Interface for schema information.
    /// </summary>
    public interface ISchema
    {
        /// <summary>
        /// Number of columns.
        /// </summary>
        int ColumnCount {
            get;
        }

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
    /// Base interface for schematized information. IDataView and IRowCursor both derive from this.
    /// </summary>
    public interface ISchematized
    {
        /// <summary>
        /// Gets an instance of Schema.
        /// </summary>
        ISchema Schema { get; }
    }

    /// <summary>
    /// The input and output of Query Operators (Transforms). This is the fundamental data pipeline
    /// type, comparable to IEnumerable for LINQ.
    /// </summary>
    public interface IDataView : ISchematized
    {
        /// <summary>
        /// Whether this IDataView supports shuffling of rows, to any degree.
        /// </summary>
        bool CanShuffle { get; }

        /// <summary>
        /// Returns the number of rows if known. Null means unknown. If lazy is true, then
        /// this is permitted to return null when it might return a non-null value on a subsequent
        /// call. This indicates, that the transform does not YET know the number of rows, but
        /// may in the future. If lazy is false, then this is permitted to do some work (no more
        /// that it would normally do for cursoring) to determine the number of rows.
        /// 
        /// Most components will return the same answer whether lazy is true or false. Some, like
        /// a cache, might return null until the cache is fully populated (when lazy is true). When
        /// lazy is false, such a cache would block until the cache was populated.
        /// </summary>
        long? GetRowCount(bool lazy = true);

        /// <summary>
        /// Get a row cursor. The active column indices are those for which needCol(col) returns true.
        /// The schema of the returned cursor will be the same as the schema of the IDataView, but getting
        /// a getter for an inactive columns will throw. The <paramref name="needCol"/> predicate must be
        /// non-null. To activate all columns, pass "col => true".
        /// </summary>
        IRowCursor GetRowCursor(Func<int, bool> needCol, IRandom rand = null);

        /// <summary>
        /// This constructs a set of parallel batch cursors. The value n is a recommended limit
        /// on cardinality. If <paramref name="n"/> is non-positive, this indicates that the caller
        /// has no recommendation, and the implementation should have some default behavior to cover
        /// this case. Note that this is strictly a recommendation: it is entirely possible that
        /// an implementation can return a different number of cursors.
        /// 
        /// The cursors should return the same data as returned through
        /// <see cref="GetRowCursor(Func{int, bool}, IRandom)"/>, except partitioned: no two cursors
        /// should return the "same" row as would have been returned through the regular serial cursor,
        /// but all rows should be returned by exactly one of the cursors returned from this cursor.
        /// The cursors can have their values reconciled downstream through the use of the
        /// <see cref="ICounted.Batch"/> property.
        /// </summary>
        /// <param name="consolidator">This is an object that can be used to reconcile the
        /// returned array of cursors. When the array of cursors is of length 1, it is legal,
        /// indeed expected, that this parameter should be null.</param>
        /// <param name="needCol">The predicate, where a column is active if this returns true.</param>
        /// <param name="n">The suggested degree of parallelism.</param>
        /// <param name="rand">An instance </param>
        /// <returns></returns>
        IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator,
            Func<int, bool> needCol, int n, IRandom rand = null);
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
        IRowCursor CreateCursor(IChannelProvider provider, IRowCursor[] inputs);
    }

    /// <summary>
    /// Delegate type to get a value. This can used for efficient access to data in an IRow
    /// or IRowCursor.
    /// </summary>
    public delegate void ValueGetter<TValue>(ref TValue value);

    /// <summary>
    /// A logical row. May be a row of an IDataView or a stand-alone row. If/when its contents
    /// change, its ICounted.Counter value is incremented.
    /// </summary>
    public interface IRow : ISchematized, ICounted
    {
        /// <summary>
        /// Returns whether the given column is active in this row.
        /// </summary>
        bool IsColumnActive(int col);

        /// <summary>
        /// Returns a value getter delegate to fetch the given column value from the row.
        /// This throws if the column is not active in this row, or if the type
        /// <typeparamref name="TValue"/> differs from this row's schema's
        /// <see cref="ISchema.GetColumnType(int)"/> on <paramref name="col"/>.
        /// </summary>
        ValueGetter<TValue> GetGetter<TValue>(int col);
    }

    /// <summary>
    /// A cursor through rows of an <see cref="IDataView"/>. Note that this includes/is an
    /// <see cref="IRow"/>, as well as an <see cref="ICursor"/>.
    /// </summary>
    public interface IRowCursor : ICursor, IRow
    {
    }
}