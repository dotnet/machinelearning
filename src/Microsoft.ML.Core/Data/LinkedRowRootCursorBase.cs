// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// A base class for a <see cref="DataViewRowCursor"/> that has an input cursor, but still needs to do work on
    /// <see cref="DataViewRowCursor.MoveNext"/>. Note that the default
    /// <see cref="LinkedRowRootCursorBase.GetGetter{TValue}(DataViewSchema.Column)"/> assumes that each input column is exposed as an
    /// output column with the same column index.
    /// </summary>
    [BestFriend]
    internal abstract class LinkedRowRootCursorBase : LinkedRootCursorBase
    {
        private readonly bool[] _active;

        /// <summary>Gets row's schema.</summary>
        public sealed override DataViewSchema Schema { get; }

        protected LinkedRowRootCursorBase(IChannelProvider provider, DataViewRowCursor input, DataViewSchema schema, bool[] active)
            : base(provider, input)
        {
            Ch.CheckValue(schema, nameof(schema));
            Ch.Check(active == null || active.Length == schema.Count);
            _active = active;
            Schema = schema;
        }

        /// <summary>
        /// Returns whether the given column is active in this row.
        /// </summary>
        public sealed override bool IsColumnActive(DataViewSchema.Column column)
        {
            Ch.Check(column.Index < Schema.Count);
            return _active == null || _active[column.Index];
        }

        /// <summary>
        /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
        /// This throws if the column is not active in this row, or if the type
        /// <typeparamref name="TValue"/> differs from this column's type.
        /// </summary>
        /// <typeparam name="TValue"> is the column's content type.</typeparam>
        /// <param name="column"> is the output column whose getter should be returned.</param>
        public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
        {
            return Input.GetGetter<TValue>(column);
        }
    }
}