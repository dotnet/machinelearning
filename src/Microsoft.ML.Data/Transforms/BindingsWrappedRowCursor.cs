// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// A class for mapping an input to an output cursor assuming no output columns
    /// are requested, given a bindings object. This can be useful for transforms
    /// utilizing the <see cref="ColumnBindingsBase"/>, but for which it is
    /// inconvenient or inefficient to handle the "no output selected" case in their
    /// own implementation.
    /// </summary>
    internal sealed class BindingsWrappedRowCursor : SynchronizedCursorBase
    {
        private readonly ColumnBindingsBase _bindings;

        public override DataViewSchema Schema => _bindings.AsSchema;

        /// <summary>
        /// Creates a wrapped version of the cursor
        /// </summary>
        /// <param name="provider">Channel provider</param>
        /// <param name="input">The input cursor</param>
        /// <param name="bindings">The bindings object, </param>
        public BindingsWrappedRowCursor(IChannelProvider provider, DataViewRowCursor input, ColumnBindingsBase bindings)
            : base(provider, input)
        {
            Ch.CheckValue(input, nameof(input));
            Ch.CheckValue(bindings, nameof(bindings));

            _bindings = bindings;
        }

        /// <summary>
        /// Returns whether the given column is active in this row.
        /// </summary>
        public override bool IsColumnActive(DataViewSchema.Column column)
        {
            Ch.Check(column.Index < _bindings.ColumnCount, nameof(column));
            bool isSrc;
            int srcColumnIndex = _bindings.MapColumnIndex(out isSrc, column.Index);
            return isSrc && Input.IsColumnActive(Schema[srcColumnIndex]);
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
            Ch.Check(IsColumnActive(column), nameof(column));
            bool isSrc;
            var index = _bindings.MapColumnIndex(out isSrc, column.Index);
            Ch.Assert(isSrc);
            return Input.GetGetter<TValue>(Input.Schema[index]);
        }
    }
}
