// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// An implementation of <see cref="DataViewRow"/> that gets its <see cref="DataViewRow.Position"/>, <see cref="DataViewRow.Batch"/>,
    /// and <see cref="DataViewRow.GetIdGetter"/> from an input row. The constructor requires a schema and array of getter
    /// delegates. A <see langword="null"/> delegate indicates an inactive column. The delegates are assumed to be
    /// of the appropriate type (this does not validate the type).
    /// REVIEW: Should this validate that the delegates are of the appropriate type? It wouldn't be difficult
    /// to do so.
    /// </summary>
    [BestFriend]
    internal sealed class SimpleRow : WrappingRow
    {
        private readonly Delegate[] _getters;
        private readonly Action _disposer;

        public override DataViewSchema Schema { get; }

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="schema">The schema for the row.</param>
        /// <param name="input">The row that is being wrapped by this row, where our <see cref="DataViewRow.Position"/>,
        /// <see cref="DataViewRow.Batch"/>, <see cref="DataViewRow.GetIdGetter"/>.</param>
        /// <param name="getters">The collection of getter delegates, whose types should map those in a schema.
        /// If one of these is <see langword="null"/>, the corresponding column is considered inactive.</param>
        /// <param name="disposer">A method that, if non-null, will be called exactly once during
        /// <see cref="IDisposable.Dispose"/>, prior to disposing <paramref name="input"/>.</param>
        public SimpleRow(DataViewSchema schema, DataViewRow input, Delegate[] getters, Action disposer = null)
            : base(input)
        {
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.CheckValue(input, nameof(input));
            Contracts.Check(Utils.Size(getters) == schema.Count);
            Contracts.CheckValueOrNull(disposer);
            Schema = schema;
            _getters = getters ?? new Delegate[0];
            _disposer = disposer;
        }

        protected override void DisposeCore(bool disposing)
        {
            if (disposing)
                _disposer?.Invoke();
        }

        public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
        {
            Contracts.CheckParam(column.Index < _getters.Length, nameof(column), "Invalid col value in GetGetter");
            Contracts.Check(IsColumnActive(column));
            if (_getters[column.Index] is ValueGetter<TValue> fn)
                return fn;
            throw Contracts.Except("Unexpected TValue in GetGetter");
        }

        /// <summary>
        /// Returns whether the given column is active in this row.
        /// </summary>
        public override bool IsColumnActive(DataViewSchema.Column column)
        {
            Contracts.Check(column.Index < _getters.Length);
            return _getters[column.Index] != null;
        }
    }
}