// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Internal.Utilities;

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

        public override ValueGetter<T> GetGetter<T>(int col)
        {
            Contracts.CheckParam(0 <= col && col < _getters.Length, nameof(col), "Invalid col value in GetGetter");
            Contracts.Check(IsColumnActive(col));
            if (_getters[col] is ValueGetter<T> fn)
                return fn;
            throw Contracts.Except("Unexpected TValue in GetGetter");
        }

        public override bool IsColumnActive(int col)
        {
            Contracts.Check(0 <= col && col < _getters.Length);
            return _getters[col] != null;
        }
    }

    public static class SimpleSchemaUtils
    {
        public static DataViewSchema Create(IExceptionContext ectx, params KeyValuePair<string, DataViewType>[] columns)
        {
            Contracts.CheckValueOrNull(ectx);
            ectx.CheckValue(columns, nameof(columns));

            var builder = new SchemaBuilder();
            builder.AddColumns(columns.Select(kvp => new DataViewSchema.DetachedColumn(kvp.Key, kvp.Value)));
            return builder.GetSchema();
        }
    }
}