// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// This implements a data view that has a schema, but no rows.
    /// </summary>
    [BestFriend]
    internal sealed class EmptyDataView : IDataView
    {
        private readonly IHost _host;

        public bool CanShuffle => true;
        public DataViewSchema Schema { get; }

        public EmptyDataView(IHostEnvironment env, DataViewSchema schema)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(EmptyDataView));
            _host.CheckValue(schema, nameof(schema));
            Schema = schema;
        }

        public long? GetRowCount() => 0;

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            _host.CheckValueOrNull(rand);
            return new Cursor(_host, Schema, columnsNeeded);
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            _host.CheckValueOrNull(rand);
            return new[] { new Cursor(_host, Schema, columnsNeeded) };
        }

        private sealed class Cursor : RootCursorBase
        {
            private readonly bool[] _active;

            public override DataViewSchema Schema { get; }
            public override long Batch => 0;

            public Cursor(IChannelProvider provider, DataViewSchema schema, IEnumerable<DataViewSchema.Column> columnsNeeded)
                : base(provider)
            {
                Ch.AssertValue(schema);
                Schema = schema;
                _active = Utils.BuildArray(Schema.Count, columnsNeeded);
            }

            public override ValueGetter<DataViewRowId> GetIdGetter()
            {
                return (ref DataViewRowId val) => throw Ch.Except(RowCursorUtils.FetchValueStateError);
            }

            protected override bool MoveNextCore() => false;

            /// <summary>
            /// Returns whether the given column is active in this row.
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column) => column.Index < _active.Length && _active[column.Index];

            /// <summary>
            /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
            /// This throws if the column is not active in this row, or if the type
            /// <typeparamref name="TValue"/> differs from this column's type.
            /// </summary>
            /// <typeparam name="TValue"> is the column's content type.</typeparam>
            /// <param name="column"> is the output column whose getter should be returned.</param>
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                Ch.CheckParam(IsColumnActive(column), nameof(column), "requested column not active");
                return (ref TValue value) => throw Ch.Except(RowCursorUtils.FetchValueStateError);
            }
        }
    }
}