// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// This implements a data view that has a schema, but no rows.
    /// </summary>
    public sealed class EmptyDataView : IDataView
    {
        private readonly IHost _host;

        public bool CanShuffle => true;
        public ISchema Schema { get; }

        public EmptyDataView(IHostEnvironment env, ISchema schema)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(EmptyDataView));
            _host.CheckValue(schema, nameof(schema));
            Schema = schema;
        }

        public long? GetRowCount(bool lazy = true) => 0;

        public IRowCursor GetRowCursor(Func<int, bool> needCol, IRandom rand = null)
        {
            _host.CheckValue(needCol, nameof(needCol));
            _host.CheckValueOrNull(rand);
            return new Cursor(_host, Schema, needCol);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> needCol, int n, IRandom rand = null)
        {
            _host.CheckValue(needCol, nameof(needCol));
            _host.CheckValueOrNull(rand);
            consolidator = null;
            return new[] { new Cursor(_host, Schema, needCol) };
        }

        private sealed class Cursor : RootCursorBase, IRowCursor
        {
            private readonly bool[] _active;

            public ISchema Schema { get; }
            public override long Batch => 0;

            public Cursor(IChannelProvider provider, ISchema schema, Func<int, bool> needCol)
                : base(provider)
            {
                Ch.AssertValue(schema);
                Ch.AssertValue(needCol);
                Schema = schema;
                _active = Utils.BuildArray(Schema.ColumnCount, needCol);
            }

            public override ValueGetter<UInt128> GetIdGetter()
            {
                return
                    (ref UInt128 val) =>
                    {
                        Ch.Assert(!IsGood);
                        throw Ch.Except("Cannot call ID getter in current state");
                    };
            }

            protected override bool MoveNextCore() => false;

            public bool IsColumnActive(int col) => 0 <= col && col < _active.Length && _active[col];

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.Check(IsColumnActive(col), "Can't get getter for inactive column");
                return
                    (ref TValue value) =>
                    {
                        Ch.Assert(State != CursorState.Good);
                        throw Ch.Except("Cannot use getter with cursor in this state");
                    };
            }
        }
    }
}