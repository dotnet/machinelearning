// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// This is a data view that is a 'zip' of several data views.
    /// The length of the zipped data view is equal to the shortest of the lengths of the components.
    /// </summary>
    public sealed class ZipDataView : IDataView
    {
        // REVIEW: there are other potential 'zip modes' that can be implemented:
        // * 'zip longest', iterate until all sources finish, and return the 'sensible missing values' for sources that ended
        // too early.
        // * 'zip longest with loop', iterate until the longest source finishes, and for those that finish earlier, restart from
        // the beginning.

        public const string RegistrationName = "ZipDataView";

        private readonly IHost _host;
        private readonly IDataView[] _sources;
        private readonly CompositeSchema _compositeSchema;

        public static IDataView Create(IHostEnvironment env, IEnumerable<IDataView> sources)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(sources, nameof(sources));

            var srcArray = sources.ToArray();
            host.CheckNonEmpty(srcArray, nameof(sources));
            if (srcArray.Length == 1)
                return srcArray[0];
            return new ZipDataView(host, srcArray);
        }

        private ZipDataView(IHost host, IDataView[] sources)
        {
            Contracts.AssertValue(host);
            _host = host;

            _host.Assert(Utils.Size(sources) > 1);
            _sources = sources;
            _compositeSchema = new CompositeSchema(_sources.Select(x => x.Schema).ToArray());
        }

        public bool CanShuffle { get { return false; } }

        public Schema Schema => _compositeSchema.AsSchema;

        public long? GetRowCount(bool lazy = true)
        {
            long min = -1;
            foreach (var source in _sources)
            {
                var cur = source.GetRowCount(lazy);
                if (cur == null)
                    return null;
                _host.Check(cur.Value >= 0, "One of the sources returned a negative row count");
                if (min < 0 || min > cur.Value)
                    min = cur.Value;
            }

            return min;
        }

        public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
        {
            _host.CheckValue(predicate, nameof(predicate));
            _host.CheckValueOrNull(rand);

            var srcPredicates = _compositeSchema.GetInputPredicates(predicate);

            // REVIEW: if we know the row counts, we could only open cursor if it has needed columns, and have the
            // outer cursor handle the early stopping. If we don't know row counts, we need to open all the cursors because
            // we don't know which one will be the shortest.
            // One reason this is not done currently is because the API has 'somewhat mutable' data views, so potentially this
            // optimization might backfire.
            var srcCursors = _sources
                .Select((dv, i) => srcPredicates[i] == null ? GetMinimumCursor(dv) : dv.GetRowCursor(srcPredicates[i], null)).ToArray();
            return new Cursor(this, srcCursors, predicate);
        }

        /// <summary>
        /// Create an <see cref="IRowCursor"/> with no requested columns on a data view.
        /// Potentially, this can be optimized by calling GetRowCount(lazy:true) first, and if the count is not known,
        /// wrapping around GetCursor().
        /// </summary>
        private IRowCursor GetMinimumCursor(IDataView dv)
        {
            _host.AssertValue(dv);
            return dv.GetRowCursor(x => false);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            consolidator = null;
            return new IRowCursor[] { GetRowCursor(predicate, rand) };
        }

        private sealed class Cursor : RootCursorBase, IRowCursor
        {
            private readonly IRowCursor[] _cursors;
            private readonly CompositeSchema _compositeSchema;
            private readonly bool[] _isColumnActive;

            public override long Batch { get { return 0; } }

            public Cursor(ZipDataView parent, IRowCursor[] srcCursors, Func<int, bool> predicate)
                : base(parent._host)
            {
                Ch.AssertNonEmpty(srcCursors);
                Ch.AssertValue(predicate);

                _cursors = srcCursors;
                _compositeSchema = parent._compositeSchema;
                _isColumnActive = Utils.BuildArray(_compositeSchema.ColumnCount, predicate);
            }

            public override void Dispose()
            {
                for (int i = _cursors.Length - 1; i >= 0; i--)
                    _cursors[i].Dispose();
                base.Dispose();
            }

            public override ValueGetter<UInt128> GetIdGetter()
            {
                return
                    (ref UInt128 val) =>
                    {
                        Ch.Check(IsGood, "Cannot call ID getter in current state");
                        val = new UInt128((ulong)Position, 0);
                    };
            }

            protected override bool MoveNextCore()
            {
                Ch.Assert(State != CursorState.Done);
                foreach (var cursor in _cursors)
                {
                    Ch.Assert(cursor.State != CursorState.Done);
                    if (!cursor.MoveNext())
                        return false;
                }

                return true;
            }

            protected override bool MoveManyCore(long count)
            {
                Ch.Assert(State != CursorState.Done);
                foreach (var cursor in _cursors)
                {
                    Ch.Assert(cursor.State != CursorState.Done);
                    if (!cursor.MoveMany(count))
                        return false;
                }

                return true;
            }

            public Schema Schema => _compositeSchema.AsSchema;

            public bool IsColumnActive(int col)
            {
                _compositeSchema.CheckColumnInRange(col);
                return _isColumnActive[col];
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                int dv;
                int srcCol;
                _compositeSchema.GetColumnSource(col, out dv, out srcCol);
                return _cursors[dv].GetGetter<TValue>(srcCol);
            }
        }
    }
}
