// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// This is a data view that is a 'zip' of several data views.
    /// The length of the zipped data view is equal to the shortest of the lengths of the components.
    /// </summary>
    [BestFriend]
    internal sealed class ZipDataView : IDataView
    {
        // REVIEW: there are other potential 'zip modes' that can be implemented:
        // * 'zip longest', iterate until all sources finish, and return the 'sensible missing values' for sources that ended
        // too early.
        // * 'zip longest with loop', iterate until the longest source finishes, and for those that finish earlier, restart from
        // the beginning.

        public const string RegistrationName = "ZipDataView";

        private readonly IHost _host;
        private readonly IDataView[] _sources;
        private readonly ZipBinding _zipBinding;

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
            _zipBinding = new ZipBinding(_sources.Select(x => x.Schema).ToArray());
        }

        public bool CanShuffle { get { return false; } }

        public DataViewSchema Schema => _zipBinding.OutputSchema;

        public long? GetRowCount()
        {
            long min = -1;
            foreach (var source in _sources)
            {
                var cur = source.GetRowCount();
                if (cur == null)
                    return null;
                _host.Check(cur.Value >= 0, "One of the sources returned a negative row count");
                if (min < 0 || min > cur.Value)
                    min = cur.Value;
            }

            return min;
        }

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, Schema);
            _host.CheckValueOrNull(rand);

            var srcPredicates = _zipBinding.GetInputPredicates(predicate);

            // REVIEW: if we know the row counts, we could only open cursor if it has needed columns, and have the
            // outer cursor handle the early stopping. If we don't know row counts, we need to open all the cursors because
            // we don't know which one will be the shortest.
            // One reason this is not done currently is because the API has 'somewhat mutable' data views, so potentially this
            // optimization might backfire.
            var srcCursors = _sources
                .Select((dv, i) => srcPredicates[i] == null ? GetMinimumCursor(dv) : dv.GetRowCursor(dv.Schema.Where(x => srcPredicates[i](x.Index)), null)).ToArray();
            return new Cursor(this, srcCursors, predicate);
        }

        /// <summary>
        /// Create an <see cref="DataViewRowCursor"/> with no requested columns on a data view.
        /// Potentially, this can be optimized by calling GetRowCount(lazy:true) first, and if the count is not known,
        /// wrapping around GetCursor().
        /// </summary>
        private DataViewRowCursor GetMinimumCursor(IDataView dv)
        {
            _host.AssertValue(dv);
            return dv.GetRowCursor();
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            return new DataViewRowCursor[] { GetRowCursor(columnsNeeded, rand) };
        }

        private sealed class Cursor : RootCursorBase
        {
            private readonly DataViewRowCursor[] _cursors;
            private readonly ZipBinding _zipBinding;
            private readonly bool[] _isColumnActive;
            private bool _disposed;

            public override long Batch { get { return 0; } }

            public Cursor(ZipDataView parent, DataViewRowCursor[] srcCursors, Func<int, bool> predicate)
                : base(parent._host)
            {
                Ch.AssertNonEmpty(srcCursors);
                Ch.AssertValue(predicate);

                _cursors = srcCursors;
                _zipBinding = parent._zipBinding;
                _isColumnActive = Utils.BuildArray(_zipBinding.ColumnCount, predicate);
            }

            protected override void Dispose(bool disposing)
            {
                if (_disposed)
                    return;
                if (disposing)
                {
                    for (int i = _cursors.Length - 1; i >= 0; i--)
                        _cursors[i].Dispose();
                }
                _disposed = true;
                base.Dispose(disposing);
            }

            public override ValueGetter<DataViewRowId> GetIdGetter()
            {
                return
                    (ref DataViewRowId val) =>
                    {
                        Ch.Check(IsGood, RowCursorUtils.FetchValueStateError);
                        val = new DataViewRowId((ulong)Position, 0);
                    };
            }

            protected override bool MoveNextCore()
            {
                foreach (var cursor in _cursors)
                {
                    if (!cursor.MoveNext())
                        return false;
                }

                return true;
            }

            public override DataViewSchema Schema => _zipBinding.OutputSchema;

            /// <summary>
            /// Returns whether the given column is active in this row.
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                _zipBinding.CheckColumnInRange(column.Index);
                return _isColumnActive[column.Index];
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
                int dv;
                int srcCol;
                _zipBinding.GetColumnSource(column.Index, out dv, out srcCol);
                var rowCursor = _cursors[dv];
                return rowCursor.GetGetter<TValue>(rowCursor.Schema[srcCol]);
            }
        }
    }
}
