// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Threading;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// This is a dataview that wraps another dataview, and does on-demand caching of the
    /// input columns. When constructed, it caches no data. Whenever a cursor is constructed
    /// that requests a column that has not yet been cached, any requested uncached columns
    /// become cached through a background thread worker. A user can provide a hint to the
    /// constructor to indicate that some columns should be pre-cached. A cursor over this
    /// dataview will block when moved to a row until such time as all requested columns
    /// have that row in cache.
    /// </summary>
    public sealed class CacheDataView : IDataView, IRowSeekable
    {
        private readonly IHost _host;
        private readonly IDataView _subsetInput;
        private long _rowCount;
        private readonly int[] _inputToSubsetColIndex;

        // Useful constants to tie together the block batching behavior for the parallel cursors.
        private const int _batchShift = 6;
        private const int _batchSize = 1 << _batchShift;
        private const int _batchMask = (1 << _batchShift) - 1;

        // REVIEW: The first version of this code was actually Task based, but this
        // was problematic. Unfortunately the only way I would see to make this work was to
        // make the process thread based again.

        /// <summary>
        /// Cursors can be opened from multiple threads simultaneously, so this is used to
        /// synchronize both at the level of ensuring that only one cache is created per
        /// column.
        /// </summary>
        private readonly object _cacheLock;

        /// <summary>
        /// Filler threads. Currently nothing is done with them. Possibly it may be good
        /// practice to join against them somehow, but IDataViews as this stage are not
        /// disposed, so it's unclear what would actually have the job of joining against
        /// them.
        /// </summary>
        private readonly ConcurrentBag<Thread> _cacheFillerThreads;

        /// <summary>
        /// One cache per column. If this column is not being cached or has been cached,
        /// this column will be null.
        /// </summary>
        private readonly ColumnCache[] _caches;

        /// <summary>
        /// A waiter used for cursors where no columns are actually requested but it's still
        /// necesssary to wait to determine the number of rows.
        /// </summary>
        private volatile OrderedWaiter _cacheDefaultWaiter;

        /// <summary>
        /// Constructs an on demand cache for the input.
        /// </summary>
        /// <param name="env">The host environment</param>
        /// <param name="input">The input dataview to cache. Note that if we do not know
        /// how to cache some columns, these columns will not appear in this dataview.</param>
        /// <param name="prefetch">A list of column indices the cacher should frontload,
        /// prior to any cursors being requested. This can be null to indicate no
        /// prefetching.</param>
        public CacheDataView(IHostEnvironment env, IDataView input, int[] prefetch)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register("Cache");

            _host.CheckValue(input, nameof(input));
            _host.CheckValueOrNull(prefetch);

            // REVIEW: The slightly more complicated alternative to this is that we
            // filter no columns at all, and if a cursor is created over something we don't
            // know how to cache, then a parallel cursor is created. However this has some
            // somewhat strange side effects (such a thing would not be shufflable).
            _subsetInput = SelectCachableColumns(input, _host, ref prefetch, out _inputToSubsetColIndex);
            _rowCount = _subsetInput.GetRowCount() ?? -1;
            // REVIEW: We could actually handle this by using something other than
            // a single array in the cache, but I don't view this as terribly urgent.
            if (_rowCount > Utils.ArrayMaxSize)
                throw _host.Except("The input data view has too many ({0}) rows. CacheDataView can only cache up to {1} rows", _rowCount, Utils.ArrayMaxSize);

            _cacheLock = new object();
            _cacheFillerThreads = new ConcurrentBag<Thread>();
            _caches = new ColumnCache[_subsetInput.Schema.ColumnCount];

            if (Utils.Size(prefetch) > 0)
                KickoffFiller(prefetch);
        }

        /// <summary>
        /// Since shuffling requires serving up items potentially out of order we need to know
        /// how to save and then copy out values that we read. This transform knows how to save
        /// and copy out only primitive and vector valued columns, but nothing else, so any
        /// other columns are dropped.
        /// </summary>
        private static IDataView SelectCachableColumns(IDataView data, IHostEnvironment env, ref int[] prefetch, out int[] inputToSubset)
        {
            List<int> columnsToDrop = null;
            var schema = data.Schema;

            // While dropping columns, match the prefetch column indices to their new indices.
            if (prefetch == null)
                prefetch = new int[0];
            else if (prefetch.Length > 0)
            {
                var tmp = new int[prefetch.Length];
                Array.Copy(prefetch, tmp, prefetch.Length);
                Array.Sort(tmp);
                prefetch = tmp;
                if (prefetch.Length > 0 && (prefetch[0] < 0 || prefetch[prefetch.Length - 1] >= schema.ColumnCount))
                    throw env.Except("Prefetch array had column indices out of range");
            }
            int ip = 0;
            inputToSubset = null;

            for (int c = 0; c < schema.ColumnCount; ++c)
            {
                var type = schema.GetColumnType(c);
                env.Assert(ip == prefetch.Length || c <= prefetch[ip]);
                if (!type.IsCachable())
                {
                    if (inputToSubset == null)
                    {
                        inputToSubset = new int[schema.ColumnCount];
                        for (int cc = 0; cc < c; ++cc)
                            inputToSubset[cc] = cc;
                    }
                    inputToSubset[c] = -1;
                    Utils.Add(ref columnsToDrop, c);
                    // Make sure we weren't asked to prefetch any dropped column.
                    if (ip < prefetch.Length && prefetch[ip] == c)
                    {
                        throw env.Except(
                            "Asked to prefetch column '{0}' into cache, but it is of unhandled type '{1}'",
                            schema.GetColumnName(c), type);
                    }
                }
                else
                {
                    while (ip < prefetch.Length && prefetch[ip] == c)
                        prefetch[ip++] -= Utils.Size(columnsToDrop);
                    if (inputToSubset != null)
                        inputToSubset[c] = c - Utils.Size(columnsToDrop);
                }
                env.Assert(ip == prefetch.Length || c < prefetch[ip]);
            }
            env.Assert(ip == prefetch.Length);
            if (Utils.Size(columnsToDrop) == 0)
                return data;

            // REVIEW: This can potentially cause hidden columns to become unhidden. See task 3739.
            var args = new ChooseColumnsByIndexTransform.Arguments();
            args.Drop = true;
            args.Index = columnsToDrop.ToArray();
            return new ChooseColumnsByIndexTransform(env, args, data);
        }

        /// <summary>
        /// While in typical cases the cache data view will know how to cache all columns,
        /// the cache data view may not know how to cache certain custom types. User code
        /// may require a mapping from input data view to cache data view column index space.
        /// </summary>
        /// <param name="inputIndex">An input data column index</param>
        /// <returns>The column index of the corresponding column in the cache data view
        /// if this was cachable, or else -1 if the column was not cachable</returns>
        public int MapInputToCacheColumnIndex(int inputIndex)
        {
            int inputIndexLim = _inputToSubsetColIndex == null ? _subsetInput.Schema.ColumnCount : _inputToSubsetColIndex.Length;
            _host.CheckParam(0 <= inputIndex && inputIndex < inputIndexLim, nameof(inputIndex), "Input column index not in range");
            var result = _inputToSubsetColIndex == null ? inputIndex : _inputToSubsetColIndex[inputIndex];
            _host.Assert(-1 <= result && result < _subsetInput.Schema.ColumnCount);
            return result;
        }

        public bool CanShuffle { get { return true; } }

        public ISchema Schema { get { return _subsetInput.Schema; } }

        public long? GetRowCount(bool lazy = true)
        {
            if (_rowCount < 0)
            {
                if (lazy)
                    return null;
                if (_cacheDefaultWaiter == null)
                    KickoffFiller(new int[0]);
                _host.Assert(_cacheDefaultWaiter != null);
                _cacheDefaultWaiter.Wait(long.MaxValue);
                _host.Assert(_rowCount >= 0);
            }
            return _rowCount;
        }

        public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
        {
            _host.CheckValue(predicate, nameof(predicate));
            _host.CheckValueOrNull(rand);
            // We have this explicit enumeration over the generic types to force different assembly
            // code to be generated for the different types, of both waiters and especially indexers.
            // Note also that these must be value types (hence the adorably clever struct wrappers),
            // because the different assembly code will not be generated if they are reference types.
            var waiter = WaiterWaiter.Create(this, predicate);
            if (waiter.IsTrivial)
                return GetRowCursorWaiterCore(TrivialWaiter.Create(this), predicate, rand);
            return GetRowCursorWaiterCore(waiter, predicate, rand);
        }

        /// <summary>
        /// Returns a permutation or null. This function will return null if either <paramref name="rand"/>
        /// is null, or if the row count of this cache exceeds the maximum array size.
        /// </summary>
        private int[] GetPermutationOrNull(IRandom rand)
        {
            if (rand == null)
                return null;
            if (_rowCount < 0)
                _cacheDefaultWaiter.Wait(long.MaxValue);
            long rc = _rowCount;
            _host.Assert(rc >= 0);
            // REVIEW: Ideally, in this situation we would report that we could not shuffle.
            if (rc > Utils.ArrayMaxSize)
                return null;
            return Utils.GetRandomPermutation(rand, (int)_rowCount);
        }

        private IRowCursor GetRowCursorWaiterCore<TWaiter>(TWaiter waiter, Func<int, bool> predicate, IRandom rand)
            where TWaiter : struct, IWaiter
        {
            _host.AssertValue(predicate);
            _host.AssertValueOrNull(rand);

            int[] perm = GetPermutationOrNull(rand);
            if (perm == null)
                return CreateCursor(predicate, SequenceIndex<TWaiter>.Create(waiter));
            return CreateCursor(predicate, RandomIndex<TWaiter>.Create(waiter, perm));
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator,
            Func<int, bool> predicate, int n, IRandom rand = null)
        {
            _host.CheckValue(predicate, nameof(predicate));
            _host.CheckValueOrNull(rand);

            n = DataViewUtils.GetThreadCount(_host, n);

            if (n <= 1)
            {
                consolidator = null;
                return new IRowCursor[] { GetRowCursor(predicate, rand) };
            }

            consolidator = new Consolidator();
            var waiter = WaiterWaiter.Create(this, predicate);
            if (waiter.IsTrivial)
                return GetRowCursorSetWaiterCore(TrivialWaiter.Create(this), predicate, n, rand);
            return GetRowCursorSetWaiterCore(waiter, predicate, n, rand);
        }

        /// <summary>
        /// Minimal consolidator.
        /// </summary>
        private sealed class Consolidator : IRowCursorConsolidator
        {
            public IRowCursor CreateCursor(IChannelProvider provider, IRowCursor[] inputs)
            {
                return DataViewUtils.ConsolidateGeneric(provider, inputs, _batchSize);
            }
        }

        private IRowCursor[] GetRowCursorSetWaiterCore<TWaiter>(TWaiter waiter, Func<int, bool> predicate, int n, IRandom rand)
            where TWaiter : struct, IWaiter
        {
            _host.AssertValue(predicate);
            _host.Assert(n > 1);
            _host.AssertValueOrNull(rand);

            var scheduler = new JobScheduler(n);
            IRowCursor[] cursors = new IRowCursor[n];
            int[] perm = GetPermutationOrNull(rand);
            for (int i = 0; i < n; ++i)
            {
                // While the counter and waiter is shared among the cursors, the indexer is not.
                if (perm == null)
                    cursors[i] = CreateCursor(predicate, BlockSequenceIndex<TWaiter>.Create(waiter, scheduler));
                else
                    cursors[i] = CreateCursor(predicate, BlockRandomIndex<TWaiter>.Create(waiter, scheduler, perm));
            }
            return cursors;
        }

        private IRowCursor CreateCursor<TIndex>(Func<int, bool> predicate, TIndex index)
            where TIndex : struct, IIndex
        {
            Contracts.AssertValue(predicate);
            return new RowCursor<TIndex>(this, predicate, index);
        }

        public IRowSeeker GetSeeker(Func<int, bool> predicate)
        {
            _host.CheckValue(predicate, nameof(predicate));
            // The seeker needs to know the row count when it validates the row index to move to.
            // Calling GetRowCount here to force a wait indirectly so that _rowCount will have a valid value.
            GetRowCount(false);
            _host.Assert(_rowCount >= 0);
            var waiter = WaiterWaiter.Create(this, predicate);
            if (waiter.IsTrivial)
                return GetSeeker(predicate, TrivialWaiter.Create(this));
            return GetSeeker(predicate, waiter);
        }

        private IRowSeeker GetSeeker<TWaiter>(Func<int, bool> predicate, TWaiter waiter)
            where TWaiter : struct, IWaiter
        {
            _host.AssertValue(predicate);
            return new RowSeeker<TWaiter>(this, predicate, waiter);
        }

        /// <summary>
        /// This is a helper method that, given a list of columns, determines the subset
        /// that are uncached, and if there are any uncached kicks off a filler worker to
        /// fill them in, over a row cursor over this subset of columns.
        /// </summary>
        /// <param name="columns">The requested set of columns</param>
        /// <seealso cref="Filler"/>
        private void KickoffFiller(int[] columns)
        {
            _host.AssertValue(columns);

            HashSet<int> taskColumns = null;
            IRowCursor cursor;
            ColumnCache[] caches;
            OrderedWaiter waiter;
            lock (_cacheLock)
            {
                for (int ic = 0; ic < columns.Length; ++ic)
                {
                    int c = columns[ic];
                    if (_caches[c] == null)
                        Utils.Add(ref taskColumns, c);
                }
                // If we will already at some point know the number of rows,
                // and there are no columns, kick off no new task.
                if (Utils.Size(taskColumns) == 0 && _cacheDefaultWaiter != null)
                    return;
                if (taskColumns == null)
                    cursor = _subsetInput.GetRowCursor(c => false);
                else
                    cursor = _subsetInput.GetRowCursor(taskColumns.Contains);
                waiter = new OrderedWaiter(firstCleared: false);
                _cacheDefaultWaiter = waiter;
                caches = new ColumnCache[Utils.Size(taskColumns)];
                if (caches.Length > 0)
                {
                    int ic = 0;
                    foreach (var c in taskColumns)
                        caches[ic++] = _caches[c] = ColumnCache.Create(this, cursor, c, waiter);
                }
            }

            // REVIEW: Exceptions of the internal cursor, or this cursor, will occur within
            // the thread which will be treated as unhandled exceptions, terminating the process.
            // They will not be caught by the big catch in the main thread, as filler is not running
            // in the main thread. Some sort of scheme by which these exceptions could be
            // cleanly handled would be more appropriate. See task 3740.
            var fillerThread = Utils.CreateBackgroundThread(() => Filler(cursor, caches, waiter));
            _cacheFillerThreads.Add(fillerThread);
            fillerThread.Start();
        }

        /// <summary>
        /// The actual body of the filler worker. The filler worker works fairly simply:
        /// for each row, it tells each <see cref="ColumnCache"/> instance in
        /// <paramref name="caches"/> to fill in the value at the current position,
        /// then increments the <paramref name="waiter"/>, then moves to the next row.
        /// When it's done, it tells <see cref="ColumnCache"/> to "freeze" itself, since
        /// there should be no more rows.
        /// <param name="cursor">The cursor over the rows to cache</param>
        /// <param name="caches">The caches we must fill and, at the end of the cursor, freeze</param>
        /// <param name="waiter">The waiter to increment as we cache each additional row</param>
        /// </summary>
        private void Filler(IRowCursor cursor, ColumnCache[] caches, OrderedWaiter waiter)
        {
            _host.AssertValue(cursor);
            _host.AssertValue(caches);
            _host.AssertValue(waiter);

            const string inconsistentError = "Inconsistent number of rows from input data view detected. This indicates a bug within the implementation of the input data view.";

            try
            {
                using (cursor)
                using (var ch = _host.Start("Cache Filler"))
                {
                    ch.Trace("Begin cache of {0} columns", caches.Length);
                    long rowCount = 0;
                    if (caches.Length > 0)
                    {
                        while (cursor.MoveNext())
                        {
                            rowCount++;
                            if (rowCount > Utils.ArrayMaxSize)
                                throw _host.Except("The input data view has too many ({0}) rows. CacheDataView can only cache up to {1} rows", rowCount, Utils.ArrayMaxSize);
                            for (int ic = 0; ic < caches.Length; ++ic)
                                caches[ic].CacheCurrent();
                            // REVIEW: Should this be a check, since we cannot control the implementation of the input?
                            _host.Assert(_rowCount == -1 || rowCount <= _rowCount, inconsistentError);
                            waiter.Increment();
                        }
                    }
                    else
                    {
                        // Slightly simplify the no-column special case.
                        while (cursor.MoveNext())
                        {
                            rowCount++;
                            if (_rowCount > Utils.ArrayMaxSize)
                                throw _host.Except("The input data view has too many ({0}) rows. CacheDataView can only cache up to {1} rows", _rowCount, Utils.ArrayMaxSize);
                            _host.Assert(_rowCount == -1 || rowCount <= _rowCount, inconsistentError);
                            waiter.Increment();
                        }
                    }
                    long rc = Interlocked.CompareExchange(ref _rowCount, rowCount, -1);
                    for (int ic = 0; ic < caches.Length; ++ic)
                        caches[ic].Freeze();
                    _host.Assert(rc == -1 || rc == rowCount, inconsistentError);
                    if (rc == -1)
                        ch.Trace("Number of rows determined as {0}", rowCount);
                    waiter.IncrementAll();
                    ch.Trace("End cache of {0} columns", caches.Length);
                }
            }
            catch (Exception ex)
            {
                waiter.SignalException(ex);
            }
        }

        // REVIEW: Consider making CacheDataView implement IDisposable.
        // Shut down the filler threads on diposal.
        /// <summary>
        /// Joins all the cache filler threads. This method is currently supposed to be called only by tests.
        /// </summary>
        internal void Wait()
        {
            if (_cacheFillerThreads != null)
            {
                foreach (var thread in _cacheFillerThreads)
                {
                    if (thread.IsAlive)
                        thread.Join();
                }
            }
        }

        private sealed class RowCursor<TIndex> : RowCursorSeekerBase, IRowCursor
            where TIndex : struct, IIndex
        {
            private CursorState _state;
            private readonly TIndex _index;

            public CursorState State { get { return _state; } }

            public long Batch { get { return _index.Batch; } }

            public RowCursor(CacheDataView parent, Func<int, bool> predicate, TIndex index)
                : base(parent, predicate)
            {
                _state = CursorState.NotStarted;
                _index = index;
            }

            public ValueGetter<UInt128> GetIdGetter()
            {
                return _index.GetIdGetter();
            }

            public ICursor GetRootCursor()
            {
                return this;
            }

            public bool MoveNext()
            {
                if (_state == CursorState.Done)
                {
                    Ch.Assert(Position == -1);
                    return false;
                }

                Ch.Assert(_state == CursorState.NotStarted || _state == CursorState.Good);
                if (_index.MoveNext())
                {
                    Position++;
                    Ch.Assert(Position >= 0);
                    _state = CursorState.Good;
                    return true;
                }

                Dispose();
                Ch.Assert(Position == -1);
                return false;
            }

            public bool MoveMany(long count)
            {
                // Note: If we decide to allow count == 0, then we need to special case
                // that MoveNext() has never been called. It's not entirely clear what the return
                // result would be in that case.
                Ch.CheckParam(count > 0, nameof(count));

                if (_state == CursorState.Done)
                {
                    Ch.Assert(Position == -1);
                    return false;
                }

                Ch.Assert(_state == CursorState.NotStarted || _state == CursorState.Good);
                if (_index.MoveMany(count))
                {
                    Position += count;
                    _state = CursorState.Good;
                    Ch.Assert(Position >= 0);
                    return true;
                }

                Dispose();
                Ch.Assert(Position == -1);
                return false;
            }

            protected override void DisposeCore()
            {
                _state = CursorState.Done;
            }

            protected override ValueGetter<TValue> CreateGetterDelegateCore<TValue>(ColumnCache<TValue> cache)
            {
                return
                    (ref TValue value) =>
                    {
                        Ch.Check(_state == CursorState.Good, "Cannot use getter with cursor in this state");
                        cache.Fetch((int)_index.GetIndex(), ref value);
                    };
            }
        }

        private sealed class RowSeeker<TWaiter> : RowCursorSeekerBase, IRowSeeker
            where TWaiter : struct, IWaiter
        {
            private readonly TWaiter _waiter;

            public long Batch { get { return 0; } }

            public ValueGetter<UInt128> GetIdGetter()
            {
                return
                    (ref UInt128 val) =>
                    {
                        Ch.Check(Position >= 0, "Cannot call ID getter in current state");
                        val = new UInt128((ulong)Position, 0);
                    };
            }

            public RowSeeker(CacheDataView parent, Func<int, bool> predicate, TWaiter waiter)
                : base(parent, predicate)
            {
                _waiter = waiter;
            }

            public bool MoveTo(long rowIndex)
            {
                if (rowIndex < 0 || !_waiter.Wait(rowIndex))
                {
                    // If requested row index is out of range, the row seeker
                    // returns false and sets its position to -1.
                    Position = -1;
                    return false;
                }

                Position = rowIndex;
                return true;
            }

            protected override void DisposeCore()
            {
            }

            protected override ValueGetter<TValue> CreateGetterDelegateCore<TValue>(ColumnCache<TValue> cache)
            {
                return (ref TValue value) => cache.Fetch((int)Position, ref value);
            }
        }

        private interface IWaiter
        {
            /// <summary>
            /// Blocks until that position is either available, or it has been
            /// determined no such position exists. Implicit in a true return value
            /// for this is that any values of <paramref name="pos"/> less than are
            /// also true, that is, if one waits on <c>i</c>, when this returns it
            /// is equivalent to also having waited on <c>i-1</c>, <c>i-2</c>, etc.
            /// Note that this is position within the cache, that is, a row index,
            /// as opposed to position within the cursor.
            ///
            /// This method should be thread safe because in the parallel cursor
            /// case it will be used by multiple threads.
            /// </summary>
            /// <param name="pos">The position to wait for, must be positive</param>
            /// <returns>True if the position can be accessed, false if not</returns>
            bool Wait(long pos);
        }

        /// <summary>
        /// A waiter for use in situations where there is no real waiting, per se, just a row limit.
        /// This should be instantiated only if the analogous <see cref="WaiterWaiter.IsTrivial"/>
        /// returned true.
        /// </summary>
        private sealed class TrivialWaiter : IWaiter
        {
            private readonly long _lim;

            private TrivialWaiter(CacheDataView parent)
            {
                Contracts.Assert(parent._rowCount >= 0);
                _lim = parent._rowCount;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public bool Wait(long pos)
            {
                Contracts.Assert(pos >= 0);
                return pos < _lim;
            }

            public static Wrapper Create(CacheDataView parent)
            {
                return new Wrapper(new TrivialWaiter(parent));
            }

            public struct Wrapper : IWaiter
            {
                private readonly TrivialWaiter _waiter;

                public Wrapper(TrivialWaiter waiter)
                {
                    Contracts.AssertValue(waiter);
                    _waiter = waiter;
                }

                public bool Wait(long pos) { return _waiter.Wait(pos); }
            }
        }

        /// <summary>
        /// A waiter that determines the necessary waiters for a set of active columns, and waits
        /// on all of them.
        /// </summary>
        private sealed class WaiterWaiter : IWaiter
        {
            private readonly OrderedWaiter[] _waiters;
            private readonly CacheDataView _parent;

            /// <summary>
            /// If this is true, then a <see cref="TrivialWaiter"/> could be used instead.
            /// </summary>
            public bool IsTrivial { get { return _waiters.Length == 0; } }

            private WaiterWaiter(CacheDataView parent, Func<int, bool> pred)
            {
                Contracts.AssertValue(parent);
                Contracts.AssertValue(pred);
                _parent = parent;

                int[] actives = Enumerable.Range(0, _parent.Schema.ColumnCount).Where(pred).ToArray();
                // Kick off the thread to fill in any requested columns.
                _parent.KickoffFiller(actives);

                HashSet<OrderedWaiter> waiters = new HashSet<OrderedWaiter>();
                foreach (int c in actives)
                {
                    Contracts.Assert(pred(c));
                    Contracts.AssertValue(_parent._caches[c]);
                    var waiter = _parent._caches[c].Waiter;
                    if (waiter != null)
                        waiters.Add(waiter);
                }
                // Make the array of waiters.
                if (_parent._rowCount < 0 && waiters.Count == 0)
                {
                    Contracts.AssertValue(_parent._cacheDefaultWaiter);
                    waiters.Add(_parent._cacheDefaultWaiter);
                }
                _waiters = new OrderedWaiter[waiters.Count];
                waiters.CopyTo(_waiters);
            }

            public bool Wait(long pos)
            {
                foreach (var w in _waiters)
                    w.Wait(pos);
                return pos < _parent._rowCount || _parent._rowCount == -1;
            }

            public static Wrapper Create(CacheDataView parent, Func<int, bool> pred)
            {
                return new Wrapper(new WaiterWaiter(parent, pred));
            }

            public struct Wrapper : IWaiter
            {
                private readonly WaiterWaiter _waiter;

                public bool IsTrivial { get { return _waiter.IsTrivial; } }

                public Wrapper(WaiterWaiter waiter)
                {
                    Contracts.AssertValue(waiter);
                    _waiter = waiter;
                }

                public bool Wait(long pos) { return _waiter.Wait(pos); }
            }
        }

        /// <summary>
        /// A collection of different simple objects that track the index into the cache at
        /// particular location. Note that this index is, in the shuffled or parallel case,
        /// very different from the position of the cursor that keeps this indexer.
        /// </summary>
        private interface IIndex
        {
            long Batch { get; }

            /// <summary>
            /// The index. Callers should never call this either before one of the move
            /// methods has returned true, or ever after either has returned false.
            /// </summary>
            long GetIndex();

            /// <summary>
            /// An ID getter, which should be based on the value that would be returned
            /// from <see cref="GetIndex"/>, if valid, and otherwise have undefined behavior.
            /// </summary>
            ValueGetter<UInt128> GetIdGetter();

            /// <summary>
            /// Moves to the next index. Once this or <see cref="MoveMany"/> has returned
            /// false, it should never be called again. (This in constrast to public
            /// <see cref="ICursor"/> objects, whose move methods are robust to that usage.)
            /// </summary>
            /// <returns>Whether the next index is available.</returns>
            bool MoveNext();

            /// <summary>
            /// Moves to the index this many forward. Once this or <see cref="MoveNext"/>
            /// has returned false, it should never be called again.
            /// </summary>
            /// <param name="count">The count.</param>
            /// <returns>Whether the index that many forward is available.</returns>
            bool MoveMany(long count);
        }

        /// <summary>
        /// An <see cref="IIndex"/> where the indices, while valid, are sequential increasing
        /// adjacent indices.
        /// </summary>
        private sealed class SequenceIndex<TWaiter> : IIndex
            where TWaiter : struct, IWaiter
        {
            // -1 means not started, -2 means finished, non-negative is the actual index.
            private long _curr;
            private readonly TWaiter _waiter;

            public long Batch { get { return 0; } }

            private SequenceIndex(TWaiter waiter)
            {
                _curr = -1;
                _waiter = waiter;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public long GetIndex()
            {
                Contracts.Assert(_curr >= 0);
                return _curr;
            }

            public ValueGetter<UInt128> GetIdGetter()
            {
                return
                    (ref UInt128 val) =>
                    {
                        Contracts.Check(_curr >= 0, "Cannot call ID getter in current state");
                        val = new UInt128((ulong)_curr, 0);
                    };
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public bool MoveNext()
            {
                Contracts.Assert(_curr >= -1); // Should not be called when _curr = -2.
                if (_waiter.Wait(++_curr))
                    return true;
                _curr = -2;
                return false;
            }

            public bool MoveMany(long count)
            {
                Contracts.Assert(_curr >= -1); // Should not be called when _curr = -2.
                if (_waiter.Wait(_curr += count))
                    return true;
                _curr = -2;
                return false;
            }

            public static Wrapper Create(TWaiter waiter)
            {
                return new Wrapper(new SequenceIndex<TWaiter>(waiter));
            }

            public struct Wrapper : IIndex
            {
                private readonly SequenceIndex<TWaiter> _index;

                public Wrapper(SequenceIndex<TWaiter> index)
                {
                    Contracts.AssertValue(index);
                    _index = index;
                }

                public long Batch { get { return _index.Batch; } }
                public long GetIndex() { return _index.GetIndex(); }
                public ValueGetter<UInt128> GetIdGetter() { return _index.GetIdGetter(); }
                public bool MoveNext() { return _index.MoveNext(); }
                public bool MoveMany(long count) { return _index.MoveMany(count); }
            }
        }

        private sealed class RandomIndex<TWaiter> : IIndex
            where TWaiter : struct, IWaiter
        {
            private int _curr;
            private readonly int[] _perm;
            private readonly TWaiter _waiter;

            public long Batch { get { return 0; } }

            private RandomIndex(TWaiter waiter, int[] perm)
            {
                Contracts.AssertValue(perm);
                _curr = -1;
                _waiter = waiter;
                _perm = perm;
            }

            public long GetIndex()
            {
                Contracts.Assert(0 <= _curr && _curr < _perm.Length);
                return _perm[_curr];
            }

            public ValueGetter<UInt128> GetIdGetter()
            {
                return
                    (ref UInt128 val) =>
                    {
                        Contracts.Check(_curr >= 0, "Cannot call ID getter in current state");
                        val = new UInt128((ulong)_perm[_curr], 0);
                    };
            }

            public bool MoveNext()
            {
                Contracts.Assert(_curr >= -1); // Should not be called when _curr = -2.
                if (++_curr < _perm.Length)
                {
                    Contracts.Assert(_perm[_curr] >= 0);
                    bool result = _waiter.Wait(_perm[_curr]);
                    // The perm array should have been constructed in a way
                    // that all indices are valid. Assert this.
                    Contracts.Assert(result);
                    return true;
                }
                _curr = -2;
                return false;
            }

            public bool MoveMany(long count)
            {
                Contracts.Assert(_curr >= -1); // Should not be called when _curr = -2.
                // Want _curr + count < _perm.Length, but this can overflow, so we have this
                // strange looking count < _perm.Length - _curr.
                if (count < _perm.Length - _curr)
                {
                    _curr += (int)count;
                    Contracts.Assert(_perm[_curr] >= 0);
                    bool result = _waiter.Wait(_perm[_curr]);
                    // The perm array should have been constructed in a way
                    // that all indices are valid. Assert this.
                    Contracts.Assert(result);
                    return true;
                }
                _curr = -2;
                return false;
            }

            public static Wrapper Create(TWaiter waiter, int[] perm)
            {
                return new Wrapper(new RandomIndex<TWaiter>(waiter, perm));
            }

            public struct Wrapper : IIndex
            {
                private readonly RandomIndex<TWaiter> _index;

                public Wrapper(RandomIndex<TWaiter> index)
                {
                    Contracts.AssertValue(index);
                    _index = index;
                }

                public long Batch { get { return _index.Batch; } }
                public long GetIndex() { return _index.GetIndex(); }
                public ValueGetter<UInt128> GetIdGetter() { return _index.GetIdGetter(); }
                public bool MoveNext() { return _index.MoveNext(); }
                public bool MoveMany(long count) { return _index.MoveMany(count); }
            }
        }

        /// <summary>
        /// A simple job scheduler that assigns available jobs (batches/blocks for processing) to
        /// free workers (cursors/threads). This scheduler takes the ids of the completed jobs into
        /// account when assigning next jobs in order to minimize workers wait time as the consumer
        /// of the completed jobs (a.k.a consolidator, see: DataViewUtils.ConsolidateGeneric) can
        /// only consume jobs in order -according to their ids-. Note that workers get assigned
        /// next job ids before they push the completed jobs to the consumer. So the workers are
        /// then subject to being blocked until their current completed jobs are fully accepted
        /// (i.e. added to the to-consume queue).
        ///
        /// How it works:
        /// Suppose we have 7 workers (w0,..,w6) and 14 jobs (j0,..,j13).
        /// Initially, jobs get assigned to workers using a shared counter.
        /// Here is an example outcome of using a shared counter:
        /// w1->j0, w6->j1, w0->j2, w3->j3, w4->j4, w5->j5, w2->j6.
        ///
        /// Suppose workers finished jobs in the following order:
        /// w5->j5, w0->j2, w6->j1, w4->j4, w3->j3,w1->j0, w2->j6.
        ///
        /// w5 finishes processing j5 first, but will be blocked until the processing of jobs
        /// j0,..,j4 completes since the consumer can consume jobs in order only.
        /// Therefore, the next available job (j7) should not be assigned to w5. It should be
        /// assigned to the worker whose job *get consumed first* (w1 since it processes j0
        /// which is the first job) *not* to the worker who completes its job first (w5 in
        /// this example).
        ///
        /// So, a shared counter can be used to assign jobs to workers initially but should
        /// not be used onwards.
        /// </summary>
        private sealed class JobScheduler
        {
            private readonly int _workersCount;
            // A counter used to assign unique job ids to workers when they are *not* assigned jobs.
            private long _c;

            public JobScheduler(int workersCount)
            {
                _workersCount = workersCount;
                _c = -1;
            }

            public long GetAvailableJob(long completedJob)
            {
                // Starts by assigning unique job ids to workers when they are not assigned jobs.
                if (completedJob == -1)
                    return Interlocked.Increment(ref _c);

                return completedJob + _workersCount;
            }
        }

        /// <summary>
        /// An <see cref="IIndex"/> that shares a counter among multiple threads. The multiple threads divy up
        /// the work by blocks of rows rather than splitting row by row simply, both to cut down on interthread
        /// communication as well as increased locality of thread data access.
        /// </summary>
        private sealed class BlockSequenceIndex<TWaiter> : IIndex
            where TWaiter : struct, IWaiter
        {
            // -1 means not started, -2 means finished, non-negative is the actual index.
            private long _curr;
            private long _batch;
            // Whether we are in a block of _blockSize we've previously reserved.
            private bool _reserved;
            private readonly JobScheduler _scheduler;
            private readonly TWaiter _waiter;

            public long Batch
            {
                get { return _batch; }
            }

            private BlockSequenceIndex(TWaiter waiter, JobScheduler scheduler)
            {
                Contracts.AssertValue(scheduler);
                _curr = -1;
                _batch = -1;
                _reserved = true;
                _waiter = waiter;
                _scheduler = scheduler;
            }

            public long GetIndex()
            {
                Contracts.Assert(_curr >= 0);
                return _curr;
            }

            public ValueGetter<UInt128> GetIdGetter()
            {
                return
                    (ref UInt128 val) =>
                    {
                        Contracts.Check(_curr >= 0, "Cannot call ID getter in current state");
                        val = new UInt128((ulong)_curr, 0);
                    };
            }

            public bool MoveNext()
            {
                Contracts.Assert(_curr >= -1);
                if ((_curr & _batchMask) == _batchMask)
                {
                    // We're at the end of a block. If we actually reached the end of a block, we
                    // should have pre-reserved this block.
                    Contracts.Assert(_reserved);
                    // Get an available block that has not yet been claimed by another thread.
                    _batch = _scheduler.GetAvailableJob(_batch);
                    _curr = _batch << _batchShift;
                    if (_waiter.Wait(_curr))
                    {
                        // See if we can "reserve" the rest of the block, somehow.
                        _reserved = _waiter.Wait(_curr + _batchMask);
                        return true;
                    }
                }
                else if (_reserved) // No need to wait, we've already determined the position is valid.
                {
                    _curr++;
                    Contracts.Assert(_waiter.Wait(_curr));
                    return true;
                }
                else if (_waiter.Wait(++_curr)) // Didn't reserve block, explicitly wait.
                {
                    // If we couldn't have reserved this block, then we shouldn't be at the last
                    // position either because if we had reached this point then we should have been
                    // able to reserve the block.
                    Contracts.Assert((_curr & _batchMask) < _batchMask);
                    return true;
                }
                // All done.
                _curr = -2;
                return false;
            }

            public bool MoveMany(long count)
            {
                // I don't know that moving many on parallel cursors is really a thing,
                // given that the order in which they serve up results among themselves
                // is non-deterministic. For now content ourselves with this trivial
                // implementation.
                Contracts.Assert(count > 0);
                while (--count >= 0 && MoveNext())
                    ;
                return _curr >= 0;
            }

            public static Wrapper Create(TWaiter waiter, JobScheduler scheduler)
            {
                return new Wrapper(new BlockSequenceIndex<TWaiter>(waiter, scheduler));
            }

            public struct Wrapper : IIndex
            {
                private readonly BlockSequenceIndex<TWaiter> _index;

                public Wrapper(BlockSequenceIndex<TWaiter> index)
                {
                    Contracts.AssertValue(index);
                    _index = index;
                }

                public long Batch { get { return _index.Batch; } }
                public long GetIndex() { return _index.GetIndex(); }
                public ValueGetter<UInt128> GetIdGetter() { return _index.GetIdGetter(); }
                public bool MoveNext() { return _index.MoveNext(); }
                public bool MoveMany(long count) { return _index.MoveMany(count); }
            }
        }

        /// <summary>
        /// An <see cref="IIndex"/> that shares a counter among multiple threads. The multiple threads divy up
        /// the work by blocks of rows rather than splitting row by row simply, to cut down on interthread
        /// communication.
        /// </summary>
        private sealed class BlockRandomIndex<TWaiter> : IIndex
            where TWaiter : struct, IWaiter
        {
            // -1 means not started, -2 means finished, non-negative is the index into _perm.
            private int _curr;
            private int _currMax;
            private readonly int[] _perm;

            private readonly JobScheduler _scheduler;
            private readonly TWaiter _waiter;

            private long _batch;

            public long Batch { get { return _batch; } }

            private BlockRandomIndex(TWaiter waiter, JobScheduler scheduler, int[] perm)
            {
                Contracts.AssertValue(scheduler);
                Contracts.AssertValue(perm);
                _curr = _currMax = -1;
                _batch = -1;
                _perm = perm;
                _waiter = waiter;
                _scheduler = scheduler;
            }

            public long GetIndex()
            {
                Contracts.Assert(0 <= _curr && _curr < _perm.Length);
                return _perm[_curr];
            }

            public ValueGetter<UInt128> GetIdGetter()
            {
                return
                    (ref UInt128 val) =>
                    {
                        Contracts.Check(_curr >= 0, "Cannot call ID getter in current state");
                        val = new UInt128((ulong)_perm[_curr], 0);
                    };
            }

            public bool MoveNext()
            {
                Contracts.Assert(_curr >= -1); // Should not be called when _curr = -2.
                if (_curr == _currMax)
                {
                    // Try to get an available block for processing.
                    _batch = _scheduler.GetAvailableJob(_batch);
                    _curr = (int)(_batch << _batchShift);
                    // We've run off the end (possibly by overflowing), exit.
                    if (_curr >= _perm.Length || _curr < 0)
                    {
                        // We're ending.
                        _curr = -2;
                        return false;
                    }
                    // Try to get the next block length.
                    _currMax = Math.Min(_perm.Length - 1, _curr + _batchMask);
                }
                else
                    _curr++;
                Contracts.Assert(0 <= _curr && _curr <= _currMax);
                bool result = _waiter.Wait(GetIndex());
                Contracts.Assert(result);
                return true;
            }

            public bool MoveMany(long count)
            {
                // I don't know that moving many on parallel cursors is really a thing,
                // given that the order in which they serve up results among themselves
                // is non-deterministic. For now content ourselves with this trivial
                // implementation.
                Contracts.Assert(count > 0);
                while (--count >= 0 && MoveNext())
                    ;
                return _curr >= 0;
            }

            public static Wrapper Create(TWaiter waiter, JobScheduler scheduler, int[] perm)
            {
                return new Wrapper(new BlockRandomIndex<TWaiter>(waiter, scheduler, perm));
            }

            public struct Wrapper : IIndex
            {
                private readonly BlockRandomIndex<TWaiter> _index;

                public Wrapper(BlockRandomIndex<TWaiter> index)
                {
                    Contracts.AssertValue(index);
                    _index = index;
                }

                public long Batch { get { return _index.Batch; } }
                public long GetIndex() { return _index.GetIndex(); }
                public ValueGetter<UInt128> GetIdGetter() { return _index.GetIdGetter(); }
                public bool MoveNext() { return _index.MoveNext(); }
                public bool MoveMany(long count) { return _index.MoveMany(count); }
            }
        }

        private abstract class RowCursorSeekerBase : IDisposable
        {
            protected readonly CacheDataView Parent;
            protected readonly IChannel Ch;

            private readonly int[] _colToActivesIndex;
            private readonly Delegate[] _getters;

            private bool _disposed;

            public ISchema Schema => Parent.Schema;

            public long Position { get; protected set; }

            protected RowCursorSeekerBase(CacheDataView parent, Func<int, bool> predicate)
            {
                Contracts.AssertValue(parent);
                Parent = parent;
                Ch = parent._host.Start("Cursor");
                Position = -1;

                // Set up the mapping from active columns.
                int colLim = Schema.ColumnCount;
                int[] actives;
                Utils.BuildSubsetMaps(colLim, predicate, out actives, out _colToActivesIndex);
                // Construct the getters. Simultaneously collect whatever "waiters"
                // we have to wait on, to ensure that the column value is actually
                // available.
                _getters = new Delegate[actives.Length];
                for (int ic = 0; ic < actives.Length; ++ic)
                {
                    int c = actives[ic];
                    // Having added this after we've spawned the filling task,
                    // all columns should have some sort of cache.
                    Ch.Assert(_colToActivesIndex[c] == ic);
                    Ch.AssertValue(Parent._caches[c]);
                    _getters[ic] = CreateGetterDelegate(c);
                }
            }

            public bool IsColumnActive(int col)
            {
                Ch.CheckParam(0 <= col && col < _colToActivesIndex.Length, nameof(col));
                return _colToActivesIndex[col] >= 0;
            }

            public void Dispose()
            {
                if (!_disposed)
                {
                    DisposeCore();
                    Position = -1;
                    Ch.Dispose();
                    _disposed = true;
                }
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                if (!IsColumnActive(col))
                    throw Ch.Except("Column #{0} is requested but not active in the cursor", col);
                var getter = _getters[_colToActivesIndex[col]] as ValueGetter<TValue>;
                if (getter == null)
                    throw Ch.Except("Invalid TValue: '{0}'", typeof(TValue));
                return getter;
            }

            private Delegate CreateGetterDelegate(int col)
            {
                Ch.Assert(0 <= col && col < _colToActivesIndex.Length);
                Ch.Assert(_colToActivesIndex[col] >= 0);
                return Utils.MarshalInvoke(CreateGetterDelegate<int>, Schema.GetColumnType(col).RawType, col);
            }

            private Delegate CreateGetterDelegate<TValue>(int col)
            {
                Ch.Assert(0 <= col && col < _colToActivesIndex.Length);
                Ch.Assert(_colToActivesIndex[col] >= 0);
                Ch.Assert(Schema.GetColumnType(col).RawType == typeof(TValue));

                var cache = (ColumnCache<TValue>)Parent._caches[col];
                return CreateGetterDelegateCore(cache);
            }

            protected abstract ValueGetter<TValue> CreateGetterDelegateCore<TValue>(ColumnCache<TValue> cache);

            protected abstract void DisposeCore();
        }

        #region Column cache pipe objects

        /// <summary>
        /// A cache of values from a single column. The filler worker fills these in row
        /// by row, and increments the associated waiter. The consumer for the cache
        /// waits on the associated waiter (if non-null), then fetches values as it
        /// determines rows are valid.
        /// </summary>
        private abstract class ColumnCache
        {
            protected IExceptionContext Ctx;
            private static volatile Type[] _pipeConstructorTypes;

            private OrderedWaiter _waiter;

            /// <summary>
            /// The ordered waiter on row indices, indicating when a row index is valid,
            /// or null if it is no longer necessary to wait on this column, that is,
            /// it is completely filled in. Multiple columns can share a single waiter
            /// since often multiple columns are being cached simultaneously, so this
            /// object is not unqiue to this column.
            /// </summary>
            public OrderedWaiter Waiter { get { return _waiter; } }

            protected ColumnCache(IExceptionContext ctx, OrderedWaiter waiter)
            {
                Contracts.AssertValue(ctx);
                Ctx = ctx;
                Ctx.AssertValue(waiter);
                _waiter = waiter;
            }

            /// <summary>
            /// Creates a cache pipe, over a particular column in a cursor.
            /// </summary>
            /// <param name="parent">The cache data view for which we are a cache</param>
            /// <param name="input">The cursor to read from</param>
            /// <param name="srcCol">The column of the cursor we are wrapping.</param>
            /// <param name="waiter">The waiter for the filler associated with this column</param>
            /// <returns></returns>
            public static ColumnCache Create(CacheDataView parent, IRowCursor input, int srcCol, OrderedWaiter waiter)
            {
                Contracts.AssertValue(parent);
                var host = parent._host;
                host.AssertValue(input);
                host.Assert(0 <= srcCol & srcCol < input.Schema.ColumnCount);
                host.Assert(input.IsColumnActive(srcCol));

                var type = input.Schema.GetColumnType(srcCol);
                Type pipeType;
                if (type.IsVector)
                    pipeType = typeof(ImplVec<>).MakeGenericType(type.ItemType.RawType);
                else
                {
                    host.Assert(type.IsPrimitive);
                    pipeType = typeof(ImplOne<>).MakeGenericType(type.RawType);
                }
                if (_pipeConstructorTypes == null)
                {
                    Interlocked.CompareExchange(ref _pipeConstructorTypes,
                        new Type[] { typeof(CacheDataView), typeof(IRowCursor), typeof(int), typeof(OrderedWaiter) }, null);
                }
                var constructor = pipeType.GetConstructor(_pipeConstructorTypes);
                return (ColumnCache)constructor.Invoke(new object[] { parent, input, srcCol, waiter });
            }

            /// <summary>
            /// Utilized by the filler worker, to fill in the cache at the current position of the cursor.
            /// The filler worker will have moved the cursor to the next row prior to calling this, so
            /// overrides will merely get the value at the current position of the cursor.
            /// </summary>
            public abstract void CacheCurrent();

            /// <summary>
            /// Utilized by the filler worker, to indicate to the cache that it will not be receiving
            /// any more values through <see cref="CacheCurrent"/>.
            /// </summary>
            public virtual void Freeze()
            {
                _waiter = null;
            }

            private sealed class ImplVec<T> : ColumnCache<VBuffer<T>>
            {
                // The number of rows cached.
                private int _rowCount;
                // For a given row [r], elements at [r] and [r+1] specify the inclusive
                // and exclusive range of values for the two big arrays. In the case
                // of indices, if that range is empty, then the corresponding stored
                // vector is dense. E.g.: row 5 would have its vector's values stored
                // at indices [_valueBoundaries[5], valueBoundaries[6]) of _values.
                // Both of these boundaries arrays have logical length _rowCount + 1.
                private long[] _indexBoundaries;
                private long[] _valueBoundaries;
                // Non-null only if the vector is of unknown size (so _uniformLength == 0),
                // and holds, per row, the length of the vector.
                private int[] _lengths;
                private readonly int _uniformLength;
                // A structure holding all indices for all stored sparse vectors.
                private readonly BigArray<int> _indices;
                // A structure holding all values for all stored vectors.
                private readonly BigArray<T> _values;

                // The source getter.
                private ValueGetter<VBuffer<T>> _getter;
                // Temporary working reusable storage for caching the source data.
                private VBuffer<T> _temp;

                public ImplVec(CacheDataView parent, IRowCursor input, int srcCol, OrderedWaiter waiter)
                    : base(parent, input, srcCol, waiter)
                {
                    var type = input.Schema.GetColumnType(srcCol);
                    Ctx.Assert(type.IsVector);
                    _uniformLength = type.VectorSize;
                    _indices = new BigArray<int>();
                    _values = new BigArray<T>();
                    _getter = input.GetGetter<VBuffer<T>>(srcCol);
                }

                public override void CacheCurrent()
                {
                    Ctx.Assert(0 <= _rowCount & _rowCount < int.MaxValue);
                    Ctx.AssertValue(Waiter);
                    Ctx.AssertValue(_getter);

                    _getter(ref _temp);
                    if (_uniformLength != 0 && _uniformLength != _temp.Length)
                        throw Ctx.Except("Caching expected vector of size {0}, but {1} encountered.", _uniformLength, _temp.Length);
                    Ctx.Assert(_uniformLength == 0 || _uniformLength == _temp.Length);
                    if (!_temp.IsDense)
                        _indices.AddRange(_temp.Indices, _temp.Count);
                    _values.AddRange(_temp.Values, _temp.Count);
                    int rowCount = _rowCount + 1;
                    Utils.EnsureSize(ref _indexBoundaries, rowCount + 1);
                    Utils.EnsureSize(ref _valueBoundaries, rowCount + 1);
                    _indexBoundaries[rowCount] = _indices.Length;
                    _valueBoundaries[rowCount] = _values.Length;

                    if (_uniformLength == 0)
                    {
                        Utils.EnsureSize(ref _lengths, rowCount);
                        _lengths[rowCount - 1] = _temp.Length;
                    }
                    _rowCount = rowCount;
                }

                public override void Fetch(int idx, ref VBuffer<T> value)
                {
                    Ctx.Assert(0 <= idx & idx < _rowCount);
                    Ctx.Assert(_rowCount < Utils.Size(_indexBoundaries));
                    Ctx.Assert(_rowCount < Utils.Size(_valueBoundaries));
                    Ctx.Assert(_uniformLength > 0 || _rowCount <= Utils.Size(_lengths));

                    Ctx.Assert(_indexBoundaries[idx + 1] - _indexBoundaries[idx] <= int.MaxValue);
                    int indexCount = (int)(_indexBoundaries[idx + 1] - _indexBoundaries[idx]);
                    Ctx.Assert(_valueBoundaries[idx + 1] - _valueBoundaries[idx] <= int.MaxValue);
                    int valueCount = (int)(_valueBoundaries[idx + 1] - _valueBoundaries[idx]);
                    Ctx.Assert(valueCount == indexCount || indexCount == 0);
                    Ctx.Assert(0 <= indexCount && indexCount <= valueCount);
                    int len = _uniformLength == 0 ? _lengths[idx] : _uniformLength;
                    Ctx.Assert(valueCount <= len);
                    Ctx.Assert(valueCount == len || indexCount == valueCount);

                    T[] values = value.Values;
                    Utils.EnsureSize(ref values, valueCount);
                    _values.CopyTo(_valueBoundaries[idx], values, valueCount);
                    int[] indices = value.Indices;

                    if (valueCount < len)
                    {
                        Utils.EnsureSize(ref indices, indexCount);
                        _indices.CopyTo(_indexBoundaries[idx], indices, indexCount);
                        value = new VBuffer<T>(len, indexCount, values, indices);
                    }
                    else
                        value = new VBuffer<T>(len, values, indices);
                }

                public override void Freeze()
                {
                    Array.Resize(ref _indexBoundaries, _rowCount + 1);
                    Array.Resize(ref _valueBoundaries, _rowCount + 1);
                    Array.Resize(ref _lengths, _rowCount);
                    _values.TrimCapacity();
                    _indices.TrimCapacity();
                    _temp = default(VBuffer<T>);
                    base.Freeze();
                    _getter = null;
                }
            }

            private sealed class ImplOne<T> : ColumnCache<T>
            {
                private int _rowCount;
                private T[] _values;
                private ValueGetter<T> _getter;

                public ImplOne(CacheDataView parent, IRowCursor input, int srcCol, OrderedWaiter waiter)
                    : base(parent, input, srcCol, waiter)
                {
                    _getter = input.GetGetter<T>(srcCol);
                    if (parent._rowCount >= 0)
                        _values = new T[(int)parent._rowCount];
                }

                public override void CacheCurrent()
                {
                    Contracts.Assert(0 <= _rowCount & _rowCount < int.MaxValue);
                    Contracts.AssertValue(Waiter);
                    Contracts.AssertValue(_getter);
                    Utils.EnsureSize(ref _values, _rowCount + 1);
                    _getter(ref _values[_rowCount]);
                    ++_rowCount;
                }

                public override void Fetch(int idx, ref T value)
                {
                    Contracts.Assert(0 <= idx & idx < _rowCount);
                    value = _values[idx];
                }

                public override void Freeze()
                {
                    Array.Resize(ref _values, _rowCount);
                    base.Freeze();
                    _getter = null;
                }
            }
        }

        private abstract class ColumnCache<T> : ColumnCache
        {
            public ColumnCache(CacheDataView parent, IRowCursor input, int srcCol, OrderedWaiter waiter)
                : base(parent._host, waiter)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= srcCol & srcCol < input.Schema.ColumnCount);
                Contracts.Assert(input.Schema.GetColumnType(srcCol).RawType == typeof(T));
            }

            /// <summary>
            /// Utilized by the consumer to get a value in the cache at an index. The
            /// consumer should coordinate with the <see cref="ColumnCache.Waiter"/> member to ensure
            /// that the row is valid.
            /// </summary>
            public abstract void Fetch(int idx, ref T value);
        }
        #endregion
    }
}
