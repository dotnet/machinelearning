// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    // REVIEW: Currently, to enable shuffling, we require the row counts of the sources to be known.
    // We can think of the shuffling in AppendRowsDataView as a two-stage process:
    // 1. the shuffling inside each source, and
    // 2. choosing a source with probability proportional to its remaining row counts when the (meta) cursor moves
    // For full-fledged shuffling, we need to know the row counts so as to choose a row uniformly at random.
    // However, this restriction could be loosened according to the desired level of randomness.
    // For instance, replacing unknown row counts with the mean or the maximum value of the known might be good
    // enough for some scenarios.

    /// <summary>
    /// This class provides the functionality to combine multiple IDataView objects which share the same schema
    /// All sources must contain the same number of columns and their column names, sizes, and item types must match.
    /// The row count of the resulting IDataView will be the sum over that of each individual.
    ///
    /// An AppendRowsDataView instance is shuffleable iff all of its sources are shuffleable and their row counts are known.
    /// </summary>
    public sealed class AppendRowsDataView : IDataView
    {
        public const string RegistrationName = "AppendRowsDataView";

        private readonly IDataView[] _sources;
        private readonly int[] _counts;
        private readonly ISchema _schema;
        private readonly IHost _host;
        private readonly bool _canShuffle;

        public bool CanShuffle { get { return _canShuffle; } }

        public ISchema Schema { get { return _schema; } }

        // REVIEW: AppendRowsDataView now only checks schema consistency up to column names and types.
        // A future task will be to ensure that the sources are consistent on the metadata level.

        /// <summary>
        /// Create a dataview by appending the rows of the sources.
        ///
        /// All sources must be consistent with the passed-in schema in the number of columns, column names,
        /// and column types. If schema is null, the first source's schema will be used.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="schema">The schema for the result. If this is null, the first source's schema will be used.</param>
        /// <param name="sources">The sources to be appended.</param>
        /// <returns>The resulting IDataView.</returns>
        public static IDataView Create(IHostEnvironment env, ISchema schema, params IDataView[] sources)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(sources, nameof(sources));
            env.CheckNonEmpty(sources, nameof(sources), "There must be at least one source.");
            env.CheckParam(sources.All(s => s != null), nameof(sources));
            env.CheckValueOrNull(schema);
            if (sources.Length == 1)
                return sources[0];
            return new AppendRowsDataView(env, schema, sources);
        }

        private AppendRowsDataView(IHostEnvironment env, ISchema schema, IDataView[] sources)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);

            _host.AssertValueOrNull(schema);
            _host.AssertValue(sources);
            _host.Assert(sources.Length >= 2);

            _sources = sources;
            _schema = schema ?? _sources[0].Schema;

            CheckSchemaConsistency();

            _canShuffle = true;
            _counts = new int[_sources.Length];
            for (int i = 0; i < _sources.Length; i++)
            {
                IDataView dv = _sources[i];
                if (!dv.CanShuffle)
                {
                    _canShuffle = false;
                    _counts = null;
                    break;
                }
                long? count = dv.GetRowCount(true);
                if (count == null || count < 0 || count > int.MaxValue)
                {
                    _canShuffle = false;
                    _counts = null;
                    break;
                }
                _counts[i] = (int)count;
            }
        }

        private void CheckSchemaConsistency()
        {
            // REVIEW: Allow schema isomorphism.
            const string errMsg = "Inconsistent schema: all source dataviews must have identical column names, sizes, and item types.";

            int startingSchemaIndex = _schema == _sources[0].Schema ? 1 : 0;
            int colCount = _schema.ColumnCount;

            // Check if the column counts are identical.
            _host.Check(_sources.All(source => source.Schema.ColumnCount == colCount), errMsg);

            for (int c = 0; c < colCount; c++)
            {
                string name = _schema.GetColumnName(c);
                ColumnType type = _schema.GetColumnType(c);
                for (int i = startingSchemaIndex; i < _sources.Length; i++)
                {
                    ISchema schema = _sources[i].Schema;
                    _host.Check(schema.GetColumnName(c) == name, errMsg);
                    _host.Check(schema.GetColumnType(c).SameSizeAndItemType(type), errMsg);
                }
            }
        }

        public long? GetRowCount(bool lazy = true)
        {
            long sum = 0;
            foreach (var source in _sources)
            {
                var cur = source.GetRowCount(lazy);
                if (cur == null)
                    return null;
                _host.Check(cur.Value >= 0, "One of the sources returned a negative row count");

                // In the case of overflow, the count is considered unknown.
                if (sum + cur.Value < sum)
                    return null;
                sum += cur.Value;
            }
            return sum;
        }

        public IRowCursor GetRowCursor(Func<int, bool> needCol, IRandom rand = null)
        {
            _host.CheckValue(needCol, nameof(needCol));
            if (rand == null || !_canShuffle)
                return new Cursor(this, needCol);
            return new RandCursor(this, needCol, rand, _counts);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            consolidator = null;
            return new IRowCursor[] { GetRowCursor(predicate, rand) };
        }

        private abstract class CursorBase : RootCursorBase, IRowCursor
        {
            protected readonly IDataView[] Sources;
            protected readonly Delegate[] Getters;

            public override long Batch => 0;

            public ISchema Schema { get; }

            public CursorBase(AppendRowsDataView parent)
                : base(parent._host)
            {
                Sources = parent._sources;
                Ch.AssertNonEmpty(Sources);
                Schema = parent._schema;
                Getters = new Delegate[Schema.ColumnCount];
            }

            protected Delegate CreateGetter(int col)
            {
                ColumnType colType = Schema.GetColumnType(col);
                Ch.AssertValue(colType);
                Func<int, Delegate> creator = CreateTypedGetter<int>;
                var typedCreator = creator.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(colType.RawType);
                return (Delegate)typedCreator.Invoke(this, new object[] { col });
            }

            protected abstract ValueGetter<TValue> CreateTypedGetter<TValue>(int col);

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.Check(IsColumnActive(col), "The column must be active against the defined predicate.");
                if (!(Getters[col] is ValueGetter<TValue>))
                    throw Ch.Except($"Invalid TValue in GetGetter: '{typeof(TValue)}'");
                return Getters[col] as ValueGetter<TValue>;
            }

            public bool IsColumnActive(int col)
            {
                Ch.Check(0 <= col && col < Schema.ColumnCount, "Column index is out of range");
                return Getters[col] != null;
            }
        }

        /// <summary>
        /// The deterministic cursor. It will scan through the sources sequentially.
        /// </summary>
        private sealed class Cursor : CursorBase
        {
            private IRowCursor _currentCursor;
            private ValueGetter<UInt128> _currentIdGetter;
            private int _currentSourceIndex;

            public Cursor(AppendRowsDataView parent, Func<int, bool> needCol)
                : base(parent)
            {
                Ch.AssertValue(needCol);

                _currentSourceIndex = 0;
                _currentCursor = Sources[_currentSourceIndex].GetRowCursor(needCol);
                _currentIdGetter = _currentCursor.GetIdGetter();
                for (int c = 0; c < Getters.Length; c++)
                {
                    if (needCol(c))
                        Getters[c] = CreateGetter(c);
                }
            }

            public override ValueGetter<UInt128> GetIdGetter()
            {
                return
                    (ref UInt128 val) =>
                    {
                        _currentIdGetter(ref val);
                        // While the union of all IDs may not be acceptable, by taking each
                        // data views IDs and combining them against their source index, the
                        // union of these IDs becomes acceptable.
                        // REVIEW: Convenience UInt128 constructor for this scenario?
                        val = val.Combine(new UInt128((ulong)_currentSourceIndex, 0));
                    };
            }

            protected override ValueGetter<TValue> CreateTypedGetter<TValue>(int col)
            {
                Ch.AssertValue(_currentCursor);
                ValueGetter<TValue> getSrc = null;
                // Whenever captured != current, we know that the captured getter is outdated.
                int capturedSourceIndex = -1;
                return
                    (ref TValue val) =>
                    {
                        Ch.Check(State == CursorState.Good, "A getter can only be used when the cursor state is Good.");
                        if (_currentSourceIndex != capturedSourceIndex)
                        {
                            Ch.Assert(0 <= _currentSourceIndex && _currentSourceIndex < Sources.Length);
                            Ch.Assert(_currentCursor != null);
                            getSrc = _currentCursor.GetGetter<TValue>(col);
                            capturedSourceIndex = _currentSourceIndex;
                        }
                        getSrc(ref val);
                    };
            }

            protected override bool MoveNextCore()
            {
                Ch.AssertValue(_currentCursor);
                while (!_currentCursor.MoveNext())
                {
                    // Mark the current cursor as finished.
                    _currentCursor.Dispose();
                    _currentCursor = null;
                    if (++_currentSourceIndex >= Sources.Length)
                        return false;
                    _currentCursor = Sources[_currentSourceIndex].GetRowCursor(c => IsColumnActive(c));
                    _currentIdGetter = _currentCursor.GetIdGetter();
                }

                return true;
            }

            public override void Dispose()
            {
                if (State != CursorState.Done)
                {
                    Ch.Dispose();
                    if (_currentCursor != null)
                        _currentCursor.Dispose();
                    base.Dispose();
                }
            }
        }

        /// <summary>
        ///  A RandCursor will ask each subordinate cursor to shuffle itself.
        /// Then, at each step, it randomly calls a subordinate to move next with probability (roughly) proportional to
        /// the number of the subordinate's remaining rows.
        /// </summary>
        private sealed class RandCursor : CursorBase
        {
            private readonly IRowCursor[] _cursorSet;
            private readonly MultinomialWithoutReplacementSampler _sampler;
            private readonly IRandom _rand;
            private int _currentSourceIndex;

            public RandCursor(AppendRowsDataView parent, Func<int, bool> needCol, IRandom rand, int[] counts)
                : base(parent)
            {
                Ch.AssertValue(needCol);
                Ch.AssertValue(rand);

                _rand = rand;
                Ch.AssertValue(counts);
                Ch.Assert(Sources.Length == counts.Length);
                _cursorSet = new IRowCursor[counts.Length];
                for (int i = 0; i < counts.Length; i++)
                {
                    Ch.Assert(counts[i] >= 0);
                    _cursorSet[i] = parent._sources[i].GetRowCursor(needCol, RandomUtils.Create(_rand));
                }
                _sampler = new MultinomialWithoutReplacementSampler(Ch, counts, rand);
                _currentSourceIndex = -1;
                for (int c = 0; c < Getters.Length; c++)
                {
                    if (needCol(c))
                        Getters[c] = CreateGetter(c);
                }
            }

            public override ValueGetter<UInt128> GetIdGetter()
            {
                ValueGetter<UInt128>[] idGetters = new ValueGetter<UInt128>[_cursorSet.Length];
                for (int i = 0; i < _cursorSet.Length; ++i)
                    idGetters[i] = _cursorSet[i].GetIdGetter();
                return
                    (ref UInt128 val) =>
                    {
                        Ch.Check(IsGood, "Cannot call ID getter in current state");
                        idGetters[_currentSourceIndex](ref val);
                        val = val.Combine(new UInt128((ulong)_currentSourceIndex, 0));
                    };
            }

            protected override ValueGetter<TValue> CreateTypedGetter<TValue>(int col)
            {
                ValueGetter<TValue>[] getSrc = new ValueGetter<TValue>[_cursorSet.Length];
                return
                    (ref TValue val) =>
                    {
                        Ch.Check(State == CursorState.Good, "A getter can only be used when the cursor state is Good.");
                        Ch.Assert(0 <= _currentSourceIndex && _currentSourceIndex < Sources.Length);
                        if (getSrc[_currentSourceIndex] == null)
                            getSrc[_currentSourceIndex] = _cursorSet[_currentSourceIndex].GetGetter<TValue>(col);
                        getSrc[_currentSourceIndex](ref val);
                    };
            }

            protected override bool MoveNextCore()
            {
                int pos;
                // Ask the sampler to select a source and move with it.
                if ((pos = _sampler.Next()) < 0)
                    return false;
                Ch.Assert(pos < _cursorSet.Length);
                _currentSourceIndex = pos;
                bool r = _cursorSet[_currentSourceIndex].MoveNext();
                Ch.Assert(r);
                return true;
            }

            public override void Dispose()
            {
                if (State != CursorState.Done)
                {
                    Ch.Dispose();
                    foreach (IRowCursor c in _cursorSet)
                        c.Dispose();
                    base.Dispose();
                }
            }
        }

        /// <summary>
        /// Given k classes with counts (N_0, N_2, N_3, ...,  N_{k-1}), the goal of this sampler is to select the i-th
        /// class with probability N_i/M, where M = N_0 + N_1 + ... + N_{k-1}.
        /// Once the i-th class is selected, its count will be updated to N_i - 1.
        ///
        /// For efficiency consideration, the sampling distribution is only an approximation of the desired distribution.
        /// </summary>
        private sealed class MultinomialWithoutReplacementSampler
        {
            // Implementation: generate a batch array of size BatchSize.
            // Each class will claim a fraction of the batch proportional to its remaining row count.
            // Shuffle the array. The sampler reads from the array one at a time until the batch is consumed.
            // The sampler then generates a new batch and repeat the process.
            private const int BatchSize = 1000;

            private readonly int[] _rowsLeft;
            private readonly IRandom _rand;
            private readonly int[] _batch;
            private readonly IExceptionContext _ectx;

            private int _batchEnd;
            private int _batchPos;
            private int _totalLeft;

            public MultinomialWithoutReplacementSampler(IExceptionContext context, int[] counts, IRandom rand)
            {
                Contracts.AssertValue(context);
                _ectx = context;
                _ectx.Assert(Utils.Size(counts) > 0);
                _rowsLeft = (int[])counts.Clone();
                _ectx.AssertValue(rand);
                _rand = rand;
                foreach (int count in _rowsLeft)
                {
                    context.Assert(count >= 0 && _totalLeft + count >= _totalLeft);
                    _totalLeft += count;
                }
                _batch = new int[BatchSize];
            }

            private void GenerateNextBatch()
            {
                _batchEnd = 0;
                for (int i = 0; i < _rowsLeft.Length && _batchEnd < BatchSize; i++)
                {
                    int newEnd;
                    if (_totalLeft <= BatchSize)
                        newEnd = _batchEnd + _rowsLeft[i];
                    else
                    {
                        // If we are content with half-way decent shuffling, using Ceiling makes more sense,
                        // as using Floor or Round might result in a second pass in order to fill up the batch.
                        newEnd = _batchEnd + (int)Math.Ceiling((double)_rowsLeft[i] * BatchSize / _totalLeft);
                        if (newEnd > BatchSize)
                            newEnd = BatchSize;
                    }

                    for (int j = _batchEnd; j < newEnd; j++)
                        _batch[j] = i;
                    _rowsLeft[i] -= newEnd - _batchEnd;
                    _batchEnd = newEnd;
                }
                _totalLeft -= _batchEnd;
                Utils.Shuffle(_rand, _batch, 0, _batchEnd);
            }

            public int Next()
            {
                if (_batchPos < _batchEnd)
                    return _batch[_batchPos++];
                else if (_totalLeft > 0)
                {
                    GenerateNextBatch();
                    _ectx.Assert(_batchEnd > 0);
                    _batchPos = 0;
                    return _batch[_batchPos++];
                }
                else
                    return -1;
            }
        }
    }
}
