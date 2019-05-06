// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
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
    [BestFriend]
    internal sealed class AppendRowsDataView : IDataView
    {
        public const string RegistrationName = "AppendRowsDataView";

        private readonly IDataView[] _sources;
        private readonly int[] _counts;
        private readonly IHost _host;

        public bool CanShuffle { get; }

        public DataViewSchema Schema { get; }

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
        public static IDataView Create(IHostEnvironment env, DataViewSchema schema, params IDataView[] sources)
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

        private AppendRowsDataView(IHostEnvironment env, DataViewSchema schema, IDataView[] sources)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);

            _host.AssertValueOrNull(schema);
            _host.AssertValue(sources);
            _host.Assert(sources.Length >= 2);

            _sources = sources;
            Schema = schema ?? _sources[0].Schema;

            CheckSchemaConsistency();

            CanShuffle = true;
            _counts = new int[_sources.Length];
            for (int i = 0; i < _sources.Length; i++)
            {
                IDataView dv = _sources[i];
                if (!dv.CanShuffle)
                {
                    CanShuffle = false;
                    _counts = null;
                    break;
                }
                long? count = dv.GetRowCount();
                if (count == null || count < 0 || count > int.MaxValue)
                {
                    CanShuffle = false;
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

            int startingSchemaIndex = Schema == _sources[0].Schema ? 1 : 0;
            int colCount = Schema.Count;

            // Check if the column counts are identical.
            _host.Check(_sources.All(source => source.Schema.Count == colCount), errMsg);

            for (int c = 0; c < colCount; c++)
            {
                string name = Schema[c].Name;
                DataViewType type = Schema[c].Type;

                for (int i = startingSchemaIndex; i < _sources.Length; i++)
                {
                    var schema = _sources[i].Schema;
                    _host.Check(schema[c].Name == name, errMsg);
                    _host.Check(schema[c].Type.SameSizeAndItemType(type), errMsg);
                }
            }
        }

        public long? GetRowCount()
        {
            long sum = 0;
            foreach (var source in _sources)
            {
                var cur = source.GetRowCount();
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

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            if (rand == null || !CanShuffle)
                return new Cursor(this, columnsNeeded);
            return new RandCursor(this, columnsNeeded, rand, _counts);
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            return new DataViewRowCursor[] { GetRowCursor(columnsNeeded, rand) };
        }

        private abstract class CursorBase : RootCursorBase
        {
            protected readonly IDataView[] Sources;
            protected readonly Delegate[] Getters;

            public override long Batch => 0;

            public sealed override DataViewSchema Schema { get; }

            public CursorBase(AppendRowsDataView parent)
                : base(parent._host)
            {
                Sources = parent._sources;
                Ch.AssertNonEmpty(Sources);
                Schema = parent.Schema;
                Getters = new Delegate[Schema.Count];
            }

            protected Delegate CreateGetter(int col)
            {
                DataViewType colType = Schema[col].Type;
                Ch.AssertValue(colType);
                Func<int, Delegate> creator = CreateTypedGetter<int>;
                var typedCreator = creator.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(colType.RawType);
                return (Delegate)typedCreator.Invoke(this, new object[] { col });
            }

            protected abstract ValueGetter<TValue> CreateTypedGetter<TValue>(int col);

            public sealed override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                Ch.CheckParam(column.Index <= Getters.Length && IsColumnActive(column), nameof(column), "requested column not active");

                if (!(Getters[column.Index] is ValueGetter<TValue>))
                    throw Ch.Except($"Invalid TValue in GetGetter: '{typeof(TValue)}'");
                return Getters[column.Index] as ValueGetter<TValue>;
            }

            /// <summary>
            /// Returns whether the given column is active in this row.
            /// </summary>
            public sealed override bool IsColumnActive(DataViewSchema.Column column)
            {
                Ch.Check(column.Index < Schema.Count, "Column index is out of range");
                return Getters[column.Index] != null;
            }
        }

        /// <summary>
        /// The deterministic cursor. It will scan through the sources sequentially.
        /// </summary>
        private sealed class Cursor : CursorBase
        {
            private DataViewRowCursor _currentCursor;
            private ValueGetter<DataViewRowId> _currentIdGetter;
            private int _currentSourceIndex;
            private bool _disposed;

            public Cursor(AppendRowsDataView parent, IEnumerable<DataViewSchema.Column> columnsNeeded)
                : base(parent)
            {
                _currentSourceIndex = 0;
                _currentCursor = Sources[_currentSourceIndex].GetRowCursor(columnsNeeded);
                _currentIdGetter = _currentCursor.GetIdGetter();

                foreach(var col in columnsNeeded)
                    Getters[col.Index] = CreateGetter(col.Index);
            }

            public override ValueGetter<DataViewRowId> GetIdGetter()
            {
                return
                    (ref DataViewRowId val) =>
                    {
                        _currentIdGetter(ref val);
                        // While the union of all IDs may not be acceptable, by taking each
                        // data views IDs and combining them against their source index, the
                        // union of these IDs becomes acceptable.
                        // REVIEW: Convenience RowId constructor for this scenario?
                        val = val.Combine(new DataViewRowId((ulong)_currentSourceIndex, 0));
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
                        Ch.Check(Position >= 0, RowCursorUtils.FetchValueStateError);
                        if (_currentSourceIndex != capturedSourceIndex)
                        {
                            Ch.Assert(0 <= _currentSourceIndex && _currentSourceIndex < Sources.Length);
                            Ch.Assert(_currentCursor != null);
                            getSrc = _currentCursor.GetGetter<TValue>(_currentCursor.Schema[col]);
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

                    var columnsNeeded = Schema.Where(col => IsColumnActive(col));
                    _currentCursor = Sources[_currentSourceIndex].GetRowCursor(columnsNeeded);
                    _currentIdGetter = _currentCursor.GetIdGetter();
                }

                return true;
            }

            protected override void Dispose(bool disposing)
            {
                if (_disposed)
                    return;
                if (disposing)
                {
                    Ch.Dispose();
                    _currentCursor?.Dispose();
                }
                _disposed = true;
                base.Dispose(disposing);
            }
        }

        /// <summary>
        ///  A RandCursor will ask each subordinate cursor to shuffle itself.
        /// Then, at each step, it randomly calls a subordinate to move next with probability (roughly) proportional to
        /// the number of the subordinate's remaining rows.
        /// </summary>
        private sealed class RandCursor : CursorBase
        {
            private readonly DataViewRowCursor[] _cursorSet;
            private readonly MultinomialWithoutReplacementSampler _sampler;
            private readonly Random _rand;
            private int _currentSourceIndex;
            private bool _disposed;

            public RandCursor(AppendRowsDataView parent, IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand, int[] counts)
                : base(parent)
            {
                Ch.AssertValue(rand);

                _rand = rand;
                Ch.AssertValue(counts);
                Ch.Assert(Sources.Length == counts.Length);
                _cursorSet = new DataViewRowCursor[counts.Length];
                for (int i = 0; i < counts.Length; i++)
                {
                    Ch.Assert(counts[i] >= 0);
                    _cursorSet[i] = parent._sources[i].GetRowCursor(columnsNeeded, RandomUtils.Create(_rand));
                }
                _sampler = new MultinomialWithoutReplacementSampler(Ch, counts, rand);
                _currentSourceIndex = -1;

                foreach(var col in columnsNeeded)
                    Getters[col.Index] = CreateGetter(col.Index);
            }

            public override ValueGetter<DataViewRowId> GetIdGetter()
            {
                ValueGetter<DataViewRowId>[] idGetters = new ValueGetter<DataViewRowId>[_cursorSet.Length];
                for (int i = 0; i < _cursorSet.Length; ++i)
                    idGetters[i] = _cursorSet[i].GetIdGetter();
                return
                    (ref DataViewRowId val) =>
                    {
                        Ch.Check(IsGood, RowCursorUtils.FetchValueStateError);
                        idGetters[_currentSourceIndex](ref val);
                        val = val.Combine(new DataViewRowId((ulong)_currentSourceIndex, 0));
                    };
            }

            protected override ValueGetter<TValue> CreateTypedGetter<TValue>(int col)
            {
                ValueGetter<TValue>[] getSrc = new ValueGetter<TValue>[_cursorSet.Length];
                return
                    (ref TValue val) =>
                    {
                        Ch.Check(Position >= 0, RowCursorUtils.FetchValueStateError);
                        Ch.Assert(0 <= _currentSourceIndex && _currentSourceIndex < Sources.Length);

                        var rowCursor = _cursorSet[_currentSourceIndex];

                        if (getSrc[_currentSourceIndex] == null)
                            getSrc[_currentSourceIndex] = rowCursor.GetGetter<TValue>(rowCursor.Schema[col]);
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

            protected override void Dispose(bool disposing)
            {
                if (_disposed)
                    return;
                if (disposing)
                {
                    Ch.Dispose();
                    foreach (DataViewRowCursor c in _cursorSet)
                        c.Dispose();
                }
                _disposed = true;
                base.Dispose(disposing);
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
            private readonly Random _rand;
            private readonly int[] _batch;
            private readonly IExceptionContext _ectx;

            private int _batchEnd;
            private int _batchPos;
            private int _totalLeft;

            public MultinomialWithoutReplacementSampler(IExceptionContext context, int[] counts, Random rand)
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
                Utils.Shuffle(_rand, _batch.AsSpan(0, _batchEnd));
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
