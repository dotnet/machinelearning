// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data.DataView
{
    internal abstract class BatchDataViewMapperBase<TInput, TBatch> : IDataView
    {
        public bool CanShuffle => false;

        public DataViewSchema Schema => SchemaBindings.AsSchema;

        private readonly IDataView _source;
        protected readonly IHost Host;

        protected BatchDataViewMapperBase(IHostEnvironment env, string registrationName, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            Host = env.Register(registrationName);
            _source = input;
        }

        public long? GetRowCount() => _source.GetRowCount();

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            Host.CheckValue(columnsNeeded, nameof(columnsNeeded));
            Host.CheckValueOrNull(rand);

            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, SchemaBindings.AsSchema);

            // If we aren't selecting any of the output columns, don't construct our cursor.
            // Note that because we cannot support random due to the inherently
            // stratified nature, neither can we allow the base data to be shuffled,
            // even if it supports shuffling.
            if (!SchemaBindings.AnyNewColumnsActive(predicate))
            {
                var activeInput = SchemaBindings.GetActiveInput(predicate);
                var inputCursor = _source.GetRowCursor(_source.Schema.Where(c => activeInput[c.Index]), null);
                return new BindingsWrappedRowCursor(Host, inputCursor, SchemaBindings);
            }
            var active = SchemaBindings.GetActive(predicate);
            Contracts.Assert(active.Length == SchemaBindings.ColumnCount);

            // REVIEW: We can get a different input predicate for the input cursor and for the lookahead cursor. The lookahead
            // cursor is only used for getting the values from the input column, so it only needs that column activated. The
            // other cursor is used to get source columns, so it needs the rest of them activated.
            var predInput = GetSchemaBindingDependencies(predicate);
            var inputCols = _source.Schema.Where(c => predInput(c.Index));
            return new Cursor(this, _source.GetRowCursor(inputCols), _source.GetRowCursor(inputCols), active);
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            return new[] { GetRowCursor(columnsNeeded, rand) };
        }

        protected abstract ColumnBindingsBase SchemaBindings { get; }
        protected abstract TBatch CreateBatch(DataViewRowCursor input);
        protected abstract void ProcessBatch(TBatch currentBatch);
        protected abstract void ProcessExample(TBatch currentBatch, TInput currentInput);
        protected abstract Func<bool> GetLastInBatchDelegate(DataViewRowCursor lookAheadCursor);
        protected abstract Func<bool> GetIsNewBatchDelegate(DataViewRowCursor lookAheadCursor);
        protected abstract ValueGetter<TInput> GetLookAheadGetter(DataViewRowCursor lookAheadCursor);
        protected abstract Delegate[] CreateGetters(DataViewRowCursor input, TBatch currentBatch, bool[] active);
        protected abstract Func<int, bool> GetSchemaBindingDependencies(Func<int, bool> predicate);

        private sealed class Cursor : RootCursorBase
        {
            private readonly BatchDataViewMapperBase<TInput, TBatch> _parent;
            private readonly DataViewRowCursor _lookAheadCursor;
            private readonly DataViewRowCursor _input;

            private readonly bool[] _active;
            private readonly Delegate[] _getters;

            private readonly TBatch _currentBatch;
            private readonly Func<bool> _lastInBatchInLookAheadCursorDel;
            private readonly Func<bool> _firstInBatchInInputCursorDel;
            private readonly ValueGetter<TInput> _inputGetterInLookAheadCursor;
            private TInput _currentInput;

            public override long Batch => 0;

            public override DataViewSchema Schema => _parent.Schema;

            public Cursor(BatchDataViewMapperBase<TInput, TBatch> parent, DataViewRowCursor input, DataViewRowCursor lookAheadCursor, bool[] active)
                : base(parent.Host)
            {
                _parent = parent;
                _input = input;
                _lookAheadCursor = lookAheadCursor;
                _active = active;

                _currentBatch = _parent.CreateBatch(_input);

                _getters = _parent.CreateGetters(_input, _currentBatch, _active);

                _lastInBatchInLookAheadCursorDel = _parent.GetLastInBatchDelegate(_lookAheadCursor);
                _firstInBatchInInputCursorDel = _parent.GetIsNewBatchDelegate(_input);
                _inputGetterInLookAheadCursor = _parent.GetLookAheadGetter(_lookAheadCursor);
            }

            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                Contracts.CheckParam(IsColumnActive(column), nameof(column), "requested column is not active");

                var col = _parent.SchemaBindings.MapColumnIndex(out bool isSrc, column.Index);
                if (isSrc)
                {
                    Contracts.AssertValue(_input);
                    return _input.GetGetter<TValue>(_input.Schema[col]);
                }

                Ch.AssertValue(_getters);
                var getter = _getters[col];
                Ch.Assert(getter != null);
                var fn = getter as ValueGetter<TValue>;
                if (fn == null)
                    throw Ch.Except($"Invalid TValue in GetGetter: '{typeof(TValue)}', " +
                            $"expected type: '{getter.GetType().GetGenericArguments().First()}'.");
                return fn;
            }

            public override ValueGetter<DataViewRowId> GetIdGetter()
            {
                return
                    (ref DataViewRowId val) =>
                    {
                        Ch.Check(IsGood, "Cannot call ID getter in current state");
                        val = new DataViewRowId((ulong)Position, 0);
                    };
            }

            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                Ch.Check(column.Index < _parent.SchemaBindings.AsSchema.Count);
                return _active[column.Index];
            }

            protected override bool MoveNextCore()
            {
                if (!_input.MoveNext())
                    return false;
                if (!_firstInBatchInInputCursorDel())
                    return true;

                // If we are here, this means that _input.MoveNext() has gotten us to the beginning of the next batch,
                // so now we need to look ahead at the entire next batch in the _lookAheadCursor.
                // The _lookAheadCursor's position should be on the last row of the previous batch (or -1).
                Ch.Assert(_lastInBatchInLookAheadCursorDel());

                var good = _lookAheadCursor.MoveNext();
                // The two cursors should have the same number of elements, so if _input.MoveNext() returned true,
                // then it must return true here too.
                Ch.Assert(good);

                do
                {
                    _inputGetterInLookAheadCursor(ref _currentInput);
                    _parent.ProcessExample(_currentBatch, _currentInput);
                } while (!_lastInBatchInLookAheadCursorDel() && _lookAheadCursor.MoveNext());

                _parent.ProcessBatch(_currentBatch);
                return true;
            }
        }
    }
}
