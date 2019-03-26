// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
namespace Microsoft.ML.Data
{
    /// <summary>
    /// This is a base implementation for a transform that in order to compute its output columns, needs to look
    /// at an entire group of consecutive input examples. For each example in the group, it looks at the value of
    /// two input columns and after seeing the entire group it computes the output column values. The output values
    /// are the same for every example in the same group.
    /// </summary>
    /// <typeparam name="TLabel">The type of the values in the first input column</typeparam>
    /// <typeparam name="TScore">The type of the values in the second input column</typeparam>
    /// <typeparam name="TState">Each class deriving from this transform should implement a state class that knows
    /// how to return the current group's output column values.</typeparam>
    internal abstract class PerGroupTransformBase<TLabel, TScore, TState> : IDataTransform
        where TState : class
    {
        /// <summary>
        /// Deriving classes only need to implement <see cref="ColumnBindingsBase.GetColumnTypeCore"/>.
        /// If any of the output columns have metadata, then the metadata methods should be overridden.
        /// </summary>
        private protected abstract class BindingsBase : ColumnBindingsBase
        {
            public readonly int LabelIndex;
            public readonly int ScoreIndex;
            public readonly int GroupIndex;

            protected BindingsBase(IExceptionContext ectx, DataViewSchema input, string labelCol, string scoreCol, string groupCol, bool user, params string[] names)
                : base(input, user, names)
            {
                ectx.AssertNonWhiteSpace(labelCol);
                ectx.AssertNonWhiteSpace(scoreCol);
                ectx.AssertNonWhiteSpace(groupCol);

                if (!input.TryGetColumnIndex(labelCol, out LabelIndex))
                {
                    throw user ?
                        ectx.ExceptParam(nameof(labelCol), "Label column '{0}' does not exist", labelCol) :
                        ectx.ExceptDecode("Label column '{0}' does not exist", labelCol);
                }
                if (!input.TryGetColumnIndex(scoreCol, out ScoreIndex))
                {
                    throw user ?
                        ectx.ExceptParam(nameof(scoreCol), "Score column '{0}' does not exist", scoreCol) :
                        ectx.ExceptDecode("Score column '{0}' does not exist", scoreCol);
                }
                if (!input.TryGetColumnIndex(groupCol, out GroupIndex))
                {
                    throw user ?
                        ectx.ExceptParam(nameof(groupCol), "Group column '{0}' does not exist", groupCol) :
                        Contracts.ExceptDecode("Group column '{0}' does not exist", groupCol);
                }
            }

            // Get a predicate for the input columns.
            public Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                Contracts.AssertValue(predicate);

                var active = new bool[Input.Count];
                for (int col = 0; col < ColumnCount; col++)
                {
                    if (!predicate(col))
                        continue;

                    bool isSrc;
                    int index = MapColumnIndex(out isSrc, col);
                    if (isSrc)
                        active[index] = true;
                    else
                    {
                        active[LabelIndex] = true;
                        active[ScoreIndex] = true;
                        active[GroupIndex] = true;
                    }
                }

                return col => 0 <= col && col < active.Length && active[col];
            }
        }

        protected readonly IHost Host;

        protected readonly string LabelCol;
        protected readonly string ScoreCol;
        protected readonly string GroupCol;

        DataViewSchema IDataView.Schema => OutputSchema;

        public DataViewSchema OutputSchema => GetBindings().AsSchema;

        public IDataView Source { get; }

        public bool CanShuffle => false;

        protected PerGroupTransformBase(IHostEnvironment env, IDataView input, string labelCol, string scoreCol, string groupCol, string registrationName)
        {
            Contracts.CheckValue(env, nameof(env));
            Host = env.Register(registrationName);
            Host.CheckValue(input, nameof(input));
            Host.CheckNonWhiteSpace(labelCol, nameof(labelCol));
            Host.CheckNonWhiteSpace(scoreCol, nameof(scoreCol));
            Host.CheckNonWhiteSpace(groupCol, nameof(groupCol));

            Source = input;
            LabelCol = labelCol;
            ScoreCol = scoreCol;
            GroupCol = groupCol;
        }

        protected PerGroupTransformBase(IHostEnvironment env, ModelLoadContext ctx, IDataView input, string registrationName)
        {
            Contracts.CheckValue(env, nameof(env));
            Host = env.Register(registrationName);
            Host.CheckValue(input, nameof(input));
            Source = input;

            // *** Binary format ***
            // int: Id of the label column name
            // int: Id of the score column name
            // int: Id of the group column name

            LabelCol = ctx.LoadNonEmptyString();
            ScoreCol = ctx.LoadNonEmptyString();
            GroupCol = ctx.LoadNonEmptyString();
        }

        void ICanSaveModel.Save(ModelSaveContext ctx) => SaveModel(ctx);

        private protected virtual void SaveModel(ModelSaveContext ctx)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // int: Id of the label column name
            // int: Id of the score column name
            // int: Id of the group column name

            ctx.SaveNonEmptyString(LabelCol);
            ctx.SaveNonEmptyString(ScoreCol);
            ctx.SaveNonEmptyString(GroupCol);
        }

        private protected abstract BindingsBase GetBindings();

        public long? GetRowCount()
        {
            return Source.GetRowCount();
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            Host.CheckValueOrNull(rand);
            return new DataViewRowCursor[] { GetRowCursor(columnsNeeded, rand) };
        }

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);

            Host.CheckValueOrNull(rand);
            // If we aren't selecting any of the output columns, don't construct our cursor.
            // Note that because we cannot support random due to the inherently
            // stratified nature, neither can we allow the base data to be shuffled,
            // even if it supports shuffling.
            var bindings = GetBindings();
            if (!bindings.AnyNewColumnsActive(predicate))
            {
                var activeInput = bindings.GetActiveInput(predicate);
                var activeCols = Source.Schema.Where(x => activeInput.Length > x.Index && activeInput[x.Index]);
                var inputCursor = Source.GetRowCursor(activeCols, null);
                return new BindingsWrappedRowCursor(Host, inputCursor, bindings);
            }
            return GetRowCursorCore(predicate);
        }

        private DataViewRowCursor GetRowCursorCore(Func<int, bool> predicate)
        {
            var bindings = GetBindings();
            var active = bindings.GetActive(predicate);
            Contracts.Assert(active.Length == bindings.ColumnCount);

            var predInput = bindings.GetDependencies(predicate);

            var cols = Source.Schema.Where(x => predInput(x.Index));

            return new Cursor(this, Source.GetRowCursor(cols, null), Source.GetRowCursor(cols, null), active);
        }

        /// <summary>
        /// Creates the getters for the transform's output columns. It can be assumed that when the getters are called, the state
        /// object contains the current values of the output columns.
        /// </summary>
        /// <param name="state">The state object, containing the current group's output values.</param>
        /// <param name="predicate">Which output columns are active.</param>
        protected abstract Delegate[] CreateGetters(TState state, Func<int, bool> predicate);

        /// <summary>
        /// Get the getter for the first input column.
        /// </summary>
        protected abstract ValueGetter<TLabel> GetLabelGetter(DataViewRow row);

        /// <summary>
        /// Get the getter for the second input column.
        /// </summary>
        protected abstract ValueGetter<TScore> GetScoreGetter(DataViewRow row);

        /// <summary>
        /// Return a new state object.
        /// </summary>
        protected abstract TState InitializeState(DataViewRow input);

        /// <summary>
        /// Update the state object with one example.
        /// </summary>
        protected abstract void ProcessExample(TState state, TLabel label, TScore score);

        /// <summary>
        /// This method is called after processing a whole group of examples. In this method the
        /// state object should compute the output values for the group just seen.
        /// </summary>
        protected abstract void UpdateState(TState state);

        private sealed class Cursor : RootCursorBase
        {
            private readonly PerGroupTransformBase<TLabel, TScore, TState> _parent;
            private readonly DataViewRowCursor _groupCursor;
            private readonly DataViewRowCursor _input;
            private readonly bool[] _active;
            private readonly Delegate[] _getters;

            private readonly TState _state;

            private readonly Func<bool> _newGroupInGroupCursorDel;
            private readonly Func<bool> _newGroupInInputCursorDel;
            private readonly ValueGetter<TLabel> _labelGetter;
            private readonly ValueGetter<TScore> _scoreGetter;

            public override DataViewSchema Schema => _parent.OutputSchema;

            public override long Batch => 0;

            public Cursor(PerGroupTransformBase<TLabel, TScore, TState> parent, DataViewRowCursor input, DataViewRowCursor groupCursor, bool[] active)
                : base(parent.Host)
            {
                Ch.AssertValue(parent);
                // REVIEW: Try to see if we can relax this requirement in case there are no
                // active columns from the input (that are not required for output computation).
                Ch.AssertValue(input);
                Ch.AssertValue(groupCursor);
                Ch.AssertValue(active);

                _parent = parent;
                _input = input;
                _groupCursor = groupCursor;
                _active = active;

                _state = _parent.InitializeState(_input);

                var bindings = _parent.GetBindings();
                _getters = _parent.CreateGetters(_state, iinfo => active[bindings.MapIinfoToCol(iinfo)]);

                _newGroupInGroupCursorDel = RowCursorUtils.GetIsNewGroupDelegate(_groupCursor, bindings.GroupIndex);
                _newGroupInInputCursorDel = RowCursorUtils.GetIsNewGroupDelegate(_input, bindings.GroupIndex);
                _labelGetter = _parent.GetLabelGetter(_groupCursor);
                _scoreGetter = _parent.GetScoreGetter(_groupCursor);
            }

            /// <summary>
            /// Returns whether the given column is active in this row.
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                Ch.Check(column.Index < _parent.GetBindings().ColumnCount);
                return _active[column.Index];
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
                Contracts.CheckParam(IsColumnActive(column), nameof(column), "requested column is not active");

                bool isSrc;
                var index = _parent.GetBindings().MapColumnIndex(out isSrc, column.Index);
                if (isSrc)
                {
                    Contracts.AssertValue(_input);
                    return _input.GetGetter<TValue>(_input.Schema[index]);
                }

                Ch.AssertValue(_getters);
                var getter = _getters[index];
                Ch.Assert(getter != null);
                var fn = getter as ValueGetter<TValue>;
                if (fn == null)
                    throw Ch.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                return fn;
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
                if (!_input.MoveNext())
                    return false;
                if (!_newGroupInInputCursorDel())
                    return true;

                // If this is the first step, we need to move next on _groupCursor. Otherwise, the position of _groupCursor is
                // at the start of the next group.
                if (_groupCursor.Position < 0)
                {
                    // The two cursors should have the same number of elements, so if _input.MoveNext() returned true,
                    // then it must return true here too.
                    var good = _groupCursor.MoveNext() && _newGroupInGroupCursorDel();
                    Ch.Assert(good);
                }
                Ch.Assert(_groupCursor.Position >= 0);

                // Read the whole group from the auxiliary cursor.
                while (_groupCursor.Position >= 0 && !_newGroupInGroupCursorDel())
                {
                    TLabel label = default;
                    TScore score = default;
                    _labelGetter(ref label);
                    _scoreGetter(ref score);
                    _parent.ProcessExample(_state, label, score);
                    _groupCursor.MoveNext();
                }

                _parent.UpdateState(_state);
                return true;
            }
        }
    }
}
