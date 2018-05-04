// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.Data
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
    public abstract class PerGroupTransformBase<TLabel, TScore, TState> : IDataTransform
        where TState : class
    {
        /// <summary>
        /// Deriving classes only need to implement <see cref="ColumnBindingsBase.GetColumnTypeCore"/>.
        /// If any of the output columns have metadata, then the metadata methods should be overridden.
        /// </summary>
        protected abstract class BindingsBase : ColumnBindingsBase
        {
            public readonly int LabelIndex;
            public readonly int ScoreIndex;
            public readonly int GroupIndex;

            protected BindingsBase(IExceptionContext ectx, ISchema input, string labelCol, string scoreCol, string groupCol, bool user, params string[] names)
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

                var active = new bool[Input.ColumnCount];
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

        public ISchema Schema => GetBindings();

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

        public virtual void Save(ModelSaveContext ctx)
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

        protected abstract BindingsBase GetBindings();

        public long? GetRowCount(bool lazy = true)
        {
            return Source.GetRowCount(lazy);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            Host.CheckValue(predicate, nameof(predicate));
            Host.CheckValueOrNull(rand);
            consolidator = null;
            return new IRowCursor[] { GetRowCursor(predicate, rand) };
        }

        public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
        {
            Host.CheckValue(predicate, nameof(predicate));
            Host.CheckValueOrNull(rand);
            // If we aren't selecting any of the output columns, don't construct our cursor.
            // Note that because we cannot support random due to the inherently
            // stratified nature, neither can we allow the base data to be shuffled,
            // even if it supports shuffling.
            var bindings = GetBindings();
            if (!bindings.AnyNewColumnsActive(predicate))
            {
                var activeInput = bindings.GetActiveInput(predicate);
                var inputCursor = Source.GetRowCursor(c => activeInput[c], null);
                return new BindingsWrappedRowCursor(Host, inputCursor, bindings);
            }
            return GetRowCursorCore(predicate);
        }

        private IRowCursor GetRowCursorCore(Func<int, bool> predicate)
        {
            var bindings = GetBindings();
            var active = bindings.GetActive(predicate);
            Contracts.Assert(active.Length == bindings.ColumnCount);

            var predInput = bindings.GetDependencies(predicate);
            return new RowCursor(this, Source.GetRowCursor(predInput, null), Source.GetRowCursor(predInput, null), active);
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
        protected abstract ValueGetter<TLabel> GetLabelGetter(IRow row);

        /// <summary>
        /// Get the getter for the second input column.
        /// </summary>
        protected abstract ValueGetter<TScore> GetScoreGetter(IRow row);

        /// <summary>
        /// Return a new state object.
        /// </summary>
        protected abstract TState InitializeState(IRow input);

        /// <summary>
        /// Update the state object with one example.
        /// </summary>
        protected abstract void ProcessExample(TState state, TLabel label, TScore score);

        /// <summary>
        /// This method is called after processing a whole group of examples. In this method the
        /// state object should compute the output values for the group just seen.
        /// </summary>
        protected abstract void UpdateState(TState state);

        private sealed class RowCursor : RootCursorBase, IRowCursor
        {
            private readonly PerGroupTransformBase<TLabel, TScore, TState> _parent;
            private readonly IRowCursor _groupCursor;
            private readonly IRowCursor _input;
            private readonly bool[] _active;
            private readonly Delegate[] _getters;

            private readonly TState _state;

            private readonly Func<bool> _newGroupInGroupCursorDel;
            private readonly Func<bool> _newGroupInInputCursorDel;
            private readonly ValueGetter<TLabel> _labelGetter;
            private readonly ValueGetter<TScore> _scoreGetter;

            public ISchema Schema { get { return _parent.GetBindings(); } }

            public override long Batch { get { return 0; } }

            public RowCursor(PerGroupTransformBase<TLabel, TScore, TState> parent, IRowCursor input, IRowCursor groupCursor, bool[] active)
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

            public bool IsColumnActive(int col)
            {
                Ch.Check(0 <= col && col < _parent.GetBindings().ColumnCount);
                return _active[col];
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Contracts.CheckParam(IsColumnActive(col), nameof(col), "requested column is not active");

                bool isSrc;
                col = _parent.GetBindings().MapColumnIndex(out isSrc, col);
                if (isSrc)
                {
                    Contracts.AssertValue(_input);
                    return _input.GetGetter<TValue>(col);
                }

                Ch.AssertValue(_getters);
                var getter = _getters[col];
                Ch.Assert(getter != null);
                var fn = getter as ValueGetter<TValue>;
                if (fn == null)
                    throw Ch.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                return fn;
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
                if (!_input.MoveNext())
                    return false;
                if (!_newGroupInInputCursorDel())
                    return true;

                // If this is the first step, we need to move next on _groupCursor. Otherwise, the position of _groupCursor is 
                // at the start of the next group.
                if (_groupCursor.State == CursorState.NotStarted)
                {
                    // The two cursors should have the same number of elements, so if _input.MoveNext() returned true,
                    // then it must return true here too.
                    var good = _groupCursor.MoveNext() && _newGroupInGroupCursorDel();
                    Ch.Assert(good);
                }

                // Read the whole group from the auxiliary cursor.
                while (_groupCursor.State != CursorState.Done && !_newGroupInGroupCursorDel())
                {
                    TLabel label = default(TLabel);
                    TScore score = default(TScore);
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
