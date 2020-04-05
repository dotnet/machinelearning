// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    internal abstract class CustomMappingFilterBase<TSrc> : IDataView
        where TSrc : class, new()
    {
        protected readonly IDataView Input;
        protected readonly TypedCursorable<TSrc> TypedSrc;
        protected readonly IHost Host;

        public abstract bool CanShuffle { get; }

        public DataViewSchema Schema => Input.Schema;

        private protected CustomMappingFilterBase(IHostEnvironment env, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            Host = env.Register("CustomFilter");
            Host.CheckValue(input, nameof(input));

            Input = input;
            TypedSrc = TypedCursorable<TSrc>.Create(Host, input, false, null);
        }

        public long? GetRowCount() => null;

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            Host.AssertValueOrNull(rand);
            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, Schema);

            Func<int, bool> inputPred = TypedSrc.GetDependencies(predicate);

            var inputCols = Input.Schema.Where(x => inputPred(x.Index));
            var input = Input.GetRowCursor(inputCols, rand);
            return GetRowCursorCore(input, Utils.BuildArray(Input.Schema.Count, inputCols));
        }

        protected abstract DataViewRowCursor GetRowCursorCore(DataViewRowCursor input, bool[] active);

        public abstract DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null);
    }

    internal sealed class CustomMappingFilter<TSrc> : CustomMappingFilterBase<TSrc>
        where TSrc : class, new()
    {
        private readonly Func<TSrc, bool> _predicate;

        public override bool CanShuffle => Input.CanShuffle;

        public CustomMappingFilter(IHostEnvironment env, IDataView input, Func<TSrc, bool> predicate)
            : base(env, input)
        {
            Host.CheckValue(predicate, nameof(predicate));
            _predicate = predicate;
        }

        protected override DataViewRowCursor GetRowCursorCore(DataViewRowCursor input, bool[] active)
        {
            return new Cursor(this, input, active);
        }

        public override DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            Host.AssertValueOrNull(rand);
            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, Schema);

            Func<int, bool> inputPred = TypedSrc.GetDependencies(predicate);

            var inputCols = Input.Schema.Where(x => inputPred(x.Index));
            var inputs = Input.GetRowCursorSet(inputCols, n, rand);
            Host.AssertNonEmpty(inputs);
            var active = Utils.BuildArray(Input.Schema.Count, inputCols);

            // No need to split if this is given 1 input cursor.
            var cursors = new DataViewRowCursor[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                cursors[i] = new Cursor(this, inputs[i], active);
            return cursors;
        }

        private sealed class Cursor : LinkedRowFilterCursorBase
        {
            private readonly Func<bool> _accept;

            public Cursor(CustomMappingFilter<TSrc> parent, DataViewRowCursor input, bool[] active)
                : base(parent.Host, input, input.Schema, active)
            {
                Contracts.AssertValue(parent);

                IRowReadableAs<TSrc> inputRow = parent.TypedSrc.GetRow(input);

                TSrc src = new TSrc();
                long lastServedPosition = -1;
                Action refresh = () =>
                {
                    if (lastServedPosition != input.Position)
                    {
                        inputRow.FillValues(src);
                        lastServedPosition = input.Position;
                    }
                };

                var predicate = parent._predicate;
                _accept = () =>
                {
                    refresh();
                    return !predicate(src);
                };
            }

            protected override bool Accept()
            {
                return _accept();
            }
        }
    }

    internal sealed class StatefulCustomMappingFilter<TSrc, TState> : CustomMappingFilterBase<TSrc>
        where TSrc : class, new()
        where TState : class, new()
    {
        private readonly Func<TSrc, TState, bool> _predicate;
        private readonly Action<TState> _stateInitAction;

        public override bool CanShuffle => false;

        public StatefulCustomMappingFilter(IHostEnvironment env, IDataView input, Func<TSrc, TState, bool> predicate, Action<TState> stateInitAction)
            : base(env, input)
        {
            Host.CheckValue(predicate, nameof(predicate));
            Host.CheckValue(stateInitAction, nameof(stateInitAction));

            _predicate = predicate;
            _stateInitAction = stateInitAction;
        }

        protected override DataViewRowCursor GetRowCursorCore(DataViewRowCursor input, bool[] active)
        {
            return new Cursor(this, input, active);
        }

        public override DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            return new[] { GetRowCursor(columnsNeeded, rand) };
        }

        private sealed class Cursor : LinkedRowFilterCursorBase
        {
            private readonly Func<bool> _accept;

            public Cursor(StatefulCustomMappingFilter<TSrc, TState> parent, DataViewRowCursor input, bool[] active)
                : base(parent.Host, input, input.Schema, active)
            {
                Contracts.AssertValue(parent);

                IRowReadableAs<TSrc> inputRow = parent.TypedSrc.GetRow(input);

                TSrc src = new TSrc();
                TState state = new TState();
                parent._stateInitAction(state);
                long lastServedPosition = -1;
                Action refresh = () =>
                {
                    if (lastServedPosition != input.Position)
                    {
                        inputRow.FillValues(src);
                        lastServedPosition = input.Position;
                    }
                };

                var predicate = parent._predicate;
                _accept = () =>
                {
                    refresh();
                    return !predicate(src, state);
                };
            }

            protected override bool Accept()
            {
                return _accept();
            }
        }
    }
}
