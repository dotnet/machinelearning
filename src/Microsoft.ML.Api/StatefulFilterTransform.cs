// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using System;
using System.IO;

namespace Microsoft.ML.Runtime.Api
{
    // REVIEW: the current interface to 'state' object may be inadequate: instead of insisting on
    // parameterless constructor, we could take a delegate that would create the state per cursor.
    /// <summary>
    /// This transform is similar to <see cref="CustomMappingTransformer{TSrc,TDst}"/>, but it allows per-cursor state,
    /// as well as the ability to 'accept' or 'filter out' some rows of the supplied <see cref="IDataView"/>.
    /// The downside is that the provided lambda is eagerly called on every row (not lazily when needed), and
    /// parallel cursors are not allowed.
    /// </summary>
    /// <typeparam name="TSrc">The type that describes what 'source' columns are consumed from the input <see cref="IDataView"/>.</typeparam>
    /// <typeparam name="TDst">The type that describes what new columns are added by this transform.</typeparam>
    /// <typeparam name="TState">The type that describes per-cursor state.</typeparam>
    internal sealed class StatefulFilterTransform<TSrc, TDst, TState> : LambdaTransformBase, ITransformTemplate
        where TSrc : class, new()
        where TDst : class, new()
        where TState : class, new()
    {
        private const string RegistrationNameTemplate = "StatefulFilterTransform<{0}, {1}>";
        private readonly IDataView _source;
        private readonly Func<TSrc, TDst, TState, bool> _filterFunc;
        private readonly Action<TState> _initStateAction;
        private readonly ColumnBindings _bindings;
        private readonly InternalSchemaDefinition _addedSchema;

        // Memorized input schema definition. Needed for re-apply.
        private readonly SchemaDefinition _inputSchemaDefinition;
        private readonly TypedCursorable<TSrc> _typedSource;

        private static string RegistrationName { get { return string.Format(RegistrationNameTemplate, typeof(TSrc).FullName, typeof(TDst).FullName); } }

        /// <summary>
        /// Create a filter transform that is savable iff <paramref name="saveAction"/> and <paramref name="loadFunc"/> are
        /// not null.
        /// </summary>
        /// <param name="env">The host environment</param>
        /// <param name="source">The dataview upon which we construct the transform</param>
        /// <param name="filterFunc">The function by which we transform source to destination columns and decide whether
        /// to keep the row.</param>
        /// <param name="initStateAction">The function that is called once per cursor to initialize state. Can be null.</param>
        /// <param name="saveAction">An action that allows us to save state to the serialization stream. May be
        /// null simultaneously with <paramref name="loadFunc"/>.</param>
        /// <param name="loadFunc">A function that given the serialization stream and a data view, returns
        /// an <see cref="ITransformTemplate"/>. The intent is, this returned object should itself be a
        /// <see cref="CustomMappingTransformer{TSrc,TDst}"/>, but this is not strictly necessary. This delegate should be
        /// a static non-lambda method that this assembly can legally call. May be null simultaneously with
        /// <paramref name="saveAction"/>.</param>
        /// <param name="inputSchemaDefinition">The schema definition overrides for <typeparamref name="TSrc"/></param>
        /// <param name="outputSchemaDefinition">The schema definition overrides for <typeparamref name="TDst"/></param>
        public StatefulFilterTransform(IHostEnvironment env, IDataView source, Func<TSrc, TDst, TState, bool> filterFunc,
            Action<TState> initStateAction,
            Action<BinaryWriter> saveAction, LambdaTransform.LoadDelegate loadFunc,
            SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            : base(env, RegistrationName, saveAction, loadFunc)
        {
            Host.AssertValue(source, "source");
            Host.AssertValue(filterFunc, "filterFunc");
            Host.AssertValueOrNull(initStateAction);
            Host.AssertValueOrNull(inputSchemaDefinition);
            Host.AssertValueOrNull(outputSchemaDefinition);

            _source = source;
            _filterFunc = filterFunc;
            _initStateAction = initStateAction;
            _inputSchemaDefinition = inputSchemaDefinition;
            _typedSource = TypedCursorable<TSrc>.Create(Host, Source, false, inputSchemaDefinition);

            var outSchema = InternalSchemaDefinition.Create(typeof(TDst), outputSchemaDefinition);
            _addedSchema = outSchema;
            _bindings = new ColumnBindings(Data.Schema.Create(Source.Schema), DataViewConstructionUtils.GetSchemaColumns(outSchema));
        }

        /// <summary>
        /// The 'reapply' constructor.
        /// </summary>
        private StatefulFilterTransform(IHostEnvironment env, StatefulFilterTransform<TSrc, TDst, TState> transform, IDataView newSource)
            : base(env, RegistrationName, transform)
        {
            Host.AssertValue(transform);
            Host.AssertValue(newSource);
            _source = newSource;
            _filterFunc = transform._filterFunc;
            _typedSource = TypedCursorable<TSrc>.Create(Host, newSource, false, transform._inputSchemaDefinition);

            _addedSchema = transform._addedSchema;
            _bindings = new ColumnBindings(Data.Schema.Create(newSource.Schema), DataViewConstructionUtils.GetSchemaColumns(_addedSchema));
        }

        public bool CanShuffle { get { return false; } }

        public Schema Schema => _bindings.Schema;

        public long? GetRowCount(bool lazy = true)
        {
            // REVIEW: currently stateful map is implemented via filter, and this is sub-optimal.
            return null;
        }

        public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
        {
            Host.CheckValue(predicate, nameof(predicate));
            Host.CheckValueOrNull(rand);

            var activeInputs = _bindings.GetActiveInput(predicate);
            Func<int, bool> srcPredicate = c => activeInputs[c];

            var input = _typedSource.GetCursor(srcPredicate, rand == null ? (int?)null : rand.Next());
            return new Cursor(this, input, predicate);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            Contracts.CheckValue(predicate, nameof(predicate));
            Contracts.CheckParam(n >= 0, nameof(n));
            Contracts.CheckValueOrNull(rand);

            // This transform is stateful, its contract is to allocate exactly one state object per cursor and call the filter function
            // on every row in sequence. Therefore, parallel cursoring is not possible.
            consolidator = null;
            return new[] { GetRowCursor(predicate, rand) };
        }

        public IDataView Source
        {
            get { return _source; }
        }

        public IDataTransform ApplyToData(IHostEnvironment env, IDataView newSource)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(newSource, nameof(newSource));
            return new StatefulFilterTransform<TSrc, TDst, TState>(env, this, newSource);
        }

        private sealed class Cursor : RootCursorBase, IRowCursor
        {
            private readonly StatefulFilterTransform<TSrc, TDst, TState> _parent;

            private readonly IRowCursor<TSrc> _input;
            // This is used to serve getters for the columns we produce.
            private readonly IRow _appendedRow;

            private readonly TSrc _src;
            private readonly TDst _dst;
            private readonly TState _state;

            private bool _disposed;

            public override long Batch
            {
                get { return _input.Batch; }
            }

            public Cursor(StatefulFilterTransform<TSrc, TDst, TState> parent, IRowCursor<TSrc> input, Func<int, bool> predicate)
                : base(parent.Host)
            {
                Ch.AssertValue(input);
                Ch.AssertValue(predicate);

                _parent = parent;
                _input = input;

                _src = new TSrc();
                _dst = new TDst();
                _state = new TState();

                CursorChannelAttribute.TrySetCursorChannel(_parent.Host, _src, Ch);
                CursorChannelAttribute.TrySetCursorChannel(_parent.Host, _dst, Ch);
                CursorChannelAttribute.TrySetCursorChannel(_parent.Host, _state, Ch);

                if (parent._initStateAction != null)
                    parent._initStateAction(_state);

                var appendedDataView = new DataViewConstructionUtils.SingleRowLoopDataView<TDst>(parent.Host, _parent._addedSchema);
                appendedDataView.SetCurrentRowObject(_dst);

                Func<int, bool> appendedPredicate =
                    col =>
                    {
                        col = _parent._bindings.AddedColumnIndices[col];
                        return predicate(col);
                    };

                _appendedRow = appendedDataView.GetRowCursor(appendedPredicate);
            }

            public override void Dispose()
            {
                if (!_disposed)
                {
                    var disposableState = _state as IDisposable;
                    var disposableSrc = _src as IDisposable;
                    var disposableDst = _dst as IDisposable;
                    if (disposableState != null)
                        disposableState.Dispose();
                    if (disposableSrc != null)
                        disposableSrc.Dispose();
                    if (disposableDst != null)
                        disposableDst.Dispose();

                    _input.Dispose();
                    base.Dispose();
                    _disposed = true;
                }
            }

            public override ValueGetter<UInt128> GetIdGetter()
            {
                return _input.GetIdGetter();
            }

            protected override bool MoveNextCore()
            {
                bool isAccepted = false;
                while (!isAccepted)
                {
                    if (!_input.MoveNext())
                        return false;
                    RunLambda(out isAccepted);
                }
                return true;
            }

            private void RunLambda(out bool isRowAccepted)
            {
                _input.FillValues(_src);
                // REVIEW: what if this throws? Maybe swallow the exception?
                isRowAccepted = _parent._filterFunc(_src, _dst, _state);
            }

            public Schema Schema => _parent._bindings.Schema;

            public bool IsColumnActive(int col)
            {
                Contracts.CheckParam(0 <= col && col < Schema.ColumnCount, nameof(col));
                bool isSrc;
                int iCol = _parent._bindings.MapColumnIndex(out isSrc, col);
                if (isSrc)
                    return _input.IsColumnActive(iCol);
                return _appendedRow.IsColumnActive(iCol);
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Contracts.CheckParam(0 <= col && col < Schema.ColumnCount, nameof(col));
                bool isSrc;
                int iCol = _parent._bindings.MapColumnIndex(out isSrc, col);
                return isSrc ?
                    _input.GetGetter<TValue>(iCol)
                    : _appendedRow.GetGetter<TValue>(iCol);
            }
        }
    }
}
