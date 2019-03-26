// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
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
            _bindings = new ColumnBindings(Source.Schema, DataViewConstructionUtils.GetSchemaColumns(outSchema));
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
            _bindings = new ColumnBindings(newSource.Schema, DataViewConstructionUtils.GetSchemaColumns(_addedSchema));
        }

        public bool CanShuffle => false;

        DataViewSchema IDataView.Schema => OutputSchema;

        public DataViewSchema OutputSchema => _bindings.Schema;

        public long? GetRowCount()
        {
            // REVIEW: currently stateful map is implemented via filter, and this is sub-optimal.
            return null;
        }

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            Host.CheckValueOrNull(rand);

            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);
            var activeInputs = _bindings.GetActiveInput(predicate);
            Func<int, bool> inputPred = c => activeInputs[c];

            var input = _typedSource.GetCursor(inputPred, rand == null ? (int?)null : rand.Next());
            return new Cursor(this, input, columnsNeeded);
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            Contracts.CheckParam(n >= 0, nameof(n));
            Contracts.CheckValueOrNull(rand);

            // This transform is stateful, its contract is to allocate exactly one state object per cursor and call the filter function
            // on every row in sequence. Therefore, parallel cursoring is not possible.
            return new[] { GetRowCursor(columnsNeeded, rand) };
        }

        public IDataView Source => _source;

        IDataTransform ITransformTemplate.ApplyToData(IHostEnvironment env, IDataView newSource)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(newSource, nameof(newSource));
            return new StatefulFilterTransform<TSrc, TDst, TState>(env, this, newSource);
        }

        private sealed class Cursor : RootCursorBase
        {
            private readonly StatefulFilterTransform<TSrc, TDst, TState> _parent;

            private readonly RowCursor<TSrc> _input;
            // This is used to serve getters for the columns we produce.
            private readonly DataViewRow _appendedRow;

            private readonly TSrc _src;
            private readonly TDst _dst;
            private readonly TState _state;

            private bool _disposed;

            public override long Batch => _input.Batch;

            public Cursor(StatefulFilterTransform<TSrc, TDst, TState> parent, RowCursor<TSrc> input, IEnumerable<DataViewSchema.Column> columnsNeeded)
                : base(parent.Host)
            {
                Ch.AssertValue(input);
                Ch.AssertValue(columnsNeeded);

                _parent = parent;
                _input = input;

                _src = new TSrc();
                _dst = new TDst();
                _state = new TState();

                CursorChannelAttribute.TrySetCursorChannel(_parent.Host, _src, Ch);
                CursorChannelAttribute.TrySetCursorChannel(_parent.Host, _dst, Ch);
                CursorChannelAttribute.TrySetCursorChannel(_parent.Host, _state, Ch);

                parent._initStateAction?.Invoke(_state);

                var appendedDataView = new DataViewConstructionUtils.SingleRowLoopDataView<TDst>(parent.Host, _parent._addedSchema);
                appendedDataView.SetCurrentRowObject(_dst);

                var columnNames = columnsNeeded.Select(c => c.Name);
                _appendedRow = appendedDataView.GetRowCursor(appendedDataView.Schema.Where(c => !c.IsHidden && columnNames.Contains(c.Name)));
            }

            protected override void Dispose(bool disposing)
            {
                if (_disposed)
                    return;
                if (disposing)
                {
                    if (_state is IDisposable disposableState)
                        disposableState.Dispose();
                    if (_src is IDisposable disposableSrc)
                        disposableSrc.Dispose();
                    if (_dst is IDisposable disposableDst)
                        disposableDst.Dispose();
                    _input.Dispose();
                }
                _disposed = true;
                base.Dispose(disposing);
            }

            public override ValueGetter<DataViewRowId> GetIdGetter()
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

            public override DataViewSchema Schema => _parent._bindings.Schema;

            /// <summary>
            /// Returns whether the given column is active in this row.
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                Contracts.CheckParam(column.Index < Schema.Count, nameof(column));
                bool isSrc;
                int iCol = _parent._bindings.MapColumnIndex(out isSrc, column.Index);
                if (isSrc)
                    return _input.IsColumnActive(_input.Schema[iCol]);
                return _appendedRow.IsColumnActive(_appendedRow.Schema[iCol]);
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
                Contracts.CheckParam(column.Index < Schema.Count, nameof(column));
                bool isSrc;
                int iCol = _parent._bindings.MapColumnIndex(out isSrc, column.Index);
                return isSrc ?
                    _input.GetGetter<TValue>(_input.Schema[iCol])
                    : _appendedRow.GetGetter<TValue>(_appendedRow.Schema[iCol]);
            }
        }
    }
}
