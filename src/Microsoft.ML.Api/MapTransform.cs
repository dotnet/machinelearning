// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Api
{
    /// <summary>
    /// This transform generates additional columns to the provided <see cref="IDataView"/>.
    /// It doesn't change the number of rows, and can be seen as a result of application of the user's function
    /// to every row of the input data.
    /// Similarly to the existing <see cref="IDataTransform"/>'s, this object can be treated as both the 'transformation' algorithm
    /// (which can be then applied to different data by calling <see cref="ApplyToData"/>), and the transformed data (which can
    /// be enumerated upon by calling <c>GetRowCursor</c> or <c>AsCursorable{TRow}</c>).
    /// </summary>
    /// <typeparam name="TSrc">The type that describes what 'source' columns are consumed from the input <see cref="IDataView"/>.</typeparam>
    /// <typeparam name="TDst">The type that describes what new columns are added by this transform.</typeparam>
    internal sealed class MapTransform<TSrc, TDst> : LambdaTransformBase, ITransformTemplate, IRowToRowMapper
        where TSrc : class, new()
        where TDst : class, new()
    {
        private const string RegistrationNameTemplate = "MapTransform<{0}, {1}>";

        private readonly IDataView _source;
        private readonly Action<TSrc, TDst> _mapAction;
        private readonly MergedSchema _schema;

        // Memorized input schema definition. Needed for re-apply.
        private readonly SchemaDefinition _inputSchemaDefinition;
        private readonly TypedCursorable<TSrc> _typedSource;

        private static string RegistrationName { get { return string.Format(RegistrationNameTemplate, typeof(TSrc).FullName, typeof(TDst).FullName); } }

        /// <summary>
        /// Create a a map transform that is savable if <paramref name="saveAction"/> and <paramref name="loadFunc"/> are
        /// not null.
        /// </summary>
        /// <param name="env">The host environment</param>
        /// <param name="source">The dataview upon which we construct the transform</param>
        /// <param name="mapAction">The action by which we map source to destination columns</param>
        /// <param name="saveAction">An action that allows us to save state to the serialization stream. May be
        /// null simultaneously with <paramref name="loadFunc"/>.</param>
        /// <param name="loadFunc">A function that given the serialization stream and a data view, returns
        /// an <see cref="ITransformTemplate"/>. The intent is, this returned object should itself be a
        /// <see cref="MapTransform{TSrc,TDst}"/>, but this is not strictly necessary. This delegate should be
        /// a static non-lambda method that this assembly can legally call. May be null simultaneously with
        /// <paramref name="saveAction"/>.</param>
        /// <param name="inputSchemaDefinition">The schema definition overrides for <typeparamref name="TSrc"/></param>
        /// <param name="outputSchemaDefinition">The schema definition overrides for <typeparamref name="TDst"/></param>
        public MapTransform(IHostEnvironment env, IDataView source, Action<TSrc, TDst> mapAction,
            Action<BinaryWriter> saveAction, LambdaTransform.LoadDelegate loadFunc,
            SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            : base(env, RegistrationName, saveAction, loadFunc)
        {
            Host.AssertValue(source, "source");
            Host.AssertValue(mapAction, "mapAction");
            Host.AssertValueOrNull(inputSchemaDefinition);
            Host.AssertValueOrNull(outputSchemaDefinition);

            _source = source;
            _mapAction = mapAction;
            _inputSchemaDefinition = inputSchemaDefinition;
            _typedSource = TypedCursorable<TSrc>.Create(Host, Source, false, inputSchemaDefinition);
            var outSchema = outputSchemaDefinition == null
               ? InternalSchemaDefinition.Create(typeof(TDst), SchemaDefinition.Direction.Write)
               : InternalSchemaDefinition.Create(typeof(TDst), outputSchemaDefinition);

            _schema = MergedSchema.Create(_source.Schema, outSchema);
        }

        /// <summary>
        /// The 'reapply' constructor.
        /// </summary>
        private MapTransform(IHostEnvironment env, MapTransform<TSrc, TDst> transform, IDataView newSource)
            : base(env, RegistrationName, transform)
        {
            Host.AssertValue(transform);
            Host.AssertValue(newSource);

            _source = newSource;
            _mapAction = transform._mapAction;
            _typedSource = TypedCursorable<TSrc>.Create(Host, newSource, false, transform._inputSchemaDefinition);

            _schema = MergedSchema.Create(newSource.Schema, transform._schema.AddedSchema);
        }

        public bool CanShuffle
        {
            get { return _source.CanShuffle; }
        }

        public ISchema Schema
        {
            get { return _schema; }
        }

        public long? GetRowCount(bool lazy = true)
        {
            return _source.GetRowCount(lazy);
        }

        public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
        {
            Host.CheckValue(predicate, nameof(predicate));
            Host.CheckValueOrNull(rand);

            IRowCursor curs;
            if (DataViewUtils.TryCreateConsolidatingCursor(out curs, this, predicate, Host, rand))
                return curs;

            var activeInputs = _schema.GetActiveInput(predicate);
            Func<int, bool> srcPredicate = c => activeInputs[c];

            var input = _typedSource.GetCursor(srcPredicate, rand == null ? (int?)null : rand.Next());
            return new Cursor(Host, this, input, predicate);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            Host.CheckValue(predicate, nameof(predicate));
            Host.CheckValueOrNull(rand);

            var activeInputs = _schema.GetActiveInput(predicate);
            Func<int, bool> srcPredicate = c => activeInputs[c];

            var inputs = _typedSource.GetCursorSet(out consolidator, srcPredicate, n, rand);
            Host.AssertNonEmpty(inputs);

            var cursors = new IRowCursor[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                cursors[i] = new Cursor(Host, this, inputs[i], predicate);
            return cursors;
        }

        public IDataView Source
        {
            get { return _source; }
        }

        public IDataTransform ApplyToData(IHostEnvironment env, IDataView newSource)
        {
            Contracts.CheckValue(env, nameof(env));
            Contracts.CheckValue(newSource, nameof(newSource));
            return new MapTransform<TSrc, TDst>(env, this, newSource);
        }

        public Func<int, bool> GetDependencies(Func<int, bool> predicate)
        {
            Host.CheckValue(predicate, nameof(predicate));
            var activeInput = _schema.GetActiveInput(predicate);
            Func<int, bool> srcPredicate =
                c =>
                {
                    Host.Check(0 <= c && c < activeInput.Length);
                    return activeInput[c];
                };
            return _typedSource.GetDependencies(srcPredicate);
        }

        public IRow GetRow(IRow input, Func<int, bool> active, out Action disposer)
        {
            Host.CheckValue(input, nameof(input));
            Host.CheckValue(active, nameof(active));
            Host.CheckParam(input.Schema == _source.Schema, nameof(input), "Schema of input row must be the same as the schema the mapper is bound to");

            var src = new TSrc();
            var dst = new TDst();

            bool disposed = false;
            disposer =
                () =>
                {
                    if (!disposed)
                    {
                        (src as IDisposable)?.Dispose();
                        (dst as IDisposable)?.Dispose();
                    }
                    disposed = true;
                };

            return new Row(_typedSource.GetRow(input), this, active, src, dst);
        }

        private IRow GetAppendedRow(Func<int, bool> active, TDst dst)
        {
            // REVIEW: This is quite odd (for a cursor to create an IDataView). Consider cleaning up your
            // programming model for this. Note that you don't use the IDataView, only a cursor around a single row that
            // is owned by this cursor. Seems like that cursor implementation could be decoupled from any IDataView class.
            var appendedDataView = new DataViewConstructionUtils.SingleRowLoopDataView<TDst>(Host, _schema.AddedSchema);
            appendedDataView.SetCurrentRowObject(dst);
            return appendedDataView.GetRowCursor(i => active(_schema.MapIinfoToCol(i)));
        }

        private sealed class Cursor : SynchronizedCursorBase<IRowCursor<TSrc>>, IRowCursor
        {
            private readonly Row _row;

            private readonly TSrc _src;
            private readonly TDst _dst;

            private bool _disposed;

            public Cursor(IHost host, MapTransform<TSrc, TDst> owner, IRowCursor<TSrc> input, Func<int, bool> predicate)
                : base(host, input)
            {
                Ch.AssertValue(owner);
                Ch.AssertValue(input);
                Ch.AssertValue(predicate);

                _src = new TSrc();
                _dst = new TDst();
                _row = new Row(input, owner, predicate, _src, _dst);

                CursorChannelAttribute.TrySetCursorChannel(host, _src, Ch);
                CursorChannelAttribute.TrySetCursorChannel(host, _dst, Ch);
            }

            public ISchema Schema => _row.Schema;

            public bool IsColumnActive(int col)
            {
                return _row.IsColumnActive(col);
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Action isGood =
                    () =>
                    {
                        if (!IsGood)
                            throw Contracts.Except("Getter is called when the cursor is {0}, which is not allowed.", Input.State);
                    };
                return _row.GetGetterCore<TValue>(col, isGood);
            }

            public override void Dispose()
            {
                if (!_disposed)
                {
                    (_src as IDisposable)?.Dispose();
                    (_dst as IDisposable)?.Dispose();
                    base.Dispose();
                    _disposed = true;
                }
            }
        }

        private sealed class Row : IRow
        {
            private readonly ISchema _schema;
            private readonly IRow<TSrc> _input;
            private readonly IRow _appendedRow;
            private readonly bool[] _active;

            private readonly MapTransform<TSrc, TDst> _parent;

            private readonly TSrc _src;
            private readonly TDst _dst;

            private long _lastServedPosition;

            public long Batch => _input.Batch;

            public long Position => _input.Position;

            public ISchema Schema => _schema;

            public Row(IRow<TSrc> input, MapTransform<TSrc, TDst> parent, Func<int, bool> active, TSrc src, TDst dst)
            {
                _input = input;
                _parent = parent;
                _schema = parent.Schema;

                _active = Utils.BuildArray(_schema.ColumnCount, active);
                _src = src;
                _dst = dst;

                _lastServedPosition = -1;
                _appendedRow = _parent.GetAppendedRow(active, _dst);
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                return GetGetterCore<TValue>(col, null);
            }

            public ValueGetter<TValue> GetGetterCore<TValue>(int col, Action checkIsGood)
            {
                bool isSrc;
                int index = _parent._schema.MapColumnIndex(out isSrc, col);
                if (isSrc)
                    return _input.GetGetter<TValue>(index);

                var appendedGetter = _appendedRow.GetGetter<TValue>(index);
                return
                    (ref TValue value) =>
                    {
                        checkIsGood?.Invoke();
                        if (_lastServedPosition != _input.Position)
                        {
                            _input.FillValues(_src);
                            // REVIEW: what if this throws? Maybe swallow the exception?
                            _parent._mapAction(_src, _dst);
                            _lastServedPosition = _input.Position;
                        }

                        appendedGetter(ref value);
                    };
            }

            public ValueGetter<UInt128> GetIdGetter()
            {
                return _input.GetIdGetter();
            }

            public bool IsColumnActive(int col)
            {
                _parent.Host.Check(0 <= col && col < _schema.ColumnCount);
                return _active[col];
            }
        }
    }
}
