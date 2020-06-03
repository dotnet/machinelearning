// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// <see cref="ITransformer"/> resulting from fitting an <see cref="StatefulCustomMappingEstimator{TSrc, TState, TDst}"/>.
    /// </summary>
    /// <typeparam name="TSrc">The class defining which columns to take from the incoming data.</typeparam>
    /// <typeparam name="TDst">The class defining which new columns are added to the data.</typeparam>
    /// <typeparam name="TState">The type that describes per-cursor state.</typeparam>
    public sealed class StatefulCustomMappingTransformer<TSrc, TDst, TState> : ITransformer
        where TSrc : class, new()
        where TDst : class, new()
        where TState : class, new()
    {
        private readonly IHost _host;
        private readonly Action<TSrc, TDst, TState> _mapAction;
        private readonly Action<TState> _stateInitAction;
        private readonly string _contractName;
        private readonly string _contractAssembly;

        internal InternalSchemaDefinition AddedSchema { get; }

        /// <summary>
        /// Whether a call to <see cref="ITransformer.GetRowToRowMapper(DataViewSchema)"/> should succeed, on an
        /// appropriate schema.
        /// </summary>
        bool ITransformer.IsRowToRowMapper => true;

        /// <summary>
        /// Create a custom mapping of input columns to output columns.
        /// </summary>
        /// <param name="env">The host environment</param>
        /// <param name="mapAction">The action by which we map source to destination columns</param>
        /// <param name="contractName">The name of the action (will be saved to the model).</param>
        /// <param name="stateInitAction">The action to initialize the state object, that is called once before the cursor is initialized.</param>
        internal StatefulCustomMappingTransformer(IHostEnvironment env, Action<TSrc, TDst, TState> mapAction, string contractName,
            Action<TState> stateInitAction)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(StatefulCustomMappingTransformer<TSrc, TDst, TState>));
            _host.CheckValue(mapAction, nameof(mapAction));
            _host.CheckValue(stateInitAction, nameof(stateInitAction));
            _host.CheckValueOrNull(contractName);

            _mapAction = mapAction;
            _stateInitAction = stateInitAction;

            _contractName = contractName;
            _contractAssembly = _mapAction.Method.DeclaringType.Assembly.FullName;
            AddedSchema = InternalSchemaDefinition.Create(typeof(TDst), SchemaDefinition.Direction.Write);
        }

        void ICanSaveModel.Save(ModelSaveContext ctx) => SaveModel(ctx);

        internal void SaveModel(ModelSaveContext ctx)
        {
            if (_contractName == null)
                throw _host.Except("Empty contract name for a transform: the transform cannot be saved");
            LambdaTransform.SaveCustomTransformer(_host, ctx, _contractName, _contractAssembly);
        }

        /// <summary>
        /// Returns the <see cref="DataViewSchema"/> which would be produced by the transformer applied to
        /// an input data with schema <paramref name="inputSchema"/>.
        /// </summary>
        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var rowToRow = new RowToRowMapper(_host, this, new EmptyDataView(_host, inputSchema));
            return rowToRow.OutputSchema;
        }

        /// <summary>
        /// Take the data in, make transformations, output the data.
        /// Note that <see cref="IDataView"/>'s are lazy, so no actual transformations happen here, just schema validation.
        /// </summary>
        public IDataView Transform(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            return new RowToRowMapper(_host, this, input);
        }

        /// <summary>
        /// Constructs a row-to-row mapper based on an input schema. If <see cref="ITransformer.IsRowToRowMapper"/>
        /// is <c>false</c>, then an exception is thrown. If the <paramref name="inputSchema"/> is in any way
        /// unsuitable for constructing the mapper, an exception is likewise thrown.
        /// </summary>
        IRowToRowMapper ITransformer.GetRowToRowMapper(DataViewSchema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            return new RowToRowMapper(_host, this, new EmptyDataView(_host, inputSchema));
        }

        private sealed class RowToRowMapper : RowToRowMapperTransformBase
        {
            private readonly StatefulCustomMappingTransformer<TSrc, TDst, TState> _parent;
            private readonly ColumnBindings _bindings;
            private readonly TypedCursorable<TSrc> _typedSrc;

            public override DataViewSchema OutputSchema => _bindings.Schema;

            public RowToRowMapper(IHostEnvironment env, StatefulCustomMappingTransformer<TSrc, TDst, TState> parent, IDataView input)
                : base(env, "StatefulCustom", input)
            {
                Host.CheckValue(parent, nameof(parent));

                _parent = parent;

                var dstRow = new DataViewConstructionUtils.InputRow<TDst>(Host, _parent.AddedSchema);
                // All the output columns of dstRow are our outputs.
                var cols = Enumerable.Range(0, dstRow.Schema.Count).Select(x => new DataViewSchema.DetachedColumn(dstRow.Schema[x])).ToArray();

                _bindings = new ColumnBindings(input.Schema, cols);

                _typedSrc = TypedCursorable<TSrc>.Create(Host, input, false, null);
            }

            public override DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
            {
                Func<int, bool> needCol = c => columnsNeeded == null ? false : columnsNeeded.Any(x => x.Index == c);
                var active = Utils.BuildArray(_bindings.Schema.Count, needCol);

                var inputCols = GetDependenciesCore(columnsNeeded);
                var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);

                DataViewRowCursor input;
                if (n > 1 && ShouldUseParallelCursors(predicate) != false)
                {
                    var inputs = Source.GetRowCursorSet(inputCols, n);
                    Host.AssertNonEmpty(inputs);

                    if (inputs.Length != 1)
                    {
                        var cursors = new DataViewRowCursor[inputs.Length];
                        for (int i = 0; i < inputs.Length; i++)
                            cursors[i] = new Cursor(this, inputs[i], active);
                        return cursors;
                    }
                    input = inputs[0];
                }
                else
                    input = Source.GetRowCursor(inputCols);

                return new DataViewRowCursor[] { new Cursor(this, input, active) };
            }

            protected override Delegate[] CreateGetters(DataViewRow input, IEnumerable<DataViewSchema.Column> activeColumns, out Action disp)
            {
                disp = null;
                var getters = new Delegate[_parent.AddedSchema.Columns.Length];

                var dstRow = new DataViewConstructionUtils.InputRow<TDst>(Host, _parent.AddedSchema);
                IRowReadableAs<TSrc> inputRow = _typedSrc.GetRow(input);

                TSrc src = new TSrc();
                TState state = new TState();
                TDst dst = new TDst();

                _parent._stateInitAction(state);
                long lastServedPosition = -1;
                Action refresh = () =>
                {
                    if (lastServedPosition != input.Position)
                    {
                        inputRow.FillValues(src);
                        _parent._mapAction(src, dst, state);
                        dstRow.ExtractValues(dst);

                        lastServedPosition = input.Position;
                    }
                };

                foreach (var col in activeColumns)
                {
                    var iinfo = _bindings.MapColumnIndex(out var isSrc, col.Index);
                    if (isSrc)
                        continue;
                    getters[iinfo] = Utils.MarshalInvoke(GetDstGetter<int>, col.Type.RawType, dstRow, col.Name, refresh);
                }

                return getters;
            }

            private Delegate GetDstGetter<T>(DataViewRow input, string colName, Action refreshAction)
            {
                var getter = input.GetGetter<T>(input.Schema[colName]);
                ValueGetter<T> combinedGetter = (ref T dst) =>
                {
                    refreshAction();
                    getter(ref dst);
                };
                return combinedGetter;
            }

            protected override IEnumerable<DataViewSchema.Column> GetDependenciesCore(IEnumerable<DataViewSchema.Column> dependingColumns)
            {
                var active = new bool[_bindings.InputSchema.Count];
                bool hasActiveOutput = false;
                foreach (var col in dependingColumns)
                {
                    bool isSrc;
                    int index = MapColumnIndex(out isSrc, col.Index);
                    if (isSrc)
                        active[index] = true;
                    else
                        hasActiveOutput = true;
                }

                Func<int, bool> inputPred = c => active[c];
                if (hasActiveOutput)
                {
                    inputPred = _typedSrc.GetDependencies(inputPred);
                }

                return Source.Schema.Where(col => inputPred(col.Index));
            }

            protected override DataViewRowCursor GetRowCursorCore(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
            {
                Func<int, bool> needCol = c => columnsNeeded == null ? false : columnsNeeded.Any(x => x.Index == c);
                var active = Utils.BuildArray(_bindings.Schema.Count, needCol);

                var inputCols = GetDependenciesCore(columnsNeeded);
                var input = Source.GetRowCursor(inputCols, rand);
                return new Cursor(this, input, active);
            }

            protected override int MapColumnIndex(out bool isSrc, int col)
            {
                return _bindings.MapColumnIndex(out isSrc, col);
            }

            protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
            {
                for (int i = 0; i < _bindings.Schema.Count; i++)
                {
                    if (predicate(i))
                    {
                        _bindings.MapColumnIndex(out var isSrc, i);
                        if (!isSrc)
                            return false;
                    }
                }
                return null;
            }

            private protected override void SaveModel(ModelSaveContext ctx)
            {
                _parent.SaveModel(ctx);
            }

            private sealed class Cursor : SynchronizedCursorBase
            {
                private readonly RowToRowMapper _parent;
                private readonly bool[] _active;

                private readonly Delegate[] _getters;

                public override DataViewSchema Schema => _parent.OutputSchema;

                public Cursor(RowToRowMapper parent, DataViewRowCursor input, bool[] active)
                    : base(parent.Host, input)
                {
                    Ch.AssertValue(parent);
                    Ch.Assert(active == null || active.Length == parent.OutputSchema.Count);

                    _parent = parent;
                    _active = active;
                    _getters = new Delegate[parent._parent.AddedSchema.Columns.Length];

                    var dstRow = new DataViewConstructionUtils.InputRow<TDst>(_parent.Host, _parent._parent.AddedSchema);
                    IRowReadableAs<TSrc> inputRow = _parent._typedSrc.GetRow(input);

                    TSrc src = new TSrc();
                    TState state = new TState();
                    TDst dst = new TDst();

                    _parent._parent._stateInitAction(state);
                    long lastServedPosition = -1;
                    Action refresh = () =>
                    {
                        if (lastServedPosition != input.Position)
                        {
                            inputRow.FillValues(src);
                            _parent._parent._mapAction(src, dst, state);
                            dstRow.ExtractValues(dst);

                            lastServedPosition = input.Position;
                        }
                    };

                    for (int i = 0; i < active.Length; i++)
                    {
                        var iinfo = _parent._bindings.MapColumnIndex(out var isSrc, i);
                        if (isSrc)
                            continue;
                        _getters[iinfo] = Utils.MarshalInvoke(_parent.GetDstGetter<int>, _parent._bindings.Schema[i].Type.RawType, dstRow, _parent._bindings.Schema[i].Name, refresh);
                    }
                }

                public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
                {
                    Ch.Check(IsColumnActive(column));

                    bool isSrc;
                    int index = _parent._bindings.MapColumnIndex(out isSrc, column.Index);
                    if (isSrc)
                        return Input.GetGetter<TValue>(Input.Schema[index]);

                    var originFn = _getters[index];
                    Ch.Assert(originFn != null);
                    var fn = originFn as ValueGetter<TValue>;
                    if (fn == null)
                        throw Ch.Except($"Invalid TValue in GetGetter: '{typeof(TValue)}', " +
                            $"expected type: '{originFn.GetType().GetGenericArguments().First()}'.");
                    return fn;
                }

                public override bool IsColumnActive(DataViewSchema.Column column)
                {
                    Ch.Check(column.Index < _parent._bindings.Schema.Count);
                    return _active == null || _active[column.Index];
                }
            }
        }
    }

    /// <summary>
    /// Applies a custom mapping function to the specified input columns, while allowing a per-cursor state. The result will be in output columns.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | No |
    /// | Input column data type | Any |
    /// | Output column data type | Any |
    /// | Exportable to ONNX | No |
    ///
    /// The resulting <xref:Microsoft.ML.Transforms.StatefulCustomMappingTransformer`3> applies a user defined mapping
    /// to one or more input columns and produces one or more output columns. This transformation doesn't change the number of rows,
    /// and can be seen as the result of applying the user's function to every row of the input data.
    ///
    /// In addition to the input and output objects, the provided custom function is given a state object that it can look at and/or modify.
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]></format>
    /// </remarks>
    /// <seealso cref="CustomMappingCatalog.StatefulCustomMapping{TSrc, TDst, TState}(TransformsCatalog, Action{TSrc, TDst, TState}, Action{TState}, string)"/>
    public sealed class StatefulCustomMappingEstimator<TSrc, TDst, TState> : TrivialEstimator<StatefulCustomMappingTransformer<TSrc, TDst, TState>>
        where TSrc : class, new()
        where TDst : class, new()
        where TState : class, new()
    {
        /// <summary>
        /// Create a custom mapping of input columns to output columns.
        /// </summary>
        /// <param name="env">The host environment</param>
        /// <param name="mapAction">The mapping action. This must be thread-safe and free from side effects.</param>
        /// <param name="contractName">The contract name, used by ML.NET for loading the model. If <c>null</c> is specified, such a trained model would not be save-able.</param>
        /// <param name="stateInitAction">The action to initialize the state object, that is called once before the cursor is initialized.</param>
        internal StatefulCustomMappingEstimator(IHostEnvironment env, Action<TSrc, TDst, TState> mapAction, string contractName,
            Action<TState> stateInitAction)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(StatefulCustomMappingEstimator<TSrc, TDst, TState>)),
                 new StatefulCustomMappingTransformer<TSrc, TDst, TState>(env, mapAction, contractName, stateInitAction))
        {
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            var addedCols = DataViewConstructionUtils.GetSchemaColumns(Transformer.AddedSchema);
            var addedSchemaShape = SchemaShape.Create(SchemaExtensions.MakeSchema(addedCols));

            var result = inputSchema.ToDictionary(x => x.Name);
            var inputDef = InternalSchemaDefinition.Create(typeof(TSrc), SchemaDefinition.Direction.Read);
            foreach (var col in inputDef.Columns)
            {
                if (!result.TryGetValue(col.ColumnName, out var column))
                    throw Contracts.ExceptSchemaMismatch(nameof(inputSchema), "input", col.ColumnName);

                SchemaShape.GetColumnTypeShape(col.ColumnType, out var vecKind, out var itemType, out var isKey);
                // Special treatment for vectors: if we expect variable vector, we also allow fixed-size vector.
                if (itemType != column.ItemType || isKey != column.IsKey
                    || vecKind == SchemaShape.Column.VectorKind.Scalar && column.Kind != SchemaShape.Column.VectorKind.Scalar
                    || vecKind == SchemaShape.Column.VectorKind.Vector && column.Kind != SchemaShape.Column.VectorKind.Vector
                    || vecKind == SchemaShape.Column.VectorKind.VariableVector && column.Kind == SchemaShape.Column.VectorKind.Scalar)
                {
                    throw Contracts.ExceptSchemaMismatch(nameof(inputSchema), "input", col.ColumnName, col.ColumnType.ToString(), column.GetTypeString());
                }
            }

            foreach (var addedCol in addedSchemaShape)
                result[addedCol.Name] = addedCol;

            return new SchemaShape(result.Values);
        }
    }
}
