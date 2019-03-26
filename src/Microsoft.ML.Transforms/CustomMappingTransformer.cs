// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// This transform generates additional columns to the provided <see cref="IDataView"/>.
    /// It doesn't change the number of rows, and can be seen as a result of application of the user's function
    /// to every row of the input data.
    /// </summary>
    /// <typeparam name="TSrc">The type that describes what 'source' columns are consumed from the input <see cref="IDataView"/>.</typeparam>
    /// <typeparam name="TDst">The type that describes what new columns are added by this transform.</typeparam>
    public sealed class CustomMappingTransformer<TSrc, TDst> : ITransformer
        where TSrc : class, new()
        where TDst : class, new()
    {
        private readonly IHost _host;
        private readonly Action<TSrc, TDst> _mapAction;
        private readonly string _contractName;

        internal InternalSchemaDefinition AddedSchema { get; }
        internal SchemaDefinition InputSchemaDefinition { get; }

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
        /// <param name="inputSchemaDefinition">Additional parameters for schema mapping between <typeparamref name="TSrc"/> and input data.</param>
        /// <param name="outputSchemaDefinition">Additional parameters for schema mapping between <typeparamref name="TDst"/> and output data.</param>
        internal CustomMappingTransformer(IHostEnvironment env, Action<TSrc, TDst> mapAction, string contractName,
            SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(CustomMappingTransformer<TSrc, TDst>));
            _host.CheckValue(mapAction, nameof(mapAction));
            _host.CheckValueOrNull(contractName);
            _host.CheckValueOrNull(inputSchemaDefinition);
            _host.CheckValueOrNull(outputSchemaDefinition);

            _mapAction = mapAction;
            InputSchemaDefinition = inputSchemaDefinition;

            var outSchema = outputSchemaDefinition == null
               ? InternalSchemaDefinition.Create(typeof(TDst), SchemaDefinition.Direction.Write)
               : InternalSchemaDefinition.Create(typeof(TDst), outputSchemaDefinition);

            _contractName = contractName;
            AddedSchema = outSchema;
        }

        void ICanSaveModel.Save(ModelSaveContext ctx) => SaveModel(ctx);

        internal void SaveModel(ModelSaveContext ctx)
        {
            if (_contractName == null)
                throw _host.Except("Empty contract name for a transform: the transform cannot be saved");
            LambdaTransform.SaveCustomTransformer(_host, ctx, _contractName);
        }

        /// <summary>
        /// Returns the <see cref="DataViewSchema"/> which would be produced by the transformer applied to
        /// an input data with schema <paramref name="inputSchema"/>.
        /// </summary>
        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var mapper = MakeRowMapper(inputSchema);
            return RowToRowMapperTransform.GetOutputSchema(inputSchema, mapper);
        }

        /// <summary>
        /// Take the data in, make transformations, output the data.
        /// Note that <see cref="IDataView"/>'s are lazy, so no actual transformations happen here, just schema validation.
        /// </summary>
        public IDataView Transform(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            return new RowToRowMapperTransform(_host, input, MakeRowMapper(input.Schema), MakeRowMapper);
        }

        /// <summary>
        /// Constructs a row-to-row mapper based on an input schema. If <see cref="ITransformer.IsRowToRowMapper"/>
        /// is <c>false</c>, then an exception is thrown. If the <paramref name="inputSchema"/> is in any way
        /// unsuitable for constructing the mapper, an exception is likewise thrown.
        /// </summary>
        IRowToRowMapper ITransformer.GetRowToRowMapper(DataViewSchema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var simplerMapper = MakeRowMapper(inputSchema);
            return new RowToRowMapperTransform(_host, new EmptyDataView(_host, inputSchema), simplerMapper, MakeRowMapper);
        }

        private IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private sealed class Mapper : IRowMapper
        {
            private readonly IHost _host;
            private readonly DataViewSchema _inputSchema;
            private readonly CustomMappingTransformer<TSrc, TDst> _parent;
            private readonly TypedCursorable<TSrc> _typedSrc;

            public Mapper(CustomMappingTransformer<TSrc, TDst> parent, DataViewSchema inputSchema)
            {
                Contracts.AssertValue(parent);
                Contracts.AssertValue(inputSchema);

                _host = parent._host.Register(nameof(Mapper));
                _parent = parent;
                _inputSchema = inputSchema;

                var emptyDataView = new EmptyDataView(_host, inputSchema);
                _typedSrc = TypedCursorable<TSrc>.Create(_host, emptyDataView, false, _parent.InputSchemaDefinition);
            }

            Delegate[] IRowMapper.CreateGetters(DataViewRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                // If no outputs are active, we short-circuit to empty array of getters.
                var result = new Delegate[_parent.AddedSchema.Columns.Length];
                if (!Enumerable.Range(0, result.Length).Any(activeOutput))
                    return result;

                var dstRow = new DataViewConstructionUtils.InputRow<TDst>(_host, _parent.AddedSchema);
                IRowReadableAs<TSrc> inputRow = _typedSrc.GetRow(input);

                TSrc src = new TSrc();
                TDst dst = new TDst();

                long lastServedPosition = -1;
                Action refresh = () =>
                {
                    if (lastServedPosition != input.Position)
                    {
                        inputRow.FillValues(src);
                        _parent._mapAction(src, dst);
                        dstRow.ExtractValues(dst);

                        lastServedPosition = input.Position;
                    }
                };

                for (int i = 0; i < result.Length; i++)
                {
                    if (!activeOutput(i))
                        continue;
                    result[i] = Utils.MarshalInvoke(GetDstGetter<int>, dstRow.Schema[i].Type.RawType, dstRow, i, refresh);
                }
                return result;
            }

            private Delegate GetDstGetter<T>(DataViewRow input, int colIndex, Action refreshAction)
            {
                var getter = input.GetGetter<T>(input.Schema[colIndex]);
                ValueGetter<T> combinedGetter = (ref T dst) =>
                {
                    refreshAction();
                    getter(ref dst);
                };
                return combinedGetter;
            }

            Func<int, bool> IRowMapper.GetDependencies(Func<int, bool> activeOutput)
            {
                if (Enumerable.Range(0, _parent.AddedSchema.Columns.Length).Any(activeOutput))
                {
                    // If any output column is requested, then we activate all input columns that we need.
                    return _typedSrc.GetDependencies(col => false);
                }
                // Otherwise, we need no input.
                return col => false;
            }

            DataViewSchema.DetachedColumn[] IRowMapper.GetOutputColumns()
            {
                var dstRow = new DataViewConstructionUtils.InputRow<TDst>(_host, _parent.AddedSchema);
                // All the output columns of dstRow are our outputs.
                return Enumerable.Range(0, dstRow.Schema.Count).Select(x => new DataViewSchema.DetachedColumn(dstRow.Schema[x])).ToArray();
            }

            void ICanSaveModel.Save(ModelSaveContext ctx)
                => _parent.SaveModel(ctx);

            public ITransformer GetTransformer()
            {
                return _parent;
            }
        }
    }

    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> to define a custom mapping of rows of an <see cref="IDataView"/>.
    /// For usage details, please see <see cref="CustomMappingCatalog.CustomMapping"/>
    /// </summary>
    /// <remarks>
    /// Calling <see cref="IEstimator{TTransformer}.Fit(IDataView)"/> in this estimator, produces an <see cref="CustomMappingTransformer{TSrc, TDst}"/>.
    /// </remarks>
    public sealed class CustomMappingEstimator<TSrc, TDst> : TrivialEstimator<CustomMappingTransformer<TSrc, TDst>>
        where TSrc : class, new()
        where TDst : class, new()
    {
        /// <summary>
        /// Create a custom mapping of input columns to output columns.
        /// </summary>
        /// <param name="env">The host environment</param>
        /// <param name="mapAction">The mapping action. This must be thread-safe and free from side effects.</param>
        /// <param name="contractName">The contract name, used by ML.NET for loading the model. If <c>null</c> is specified, such a trained model would not be save-able.</param>
        /// <param name="inputSchemaDefinition">Additional parameters for schema mapping between <typeparamref name="TSrc"/> and input data.</param>
        /// <param name="outputSchemaDefinition">Additional parameters for schema mapping between <typeparamref name="TDst"/> and output data.</param>
        internal CustomMappingEstimator(IHostEnvironment env, Action<TSrc, TDst> mapAction, string contractName,
                SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(CustomMappingEstimator<TSrc, TDst>)),
                 new CustomMappingTransformer<TSrc, TDst>(env, mapAction, contractName, inputSchemaDefinition, outputSchemaDefinition))
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
            var inputDef = InternalSchemaDefinition.Create(typeof(TSrc), Transformer.InputSchemaDefinition);
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
