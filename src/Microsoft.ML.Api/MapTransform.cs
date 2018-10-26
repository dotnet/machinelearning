// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using System;
using System.Linq;

namespace Microsoft.ML.Runtime.Api
{
    /// <summary>
    /// This transform generates additional columns to the provided <see cref="IDataView"/>.
    /// It doesn't change the number of rows, and can be seen as a result of application of the user's function
    /// to every row of the input data.
    /// </summary>
    /// <typeparam name="TSrc">The type that describes what 'source' columns are consumed from the input <see cref="IDataView"/>.</typeparam>
    /// <typeparam name="TDst">The type that describes what new columns are added by this transform.</typeparam>
    public sealed class CustomMappingTransformer<TSrc, TDst> : ITransformer, ICanSaveModel
        where TSrc : class, new()
        where TDst : class, new()
    {
        private readonly IHost _host;
        private readonly Action<TSrc, TDst> _mapAction;
        private readonly InternalSchemaDefinition _addedSchema;
        private readonly string _contractName;

        internal InternalSchemaDefinition AddedSchema => _addedSchema;

        public bool IsRowToRowMapper => true;
        private readonly SchemaDefinition _inputSchemaDefinition;
        /// <summary>
        /// Create a a map transform.
        /// </summary>
        /// <param name="env">The host environment</param>
        /// <param name="mapAction">The action by which we map source to destination columns</param>
        /// <param name="contractName">The name of the action (will be saved to the model).</param>
        /// <param name="inputSchemaDefinition">The schema definition overrides for <typeparamref name="TSrc"/></param>
        /// <param name="outputSchemaDefinition">The schema definition overrides for <typeparamref name="TDst"/></param>
        public CustomMappingTransformer(IHostEnvironment env, Action<TSrc, TDst> mapAction, string contractName,
            SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(CustomMappingTransformer<TSrc, TDst>));
            _host.CheckValue(mapAction, nameof(mapAction));
            _host.CheckValueOrNull(contractName);
            _host.CheckValueOrNull(inputSchemaDefinition);
            _host.CheckValueOrNull(outputSchemaDefinition);

            _mapAction = mapAction;
            _inputSchemaDefinition = inputSchemaDefinition;

            var outSchema = outputSchemaDefinition == null
               ? InternalSchemaDefinition.Create(typeof(TDst), SchemaDefinition.Direction.Write)
               : InternalSchemaDefinition.Create(typeof(TDst), outputSchemaDefinition);

            _contractName = contractName;
            _addedSchema = outSchema;
        }

        public void Save(ModelSaveContext ctx)
        {
            if (_contractName == null)
                throw _host.Except("Empty contract name for a transform: the transform cannot be saved");
            LambdaTransform.SaveCustomTransformer(_host, ctx, _contractName);
        }

        public Schema GetOutputSchema(Schema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var mapper = MakeRowMapper(inputSchema);
            return RowToRowMapperTransform.GetOutputSchema(inputSchema, mapper);
        }

        public IDataView Transform(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            return new RowToRowMapperTransform(_host, input, MakeRowMapper(input.Schema));
        }

        public IRowToRowMapper GetRowToRowMapper(Schema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var simplerMapper = MakeRowMapper(inputSchema);
            return new RowToRowMapperTransform(_host, new EmptyDataView(_host, inputSchema), simplerMapper);
        }

        private IRowMapper MakeRowMapper(Schema schema) => new Mapper(this, schema);

        private sealed class Mapper : IRowMapper
        {
            private readonly IHost _host;
            private readonly Schema _inputSchema;
            private readonly CustomMappingTransformer<TSrc, TDst> _parent;
            private readonly TypedCursorable<TSrc> _typedSrc;

            public Mapper(CustomMappingTransformer<TSrc, TDst> parent, Schema inputSchema)
            {
                Contracts.AssertValue(parent);
                Contracts.AssertValue(inputSchema);

                _host = parent._host.Register(nameof(Mapper));
                _parent = parent;
                _inputSchema = inputSchema;

                var emptyDataView = new EmptyDataView(_host, inputSchema);
                _typedSrc = TypedCursorable<TSrc>.Create(_host, emptyDataView, false, _parent._inputSchemaDefinition);
            }

            public Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                // If no outputs are active, we short-circuit to empty array of getters.
                var result = new Delegate[_parent._addedSchema.Columns.Length];
                if (!Enumerable.Range(0, result.Length).Any(activeOutput))
                    return result;

                var dstRow = new DataViewConstructionUtils.InputRow<TDst>(_host, _parent._addedSchema);
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

            private Delegate GetDstGetter<T>(IRow input, int colIndex, Action refreshAction)
            {
                var getter = input.GetGetter<T>(colIndex);
                ValueGetter<T> combinedGetter = (ref T dst) =>
                {
                    refreshAction();
                    getter(ref dst);
                };
                return combinedGetter;
            }

            public Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
            {
                if (Enumerable.Range(0, _parent._addedSchema.Columns.Length).Any(activeOutput))
                {
                    // If any output column is requested, then we activate all input columns that we need.
                    return _typedSrc.GetDependencies(col => false);
                }
                // Otherwise, we need no input.
                return col => false;
            }

            public Schema.Column[] GetOutputColumns()
            {
                var dstRow = new DataViewConstructionUtils.InputRow<TDst>(_host, _parent._addedSchema);
                // All the output columns of dstRow are our outputs.
                return Enumerable.Range(0, dstRow.Schema.ColumnCount).Select(x => dstRow.Schema[x]).ToArray();
            }

            public void Save(ModelSaveContext ctx)
                => _parent.Save(ctx);
        }
    }

    public sealed class CustomMappingEstimator<TSrc, TDst> : TrivialEstimator<CustomMappingTransformer<TSrc, TDst>>
        where TSrc : class, new()
        where TDst : class, new()
    {
        public CustomMappingEstimator(IHostEnvironment env, Action<TSrc, TDst> mapAction, string contractName,
                SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(CustomMappingEstimator<TSrc, TDst>)),
                 new CustomMappingTransformer<TSrc, TDst>(env, mapAction, contractName, inputSchemaDefinition, outputSchemaDefinition))
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            var addedCols = DataViewConstructionUtils.GetSchemaColumns(Transformer.AddedSchema);
            var addedSchemaShape = SchemaShape.Create(new Schema(addedCols));

            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var addedCol in addedSchemaShape.Columns)
                result[addedCol.Name] = addedCol;

            return new SchemaShape(result.Values);
        }
    }
}
