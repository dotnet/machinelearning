﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Microsoft.ML.TimeSeries
{
    internal interface IStatefulRowToRowMapper : IRowToRowMapper
    {
    }

    internal interface IStatefulTransformer : ITransformer
    {
        IRowToRowMapper GetStatefulRowToRowMapper(Schema inputSchema);

        IStatefulTransformer Clone();
    }

    internal abstract class StatefulRow : Row
    {
        public abstract Action<long> GetPinger();
    }

    internal interface IStatefulRowMapper : IRowMapper
    {
        void CloneState();

        Action<long> CreatePinger(Row input, Func<int, bool> activeOutput, out Action disposer);
    }

    /// <summary>
    /// A class that runs the previously trained model (and the preceding transform pipeline) on the
    /// in-memory data, one example at a time.
    /// This can also be used with trained pipelines that do not end with a predictor: in this case, the
    /// 'prediction' will be just the outcome of all the transformations.
    /// </summary>
    /// <typeparam name="TSrc">The user-defined type that holds the example.</typeparam>
    /// <typeparam name="TDst">The user-defined type that holds the prediction.</typeparam>
    public sealed class TimeSeriesPredictionFunction<TSrc, TDst> : PredictionEngineBase<TSrc, TDst>
        where TSrc : class
        where TDst : class, new()
    {
        private Action<long> _pinger;
        private long _rowPosition;
        private ITransformer InputTransformer { get; set; }

        /// <summary>
        /// Checkpoints <see cref="TimeSeriesPredictionFunction{TSrc, TDst}"/> to disk with the updated
        /// state.
        /// </summary>
        /// <param name="env">Usually <see cref="MLContext"/>.</param>
        /// <param name="modelPath">Path to file on disk where the updated model needs to be saved.</param>
        public void CheckPoint(IHostEnvironment env, string modelPath)
        {
            using (var file = File.Create(modelPath))
                if (Transformer is ITransformerChainAccessor )
                {

                    new TransformerChain<ITransformer>
                    (((ITransformerChainAccessor )Transformer).Transformers,
                    ((ITransformerChainAccessor )Transformer).Scopes).SaveTo(env, file);
                }
                else
                    Transformer.SaveTo(env, file);
        }

        private static ITransformer CloneTransformers(ITransformer transformer)
        {
            ITransformer[] transformersClone = null;
            TransformerScope[] scopeClone = null;
            if (transformer is ITransformerChainAccessor )
            {
                ITransformerChainAccessor  accessor = (ITransformerChainAccessor )transformer;
                transformersClone = accessor.Transformers.Select(x => x).ToArray();
                scopeClone = accessor.Scopes.Select(x => x).ToArray();
                int index = 0;
                foreach (var xf in transformersClone)
                    transformersClone[index++] = xf is IStatefulTransformer ? ((IStatefulTransformer)xf).Clone() : xf;

                return new TransformerChain<ITransformer>(transformersClone, scopeClone);
            }
            else
                return transformer is IStatefulTransformer ? ((IStatefulTransformer)transformer).Clone() : transformer;
        }

        public TimeSeriesPredictionFunction(IHostEnvironment env, ITransformer transformer, bool ignoreMissingColumns,
            SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null) :
            base(env, CloneTransformers(transformer), ignoreMissingColumns, inputSchemaDefinition, outputSchemaDefinition)
        {
        }

        internal Row GetStatefulRows(Row input, IRowToRowMapper mapper, Func<int, bool> active,
            List<StatefulRow> rows, out Action disposer)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckValue(active, nameof(active));

            disposer = null;
            IRowToRowMapper[] innerMappers = new IRowToRowMapper[0];
            if (mapper is CompositeRowToRowMapper)
                innerMappers = ((CompositeRowToRowMapper)mapper).InnerMappers;

            if (innerMappers.Length == 0)
            {
                bool differentActive = false;
                for (int c = 0; c < input.Schema.ColumnCount; ++c)
                {
                    bool wantsActive = active(c);
                    bool isActive = input.IsColumnActive(c);
                    differentActive |= wantsActive != isActive;

                    if (wantsActive && !isActive)
                        throw Contracts.ExceptParam(nameof(input), $"Mapper required column '{input.Schema.GetColumnName(c)}' active but it was not.");
                }

                var row = mapper.GetRow(input, active, out disposer);
                if (row is StatefulRow statefulRow)
                    rows.Add(statefulRow);

                return row;
            }

            // For each of the inner mappers, we will be calling their GetRow method, but to do so we need to know
            // what we need from them. The last one will just have the input, but the rest will need to be
            // computed based on the dependencies of the next one in the chain.
            var deps = new Func<int, bool>[innerMappers.Length];
            deps[deps.Length - 1] = active;
            for (int i = deps.Length - 1; i >= 1; --i)
                deps[i - 1] = innerMappers[i].GetDependencies(deps[i]);

            Row result = input;
            for (int i = 0; i < innerMappers.Length; ++i)
            {
                Action localDisp;
                result = GetStatefulRows(result, innerMappers[i], deps[i], rows, out localDisp);
                if (result is StatefulRow statefulResult)
                    rows.Add(statefulResult);

                if (localDisp != null)
                {
                    if (disposer == null)
                        disposer = localDisp;
                    else
                        disposer = localDisp + disposer;
                    // We want the last disposer to be called first, so the order of the addition here is important.
                }
            }

            return result;
        }

        private Action<long> CreatePinger(List<StatefulRow> rows)
        {
            if (rows.Count == 0)
                return position => { };
            Action<long> pinger = null;
            foreach (var row in rows)
                pinger += row.GetPinger();
            return pinger;
        }

        internal override void PredictionEngineCore(IHostEnvironment env, DataViewConstructionUtils.InputRow<TSrc> inputRow, IRowToRowMapper mapper, bool ignoreMissingColumns,
                 SchemaDefinition inputSchemaDefinition, SchemaDefinition outputSchemaDefinition, out Action disposer, out IRowReadableAs<TDst> outputRow)
        {
            List<StatefulRow> rows = new List<StatefulRow>();
            Row outputRowLocal = outputRowLocal = GetStatefulRows(inputRow, mapper, col => true, rows, out disposer);
            var cursorable = TypedCursorable<TDst>.Create(env, new EmptyDataView(env, mapper.OutputSchema), ignoreMissingColumns, outputSchemaDefinition);
            _pinger = CreatePinger(rows);
            outputRow = cursorable.GetRow(outputRowLocal);
        }

        private bool IsRowToRowMapper(ITransformer transformer)
        {
            if (transformer is ITransformerChainAccessor )
                return ((ITransformerChainAccessor )transformer).Transformers.All(t => t.IsRowToRowMapper || t is IStatefulTransformer);
            else
                return transformer.IsRowToRowMapper || transformer is IStatefulTransformer;
        }

        private IRowToRowMapper GetRowToRowMapper(Schema inputSchema)
        {
            Contracts.CheckValue(inputSchema, nameof(inputSchema));
            Contracts.Check(IsRowToRowMapper(InputTransformer), nameof(GetRowToRowMapper) +
                " method called despite " + nameof(IsRowToRowMapper) + " being false. or transformer not being " + nameof(IStatefulTransformer));

            if (!(InputTransformer is ITransformerChainAccessor ))
                if (InputTransformer is IStatefulTransformer)
                    return ((IStatefulTransformer)InputTransformer).GetStatefulRowToRowMapper(inputSchema);
                else
                    return InputTransformer.GetRowToRowMapper(inputSchema);

            Contracts.Check(InputTransformer is ITransformerChainAccessor );

            var transformers = ((ITransformerChainAccessor )InputTransformer).Transformers;
            IRowToRowMapper[] mappers = new IRowToRowMapper[transformers.Length];
            Schema schema = inputSchema;
            for (int i = 0; i < mappers.Length; ++i)
            {
                if (transformers[i] is IStatefulTransformer)
                    mappers[i] = ((IStatefulTransformer)transformers[i]).GetStatefulRowToRowMapper(schema);
                else
                    mappers[i] = transformers[i].GetRowToRowMapper(schema);

                schema = mappers[i].OutputSchema;
            }
            return new CompositeRowToRowMapper(inputSchema, mappers);
        }

        protected override Func<Schema, IRowToRowMapper> TransformerChecker(IExceptionContext ectx, ITransformer transformer)
        {
            ectx.CheckValue(transformer, nameof(transformer));
            ectx.CheckParam(IsRowToRowMapper(transformer), nameof(transformer), "Must be a row to row mapper or " + nameof(IStatefulTransformer));
            InputTransformer = transformer;
            return GetRowToRowMapper;
        }

        /// <summary>
        /// Run prediction pipeline on one example.
        /// </summary>
        /// <param name="example">The example to run on.</param>
        /// <param name="prediction">The object to store the prediction in. If it's <c>null</c>, a new one will be created, otherwise the old one
        /// is reused.</param>
        public override void Predict(TSrc example, ref TDst prediction)
        {
            Contracts.CheckValue(example, nameof(example));
            ExtractValues(example);
            if (prediction == null)
                prediction = new TDst();

            // Update state.
            _pinger(_rowPosition);

            // Predict.
            FillValues(prediction);

            _rowPosition++;
        }
    }

    public static class PredictionFunctionExtensions
    {
        /// <summary>
        /// <see cref="TimeSeriesPredictionFunction{TSrc, TDst}"/> creates a prediction function/engine for a time series pipeline
        /// It updates the state of time series model with observations seen at prediction phase and allows checkpointing the model.
        /// </summary>
        /// <typeparam name="TSrc">Class describing input schema to the model.</typeparam>
        /// <typeparam name="TDst">Class describing the output schema of the prediction.</typeparam>
        /// <param name="transformer">The time series pipeline in the form of a <see cref="ITransformer"/>.</param>
        /// <param name="env">Usually <see cref="MLContext"/></param>
        /// <param name="ignoreMissingColumns">To ignore missing columns. Default is false.</param>
        /// <param name="inputSchemaDefinition">Input schema definition. Default is null.</param>
        /// <param name="outputSchemaDefinition">Output schema definition. Default is null.</param>
        /// <p>Example code can be found by searching for <i>TimeSeriesPredictionFunction</i> in <a href='https://github.com/dotnet/machinelearning'>ML.NET.</a></p>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[MF](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/IidSpikeDetectorTransform.cs)]
        /// [!code-csharp[MF](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/IidChangePointDetectorTransform.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static TimeSeriesPredictionFunction<TSrc, TDst> CreateTimeSeriesPredictionFunction<TSrc, TDst>(this ITransformer transformer, IHostEnvironment env,
            bool ignoreMissingColumns = false, SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            where TSrc : class
            where TDst : class, new()
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(transformer, nameof(transformer));
            env.CheckValueOrNull(inputSchemaDefinition);
            env.CheckValueOrNull(outputSchemaDefinition);
            return new TimeSeriesPredictionFunction<TSrc, TDst>(env, transformer, ignoreMissingColumns, inputSchemaDefinition, outputSchemaDefinition);
        }
    }
}
