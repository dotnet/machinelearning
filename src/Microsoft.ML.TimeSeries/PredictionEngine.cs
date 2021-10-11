// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms.TimeSeries
{
    internal interface IStatefulRowToRowMapper : IRowToRowMapper
    {
    }

    internal interface IStatefulTransformer : ITransformer
    {
        /// <summary>
        /// Same as <see cref="ITransformer.GetRowToRowMapper(DataViewSchema)"/> but also supports mechanism to save the state.
        /// </summary>
        /// <param name="inputSchema">The input schema for which we should get the mapper.</param>
        /// <returns>The row to row mapper.</returns>
        IRowToRowMapper GetStatefulRowToRowMapper(DataViewSchema inputSchema);

        /// <summary>
        /// Creates a clone of the transfomer. Used for taking the snapshot of the state.
        /// This is used to create multiple time series with their own state.
        /// </summary>
        /// <returns></returns>
        IStatefulTransformer Clone();
    }

    internal abstract class StatefulRow : DataViewRow
    {
        public abstract Action<PingerArgument> GetPinger();
    }

    internal interface IStatefulRowMapper : IRowMapper
    {
        void CloneState();

        Action<PingerArgument> CreatePinger(DataViewRow input, Func<int, bool> activeOutput, out Action disposer);
    }

    internal class PingerArgument
    {
        public long RowPosition;
        public float? ConfidenceLevel;
        public int? Horizon;
        public bool DontConsumeSource;
    }

    /// <summary>
    /// A class that runs the previously trained model (and the preceding transform pipeline) on the
    /// in-memory data, one example at a time.
    /// This can also be used with trained pipelines that do not end with a predictor: in this case, the
    /// 'prediction' will be just the outcome of all the transformations.
    /// </summary>
    /// <typeparam name="TSrc">The user-defined type that holds the example.</typeparam>
    /// <typeparam name="TDst">The user-defined type that holds the prediction.</typeparam>
    public sealed class TimeSeriesPredictionEngine<TSrc, TDst> : PredictionEngineBase<TSrc, TDst>
        where TSrc : class
        where TDst : class, new()
    {
        private Action<PingerArgument> _pinger;
        private long _rowPosition;
        private ITransformer InputTransformer { get; set; }

        /// <summary>
        /// Checkpoints <see cref="TimeSeriesPredictionEngine{TSrc, TDst}"/> to disk with the updated
        /// state.
        /// </summary>
        /// <param name="env">Usually <see cref="MLContext"/>.</param>
        /// <param name="modelPath">Path to file on disk where the updated model needs to be saved.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// This is an example for checkpointing time series that detects change point using Singular Spectrum Analysis (SSA) model.
        /// [!code-csharp[Checkpoint](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/TimeSeries/DetectChangePointBySsa.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public void CheckPoint(IHostEnvironment env, string modelPath)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckParam(!string.IsNullOrEmpty(modelPath), nameof(modelPath));

            using (var file = File.Create(modelPath))
                CheckPoint(env, file);
        }

        /// <summary>
        /// Checkpoints <see cref="TimeSeriesPredictionEngine{TSrc, TDst}"/> to a <see cref="Stream"/> with the updated
        /// state.
        /// </summary>
        /// <param name="env">Usually <see cref="MLContext"/>.</param>
        /// <param name="stream">Stream where the updated model needs to be saved.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// This is an example for checkpointing time series that detects change point using Singular Spectrum Analysis (SSA) model.
        /// [!code-csharp[Checkpoint](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/TimeSeries/DetectChangePointBySsaStream.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public void CheckPoint(IHostEnvironment env, Stream stream)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckParam(stream != null, nameof(stream));

            if (Transformer is ITransformerChainAccessor transformerChainAccessor)
            {

                new TransformerChain<ITransformer>(transformerChainAccessor.Transformers, transformerChainAccessor.Scopes).SaveTo(env, stream);
            }
            else
                Transformer.SaveTo(env, stream);
        }

        private static ITransformer CloneTransformers(ITransformer transformer)
        {
            ITransformer[] transformersClone = null;
            TransformerScope[] scopeClone = null;
            if (transformer is ITransformerChainAccessor transformerChainAccessor)
            {
                ITransformerChainAccessor accessor = transformerChainAccessor;
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

        /// <summary>
        /// Contructor for creating time series specific prediction engine. It allows the time series model to be updated with the observations
        /// seen at prediction time via <see cref="CheckPoint(IHostEnvironment, string)"/>
        /// </summary>
        public TimeSeriesPredictionEngine(IHostEnvironment env, ITransformer transformer, bool ignoreMissingColumns,
            SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null) :
            base(env, CloneTransformers(transformer), ignoreMissingColumns, inputSchemaDefinition, outputSchemaDefinition)
        {
        }

        /// <summary>
        /// Contructor for creating time series specific prediction engine. It allows the time series model to be updated with the observations
        /// seen at prediction time via <see cref="CheckPoint(IHostEnvironment, string)"/>
        /// </summary>
        internal TimeSeriesPredictionEngine(IHostEnvironment env, ITransformer transformer, PredictionEngineOptions options) :
            base(env, CloneTransformers(transformer), options.IgnoreMissingColumns, options.InputSchemaDefinition, options.OutputSchemaDefinition, options.OwnsTransformer)
        {
        }

        internal DataViewRow GetStatefulRows(DataViewRow input, IRowToRowMapper mapper, IEnumerable<DataViewSchema.Column> activeColumns, List<StatefulRow> rows)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckValue(activeColumns, nameof(activeColumns));

            IRowToRowMapper[] innerMappers = new IRowToRowMapper[0];
            if (mapper is CompositeRowToRowMapper compositeMapper)
                innerMappers = compositeMapper.InnerMappers;

            var activeIndices = new HashSet<int>(activeColumns.Select(c => c.Index));
            if (innerMappers.Length == 0)
            {
                bool differentActive = false;
                for (int c = 0; c < input.Schema.Count; ++c)
                {
                    bool wantsActive = activeIndices.Contains(c);
                    bool isActive = input.IsColumnActive(input.Schema[c]);
                    differentActive |= wantsActive != isActive;

                    if (wantsActive && !isActive)
                        throw Contracts.ExceptParam(nameof(input), $"Mapper required column '{input.Schema[c].Name}' active but it was not.");
                }

                var row = mapper.GetRow(input, activeColumns);
                if (row is StatefulRow statefulRow)
                    rows.Add(statefulRow);
                return row;
            }

            // For each of the inner mappers, we will be calling their GetRow method, but to do so we need to know
            // what we need from them. The last one will just have the input, but the rest will need to be
            // computed based on the dependencies of the next one in the chain.
            var deps = new IEnumerable<DataViewSchema.Column>[innerMappers.Length];
            deps[deps.Length - 1] = activeColumns;
            for (int i = deps.Length - 1; i >= 1; --i)
                deps[i - 1] = innerMappers[i].GetDependencies(deps[i]);

            DataViewRow result = input;
            for (int i = 0; i < innerMappers.Length; ++i)
            {
                result = GetStatefulRows(result, innerMappers[i], deps[i], rows);
                if (result is StatefulRow statefulResult)
                    rows.Add(statefulResult);
            }
            return result;
        }

        internal Action<PingerArgument> CreatePinger(List<StatefulRow> rows)
        {
            if (rows.Count == 0)
                return position => { };
            Action<PingerArgument> pinger = null;
            foreach (var row in rows)
                pinger += row.GetPinger();
            return pinger;
        }

        private protected override void PredictionEngineCore(IHostEnvironment env, DataViewConstructionUtils.InputRow<TSrc> inputRow,
            IRowToRowMapper mapper, bool ignoreMissingColumns, SchemaDefinition outputSchemaDefinition, out Action disposer, out IRowReadableAs<TDst> outputRow)
        {
            List<StatefulRow> rows = new List<StatefulRow>();
            DataViewRow outputRowLocal = outputRowLocal = GetStatefulRows(inputRow, mapper, mapper.OutputSchema, rows);
            var cursorable = TypedCursorable<TDst>.Create(env, new EmptyDataView(env, mapper.OutputSchema), ignoreMissingColumns, outputSchemaDefinition);
            _pinger = CreatePinger(rows);
            disposer = outputRowLocal.Dispose;
            outputRow = cursorable.GetRow(outputRowLocal);
        }

        private bool IsRowToRowMapper(ITransformer transformer)
        {
            if (transformer is ITransformerChainAccessor transformerChainAccessor)
                return transformerChainAccessor.Transformers.All(t => t.IsRowToRowMapper || t is IStatefulTransformer);
            else
                return transformer.IsRowToRowMapper || transformer is IStatefulTransformer;
        }

        private IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        {
            Contracts.CheckValue(inputSchema, nameof(inputSchema));
            Contracts.Check(IsRowToRowMapper(InputTransformer), nameof(GetRowToRowMapper) +
                " method called despite " + nameof(IsRowToRowMapper) + " being false. or transformer not being " + nameof(IStatefulTransformer));

            if (!(InputTransformer is ITransformerChainAccessor))
                if (InputTransformer is IStatefulTransformer statefulTransformer)
                    return statefulTransformer.GetStatefulRowToRowMapper(inputSchema);
                else
                    return InputTransformer.GetRowToRowMapper(inputSchema);

            Contracts.Check(InputTransformer is ITransformerChainAccessor);

            var transformers = ((ITransformerChainAccessor)InputTransformer).Transformers;
            IRowToRowMapper[] mappers = new IRowToRowMapper[transformers.Length];
            DataViewSchema schema = inputSchema;
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

        private protected override Func<DataViewSchema, IRowToRowMapper> TransformerChecker(IExceptionContext ectx, ITransformer transformer)
        {
            ectx.CheckValue(transformer, nameof(transformer));
            ectx.CheckParam(IsRowToRowMapper(transformer), nameof(transformer), "Must be a row to row mapper or " + nameof(IStatefulTransformer));
            InputTransformer = transformer;
            return GetRowToRowMapper;
        }

        /// <summary>
        /// Performs prediction. In the case of forecasting only task <paramref name="example"/> can be left as null.
        /// If <paramref name="example"/> is not null then it could be used to update forecasting models with new obervation.
        /// For anomaly detection the model is always updated with <paramref name="example"/>.
        /// </summary>
        /// <param name="example">Input to the prediction engine.</param>
        /// <param name="prediction">Forecasting/Prediction from the engine.</param>
        /// <param name="horizon">Used to indicate the number of values to forecast.</param>
        /// <param name="confidenceLevel">Used in forecasting model for confidence.</param>
        public void Predict(TSrc example, ref TDst prediction, int? horizon = null, float? confidenceLevel = null)
        {
            if (example != null && prediction != null)
            {
                //Update models and make a prediction after updating.
                Contracts.CheckValue(example, nameof(example));
                ExtractValues(example);

                // Update state.
                _pinger(new PingerArgument()
                {
                    RowPosition = _rowPosition,
                    ConfidenceLevel = confidenceLevel,
                    Horizon = horizon
                });

                // Predict.
                FillValues(prediction);

                _rowPosition++;
            }
            else if (prediction != null)
            {
                //Forecast.

                // Signal all time series models to not fetch src values in getters.
                _pinger(new PingerArgument()
                {
                    RowPosition = _rowPosition,
                    DontConsumeSource = true,
                    ConfidenceLevel = confidenceLevel,
                    Horizon = horizon
                });

                // Predict. The expectation is user has asked for columns that are
                // forecasting columns and hence will not trigger a getter that needs an input.
                FillValues(prediction);
            }
            else if (example != null)
            {
                //Update models.

                //Extract value that needs to propagated to all the models.
                Contracts.CheckValue(example, nameof(example));
                ExtractValues(example);

                // Update state.
                _pinger(new PingerArgument()
                {
                    RowPosition = _rowPosition,
                    ConfidenceLevel = confidenceLevel,
                    Horizon = horizon
                });

                _rowPosition++;
            }
        }

        /// <summary>
        /// Performs prediction. In the case of forecasting only task <paramref name="example"/> can be left as null.
        /// If <paramref name="example"/> is not null then it could be used to update forecasting models with new obervation.
        /// For anomaly detection the model is always updated with <paramref name="example"/>.
        /// </summary>
        /// <param name="example">Input to the prediction engine.</param>
        /// <param name="prediction">Forecasting/Prediction from the engine.</param>
        public override void Predict(TSrc example, ref TDst prediction) => Predict(example, ref prediction);

        /// <summary>
        /// Performs prediction. In the case of forecasting only task <paramref name="example"/> can be left as null.
        /// If <paramref name="example"/> is not null then it could be used to update forecasting models with new obervation.
        /// </summary>
        /// <param name="example">Input to the prediction engine.</param>
        /// <param name="horizon">Number of values to forecast.</param>
        /// <param name="confidenceLevel">Confidence level for forecasting.</param>
        /// <returns>Prediction/Forecasting after the model has been updated with <paramref name="example"/></returns>
        public TDst Predict(TSrc example, int? horizon = null, float? confidenceLevel = null)
        {
            TDst dst = new TDst();
            Predict(example, ref dst, horizon, confidenceLevel);
            return dst;
        }

        /// <summary>
        /// Forecasting only task.
        /// </summary>
        /// <param name="horizon">Number of values to forecast.</param>
        /// <param name="confidenceLevel">Confidence level for forecasting.</param>
        public TDst Predict(int? horizon = null, float? confidenceLevel = null)
        {
            TDst dst = new TDst();
            Predict(null, ref dst, horizon, confidenceLevel);
            return dst;
        }
    }

    public static class PredictionFunctionExtensions
    {
        /// <summary>
        /// <see cref="TimeSeriesPredictionEngine{TSrc, TDst}"/> creates a prediction engine for a time series pipeline.
        /// It updates the state of time series model with observations seen at prediction phase and allows checkpointing the model.
        /// </summary>
        /// <typeparam name="TSrc">Class describing input schema to the model.</typeparam>
        /// <typeparam name="TDst">Class describing the output schema of the prediction.</typeparam>
        /// <param name="transformer">The time series pipeline in the form of a <see cref="ITransformer"/>.</param>
        /// <param name="env">Usually <see cref="MLContext"/></param>
        /// <param name="ignoreMissingColumns">To ignore missing columns. Default is false.</param>
        /// <param name="inputSchemaDefinition">Input schema definition. Default is null.</param>
        /// <param name="outputSchemaDefinition">Output schema definition. Default is null.</param>
        /// <p>Example code can be found by searching for <i>TimeSeriesPredictionEngine</i> in <a href='https://github.com/dotnet/machinelearning'>ML.NET.</a></p>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// This is an example for detecting change point using Singular Spectrum Analysis (SSA) model.
        /// [!code-csharp[MF](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/TimeSeries/DetectChangePointBySsa.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static TimeSeriesPredictionEngine<TSrc, TDst> CreateTimeSeriesEngine<TSrc, TDst>(this ITransformer transformer, IHostEnvironment env,
            bool ignoreMissingColumns = false, SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            where TSrc : class
            where TDst : class, new()
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(transformer, nameof(transformer));
            env.CheckValueOrNull(inputSchemaDefinition);
            env.CheckValueOrNull(outputSchemaDefinition);
            return new TimeSeriesPredictionEngine<TSrc, TDst>(env, transformer, ignoreMissingColumns, inputSchemaDefinition, outputSchemaDefinition);
        }

        /// <summary>
        /// <see cref="TimeSeriesPredictionEngine{TSrc, TDst}"/> creates a prediction engine for a time series pipeline.
        /// It updates the state of time series model with observations seen at prediction phase and allows checkpointing the model.
        /// </summary>
        /// <typeparam name="TSrc">Class describing input schema to the model.</typeparam>
        /// <typeparam name="TDst">Class describing the output schema of the prediction.</typeparam>
        /// <param name="transformer">The time series pipeline in the form of a <see cref="ITransformer"/>.</param>
        /// <param name="env">Usually <see cref="MLContext"/></param>
        /// <param name="options">Advanced configuration options.</param>
        /// <p>Example code can be found by searching for <i>TimeSeriesPredictionEngine</i> in <a href='https://github.com/dotnet/machinelearning'>ML.NET.</a></p>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// This is an example for detecting change point using Singular Spectrum Analysis (SSA) model.
        /// [!code-csharp[MF](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/TimeSeries/DetectChangePointBySsa.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static TimeSeriesPredictionEngine<TSrc, TDst> CreateTimeSeriesEngine<TSrc, TDst>(this ITransformer transformer, IHostEnvironment env,
            PredictionEngineOptions options)
            where TSrc : class
            where TDst : class, new()
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            return new TimeSeriesPredictionEngine<TSrc, TDst>(env, transformer, options);
        }
    }
}
