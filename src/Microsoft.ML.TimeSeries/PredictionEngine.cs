using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Microsoft.ML.TimeSeries
{
    /// <summary>
    /// A class that runs the previously trained model (and the preceding transform pipeline) on the
    /// in-memory data, one example at a time.
    /// This can also be used with trained pipelines that do not end with a predictor: in this case, the
    /// 'prediction' will be just the outcome of all the transformations.
    /// </summary>
    /// <typeparam name="TSrc">The user-defined type that holds the example.</typeparam>
    /// <typeparam name="TDst">The user-defined type that holds the prediction.</typeparam>
    public sealed class PredictionEngine<TSrc, TDst>
        where TSrc : class
        where TDst : class, new()
    {
        private readonly DataViewConstructionUtils.InputRow<TSrc> _inputRow;
        private readonly IRowReadableAs<TDst> _outputRow;
        private readonly IStatefulRowReadableAs<TDst>[] _statefulRows;
        private readonly Action _disposer;

        internal PredictionEngine(IHostEnvironment env, Stream modelStream, bool ignoreMissingColumns,
            SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            : this(env, StreamChecker(env, modelStream), ignoreMissingColumns, inputSchemaDefinition, outputSchemaDefinition)
        {
        }

        private static Func<Schema, IRowToRowMapper> StreamChecker(IHostEnvironment env, Stream modelStream)
        {
            env.CheckValue(modelStream, nameof(modelStream));
            return schema =>
            {
                var pipe = DataViewConstructionUtils.LoadPipeWithPredictor(env, modelStream, new EmptyDataView(env, schema));
                var transformer = new TransformWrapper(env, pipe);
                env.CheckParam(transformer.IsRowToRowMapper, nameof(transformer), "Must be a row to row mapper");
                return transformer.GetRowToRowMapper(schema);
            };
        }

        internal PredictionEngine(IHostEnvironment env, IDataView dataPipe, bool ignoreMissingColumns,
            SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            : this(env, new TransformWrapper(env, env.CheckRef(dataPipe, nameof(dataPipe))), ignoreMissingColumns, inputSchemaDefinition, outputSchemaDefinition)
        {
        }

        internal PredictionEngine(IHostEnvironment env, ITransformer transformer, bool ignoreMissingColumns,
            SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            : this(env, TransformerChecker(env, transformer), ignoreMissingColumns, inputSchemaDefinition, outputSchemaDefinition)
        {
        }

        private static Func<Schema, IRowToRowMapper> TransformerChecker(IExceptionContext ectx, ITransformer transformer)
        {
            ectx.CheckValue(transformer, nameof(transformer));
            ectx.CheckParam(transformer.IsRowToRowMapper, nameof(transformer), "Must be a row to row mapper");
            return transformer.GetRowToRowMapper;
        }

        public void GetStatefulRows(IRow input, IRowToRowMapper mapper, Func<int, bool> active,
            List<IStatefulRow> rows, out Action disposer, out IRow outRow)
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

                outRow = input;
            }

            // For each of the inner mappers, we will be calling their GetRow method, but to do so we need to know
            // what we need from them. The last one will just have the input, but the rest will need to be
            // computed based on the dependencies of the next one in the chain.
            var deps = new Func<int, bool>[innerMappers.Length];
            deps[deps.Length - 1] = active;
            for (int i = deps.Length - 1; i >= 1; --i)
                deps[i - 1] = innerMappers[i].GetDependencies(deps[i]);

            IRow result = input;
            for (int i = 0; i < innerMappers.Length; ++i)
            {
                Action localDisp;
                if (innerMappers[i] is CompositeRowToRowMapper)
                    GetStatefulRows(result, innerMappers[i], deps[i], rows, out localDisp, out result);
                else
                    result = innerMappers[i].GetRow(result, deps[i], out localDisp);

                if (result is IStatefulRow)
                    rows.Add((IStatefulRow)input);

                if (localDisp != null)
                {
                    if (disposer == null)
                        disposer = localDisp;
                    else
                        disposer = localDisp + disposer;
                    // We want the last disposer to be called first, so the order of the addition here is important.
                }
            }

            outRow = input;
        }

        private PredictionEngine(IHostEnvironment env, Func<Schema, IRowToRowMapper> makeMapper, bool ignoreMissingColumns,
                 SchemaDefinition inputSchemaDefinition, SchemaDefinition outputSchemaDefinition)
        {
            Contracts.CheckValue(env, nameof(env));
            env.AssertValue(makeMapper);

            _inputRow = DataViewConstructionUtils.CreateInputRow<TSrc>(env, inputSchemaDefinition);
            var mapper = makeMapper(_inputRow.Schema);

            List<IStatefulRow> rows = new List<IStatefulRow>();
            if (mapper is CompositeRowToRowMapper)
                GetStatefulRows(_inputRow, mapper, col => true, rows, out _disposer, out var outRow);

            var cursorable = TypedCursorable<TDst>.Create(env, new EmptyDataView(env, mapper.Schema), ignoreMissingColumns, outputSchemaDefinition);
            var outputRow = mapper.GetRow(_inputRow, col => true, out _disposer);

            if (rows.Count == 0 && outputRow is IStatefulRow)
                rows.Add((IStatefulRow)outputRow);

            _statefulRows = rows.Select(row => cursorable.GetRow(row)).ToArray();

            _outputRow = cursorable.GetRow(outputRow);
        }

        ~PredictionEngine()
        {
            _disposer?.Invoke();
        }

        /// <summary>
        /// Run prediction pipeline on one example.
        /// </summary>
        /// <param name="example">The example to run on.</param>
        /// <returns>The result of prediction. A new object is created for every call.</returns>
        public TDst Predict(TSrc example)
        {
            var result = new TDst();
            Predict(example, ref result);
            return result;
        }

        /// <summary>
        /// Run prediction pipeline on one example.
        /// </summary>
        /// <param name="example">The example to run on.</param>
        /// <param name="prediction">The object to store the prediction in. If it's <c>null</c>, a new one will be created, otherwise the old one
        /// is reused.</param>
        public void Predict(TSrc example, ref TDst prediction)
        {
            Contracts.CheckValue(example, nameof(example));
            _inputRow.ExtractValues(example);
            if (prediction == null)
                prediction = new TDst();

            foreach (var row in _statefulRows)
                row.PingValues(prediction);

            _outputRow.FillValues(prediction);
        }
    }

    /// <summary>
    /// A prediction engine class, that takes instances of <typeparamref name="TSrc"/> through
    /// the transformer pipeline and produces instances of <typeparamref name="TDst"/> as outputs.
    /// </summary>
    public sealed class PredictionFunction<TSrc, TDst>
                where TSrc : class
                where TDst : class, new()
    {
        private readonly PredictionEngine<TSrc, TDst> _engine;

        /// <summary>
        /// Create an instance of <see cref="PredictionFunction{TSrc, TDst}"/>.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="transformer">The model (transformer) to use for prediction.</param>
        public PredictionFunction(IHostEnvironment env, ITransformer transformer)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(transformer, nameof(transformer));

            IDataView dv = env.CreateDataView(new TSrc[0]);
            _engine = env.CreateTimeSeriesPredictionEngine<TSrc, TDst>(transformer);
        }

        /// <summary>
        /// Perform one prediction using the model.
        /// </summary>
        /// <param name="example">The object that holds values to predict from.</param>
        /// <returns>The object populated with prediction results.</returns>
        public TDst Predict(TSrc example) => _engine.Predict(example);

        /// <summary>
        /// Perform one prediction using the model.
        /// Reuses the provided prediction object, which is more efficient in high-load scenarios.
        /// </summary>
        /// <param name="example">The object that holds values to predict from.</param>
        /// <param name="prediction">The object to store the predictions in. If it's <c>null</c>, a new object is created,
        /// otherwise the provided object is used.</param>
        public void Predict(TSrc example, ref TDst prediction) => _engine.Predict(example, ref prediction);
    }

    public static class PredictionFunctionExtensions
    {
        public static PredictionEngine<TSrc, TDst> CreateTimeSeriesPredictionEngine<TSrc, TDst>(this IHostEnvironment env, ITransformer transformer,
    bool ignoreMissingColumns = false, SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
    where TSrc : class
    where TDst : class, new()
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(transformer, nameof(transformer));
            env.CheckValueOrNull(inputSchemaDefinition);
            env.CheckValueOrNull(outputSchemaDefinition);
            return new PredictionEngine<TSrc, TDst>(env, transformer, ignoreMissingColumns, inputSchemaDefinition, outputSchemaDefinition);
        }

        /// <summary>
        /// Create an instance of the 'prediction function', or 'prediction machine', from a model
        /// denoted by <paramref name="transformer"/>.
        /// It will be accepting instances of <typeparamref name="TSrc"/> as input, and produce
        /// instances of <typeparamref name="TDst"/> as output.
        /// </summary>
        public static PredictionFunction<TSrc, TDst> MakeTimeSeriesPredictionFunction<TSrc, TDst>(this ITransformer transformer, IHostEnvironment env)
                where TSrc : class
                where TDst : class, new()
            => new PredictionFunction<TSrc, TDst>(env, transformer);
    }
}
