using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.TimeSeries
{
    public interface IStatefulTransformer : ITransformer
    {
        IStatefulTransformer Clone();
    }

    public delegate void ValuePinger(ref bool value, int rowPosition);

    public interface IStatefulRow : IRow
    {
        ValuePinger[] GetPingers();
    }

    public interface IStatefulRowMapper : IRowMapper
    {
        Delegate[] CreatePingers(IRow input, Func<int, bool> activeOutput, out Action disposer);
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
        private ValuePinger[][] _pingers;
        private int _rowPosition;
        public ITransformer CheckPoint => Transformer;

        private static ITransformer CloneTransformers(ITransformer transformer)
        {
            ITransformer[] transformersClone = null;
            TransformerScope[] scopeClone = null;
            if (transformer is ITransformerAccessor)
            {
                ITransformerAccessor accessor = (ITransformerAccessor)transformer;
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

        public TimeSeriesPredictionEngine(IHostEnvironment env, ITransformer transformer, bool ignoreMissingColumns,
            SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null) :
            base(env, CloneTransformers(transformer), ignoreMissingColumns, inputSchemaDefinition, outputSchemaDefinition)
        {
        }

        internal override ITransformer ProcessTransformer(ITransformer transformer) => CloneTransformers(transformer);

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
                return;
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
                    rows.Add((IStatefulRow)result);

                if (localDisp != null)
                {
                    if (disposer == null)
                        disposer = localDisp;
                    else
                        disposer = localDisp + disposer;
                    // We want the last disposer to be called first, so the order of the addition here is important.
                }
            }

            outRow = result;
        }

        private ValuePinger[][] CreatePingers(List<IStatefulRow> rows)
        {
            ValuePinger[][] pingers = new ValuePinger[rows.Count][];
            int index = 0;
            foreach(var row in rows)
                pingers[index++] = row.GetPingers();

            return pingers;
        }

        internal override void PredictionEngineCore(IHostEnvironment env, DataViewConstructionUtils.InputRow<TSrc> inputRow, IRowToRowMapper mapper, bool ignoreMissingColumns,
                 SchemaDefinition inputSchemaDefinition, SchemaDefinition outputSchemaDefinition, out Action disposer, out IRowReadableAs<TDst> outputRow)
        {
            IRow outputRowLocal = null;
            List<IStatefulRow> rows = new List<IStatefulRow>();
            if (mapper is CompositeRowToRowMapper)
                GetStatefulRows(inputRow, mapper, col => true, rows, out disposer, out outputRowLocal);
            else
                outputRowLocal = mapper.GetRow(inputRow, col => true, out disposer);

            var cursorable = TypedCursorable<TDst>.Create(env, new EmptyDataView(env, mapper.Schema), ignoreMissingColumns, outputSchemaDefinition);
            if (rows.Count == 0 && outputRowLocal is IStatefulRow)
                rows.Add((IStatefulRow)outputRowLocal);

            _pingers = CreatePingers(rows);
            outputRow = cursorable.GetRow(outputRowLocal);
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

            //Update State.
            bool status = false;
            foreach (var row in _pingers)
                foreach(var pinger in row)
                    pinger(ref status, _rowPosition);

            //Predict.
            FillValues(prediction);

            _rowPosition++;
        }
    }

    public sealed class TimeSeriesPredictionFunction<TSrc, TDst> : PredictionFunctionBase<TSrc, TDst>
         where TSrc : class
         where TDst : class, new()
    {
        public TimeSeriesPredictionFunction(IHostEnvironment env, ITransformer transformer) : base(env, transformer) { }
        private TimeSeriesPredictionEngine<TSrc, TDst> _engine;
        public ITransformer CheckPoint() => _engine.CheckPoint;
        internal override void CreatePredictionEngine(IHostEnvironment env, ITransformer transformer, out PredictionEngineBase<TSrc, TDst> engine) =>
            engine = _engine = env.CreateTimeSeriesPredictionEngine<TSrc, TDst>(transformer);
    }

    public static class PredictionFunctionExtensions
    {
        public static TimeSeriesPredictionEngine<TSrc, TDst> CreateTimeSeriesPredictionEngine<TSrc, TDst>(this IHostEnvironment env, ITransformer transformer,
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
        /// Create an instance of the 'prediction function', or 'prediction machine', from a model
        /// denoted by <paramref name="transformer"/>.
        /// It will be accepting instances of <typeparamref name="TSrc"/> as input, and produce
        /// instances of <typeparamref name="TDst"/> as output.
        /// </summary>
        public static TimeSeriesPredictionFunction<TSrc, TDst> MakeTimeSeriesPredictionFunction<TSrc, TDst>(this ITransformer transformer, IHostEnvironment env)
                where TSrc : class
                where TDst : class, new()
            => new TimeSeriesPredictionFunction<TSrc, TDst>(env, transformer);
    }
}
