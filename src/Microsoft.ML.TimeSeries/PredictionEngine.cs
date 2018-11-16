using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;

namespace Microsoft.ML.TimeSeries
{
    public interface IStatefulRow : IRow
    {
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
        private Action[][] _pingers;

        internal TimeSeriesPredictionEngine(IHostEnvironment env, Stream modelStream, bool ignoreMissingColumns,
            SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            : base(env, modelStream, ignoreMissingColumns, inputSchemaDefinition, outputSchemaDefinition)
        {
        }

        internal TimeSeriesPredictionEngine(IHostEnvironment env, IDataView dataPipe, bool ignoreMissingColumns,
            SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            : base(env, dataPipe, ignoreMissingColumns, inputSchemaDefinition, outputSchemaDefinition)
        {
        }

        internal TimeSeriesPredictionEngine(IHostEnvironment env, ITransformer transformer, bool ignoreMissingColumns,
            SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            : base(env, transformer, ignoreMissingColumns, inputSchemaDefinition, outputSchemaDefinition)
        {
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

        private Action[][] CreatePingers(List<IStatefulRow> rows)
        {
            List<Action[]> pingers = new List<Action[]>();
            foreach(var row in rows)
            {
                var list = new List<Action>();
                for (int i = 0; i < row.Schema.ColumnCount; i++)
                {
                    var colType = row.Schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.TimeSeriesColumn, i);
                    if (colType != null)
                    {
                        colType = row.Schema.GetColumnType(i);
                        MethodInfo meth = ((Func<IRow, int, Action>)CreateGetter<int>).GetMethodInfo().
                            GetGenericMethodDefinition().MakeGenericMethod(colType.RawType);

                        list.Add((Action)meth.Invoke(this, new object[] { row, i }));
                    }
                }

                pingers.Add(list.ToArray());
            }

            return pingers.ToArray();
        }

        internal override void PredictionEngineCore(IHostEnvironment env, DataViewConstructionUtils.InputRow<TSrc> inputRow, IRowToRowMapper mapper, bool ignoreMissingColumns,
                 SchemaDefinition inputSchemaDefinition, SchemaDefinition outputSchemaDefinition, out Action disposer, out IRowReadableAs<TDst> outputRow)
        {

            List<IStatefulRow> rows = new List<IStatefulRow>();
            if (mapper is CompositeRowToRowMapper)
                GetStatefulRows(inputRow, mapper, col => true, rows, out disposer, out var outRow);

            var cursorable = TypedCursorable<TDst>.Create(env, new EmptyDataView(env, mapper.Schema), ignoreMissingColumns, outputSchemaDefinition);
            var outputRowLocal = mapper.GetRow(inputRow, col => true, out disposer);

            if (rows.Count == 0 && outputRowLocal is IStatefulRow)
                rows.Add((IStatefulRow)outputRowLocal);

            _pingers = CreatePingers(rows);
            outputRow = cursorable.GetRow(outputRowLocal);
        }

        private static Action CreateGetter<T>(IRow input, int col)
        {
            var getter = input.GetGetter<T>(col);
            T value = default(T);
            return () =>
            {
                getter(ref value);
            };
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

            foreach (var row in _pingers)
                foreach(var pinger in row)
                    pinger();

            FillValues(prediction);
        }
    }

    public sealed class TimeSeriesPredictionFunction<TSrc, TDst> : PredictionFunctionBase<TSrc, TDst>
         where TSrc : class
         where TDst : class, new()
    {
        public TimeSeriesPredictionFunction(IHostEnvironment env, ITransformer transformer) : base(env, transformer) { }
        internal override void CreatePredictionEngine(IHostEnvironment env, ITransformer transformer, out PredictionEngineBase<TSrc, TDst> engine) =>
            engine = env.CreateTimeSeriesPredictionEngine<TSrc, TDst>(transformer);
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
