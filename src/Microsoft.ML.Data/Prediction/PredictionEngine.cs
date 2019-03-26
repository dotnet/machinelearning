// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML
{

    /// <summary>
    /// Utility class to run the pipeline to completion and produce a strongly-typed IEnumerable as a result.
    /// Doesn't allocate memory for every row: instead, yields the same row object on every step.
    /// </summary>
    internal sealed class PipeEngine<TDst>
        where TDst : class, new()
    {
        private readonly ICursorable<TDst> _cursorablePipe;
        private long _counter;

        internal PipeEngine(IHostEnvironment env, IDataView pipe, bool ignoreMissingColumns, SchemaDefinition schemaDefinition = null)
        {
            Contracts.AssertValue(env);
            env.AssertValue(pipe);
            env.AssertValueOrNull(schemaDefinition);

            _cursorablePipe = env.AsCursorable<TDst>(pipe, ignoreMissingColumns, schemaDefinition);
            _counter = 0;
        }

        public IEnumerable<TDst> RunPipe(bool reuseRowObject)
        {
            var curCounter = _counter;
            using (var cursor = _cursorablePipe.GetCursor())
            {
                TDst row = null;
                while (cursor.MoveNext())
                {
                    if (!reuseRowObject || row == null)
                        row = new TDst();

                    cursor.FillValues(row);
                    yield return row;
                    if (curCounter != _counter)
                        throw Contracts.Except("An attempt was made to keep iterating after the pipe has been reset.");
                }
            }
        }

        public void Reset()
        {
            _counter++;
        }
    }

    public sealed class PredictionEngine<TSrc, TDst> : PredictionEngineBase<TSrc, TDst>
       where TSrc : class
       where TDst : class, new()
    {
        internal PredictionEngine(IHostEnvironment env, ITransformer transformer, bool ignoreMissingColumns,
            SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            : base(env, transformer, ignoreMissingColumns, inputSchemaDefinition, outputSchemaDefinition)
        {
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

            FillValues(prediction);
        }
    }

    /// <summary>
    /// A class that runs the previously trained model (and the preceding transform pipeline) on the
    /// in-memory data, one example at a time.
    /// This can also be used with trained pipelines that do not end with a predictor: in this case, the
    /// 'prediction' will be just the outcome of all the transformations.
    /// </summary>
    /// <typeparam name="TSrc">The user-defined type that holds the example.</typeparam>
    /// <typeparam name="TDst">The user-defined type that holds the prediction.</typeparam>
    public abstract class PredictionEngineBase<TSrc, TDst> : IDisposable
        where TSrc : class
        where TDst : class, new()
    {
        private readonly DataViewConstructionUtils.InputRow<TSrc> _inputRow;
        private readonly IRowReadableAs<TDst> _outputRow;
        private readonly Action _disposer;
        private bool _disposed;

        /// <summary>
        /// Provides output schema.
        /// </summary>
        public DataViewSchema OutputSchema { get; }

        [BestFriend]
        private protected ITransformer Transformer { get; }

        [BestFriend]
        private static Func<DataViewSchema, IRowToRowMapper> StreamChecker(IHostEnvironment env, Stream modelStream)
        {
            env.CheckValue(modelStream, nameof(modelStream));
            return schema =>
            {
                var pipe = DataViewConstructionUtils.LoadPipeWithPredictor(env, modelStream, new EmptyDataView(env, schema));
                var transformer = new TransformWrapper(env, pipe);
                env.CheckParam(((ITransformer)transformer).IsRowToRowMapper, nameof(transformer), "Must be a row to row mapper");
                return ((ITransformer)transformer).GetRowToRowMapper(schema);
            };
        }

        [BestFriend]
        private protected PredictionEngineBase(IHostEnvironment env, ITransformer transformer, bool ignoreMissingColumns,
            SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
        {
            Contracts.CheckValue(env, nameof(env));
            env.AssertValue(transformer);
            Transformer = transformer;
            var makeMapper = TransformerChecker(env, transformer);
            env.AssertValue(makeMapper);
            _inputRow = DataViewConstructionUtils.CreateInputRow<TSrc>(env, inputSchemaDefinition);
            PredictionEngineCore(env, _inputRow, makeMapper(_inputRow.Schema), ignoreMissingColumns, outputSchemaDefinition, out _disposer, out _outputRow);
            OutputSchema = Transformer.GetOutputSchema(_inputRow.Schema);
        }

        [BestFriend]
        private protected virtual void PredictionEngineCore(IHostEnvironment env, DataViewConstructionUtils.InputRow<TSrc> inputRow,
            IRowToRowMapper mapper, bool ignoreMissingColumns, SchemaDefinition outputSchemaDefinition, out Action disposer, out IRowReadableAs<TDst> outputRow)
        {
            var cursorable = TypedCursorable<TDst>.Create(env, new EmptyDataView(env, mapper.OutputSchema), ignoreMissingColumns, outputSchemaDefinition);
            var outputRowLocal = mapper.GetRow(inputRow, mapper.OutputSchema);
            outputRow = cursorable.GetRow(outputRowLocal);
            disposer = inputRow.Dispose;
        }

        private protected virtual Func<DataViewSchema, IRowToRowMapper> TransformerChecker(IExceptionContext ectx, ITransformer transformer)
        {
            ectx.CheckValue(transformer, nameof(transformer));
            ectx.CheckParam(transformer.IsRowToRowMapper, nameof(transformer), "Must be a row to row mapper");
            return transformer.GetRowToRowMapper;
        }

        public void Dispose()
        {
            Disposing(true);
            GC.SuppressFinalize(this);
        }

        [BestFriend]
        private protected void Disposing(bool disposing)
        {
            if (_disposed)
                return;
            if (disposing)
                _disposer?.Invoke();
            _disposed = true;
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

        [BestFriend]
        private protected void ExtractValues(TSrc example) => _inputRow.ExtractValues(example);

        [BestFriend]
        private protected void FillValues(TDst prediction) => _outputRow.FillValues(prediction);

        /// <summary>
        /// Run prediction pipeline on one example.
        /// </summary>
        /// <param name="example">The example to run on.</param>
        /// <param name="prediction">The object to store the prediction in. If it's <c>null</c>, a new one will be created, otherwise the old one
        /// is reused.</param>
        public abstract void Predict(TSrc example, ref TDst prediction);
    }
}