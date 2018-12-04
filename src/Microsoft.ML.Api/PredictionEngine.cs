// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.IO;

namespace Microsoft.ML.Runtime.Api
{
    /// <summary>
    /// A class that runs the previously trained model (and the preceding transform pipeline) on the
    /// in-memory data in batch mode.
    /// This can also be used with trained pipelines that do not end with a predictor: in this case, the
    /// 'prediction' will be just the outcome of all the transformations.
    /// </summary>
    /// <typeparam name="TSrc">The user-defined type that holds the example.</typeparam>
    /// <typeparam name="TDst">The user-defined type that holds the prediction.</typeparam>
    public sealed class BatchPredictionEngine<TSrc, TDst>
        where TSrc : class
        where TDst : class, new()
    {
        // The source data view.
        private readonly DataViewConstructionUtils.StreamingDataView<TSrc> _srcDataView;
        // The transformation engine.
        private readonly PipeEngine<TDst> _pipeEngine;

        internal BatchPredictionEngine(IHostEnvironment env, Stream modelStream, bool ignoreMissingColumns,
            SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
        {
            Contracts.AssertValue(env);
            Contracts.AssertValue(modelStream);
            Contracts.AssertValueOrNull(inputSchemaDefinition);
            Contracts.AssertValueOrNull(outputSchemaDefinition);

            // Initialize pipe.
            _srcDataView = DataViewConstructionUtils.CreateFromEnumerable(env, new TSrc[] { }, inputSchemaDefinition);
            var pipe = DataViewConstructionUtils.LoadPipeWithPredictor(env, modelStream, _srcDataView);
            _pipeEngine = new PipeEngine<TDst>(env, pipe, ignoreMissingColumns, outputSchemaDefinition);
        }

        internal BatchPredictionEngine(IHostEnvironment env, IDataView dataPipeline, bool ignoreMissingColumns,
           SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
        {
            Contracts.AssertValue(env);
            Contracts.AssertValue(dataPipeline);
            Contracts.AssertValueOrNull(inputSchemaDefinition);
            Contracts.AssertValueOrNull(outputSchemaDefinition);

            // Initialize pipe.
            _srcDataView = DataViewConstructionUtils.CreateFromEnumerable(env, new TSrc[] { }, inputSchemaDefinition);
            var pipe = ApplyTransformUtils.ApplyAllTransformsToData(env, dataPipeline, _srcDataView);

            _pipeEngine = new PipeEngine<TDst>(env, pipe, ignoreMissingColumns, outputSchemaDefinition);
        }

        /// <summary>
        /// Run the prediction pipe. This will enumerate the <paramref name="examples"/> exactly once,
        /// cache all the examples (by reference) into its internal representation and then run
        /// the transformation pipe.
        /// </summary>
        /// <param name="examples">The examples to run the prediction on.</param>
        /// <param name="reuseRowObjects">If <c>true</c>, the engine will not allocate memory per output, and
        /// the returned <typeparamref name="TDst"/> objects will actually always be the same object. The user is
        /// expected to clone the values himself if needed.</param>
        /// <returns>The <see cref="IEnumerable{TDst}"/> that contains all the pipeline results.</returns>
        public IEnumerable<TDst> Predict(IEnumerable<TSrc> examples, bool reuseRowObjects)
        {
            Contracts.CheckValue(examples, nameof(examples));

            _pipeEngine.Reset();
            _srcDataView.SetData(examples);
            return _pipeEngine.RunPipe(reuseRowObjects);
        }
    }

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

            _cursorablePipe = pipe.AsCursorable<TDst>(env, ignoreMissingColumns, schemaDefinition);
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
    public abstract class PredictionEngineBase<TSrc, TDst>
        where TSrc : class
        where TDst : class, new()
    {
        private readonly DataViewConstructionUtils.InputRow<TSrc> _inputRow;
        private readonly IRowReadableAs<TDst> _outputRow;
        private readonly Action _disposer;
        [BestFriend]
        private protected ITransformer Transformer { get; }

        [BestFriend]
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
            PredictionEngineCore(env, _inputRow, makeMapper(_inputRow.Schema), ignoreMissingColumns, inputSchemaDefinition, outputSchemaDefinition, out _disposer, out _outputRow);
        }

        internal virtual void PredictionEngineCore(IHostEnvironment env, DataViewConstructionUtils.InputRow<TSrc> inputRow, IRowToRowMapper mapper, bool ignoreMissingColumns,
                 SchemaDefinition inputSchemaDefinition, SchemaDefinition outputSchemaDefinition, out Action disposer, out IRowReadableAs<TDst> outputRow)
        {
            var cursorable = TypedCursorable<TDst>.Create(env, new EmptyDataView(env, mapper.OutputSchema), ignoreMissingColumns, outputSchemaDefinition);
            var outputRowLocal = mapper.GetRow(_inputRow, col => true, out disposer);
            outputRow = cursorable.GetRow(outputRowLocal);
        }

        protected virtual Func<Schema, IRowToRowMapper> TransformerChecker(IExceptionContext ectx, ITransformer transformer)
        {
            ectx.CheckValue(transformer, nameof(transformer));
            ectx.CheckParam(transformer.IsRowToRowMapper, nameof(transformer), "Must be a row to row mapper");
            return transformer.GetRowToRowMapper;
        }

        ~PredictionEngineBase()
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

        protected void ExtractValues(TSrc example) => _inputRow.ExtractValues(example);

        protected void FillValues(TDst prediction) => _outputRow.FillValues(prediction);

        /// <summary>
        /// Run prediction pipeline on one example.
        /// </summary>
        /// <param name="example">The example to run on.</param>
        /// <param name="prediction">The object to store the prediction in. If it's <c>null</c>, a new one will be created, otherwise the old one
        /// is reused.</param>
        public abstract void Predict(TSrc example, ref TDst prediction);
    }
}