// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.CommandLine;
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

        internal PipeEngine(IHostEnvironment env, IDataView pipe, bool ignoreMissingColumns, SchemaDefinition schemaDefinition = null)
        {
            Contracts.AssertValue(env);
            env.AssertValue(pipe);
            env.AssertValueOrNull(schemaDefinition);

            _cursorablePipe = env.AsCursorable<TDst>(pipe, ignoreMissingColumns, schemaDefinition);
        }

        public IEnumerable<TDst> RunPipe(bool reuseRowObject)
        {
            using (var cursor = _cursorablePipe.GetCursor())
            {
                TDst row = null;
                while (cursor.MoveNext())
                {
                    if (!reuseRowObject || row == null)
                        row = new TDst();

                    cursor.FillValues(row);
                    yield return row;
                }
            }
        }
    }

    /// <summary>
    /// Class for making single predictions on a previously trained model (and preceding transform pipeline).
    /// </summary>
    /// <remarks>
    /// This class can also be used with trained pipelines that do not end with a predictor: in this case, the
    /// 'prediction' will be just the outcome of all the transformations.
    ///
    /// The PredictionEngine is NOT thread safe. Using it in a threaded environment can cause unexpected issues.
    /// </remarks>
    public sealed class PredictionEngine<TSrc, TDst> : PredictionEngineBase<TSrc, TDst>
       where TSrc : class
       where TDst : class, new()
    {
        internal PredictionEngine(IHostEnvironment env, ITransformer transformer, bool ignoreMissingColumns,
            SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null, bool ownsTransformer = true)
            : base(env, transformer, ignoreMissingColumns, inputSchemaDefinition, outputSchemaDefinition, ownsTransformer)
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
    /// Base class for making single predictions on a previously trained model (and the preceding transform pipeline).
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
        private readonly bool _ownsTransformer;
        private bool _disposed;

        /// <summary>
        /// Provides output schema.
        /// </summary>
        public DataViewSchema OutputSchema { get; }

        [BestFriend]
        private protected ITransformer Transformer { get; }

        [BestFriend]
        private protected PredictionEngineBase(IHostEnvironment env, ITransformer transformer, bool ignoreMissingColumns,
            SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null, bool ownsTransformer = true)
        {
            Contracts.CheckValue(env, nameof(env));
            env.AssertValue(transformer);
            Transformer = transformer;
            var makeMapper = TransformerChecker(env, transformer);
            env.AssertValue(makeMapper);
            _inputRow = DataViewConstructionUtils.CreateInputRow<TSrc>(env, inputSchemaDefinition);
            _ownsTransformer = ownsTransformer;
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
            if (_disposed)
                return;

            _disposer?.Invoke();

            if (_ownsTransformer)
                (Transformer as IDisposable)?.Dispose();

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

    /// <summary>
    /// Options for the <see cref="PredictionEngine{TSrc, TDst}"/>
    /// </summary>
    public sealed class PredictionEngineOptions
    {
        [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to throw an error if a column exists in the output schema but not the output object.", ShortName = "ignore", SortOrder = 50)]
        public bool IgnoreMissingColumns = Defaults.IgnoreMissingColumns;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Additional settings of the input schema.", ShortName = "input", SortOrder = 50)]
        public SchemaDefinition InputSchemaDefinition = Defaults.InputSchemaDefinition;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Additional settings of the output schema.", ShortName = "output")]
        public SchemaDefinition OutputSchemaDefinition = Defaults.OutputSchemaDefinition;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the prediction engine owns the transformer and should dispose of it.", ShortName = "own")]
        public bool OwnsTransformer = Defaults.OwnsTransformer;

        internal static class Defaults
        {
            public const bool IgnoreMissingColumns = true;
            public const SchemaDefinition InputSchemaDefinition = null;
            public const SchemaDefinition OutputSchemaDefinition = null;
            public const bool OwnsTransformer = true;
        }
    }
}
