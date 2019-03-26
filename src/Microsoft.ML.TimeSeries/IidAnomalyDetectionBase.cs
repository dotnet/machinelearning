// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms.TimeSeries
{
    /// <summary>
    /// The is the wrapper to <see cref="IidAnomalyDetectionBase"/> that computes the p-values and martingale scores for a supposedly i.i.d input sequence of floats. In other words, it assumes
    /// the input sequence represents the raw anomaly score which might have been computed via another process.
    /// </summary>
    public class IidAnomalyDetectionBaseWrapper : IStatefulTransformer, ICanSaveModel
    {
        /// <summary>
        /// Whether a call to <see cref="ITransformer.GetRowToRowMapper(DataViewSchema)"/> should succeed, on an
        /// appropriate schema.
        /// </summary>
        bool ITransformer.IsRowToRowMapper => ((ITransformer)InternalTransform).IsRowToRowMapper;

        /// <summary>
        /// Creates a clone of the transfomer. Used for taking the snapshot of the state.
        /// </summary>
        /// <returns></returns>
        IStatefulTransformer IStatefulTransformer.Clone() => InternalTransform.Clone();

        /// <summary>
        /// Schema propagation for transformers.
        /// Returns the output schema of the data, if the input schema is like the one provided.
        /// </summary>
        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => InternalTransform.GetOutputSchema(inputSchema);

        /// <summary>
        /// Constructs a row-to-row mapper based on an input schema. If <see cref="ITransformer.IsRowToRowMapper"/>
        /// is <c>false</c>, then an exception should be thrown. If the input schema is in any way
        /// unsuitable for constructing the mapper, an exception should likewise be thrown.
        /// </summary>
        /// <param name="inputSchema">The input schema for which we should get the mapper.</param>
        /// <returns>The row to row mapper.</returns>
        IRowToRowMapper ITransformer.GetRowToRowMapper(DataViewSchema inputSchema)
            => ((ITransformer)InternalTransform).GetRowToRowMapper(inputSchema);

        /// <summary>
        /// Same as <see cref="ITransformer.GetRowToRowMapper(DataViewSchema)"/> but also supports mechanism to save the state.
        /// </summary>
        /// <param name="inputSchema">The input schema for which we should get the mapper.</param>
        /// <returns>The row to row mapper.</returns>
        public IRowToRowMapper GetStatefulRowToRowMapper(DataViewSchema inputSchema)
            => ((IStatefulTransformer)InternalTransform).GetStatefulRowToRowMapper(inputSchema);

        /// <summary>
        /// Take the data in, make transformations, output the data.
        /// Note that <see cref="IDataView"/>'s are lazy, so no actual transformations happen here, just schema validation.
        /// </summary>
        public IDataView Transform(IDataView input) => InternalTransform.Transform(input);

        /// <summary>
        /// For saving a model into a repository.
        /// </summary>
        void ICanSaveModel.Save(ModelSaveContext ctx) => SaveModel(ctx);

        private protected virtual void SaveModel(ModelSaveContext ctx)
        {
            InternalTransform.SaveThis(ctx);
        }

        /// <summary>
        /// Creates a row mapper from Schema.
        /// </summary>
        internal IStatefulRowMapper MakeRowMapper(DataViewSchema schema) => InternalTransform.MakeRowMapper(schema);

        /// <summary>
        /// Creates an IDataTransform from an IDataView.
        /// </summary>
        internal IDataTransform MakeDataTransform(IDataView input) => InternalTransform.MakeDataTransform(input);

        internal IidAnomalyDetectionBase InternalTransform;

        internal IidAnomalyDetectionBaseWrapper(ArgumentsBase args, string name, IHostEnvironment env)
        {
            InternalTransform = new IidAnomalyDetectionBase(args, name, env, this);
        }

        internal IidAnomalyDetectionBaseWrapper(IHostEnvironment env, ModelLoadContext ctx, string name)
        {
            InternalTransform = new IidAnomalyDetectionBase(env, ctx, name, this);
        }

        /// <summary>
        /// This transform computes the p-values and martingale scores for a supposedly i.i.d input sequence of floats. In other words, it assumes
        /// the input sequence represents the raw anomaly score which might have been computed via another process.
        /// </summary>
        internal class IidAnomalyDetectionBase : SequentialAnomalyDetectionTransformBase<Single, IidAnomalyDetectionBase.State>
        {
            internal IidAnomalyDetectionBaseWrapper Parent;

            public IidAnomalyDetectionBase(ArgumentsBase args, string name, IHostEnvironment env, IidAnomalyDetectionBaseWrapper parent)
                : base(args, name, env)
            {
                InitialWindowSize = 0;
                StateRef = new State();
                StateRef.InitState(WindowSize, InitialWindowSize, this, Host);
                Parent = parent;
            }

            public IidAnomalyDetectionBase(IHostEnvironment env, ModelLoadContext ctx, string name, IidAnomalyDetectionBaseWrapper parent)
                : base(env, ctx, name)
            {
                Host.CheckDecode(InitialWindowSize == 0);
                StateRef = new State(ctx.Reader);
                StateRef.InitState(this, Host);
                Parent = parent;
            }

            public override DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
            {
                Host.CheckValue(inputSchema, nameof(inputSchema));

                if (!inputSchema.TryGetColumnIndex(InputColumnName, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", InputColumnName);

                var colType = inputSchema[col].Type;
                if (colType != NumberDataViewType.Single)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", InputColumnName, NumberDataViewType.Single.ToString(), colType.ToString());

                return Transform(new EmptyDataView(Host, inputSchema)).Schema;
            }

            private protected override void SaveModel(ModelSaveContext ctx)
            {
                ((ICanSaveModel)Parent).Save(ctx);
            }

            internal void SaveThis(ModelSaveContext ctx)
            {
                ctx.CheckAtModel();
                Host.Assert(InitialWindowSize == 0);
                base.SaveModel(ctx);

                // *** Binary format ***
                // <base>
                // State: StateRef
                StateRef.Save(ctx.Writer);
            }

            internal sealed class State : AnomalyDetectionStateBase
            {
                public State()
                {
                }

                internal State(BinaryReader reader) : base(reader)
                {
                    WindowedBuffer = TimeSeriesUtils.DeserializeFixedSizeQueueSingle(reader, Host);
                    InitialWindowedBuffer = TimeSeriesUtils.DeserializeFixedSizeQueueSingle(reader, Host);
                }

                internal override void Save(BinaryWriter writer)
                {
                    base.Save(writer);
                    TimeSeriesUtils.SerializeFixedSizeQueue(WindowedBuffer, writer);
                    TimeSeriesUtils.SerializeFixedSizeQueue(InitialWindowedBuffer, writer);
                }

                private protected override void CloneCore(State state)
                {
                    base.CloneCore(state);
                    Contracts.Assert(state is State);
                    var stateLocal = state as State;
                    stateLocal.WindowedBuffer = WindowedBuffer.Clone();
                    stateLocal.InitialWindowedBuffer = InitialWindowedBuffer.Clone();
                }

                private protected override void LearnStateFromDataCore(FixedSizeQueue<Single> data)
                {
                    // This method is empty because there is no need for initial tuning for this transform.
                }

                private protected override void InitializeAnomalyDetector()
                {
                    // This method is empty because there is no need for any extra initialization for this transform.
                }

                private protected override double ComputeRawAnomalyScore(ref Single input, FixedSizeQueue<Single> windowedBuffer, long iteration)
                {
                    // This transform treats the input sequenence as the raw anomaly score.
                    return (double)input;
                }

                public override void Consume(float value)
                {
                }
            }
        }
    }
}
