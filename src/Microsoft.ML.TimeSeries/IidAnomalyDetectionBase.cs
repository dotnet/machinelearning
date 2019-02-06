// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.TimeSeries;

namespace Microsoft.ML.TimeSeriesProcessing
{
    /// <summary>
    /// The wrapper to transform that computes the p-values and martingale scores for a supposedly i.i.d input sequence of floats. In other words, it assumes
    /// the input sequence represents the raw anomaly score which might have been computed via another process.
    /// </summary>
    public class IidAnomalyDetectionBaseWrapper : IStatefulTransformer, ICanSaveModel
    {
        public bool IsRowToRowMapper => Base.IsRowToRowMapper;

        IStatefulTransformer IStatefulTransformer.Clone() => Base.Clone();

        public Schema GetOutputSchema(Schema inputSchema) => Base.GetOutputSchema(inputSchema);

        public IRowToRowMapper GetRowToRowMapper(Schema inputSchema) => Base.GetRowToRowMapper(inputSchema);

        public IRowToRowMapper GetStatefulRowToRowMapper(Schema inputSchema) => ((IStatefulTransformer)Base).GetStatefulRowToRowMapper(inputSchema);

        public IDataView Transform(IDataView input) => Base.Transform(input);

        public virtual void Save(ModelSaveContext ctx)
        {
            Base.SaveThis(ctx);
        }

        internal IStatefulRowMapper MakeRowMapper(Schema schema) => Base.MakeRowMapper(schema);

        internal IDataTransform MakeDataTransform(IDataView input) => Base.MakeDataTransform(input);

        internal IidAnomalyDetectionBase Base;
        public IidAnomalyDetectionBaseWrapper(ArgumentsBase args, string name, IHostEnvironment env)
        {
            Base = new IidAnomalyDetectionBase(args, name, env, this);
        }

        public IidAnomalyDetectionBaseWrapper(IHostEnvironment env, ModelLoadContext ctx, string name)
        {
            Base = new IidAnomalyDetectionBase(env, ctx, name, this);
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

            public override Schema GetOutputSchema(Schema inputSchema)
            {
                Host.CheckValue(inputSchema, nameof(inputSchema));

                if (!inputSchema.TryGetColumnIndex(InputColumnName, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", InputColumnName);

                var colType = inputSchema[col].Type;
                if (colType != NumberType.R4)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", InputColumnName, NumberType.R4.ToString(), colType.ToString());

                return Transform(new EmptyDataView(Host, inputSchema)).Schema;
            }

            private protected override void SaveModel(ModelSaveContext ctx)
            {
                Parent.SaveModel(ctx);
            }

            internal void SaveThis(ModelSaveContext ctx)
            {
                ctx.CheckAtModel();
                Host.Assert(InitialWindowSize == 0);
                base.Save(ctx);

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
