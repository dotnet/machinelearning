// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.TimeSeriesProcessing
{
    /// <summary>
    /// This transform computes the p-values and martingale scores for a supposedly i.i.d input sequence of floats. In other words, it assumes
    /// the input sequence represents the raw anomaly score which might have been computed via another process.
    /// </summary>
    public abstract class IidAnomalyDetectionBase : SequentialAnomalyDetectionTransformBase<Single, IidAnomalyDetectionBase.State>
    {
        public IidAnomalyDetectionBase(ArgumentsBase args, string name, IHostEnvironment env)
            : base(args, name, env)
        {
            InitialWindowSize = 0;
            StateRef = new State();
            StateRef.InitState(WindowSize, InitialWindowSize, this, Host);
        }

        public IidAnomalyDetectionBase(IHostEnvironment env, ModelLoadContext ctx, string name)
            : base(env, ctx, name)
        {
            Host.CheckDecode(InitialWindowSize == 0);
            StateRef = new State(ctx);
            StateRef.InitState(WindowSize, InitialWindowSize, this, Host);
        }

        public override Schema GetOutputSchema(Schema inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            if (!inputSchema.TryGetColumnIndex(InputColumnName, out var col))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", InputColumnName);

            var colType = inputSchema.GetColumnType(col);
            if (colType != NumberType.R4)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", InputColumnName, NumberType.R4.ToString(), colType.ToString());

            return Transform(new EmptyDataView(Host, inputSchema)).Schema;
        }

        public override void Save(ModelSaveContext ctx)
        {
            ctx.CheckAtModel();
            Host.Assert(InitialWindowSize == 0);
            base.Save(ctx);

            // *** Binary format ***
            // <base>
            //bool: StateRef != null
            // State: StateRef
            StateRef.Save(ctx);
            StateRef.InitState(this, Host);
        }

        public sealed class State : AnomalyDetectionStateBase
        {
            public State()
            {

            }

            public State(ModelLoadContext ctx) : base(ctx)
            {
                WindowedBuffer = new FixedSizeQueue<float>(
                    ctx.Reader.ReadInt32(), ctx.Reader.ReadInt32(), ctx.Reader.ReadSingleArray());

                InitialWindowedBuffer = new FixedSizeQueue<float>(
                    ctx.Reader.ReadInt32(), ctx.Reader.ReadInt32(), ctx.Reader.ReadSingleArray());
            }

            public override void Save(ModelSaveContext ctx)
            {
                base.Save(ctx);

                ctx.Writer.Write(WindowedBuffer.Capacity);
                ctx.Writer.Write(WindowedBuffer.StartIndex);
                ctx.Writer.WriteSingleArray(WindowedBuffer.Buffer);

                ctx.Writer.Write(InitialWindowedBuffer.Capacity);
                ctx.Writer.Write(InitialWindowedBuffer.StartIndex);
                ctx.Writer.WriteSingleArray(InitialWindowedBuffer.Buffer);
            }

            public override void CloneCore(StateBase state)
            {
                base.CloneCore(state);
                Contracts.Assert(state is State);
                var stateLocal = state as State;
                stateLocal.WindowedBuffer = (FixedSizeQueue<Single>)WindowedBuffer.Clone();
                stateLocal.InitialWindowedBuffer = (FixedSizeQueue<Single>)InitialWindowedBuffer.Clone();
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
