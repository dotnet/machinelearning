// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Threading;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms.TimeSeries
{
    internal abstract class SrCnnArgumentBase
    {
        [Argument(ArgumentType.Required, HelpText = "The name of the source column.", ShortName = "src",
            SortOrder = 1, Purpose = SpecialPurpose.ColumnName)]
        public string Source;

        [Argument(ArgumentType.Required, HelpText = "The name of the new column.",
            SortOrder = 2)]
        public string Name;

        [Argument(ArgumentType.AtMostOnce, HelpText = "The size of the sliding window for computing spectral residual", ShortName = "wnd",
            SortOrder = 3)]
        public int WindowSize = 24;

        [Argument(ArgumentType.AtMostOnce, HelpText = "The size of the initial window for computing. The default value is set to 0, which means there is no initial window considered.", ShortName = "iwnd",
            SortOrder = 4)]
        public int InitialWindowSize = 0;

        [Argument(ArgumentType.AtMostOnce, HelpText = "The number of points to the back of training window.",
            ShortName = "backwnd", SortOrder = 5)]
        public int BackAddWindowSize = 5;

        [Argument(ArgumentType.AtMostOnce, HelpText = "The number of pervious points used in prediction.",
            ShortName = "aheadwnd", SortOrder = 6)]
        public int LookaheadWindowSize = 5;

        [Argument(ArgumentType.Required, HelpText = "The size of sliding window to generate a saliency map for the series.",
            ShortName = "avgwnd", SortOrder = 7)]
        public int AvergingWindowSize = 3;

        [Argument(ArgumentType.Required, HelpText = "The size of sliding window to generate a saliency map for the series.",
            ShortName = "jdgwnd", SortOrder = 8)]
        public int JudgementWindowSize = 21;

        [Argument(ArgumentType.AtMostOnce, HelpText = "The threshold to determine anomaly, score larger than the threshold is considered as anomaly.",
            ShortName = "thre", SortOrder = 9)]
        public double Threshold = 0.3;
    }

    internal abstract class SrCnnTransformBase<TInput, TState> : SequentialTransformerBase<TInput, VBuffer<Double>, TState>
        where TState : SrCnnTransformBase<TInput, TState>.SrCnnStateBase, new()
    {
        internal int BackAddWindowSize { get; }

        internal int LookaheadWindowSize { get; }

        internal int AvergingWindowSize { get; }

        internal int JudgementWindowSize { get; }

        internal double AlertThreshold { get; }

        internal int OutputLength { get; }

        private protected SrCnnTransformBase(int windowSize, int initialWindowSize, string inputColumnName, string outputColumnName, string name, IHostEnvironment env,
            int backAddWindowSize, int lookaheadWindowSize, int averagingWindowSize, int judgementWindowSize, Double alertThreshold)
            : base(Contracts.CheckRef(env, nameof(env)).Register(name), windowSize, initialWindowSize, outputColumnName, inputColumnName, new VectorDataViewType(NumberDataViewType.Double, 3))
        {
            Host.CheckUserArg(backAddWindowSize > 0, nameof(SrCnnArgumentBase.BackAddWindowSize), "Must be non-negative");
            Host.CheckUserArg(lookaheadWindowSize > 0 && lookaheadWindowSize <= windowSize, nameof(SrCnnArgumentBase.LookaheadWindowSize), "Must be non-negative and not larger than window size");
            Host.CheckUserArg(averagingWindowSize > 0 && averagingWindowSize <= windowSize, nameof(SrCnnArgumentBase.AvergingWindowSize), "Must be non-negative and not larger than window size");
            Host.CheckUserArg(judgementWindowSize > 0 && judgementWindowSize <= windowSize, nameof(SrCnnArgumentBase.JudgementWindowSize), "Must be non-negative and not larger than window size");
            Host.CheckUserArg(alertThreshold > 0 && alertThreshold < 1, nameof(SrCnnArgumentBase.Threshold), "Must be in (0,1)");

            BackAddWindowSize = backAddWindowSize;
            LookaheadWindowSize = lookaheadWindowSize;
            AvergingWindowSize = averagingWindowSize;
            JudgementWindowSize = judgementWindowSize;
            AlertThreshold = alertThreshold;

            OutputLength = 3;
        }

        private protected SrCnnTransformBase(IHostEnvironment env, ModelLoadContext ctx, string name)
            : base(Contracts.CheckRef(env, nameof(env)).Register(name), ctx)
        {
            OutputLength = 3;

            byte temp;
            temp = ctx.Reader.ReadByte();
            BackAddWindowSize = (int)temp;
            Host.CheckDecode(BackAddWindowSize > 0);

            temp = ctx.Reader.ReadByte();
            LookaheadWindowSize = (int)temp;
            Host.CheckDecode(LookaheadWindowSize > 0);

            temp = ctx.Reader.ReadByte();
            AvergingWindowSize = (int)temp;
            Host.CheckDecode(AvergingWindowSize > 0);

            temp = ctx.Reader.ReadByte();
            JudgementWindowSize = (int)temp;
            Host.CheckDecode(JudgementWindowSize > 0);

            AlertThreshold = ctx.Reader.ReadDouble();
            Host.CheckDecode(AlertThreshold >= 0 && AlertThreshold <= 1);
        }

        private protected SrCnnTransformBase(SrCnnArgumentBase args, string name, IHostEnvironment env)
            : this(args.WindowSize, args.InitialWindowSize, args.Source, args.Name,
                  name, env, args.BackAddWindowSize, args.LookaheadWindowSize, args.AvergingWindowSize, args.JudgementWindowSize, args.Threshold)
        {
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();

            Host.Assert(WindowSize > 0);
            Host.Assert(InitialWindowSize == WindowSize);
            Host.Assert(BackAddWindowSize > 0);
            Host.Assert(LookaheadWindowSize > 0);
            Host.Assert(AvergingWindowSize > 0);
            Host.Assert(JudgementWindowSize > 0);
            Host.Assert(AlertThreshold >= 0 && AlertThreshold <= 1);

            base.SaveModel(ctx);
            ctx.Writer.Write((byte)BackAddWindowSize);
            ctx.Writer.Write((byte)LookaheadWindowSize);
            ctx.Writer.Write((byte)AvergingWindowSize);
            ctx.Writer.Write((byte)JudgementWindowSize);
            ctx.Writer.Write(AlertThreshold);
        }

        internal override IStatefulRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(Host, this, schema);

        internal sealed class Mapper : IStatefulRowMapper
        {
            private readonly IHost _host;
            private readonly SrCnnTransformBase<TInput, TState> _parent;
            private readonly DataViewSchema _parentSchema;
            private readonly int _inputColumnIndex;
            private readonly VBuffer<ReadOnlyMemory<Char>> _slotNames;
            private SrCnnStateBase State { get; set; }

            public Mapper(IHostEnvironment env, SrCnnTransformBase<TInput, TState> parent, DataViewSchema inputSchema)
            {
                Contracts.CheckValue(env, nameof(env));
                _host = env.Register(nameof(Mapper));
                _host.CheckValue(inputSchema, nameof(inputSchema));
                _host.CheckValue(parent, nameof(parent));

                if (!inputSchema.TryGetColumnIndex(parent.InputColumnName, out _inputColumnIndex))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", parent.InputColumnName);

                var colType = inputSchema[_inputColumnIndex].Type;
                if (colType != NumberDataViewType.Single)
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", parent.InputColumnName, "Single", colType.ToString());

                _parent = parent;
                _parentSchema = inputSchema;
                _slotNames = new VBuffer<ReadOnlyMemory<char>>(_parent.OutputLength, new[] { "Alert".AsMemory(), "Raw Score".AsMemory(),
                    "Mag".AsMemory()});

                State = (SrCnnStateBase)_parent.StateRef;
            }

            public DataViewSchema.DetachedColumn[] GetOutputColumns()
            {
                var meta = new DataViewSchema.Annotations.Builder();
                meta.AddSlotNames(_parent.OutputLength, GetSlotNames);
                var info = new DataViewSchema.DetachedColumn[1];
                info[0] = new DataViewSchema.DetachedColumn(_parent.OutputColumnName, new VectorDataViewType(NumberDataViewType.Double, _parent.OutputLength), meta.ToAnnotations());
                return info;
            }

            public void GetSlotNames(ref VBuffer<ReadOnlyMemory<char>> dst) => _slotNames.CopyTo(ref dst, 0, _parent.OutputLength);

            public Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
            {
                if (activeOutput(0))
                    return col => col == _inputColumnIndex;
                else
                    return col => false;
            }

            void ICanSaveModel.Save(ModelSaveContext ctx) => _parent.SaveModel(ctx);

            public Delegate[] CreateGetters(DataViewRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                var getters = new Delegate[1];
                if (activeOutput(0))
                    getters[0] = MakeGetter(input, State);

                return getters;
            }

            private delegate void ProcessData(ref TInput src, ref VBuffer<double> dst);

            private Delegate MakeGetter(DataViewRow input, SrCnnStateBase state)
            {
                _host.AssertValue(input);
                var srcGetter = input.GetGetter<TInput>(input.Schema[_inputColumnIndex]);
                ProcessData processData = _parent.WindowSize > 0 ?
                    (ProcessData)state.Process : state.ProcessWithoutBuffer;

                ValueGetter<VBuffer<double>> valueGetter = (ref VBuffer<double> dst) =>
                {
                    TInput src = default;
                    srcGetter(ref src);
                    processData(ref src, ref dst);
                };
                return valueGetter;
            }

            public Action<PingerArgument> CreatePinger(DataViewRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                Action<PingerArgument> pinger = null;
                if (activeOutput(0))
                    pinger = MakePinger(input, State);

                return pinger;
            }

            private Action<PingerArgument> MakePinger(DataViewRow input, SrCnnStateBase state)
            {
                _host.AssertValue(input);
                var srcGetter = input.GetGetter<TInput>(input.Schema[_inputColumnIndex]);
                Action<PingerArgument> pinger = (PingerArgument args) =>
                {
                    if (args.DontConsumeSource)
                        return;

                    TInput src = default;
                    srcGetter(ref src);
                    state.UpdateState(ref src, args.RowPosition, _parent.WindowSize > 0);
                };
                return pinger;
            }

            public void CloneState()
            {
                if (Interlocked.Increment(ref _parent.StateRefCount) > 1)
                {
                    State = (SrCnnStateBase)_parent.StateRef.Clone();
                }
            }

            public ITransformer GetTransformer()
            {
                return _parent;
            }
        }

        internal abstract class SrCnnStateBase : SequentialTransformerBase<TInput, VBuffer<Double>, TState>.StateBase
        {
            protected SrCnnTransformBase<TInput, TState> Parent;

            private protected SrCnnStateBase() { }

            private protected override void CloneCore(TState state)
            {
                base.CloneCore(state);
                Contracts.Assert(state is SrCnnStateBase);
            }

            private protected SrCnnStateBase(BinaryReader reader) : base(reader)
            {
            }

            internal override void Save(BinaryWriter writer)
            {
                base.Save(writer);
            }

            private protected override void SetNaOutput(ref VBuffer<double> dst)
            {
                var outputLength = Parent.OutputLength;
                var editor = VBufferEditor.Create(ref dst, outputLength);

                for (int i = 0; i < outputLength; ++i)
                    editor.Values[i] = 0;

                dst = editor.Commit();
            }

            public sealed override void TransformCore(ref TInput input, FixedSizeQueue<TInput> windowedBuffer, long iteration, ref VBuffer<double> dst)
            {
                var outputLength = Parent.OutputLength;

                var result = VBufferEditor.Create(ref dst, outputLength);
                result.Values.Fill(Double.NaN);

                SpectralResidual(input, windowedBuffer, ref result);

                dst = result.Commit();
            }

            private protected sealed override void InitializeStateCore(bool disk = false)
            {
                Parent = (SrCnnTransformBase<TInput, TState>)ParentTransform;
            }

            private protected override void LearnStateFromDataCore(FixedSizeQueue<TInput> data)
            {
            }

            private protected virtual void SpectralResidual(TInput input, FixedSizeQueue<TInput> data, ref VBufferEditor<double> result)
            {
            }
        }
    }
}
