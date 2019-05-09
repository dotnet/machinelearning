// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
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

        [Argument(ArgumentType.AtMostOnce, HelpText = "The size of the initial window for computingd. The default value is set to 0, which means there is no initial window considered.", ShortName = "iwnd",
            SortOrder = 4)]
        public int InitialWindowSize = 0;

        [Argument(ArgumentType.AtMostOnce, HelpText = "The number of points to the back of training window.",
            ShortName = "bwnd", SortOrder = 5)]
        public int BackAddWindowSize = 5;

        [Argument(ArgumentType.AtMostOnce, HelpText = "The number of pervious points used in prediction.",
            ShortName = "awnd", SortOrder = 6)]
        public int LookaheadWindowSize = 5;

        [Argument(ArgumentType.AtMostOnce, HelpText = "The threshold to determine anomaly, score larger than the threshold is considered as anomaly.",
            ShortName = "thre", SortOrder = 7)]
        public double Threshold = 0.3;
    }

    internal abstract class SrCnnTransformBase<TInput, TState> : SequentialTransformerBase<TInput, VBuffer<Double>, TState>
        where TState : SrCnnTransformBase<TInput, TState>.SrCnnStateBase, new()
    {
        internal int BackAddWindowSize;

        internal int LookaheadWindowSize;

        internal Double AlertThreshold;

        internal int OutputLength;

        private protected SrCnnTransformBase(int windowSize, int initialWindowSize, string inputColumnName, string outputColumnName, string name, IHostEnvironment env,
            int backAddWindowSize, int lookaheadWindowSize, Double alertThreshold)
            : base(Contracts.CheckRef(env, nameof(env)).Register(name), windowSize, initialWindowSize, outputColumnName, inputColumnName, new VectorDataViewType(NumberDataViewType.Double, 3))
        {
            //TODO:

            BackAddWindowSize = backAddWindowSize;
            LookaheadWindowSize = lookaheadWindowSize;
            AlertThreshold = alertThreshold;
        }

        private protected SrCnnTransformBase(IHostEnvironment env, ModelLoadContext ctx, string name)
            : base(Contracts.CheckRef(env, nameof(env)).Register(name), ctx)
        {
            //TODO:
        }

        private protected SrCnnTransformBase(SrCnnArgumentBase args, string name, IHostEnvironment env)
            : this(args.WindowSize, args.InitialWindowSize, args.Source, args.Name,
                  name, env, args.BackAddWindowSize, args.LookaheadWindowSize, args.Threshold)
        {
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            //TODO:
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
                _slotNames = new VBuffer<ReadOnlyMemory<char>>(2, new[] { "Alert".AsMemory(), "Raw Score".AsMemory(),
                    "Mag".AsMemory()});

                State = (SrCnnStateBase)_parent.StateRef;
            }

            public DataViewSchema.DetachedColumn[] GetOutputColumns()
            {
                //TODO:
                throw new NotImplementedException();
            }

            public void GetSlotNames(ref VBuffer<ReadOnlyMemory<char>> dst)
            {
                //TODO:
                throw new NotImplementedException();
            }

            public Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
            {
                //TODO:
                throw new NotImplementedException();
            }

            void ICanSaveModel.Save(ModelSaveContext ctx) => _parent.SaveModel(ctx);

            public Delegate[] CreateGetters(DataViewRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                //TODO:
                throw new NotImplementedException();
            }

            private delegate void ProcessData(ref TInput src, ref VBuffer<double> dst);

            private Delegate MakeGetter(DataViewRow input, SrCnnStateBase state)
            {
                //TODO:
                throw new NotImplementedException();
            }

            public Action<long> CreatePinger(DataViewRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                //TODO:
                throw new NotImplementedException();
            }

            private Action<long> MakePinger(DataViewRow input, SrCnnStateBase state)
            {
                //TODO:
                throw new NotImplementedException();
            }

            public void CloneState()
            {
                //TODO:
            }

            public ITransformer GetTransformer()
            {
                //TODO:
                throw new NotImplementedException();
            }
        }

        internal abstract class SrCnnStateBase : SequentialTransformerBase<TInput, VBuffer<Double>, TState>.StateBase
        {
            protected SrCnnTransformBase<TInput, TState> Parent;

            private protected SrCnnStateBase() { }

            private protected override void CloneCore(TState state)
            {
                //TODO:
            }

            private protected SrCnnStateBase(BinaryReader reader) : base(reader)
            {
                //TODO:
            }

            internal override void Save(BinaryWriter writer)
            {
                //TODO:
            }

            private protected override void SetNaOutput(ref VBuffer<double> dst)
            {
                //TODO:
            }

            private protected sealed override void TransformCore(ref TInput input, FixedSizeQueue<TInput> windowedBuffer, long iteration, ref VBuffer<double> dst)
            {
                //TODO:
            }

            private protected sealed override void InitializeStateCore(bool disk = false)
            {
                //TODO:
            }
        }
    }
}
