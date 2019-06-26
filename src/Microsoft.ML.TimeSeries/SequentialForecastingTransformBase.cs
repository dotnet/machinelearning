// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Threading;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms.TimeSeries
{

    /// <summary>
    /// The base class that can be inherited by the 'Argument' classes in the derived classes containing the shared input parameters.
    /// </summary>
    internal abstract class ForecastingArgumentsBase
    {
        [Argument(ArgumentType.Required, HelpText = "The name of the source column", ShortName = "src",
            SortOrder = 1, Purpose = SpecialPurpose.ColumnName)]
        public string Source;

        [Argument(ArgumentType.Required, HelpText = "The name of the new column", ShortName = "name",
            SortOrder = 2)]
        public string Name;

        [Argument(ArgumentType.AtMostOnce, HelpText = "The size of the sliding window for computing the p-value.", ShortName = "wnd",
            SortOrder = 3)]
        public int TrainSize = 1;

        [Argument(ArgumentType.AtMostOnce, HelpText = "The size of the initial window for computing the p-value as well as" +
            " training if needed. The default value is set to 0, which means there is no initial window considered.",
            ShortName = "initwnd", SortOrder = 5)]
        public int SeriesLength = 0;
    }

    // REVIEW: This base class and its children classes generate one output column of type VBuffer<Double> to output 3 different anomaly scores as well as
    // the alert flag. Ideally these 4 output information should be put in four seaparate columns instead of one VBuffer<> column. However, this is not currently
    // possible due to our design restriction. This must be fixed in the next version and will potentially affect the children classes.
    /// <summary>
    /// The base class for sequential anomaly detection transforms that supports the p-value as well as the martingales scores computation from the sequence of
    /// raw anomaly scores whose calculation is specified by the children classes. This class also provides mechanism for the threshold-based alerting on
    /// the raw anomaly score, the p-value score or the martingale score. Currently, this class supports Power and Mixture martingales.
    /// For more details, please refer to http://arxiv.org/pdf/1204.3251.pdf
    /// </summary>
    /// <typeparam name="TInput">The type of the input sequence</typeparam>
    /// <typeparam name="TState">The type of the input sequence</typeparam>
    internal abstract class SequentialForecastingTransformBase<TInput, TState> : SequentialTransformerBase<TInput, VBuffer<Double>, TState>
    where TState : SequentialForecastingTransformBase<TInput, TState>.ForecastingStateBase, new()
    {

        // The size of the VBuffer in the dst column.
        internal int OutputLength;

        private protected SequentialForecastingTransformBase(int windowSize, int initialWindowSize, string inputColumnName, string outputColumnName, string name, IHostEnvironment env)
            : base(Contracts.CheckRef(env, nameof(env)).Register(name), windowSize, initialWindowSize, outputColumnName, inputColumnName, new VectorDataViewType(NumberDataViewType.Double, 0))
        {
        }

        private protected SequentialForecastingTransformBase(ForecastingArgumentsBase args, string name, IHostEnvironment env)
            : this(args.TrainSize, args.SeriesLength, args.Source, args.Name, name, env)
        {
        }

        private protected SequentialForecastingTransformBase(IHostEnvironment env, ModelLoadContext ctx, string name)
            : base(Contracts.CheckRef(env, nameof(env)).Register(name), ctx)
        {
            // *** Binary format ***
            // <base>
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();

            // *** Binary format ***
            // <base>

            base.SaveModel(ctx);
        }

        internal override IStatefulRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(Host, this, schema);

        internal sealed class Mapper : IStatefulRowMapper
        {
            private readonly IHost _host;
            private readonly SequentialForecastingTransformBase<TInput, TState> _parent;
            private readonly DataViewSchema _parentSchema;
            private readonly int _inputColumnIndex;
            private readonly VBuffer<ReadOnlyMemory<Char>> _slotNames;
            private ForecastingStateBase State { get; set; }

            public Mapper(IHostEnvironment env, SequentialForecastingTransformBase<TInput, TState> parent, DataViewSchema inputSchema)
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
                _slotNames = new VBuffer<ReadOnlyMemory<char>>(_parent.OutputLength, new[] {"".AsMemory()});

                State = (ForecastingStateBase)_parent.StateRef;
            }

            public DataViewSchema.DetachedColumn[] GetOutputColumns()
            {
                var meta = new DataViewSchema.Annotations.Builder();
                meta.AddSlotNames(_parent.OutputLength, GetSlotNames);
                var info = new DataViewSchema.DetachedColumn[3];
                info[0] = new DataViewSchema.DetachedColumn(_parent.OutputColumnName, new VectorDataViewType(NumberDataViewType.Double, 0), meta.ToAnnotations());
                info[1] = new DataViewSchema.DetachedColumn("Min", new VectorDataViewType(NumberDataViewType.Double, 0), meta.ToAnnotations());
                info[2] = new DataViewSchema.DetachedColumn("Max", new VectorDataViewType(NumberDataViewType.Double, 0), meta.ToAnnotations());
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
                var getters = new Delegate[3];
                if (activeOutput(0))
                    getters[0] = MakeGetter(input, State);

                getters[1] = getters[2] = getters[0];
                return getters;
            }

            private delegate void ProcessData(ref TInput src, ref VBuffer<double> dst);

            private Delegate MakeGetter(DataViewRow input, ForecastingStateBase state)
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

            public Action<long> CreatePinger(DataViewRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                Action<long> pinger = null;
                if (activeOutput(0))
                    pinger = MakePinger(input, State);

                return pinger;
            }

            private Action<long> MakePinger(DataViewRow input, ForecastingStateBase state)
            {
                _host.AssertValue(input);
                var srcGetter = input.GetGetter<TInput>(input.Schema[_inputColumnIndex]);
                Action<long> pinger = (long rowPosition) =>
                {
                    TInput src = default;
                    srcGetter(ref src);
                    state.UpdateState(ref src, rowPosition, _parent.WindowSize > 0);
                };
                return pinger;
            }

            public void CloneState()
            {
                if (Interlocked.Increment(ref _parent.StateRefCount) > 1)
                {
                    State = (ForecastingStateBase)_parent.StateRef.Clone();
                }
            }

            public ITransformer GetTransformer()
            {
                return _parent;
            }
        }
        /// <summary>
        /// The base state class for sequential anomaly detection: this class implements the p-values and martinagle calculations for anomaly detection
        /// given that the raw anomaly score calculation is specified by the derived classes.
        /// </summary>
        internal abstract class ForecastingStateBase : SequentialTransformerBase<TInput, VBuffer<Double>, TState>.StateBase
        {
            // A reference to the parent transform.
            protected SequentialForecastingTransformBase<TInput, TState> Parent;

            private protected ForecastingStateBase() { }

            private protected override void CloneCore(TState state)
            {
                base.CloneCore(state);
            }

            private protected ForecastingStateBase(BinaryReader reader) : base(reader)
            {
            }

            internal override void Save(BinaryWriter writer)
            {
                base.Save(writer);
            }

            private protected override void SetNaOutput(ref VBuffer<Double> dst)
            {
                var outputLength = Parent.OutputLength;
                var editor = VBufferEditor.Create(ref dst, outputLength);

                for (int i = 0; i < outputLength; ++i)
                    editor.Values[i] = Double.NaN;

                dst = editor.Commit();
            }

            private protected sealed override void InitializeStateCore(bool disk = false)
            {
                Parent = (SequentialForecastingTransformBase<TInput, TState>)ParentTransform;
                Host.Assert(WindowSize >= 0);
                InitializeAnomalyDetector();
            }

            /// <summary>
            /// The abstract method that realizes the initialization functionality for the anomaly detector.
            /// </summary>
            private protected abstract void InitializeAnomalyDetector();
        }
    }
}
