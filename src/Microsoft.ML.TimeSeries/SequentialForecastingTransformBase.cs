// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Threading;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using static Microsoft.ML.DataViewSchema;

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

        [Argument(ArgumentType.Required, HelpText = "The name of the confidence interval lower bound column.", ShortName = "cnfminname",
            SortOrder = 2)]
        public string ConfidenceLowerBoundColumn;

        [Argument(ArgumentType.Required, HelpText = "The name of the confidence interval upper bound column.", ShortName = "cnfmaxnname",
            SortOrder = 2)]
        public string ConfidenceUpperBoundColumn;

        [Argument(ArgumentType.AtMostOnce, HelpText = "The length of series from the beginning used for training.", ShortName = "wnd",
            SortOrder = 3)]
        public int TrainSize = 1;

        [Argument(ArgumentType.AtMostOnce, HelpText = "The size of the initial window. The default value " +
            "is set to 0, which means there is no initial window considered.", ShortName = "initwnd", SortOrder = 5)]
        public int SeriesLength = 0;
    }

    /// <summary>
    /// The base class for forecasting transforms that also supports confidence intervals for each forecasted value.
    /// For more details, please refer to http://arxiv.org/pdf/1204.3251.pdf
    /// </summary>
    /// <typeparam name="TInput">The type of the input sequence</typeparam>
    /// <typeparam name="TState">The type of the input sequence</typeparam>
    internal abstract class SequentialForecastingTransformBase<TInput, TState> : SequentialTransformerBase<TInput, VBuffer<float>, TState>
    where TState : SequentialForecastingTransformBase<TInput, TState>.ForecastingStateBase, new()
    {

        // The size of the VBuffer in the dst column.
        private readonly int _outputLength;

        private protected SequentialForecastingTransformBase(int windowSize, int initialWindowSize,
            string inputColumnName, string outputColumnName, string confidenceLowerBoundColumn,
                string confidenceUpperBoundColumn, string name, int outputLength, IHostEnvironment env)
            : base(Contracts.CheckRef(env, nameof(env)).Register(name), windowSize, initialWindowSize,
                  outputColumnName, confidenceLowerBoundColumn,
                  confidenceUpperBoundColumn, inputColumnName, new VectorDataViewType(NumberDataViewType.Single, outputLength))
        {
            _outputLength = outputLength;
        }

        private protected SequentialForecastingTransformBase(ForecastingArgumentsBase args, string name, int outputLength, IHostEnvironment env)
            : this(args.TrainSize, args.SeriesLength, args.Source, args.ConfidenceLowerBoundColumn,
                  args.ConfidenceUpperBoundColumn, args.Name, name, outputLength, env)
        {
        }

        private protected SequentialForecastingTransformBase(IHostEnvironment env, ModelLoadContext ctx, string name)
            : base(Contracts.CheckRef(env, nameof(env)).Register(name), ctx)
        {
            _outputLength = ctx.Reader.ReadInt32();
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
            ctx.Writer.Write(_outputLength);
        }

        internal override IStatefulRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(Host, this, schema);

        internal sealed class Mapper : IStatefulRowMapper
        {
            private readonly IHost _host;
            private readonly SequentialForecastingTransformBase<TInput, TState> _parent;
            private readonly DataViewSchema _parentSchema;
            private readonly int _inputColumnIndex;
            private ForecastingStateBase State { get; set; }
            private bool _dontFetchSrcValue;

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
                State = (ForecastingStateBase)_parent.StateRef;
                _dontFetchSrcValue = false;
            }

            public DataViewSchema.DetachedColumn[] GetOutputColumns()
            {
                DetachedColumn[] info;

                if (!string.IsNullOrEmpty(_parent.ConfidenceUpperBoundColumn))
                {
                    info = new DetachedColumn[3];
                    info[0] = new DetachedColumn(_parent.OutputColumnName, new VectorDataViewType(NumberDataViewType.Single, _parent._outputLength));
                    info[1] = new DetachedColumn(_parent.ConfidenceLowerBoundColumn, new VectorDataViewType(NumberDataViewType.Single, _parent._outputLength));
                    info[2] = new DetachedColumn(_parent.ConfidenceUpperBoundColumn, new VectorDataViewType(NumberDataViewType.Single, _parent._outputLength));
                }
                else
                {
                    info = new DetachedColumn[1];
                    info[0] = new DetachedColumn(_parent.OutputColumnName, new VectorDataViewType(NumberDataViewType.Single, _parent._outputLength));
                }

                return info;
            }

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
                var getters = string.IsNullOrEmpty(_parent.ConfidenceUpperBoundColumn) ? new Delegate[1] : new Delegate[3];

                if (activeOutput(0))
                {
                    ValueGetter<VBuffer<float>> valueGetter = (ref VBuffer<float> dst) =>
                    {
                        State.Forecast(ref dst);
                    };

                    getters[0] = valueGetter;
                }

                if (!string.IsNullOrEmpty(_parent.ConfidenceUpperBoundColumn))
                {
                    if (activeOutput(1))
                    {
                        ValueGetter<VBuffer<float>> valueGetter = (ref VBuffer<float> dst) =>
                        {
                            State.ConfidenceIntervalLowerBound(ref dst);
                        };

                        getters[1] = valueGetter;
                    }

                    if (activeOutput(2))
                    {
                        ValueGetter<VBuffer<float>> valueGetter = (ref VBuffer<float> dst) =>
                        {
                            State.ConfidenceIntervalUpperBound(ref dst);
                        };

                        getters[2] = valueGetter;
                    }
                }
                return getters;
            }

            private delegate void ProcessData(ref TInput src, ref VBuffer<float> dst);

            private Delegate MakeGetter(DataViewRow input, ForecastingStateBase state)
            {
                _host.AssertValue(input);
                var srcGetter = input.GetGetter<TInput>(input.Schema[_inputColumnIndex]);
                ProcessData processData = _parent.WindowSize > 0 ?
                    (ProcessData)state.Process : state.ProcessWithoutBuffer;

                ValueGetter<VBuffer<float>> valueGetter = (ref VBuffer<float> dst) =>
                {
                    TInput src = default;
                    if (_dontFetchSrcValue)
                    {
                        state.TransformCore(ref src, null, 0, ref dst);
                        return;
                    }

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

            private Action<PingerArgument> MakePinger(DataViewRow input, ForecastingStateBase state)
            {
                _host.AssertValue(input);
                var srcGetter = input.GetGetter<TInput>(input.Schema[_inputColumnIndex]);
                Action<PingerArgument> pinger = (PingerArgument args) =>
                {
                    state.LocalConfidenceLevel = args.ConfidenceLevel;
                    state.LocalHorizon = args.Horizon;

                    // This means don't call srcGetter in getters.
                    if (args.DontConsumeSource)
                    {
                        _dontFetchSrcValue = true;
                        return;
                    }

                    _dontFetchSrcValue = false;
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
        internal abstract class ForecastingStateBase : SequentialTransformerBase<TInput, VBuffer<float>, TState>.StateBase
        {
            // A reference to the parent transform.
            protected SequentialForecastingTransformBase<TInput, TState> Parent;
            internal int? LocalHorizon;
            internal float? LocalConfidenceLevel;

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

            private protected override void SetNaOutput(ref VBuffer<float> dst)
            {
                var outputLength = Parent._outputLength;
                var editor = VBufferEditor.Create(ref dst, outputLength);

                for (int i = 0; i < outputLength; ++i)
                    editor.Values[i] = float.NaN;

                dst = editor.Commit();
            }

            private protected sealed override void InitializeStateCore(bool disk = false)
            {
                Parent = (SequentialForecastingTransformBase<TInput, TState>)ParentTransform;
                Host.Assert(WindowSize >= 0);
                InitializeForecaster();
            }

            /// <summary>
            /// The abstract method that realizes the initialization functionality for the forecaster.
            /// </summary>
            private protected abstract void InitializeForecaster();
        }
    }
}
