// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms.TimeSeries
{
    /// <summary>
    /// The wrapper to <see cref="SsaForecastingBase"/> that implements the general anomaly detection transform based on Singular Spectrum modeling of the time-series.
    /// For the details of the Singular Spectrum Analysis (SSA), refer to http://arxiv.org/pdf/1206.6910.pdf.
    /// </summary>
    public class SsaForecastingBaseWrapper : IStatefulTransformer, ICanSaveModel
    {
        /// <summary>
        /// Whether a call to <see cref="ITransformer.GetRowToRowMapper(DataViewSchema)"/> should succeed, on an
        /// appropriate schema.
        /// </summary>
        bool ITransformer.IsRowToRowMapper => ((ITransformer)InternalTransform).IsRowToRowMapper;

        /// <summary>
        /// Creates a clone of the transformer. Used for taking the snapshot of the state.
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

        private protected virtual void SaveModel(ModelSaveContext ctx) => InternalTransform.SaveThis(ctx);

        /// <summary>
        /// Creates a row mapper from Schema.
        /// </summary>
        internal IStatefulRowMapper MakeRowMapper(DataViewSchema schema) => InternalTransform.MakeRowMapper(schema);

        /// <summary>
        /// Creates an IDataTransform from an IDataView.
        /// </summary>
        internal IDataTransform MakeDataTransform(IDataView input) => InternalTransform.MakeDataTransform(input);

        /// <summary>
        /// Options for SSA Anomaly Detection.
        /// </summary>
        internal abstract class SsaForecastingOptions : ForecastingArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The discount factor in [0, 1]", ShortName = "disc", SortOrder = 12)]
            public Single DiscountFactor = 1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The function used to compute the error between the expected and the observed value", ShortName = "err", SortOrder = 13)]
            public ErrorFunction ErrorFunction = ErrorFunction.SignedDifference;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The flag determing whether the model is adaptive", ShortName = "adp", SortOrder = 14)]
            public bool IsAdaptive = false;
            public int WindowSize;
            public RankSelectionMethod RankSelectionMethod;
            public int? Rank;
            public int? MaxRank;
            public bool ShouldStablize;
            public bool ShouldMaintainInfo;
            public GrowthRatio? MaxGrowth;
            public int Horizon;
            public float ConfidenceLevel;
            public bool VariableHorizon;
        }

        internal SsaForecastingBase InternalTransform;

        internal SsaForecastingBaseWrapper(SsaForecastingOptions options, string name, IHostEnvironment env)
        {
            InternalTransform = new SsaForecastingBase(options, name, env, this);
        }
        //string forecastingConfidenceIntervalMinOutputColumnName, string forecastingConfidenceIntervalMaxOutputColumnName, int horizon, bool computeConfidenceIntervals
        internal SsaForecastingBaseWrapper(IHostEnvironment env, ModelLoadContext ctx, string name)
        {
            InternalTransform = new SsaForecastingBase(env, ctx, name);
        }

        /// <summary>
        /// This base class that implements the general anomaly detection transform based on Singular Spectrum modeling of the time-series.
        /// For the details of the Singular Spectrum Analysis (SSA), refer to http://arxiv.org/pdf/1206.6910.pdf.
        /// </summary>
        internal sealed class SsaForecastingBase : SequentialForecastingTransformBase<float, SsaForecastingBase.State>
        {
            internal SsaForecastingBaseWrapper Parent;
            internal readonly bool IsAdaptive;
            internal readonly int Horizon;
            internal readonly float ConfidenceLevel;
            internal SequenceModelerBase<Single, Single> Model;

            public SsaForecastingBase(SsaForecastingOptions options, string name, IHostEnvironment env, SsaForecastingBaseWrapper parent)
                : base(options.TrainSize, 0, options.Source, options.Name, options.ConfidenceLowerBoundColumn,
                      options.ConfidenceUpperBoundColumn, name, options.VariableHorizon ? 0 : options.Horizon, env)
            {
                Host.CheckUserArg(0 <= options.DiscountFactor && options.DiscountFactor <= 1, nameof(options.DiscountFactor), "Must be in the range [0, 1].");
                IsAdaptive = options.IsAdaptive;
                Horizon = options.Horizon;
                ConfidenceLevel = options.ConfidenceLevel;
                // Creating the master SSA model
                Model = new AdaptiveSingularSpectrumSequenceModelerInternal(Host, options.TrainSize, options.SeriesLength, options.WindowSize,
                    options.DiscountFactor, options.RankSelectionMethod, options.Rank, options.MaxRank, !string.IsNullOrEmpty(options.ConfidenceLowerBoundColumn),
                    options.ShouldStablize, options.ShouldMaintainInfo, options.MaxGrowth);

                StateRef = new State();
                StateRef.InitState(WindowSize, InitialWindowSize, this, Host);
                Parent = parent;
            }

            public SsaForecastingBase(IHostEnvironment env, ModelLoadContext ctx, string name) : base(env, ctx, name)
            {
                // *** Binary format ***
                // <base>
                // bool: _isAdaptive
                // int32: Horizon
                // bool: ComputeConfidenceIntervals
                // State: StateRef
                // AdaptiveSingularSpectrumSequenceModeler: _model

                Host.CheckDecode(InitialWindowSize == 0);

                IsAdaptive = ctx.Reader.ReadBoolean();
                Horizon = ctx.Reader.ReadInt32();
                ConfidenceLevel = ctx.Reader.ReadFloat();
                StateRef = new State(ctx.Reader);

                ctx.LoadModel<SequenceModelerBase<Single, Single>, SignatureLoadModel>(env, out Model, "SSA");
                Host.CheckDecode(Model != null);
                StateRef.InitState(this, Host);
            }

            public override DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
            {
                Host.CheckValue(inputSchema, nameof(inputSchema));

                if (!inputSchema.TryGetColumnIndex(InputColumnName, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", InputColumnName);

                var colType = inputSchema[col].Type;
                if (colType != NumberDataViewType.Single)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", InputColumnName, "Single", colType.ToString());

                return Transform(new EmptyDataView(Host, inputSchema)).Schema;
            }

            private protected override void SaveModel(ModelSaveContext ctx)
            {
                ((ICanSaveModel)Parent).Save(ctx);
            }

            internal void SaveThis(ModelSaveContext ctx)
            {
                Host.CheckValue(ctx, nameof(ctx));
                ctx.CheckAtModel();

                Host.Assert(InitialWindowSize == 0);
                Host.Assert(Model != null);

                // *** Binary format ***
                // <base>
                // bool: _isAdaptive
                // int32: Horizon
                // State: StateRef
                // AdaptiveSingularSpectrumSequenceModeler: _model

                base.SaveModel(ctx);
                ctx.Writer.Write(IsAdaptive);
                ctx.Writer.Write(Horizon);
                ctx.Writer.Write(ConfidenceLevel);
                StateRef.Save(ctx.Writer);

                ctx.SaveModel(Model, "SSA");
            }

            internal sealed class State : ForecastingStateBase
            {
                private SequenceModelerBase<Single, Single> _model;
                private SsaForecastingBase _parentForecaster;

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
                    if (_model != null)
                    {
                        _parentForecaster.Model = _parentForecaster.Model.Clone();
                        _model = _parentForecaster.Model;
                    }
                }

                private protected override void LearnStateFromDataCore(FixedSizeQueue<Single> data)
                {
                    // This method is empty because there is no need to implement a training logic here.
                }

                private protected override void InitializeForecaster()
                {
                    _parentForecaster = (SsaForecastingBase)Parent;
                    _model = _parentForecaster.Model;
                }

                public override void TransformCore(ref float input, FixedSizeQueue<float> windowedBuffer, long iteration, ref VBuffer<float> dst)
                {
                    // Forecasting is being done without prediction engine. Update the model
                    // with the observation.
                    if (PreviousPosition == -1)
                        Consume(input);

                    dst = new VBuffer<float>(LocalHorizon ?? _parentForecaster.Horizon,
                        ((AdaptiveSingularSpectrumSequenceModelerInternal)_model).Forecast(_parentForecaster.Horizon));
                }

                private protected override void TransformCore(ref float input, FixedSizeQueue<float> windowedBuffer, long iteration,
                    ref VBuffer<float> dst1, ref VBuffer<float> dst2, ref VBuffer<float> dst3)
                {
                    // Forecasting is being done without prediction engine. Update the model
                    // with the observation.
                    if (PreviousPosition == -1)
                        Consume(input);

                    ((AdaptiveSingularSpectrumSequenceModelerInternal)_model).ForecastWithConfidenceIntervals(LocalHorizon ?? _parentForecaster.Horizon,
                        out float[] forecast, out float[] min, out float[] max, LocalConfidenceLevel ?? _parentForecaster.ConfidenceLevel);

                    dst1 = new VBuffer<float>(_parentForecaster.Horizon, forecast);
                    dst2 = new VBuffer<float>(_parentForecaster.Horizon, min);
                    dst3 = new VBuffer<float>(_parentForecaster.Horizon, max);
                }

                public override void Forecast(ref VBuffer<float> dst)
                {
                    int horizon = LocalHorizon ?? _parentForecaster.Horizon;
                    dst = new VBuffer<float>(horizon, ((AdaptiveSingularSpectrumSequenceModelerInternal)_model).Forecast(horizon));
                }

                public override void ConfidenceIntervalLowerBound(ref VBuffer<float> dst)
                {
                    int horizon = LocalHorizon ?? _parentForecaster.Horizon;
                    ((AdaptiveSingularSpectrumSequenceModelerInternal)_model).ForecastWithConfidenceIntervals(horizon,
                        out float[] forecast, out float[] min, out float[] max, LocalConfidenceLevel ?? _parentForecaster.ConfidenceLevel);

                    dst = new VBuffer<float>(horizon, min);
                }

                public override void ConfidenceIntervalUpperBound(ref VBuffer<float> dst)
                {
                    int horizon = LocalHorizon ?? _parentForecaster.Horizon;
                    ((AdaptiveSingularSpectrumSequenceModelerInternal)_model).ForecastWithConfidenceIntervals(horizon,
                        out float[] forecast, out float[] min, out float[] max, LocalConfidenceLevel ?? _parentForecaster.ConfidenceLevel);

                    dst = new VBuffer<float>(horizon, max);
                }

                public override void Consume(Single input) => _model.Consume(ref input, _parentForecaster.IsAdaptive);
            }
        }
    }
}
