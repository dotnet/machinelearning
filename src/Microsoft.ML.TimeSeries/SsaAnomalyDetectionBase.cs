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
    public enum ErrorFunction : byte
    {
        SignedDifference,
        AbsoluteDifference,
        SignedProportion,
        AbsoluteProportion,
        SquaredDifference
    }

    /// <summary>
    /// Provides the utility functions for different error functions for computing deviation.
    /// </summary>
    internal static class ErrorFunctionUtils
    {
        public const string ErrorFunctionHelpText = "The error function should be either (0) SignedDifference, (1) AbsoluteDifference, (2) SignedProportion" +
                                                     " (3) AbsoluteProportion or (4) SquaredDifference.";

        public static double SignedDifference(double actual, Double predicted)
        {
            return actual - predicted;
        }

        public static Double AbsoluteDifference(Double actual, Double predicted)
        {
            return Math.Abs(actual - predicted);
        }

        public static Double SignedProportion(Double actual, Double predicted)
        {
            return predicted == 0 ? 0 : (actual - predicted) / predicted;
        }

        public static Double AbsoluteProportion(Double actual, Double predicted)
        {
            return predicted == 0 ? 0 : Math.Abs((actual - predicted) / predicted);
        }

        public static Double SquaredDifference(Double actual, Double predicted)
        {
            Double temp = actual - predicted;
            return temp * temp;
        }

        public static Func<Double, Double, Double> GetErrorFunction(ErrorFunction errorFunction)
        {
            switch (errorFunction)
            {
                case ErrorFunction.SignedDifference:
                    return SignedDifference;

                case ErrorFunction.AbsoluteDifference:
                    return AbsoluteDifference;

                case ErrorFunction.SignedProportion:
                    return SignedProportion;

                case ErrorFunction.AbsoluteProportion:
                    return AbsoluteProportion;

                case ErrorFunction.SquaredDifference:
                    return SquaredDifference;

                default:
                    throw Contracts.Except(ErrorFunctionHelpText);
            }
        }
    }

    /// <summary>
    /// The wrapper to <see cref="SsaAnomalyDetectionBase"/> that implements the general anomaly detection transform based on Singular Spectrum modeling of the time-series.
    /// For the details of the Singular Spectrum Analysis (SSA), refer to http://arxiv.org/pdf/1206.6910.pdf.
    /// </summary>
    public class SsaAnomalyDetectionBaseWrapper : IStatefulTransformer, ICanSaveModel
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
        internal abstract class SsaOptions : ArgumentsBase
        {
            [Argument(ArgumentType.Required, HelpText = "The inner window size for SSA in [2, windowSize]", ShortName = "swnd", SortOrder = 11)]
            public int SeasonalWindowSize;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The discount factor in [0, 1]", ShortName = "disc", SortOrder = 12)]
            public Single DiscountFactor = 1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The function used to compute the error between the expected and the observed value", ShortName = "err", SortOrder = 13)]
            public ErrorFunction ErrorFunction = ErrorFunction.SignedDifference;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The flag determing whether the model is adaptive", ShortName = "adp", SortOrder = 14)]
            public bool IsAdaptive = false;
        }

        internal SsaAnomalyDetectionBase InternalTransform;

        internal SsaAnomalyDetectionBaseWrapper(SsaOptions options, string name, IHostEnvironment env)
        {
            InternalTransform = new SsaAnomalyDetectionBase(options, name, env, this);
        }

        internal SsaAnomalyDetectionBaseWrapper(IHostEnvironment env, ModelLoadContext ctx, string name)
        {
            InternalTransform = new SsaAnomalyDetectionBase(env, ctx, name);
        }

        /// <summary>
        /// This base class that implements the general anomaly detection transform based on Singular Spectrum modeling of the time-series.
        /// For the details of the Singular Spectrum Analysis (SSA), refer to http://arxiv.org/pdf/1206.6910.pdf.
        /// </summary>
        internal sealed class SsaAnomalyDetectionBase : SequentialAnomalyDetectionTransformBase<float, SsaAnomalyDetectionBase.State>
        {
            internal SsaAnomalyDetectionBaseWrapper Parent;
            internal readonly int SeasonalWindowSize;
            internal readonly Single DiscountFactor;
            internal readonly bool IsAdaptive;
            internal readonly ErrorFunction ErrorFunction;
            internal readonly Func<Double, Double, Double> ErrorFunc;
            internal SequenceModelerBase<Single, Single> Model;

            public SsaAnomalyDetectionBase(SsaOptions options, string name, IHostEnvironment env, SsaAnomalyDetectionBaseWrapper parent)
                : base(options.WindowSize, 0, options.Source, options.Name, name, env, options.Side, options.Martingale, options.AlertOn, options.PowerMartingaleEpsilon, options.AlertThreshold)
            {
                Host.CheckUserArg(2 <= options.SeasonalWindowSize, nameof(options.SeasonalWindowSize), "Must be at least 2.");
                Host.CheckUserArg(0 <= options.DiscountFactor && options.DiscountFactor <= 1, nameof(options.DiscountFactor), "Must be in the range [0, 1].");
                Host.CheckUserArg(Enum.IsDefined(typeof(ErrorFunction), options.ErrorFunction), nameof(options.ErrorFunction), ErrorFunctionUtils.ErrorFunctionHelpText);

                SeasonalWindowSize = options.SeasonalWindowSize;
                DiscountFactor = options.DiscountFactor;
                ErrorFunction = options.ErrorFunction;
                ErrorFunc = ErrorFunctionUtils.GetErrorFunction(ErrorFunction);
                IsAdaptive = options.IsAdaptive;
                // Creating the master SSA model
                Model = new AdaptiveSingularSpectrumSequenceModeler(Host, options.InitialWindowSize, SeasonalWindowSize + 1, SeasonalWindowSize,
                    DiscountFactor, AdaptiveSingularSpectrumSequenceModeler.RankSelectionMethod.Exact, null, SeasonalWindowSize / 2, false, false);

                StateRef = new State();
                StateRef.InitState(WindowSize, InitialWindowSize, this, Host);
                Parent = parent;
            }

            public SsaAnomalyDetectionBase(IHostEnvironment env, ModelLoadContext ctx, string name)
                : base(env, ctx, name)
            {
                // *** Binary format ***
                // <base>
                // int: _seasonalWindowSize
                // float: _discountFactor
                // byte: _errorFunction
                // bool: _isAdaptive
                // AdaptiveSingularSpectrumSequenceModeler: _model

                Host.CheckDecode(InitialWindowSize == 0);

                SeasonalWindowSize = ctx.Reader.ReadInt32();
                Host.CheckDecode(2 <= SeasonalWindowSize);

                DiscountFactor = ctx.Reader.ReadSingle();
                Host.CheckDecode(0 <= DiscountFactor && DiscountFactor <= 1);

                byte temp;
                temp = ctx.Reader.ReadByte();
                Host.CheckDecode(Enum.IsDefined(typeof(ErrorFunction), temp));
                ErrorFunction = (ErrorFunction)temp;
                ErrorFunc = ErrorFunctionUtils.GetErrorFunction(ErrorFunction);

                IsAdaptive = ctx.Reader.ReadBoolean();
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
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", InputColumnName, "float", colType.ToString());

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
                Host.Assert(2 <= SeasonalWindowSize);
                Host.Assert(0 <= DiscountFactor && DiscountFactor <= 1);
                Host.Assert(Enum.IsDefined(typeof(ErrorFunction), ErrorFunction));
                Host.Assert(Model != null);

                // *** Binary format ***
                // <base>
                // int: _seasonalWindowSize
                // float: _discountFactor
                // byte: _errorFunction
                // bool: _isAdaptive
                // State: StateRef
                // AdaptiveSingularSpectrumSequenceModeler: _model

                base.SaveModel(ctx);
                ctx.Writer.Write(SeasonalWindowSize);
                ctx.Writer.Write(DiscountFactor);
                ctx.Writer.Write((byte)ErrorFunction);
                ctx.Writer.Write(IsAdaptive);
                StateRef.Save(ctx.Writer);

                ctx.SaveModel(Model, "SSA");
            }

            internal sealed class State : AnomalyDetectionStateBase
            {
                private SequenceModelerBase<Single, Single> _model;
                private SsaAnomalyDetectionBase _parentAnomalyDetector;

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
                        _parentAnomalyDetector.Model = _parentAnomalyDetector.Model.Clone();
                        _model = _parentAnomalyDetector.Model;
                    }
                }

                private protected override void LearnStateFromDataCore(FixedSizeQueue<Single> data)
                {
                    // This method is empty because there is no need to implement a training logic here.
                }

                private protected override void InitializeAnomalyDetector()
                {
                    _parentAnomalyDetector = (SsaAnomalyDetectionBase)Parent;
                    _model = _parentAnomalyDetector.Model;
                }

                private protected override double ComputeRawAnomalyScore(ref Single input, FixedSizeQueue<Single> windowedBuffer, long iteration)
                {
                    // Get the prediction for the next point opn the series
                    Single expectedValue = 0;
                    _model.PredictNext(ref expectedValue);

                    if (PreviousPosition == -1)
                        // Feed the current point to the model
                        _model.Consume(ref input, _parentAnomalyDetector.IsAdaptive);

                    // Return the error as the raw anomaly score
                    return _parentAnomalyDetector.ErrorFunc(input, expectedValue);
                }

                public override void Consume(Single input) => _model.Consume(ref input, _parentAnomalyDetector.IsAdaptive);
            }
        }
    }
}
