// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.TimeSeriesProcessing
{
    /// <summary>
    /// Provides the utility functions for different error functions for computing deviation.
    /// </summary>
    public static class ErrorFunctionUtils
    {
        public const string ErrorFunctionHelpText = "The error function should be either (0) SignedDifference, (1) AbsoluteDifference, (2) SignedProportion" +
                                                     " (3) AbsoluteProportion or (4) SquaredDifference.";

        public enum ErrorFunction : byte
        {
            SignedDifference,
            AbsoluteDifference,
            SignedProportion,
            AbsoluteProportion,
            SquaredDifference
        }

        public static Double SignedDifference(Double actual, Double predicted)
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
    /// This base class that implements the general anomaly detection transform based on Singular Spectrum modeling of the time-series.
    /// For the details of the Singular Spectrum Analysis (SSA), refer to http://arxiv.org/pdf/1206.6910.pdf.
    /// </summary>
    public abstract class SsaAnomalyDetectionBase : SequentialAnomalyDetectionTransformBase<Single, SsaAnomalyDetectionBase.State>
    {
        public abstract class SsaArguments : ArgumentsBase
        {
            [Argument(ArgumentType.Required, HelpText = "The inner window size for SSA in [2, windowSize]", ShortName = "swnd", SortOrder = 11)]
            public int SeasonalWindowSize;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The discount factor in [0, 1]", ShortName = "disc", SortOrder = 12)]
            public Single DiscountFactor = 1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The function used to compute the error between the expected and the observed value", ShortName = "err", SortOrder = 13)]
            public ErrorFunctionUtils.ErrorFunction ErrorFunction = ErrorFunctionUtils.ErrorFunction.SignedDifference;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The flag determing whether the model is adaptive", ShortName = "adp", SortOrder = 14)]
            public bool IsAdaptive = false;
        }

        protected readonly int SeasonalWindowSize;
        protected readonly Single DiscountFactor;
        protected readonly bool IsAdaptive;
        protected readonly ErrorFunctionUtils.ErrorFunction ErrorFunction;
        protected readonly Func<Double, Double, Double> ErrorFunc;
        protected readonly ISequenceModeler<Single, Single> Model;

        public SsaAnomalyDetectionBase(SsaArguments args, string name, IHostEnvironment env)
            : base(args.WindowSize, 0, args.Source, args.Name, name, env, args.Side, args.Martingale, args.AlertOn, args.PowerMartingaleEpsilon, args.AlertThreshold)
        {
            Host.CheckUserArg(2 <= args.SeasonalWindowSize, nameof(args.SeasonalWindowSize), "Must be at least 2.");
            Host.CheckUserArg(0 <= args.DiscountFactor && args.DiscountFactor <= 1, nameof(args.DiscountFactor), "Must be in the range [0, 1].");
            Host.CheckUserArg(Enum.IsDefined(typeof(ErrorFunctionUtils.ErrorFunction), args.ErrorFunction), nameof(args.ErrorFunction), ErrorFunctionUtils.ErrorFunctionHelpText);

            SeasonalWindowSize = args.SeasonalWindowSize;
            DiscountFactor = args.DiscountFactor;
            ErrorFunction = args.ErrorFunction;
            ErrorFunc = ErrorFunctionUtils.GetErrorFunction(ErrorFunction);
            IsAdaptive = args.IsAdaptive;
            // Creating the master SSA model
            Model = new AdaptiveSingularSpectrumSequenceModeler(Host, args.InitialWindowSize, SeasonalWindowSize + 1, SeasonalWindowSize,
                DiscountFactor, null, AdaptiveSingularSpectrumSequenceModeler.RankSelectionMethod.Exact, null, SeasonalWindowSize / 2, false, false);
        }

        public void Fit(IDataView input)
        {
            // Training the master SSA model
            var data = new RoleMappedData(input, null, InputColumnName);
            Model.Train(data);
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
            Host.CheckDecode(Enum.IsDefined(typeof(ErrorFunctionUtils.ErrorFunction), temp));
            ErrorFunction = (ErrorFunctionUtils.ErrorFunction)temp;
            ErrorFunc = ErrorFunctionUtils.GetErrorFunction(ErrorFunction);

            IsAdaptive = ctx.Reader.ReadBoolean();

            ctx.LoadModel<ISequenceModeler<Single, Single>, SignatureLoadModel>(env, out Model, "SSA");
            Host.CheckDecode(Model != null);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();

            Host.Assert(InitialWindowSize == 0);
            Host.Assert(2 <= SeasonalWindowSize);
            Host.Assert(0 <= DiscountFactor && DiscountFactor <= 1);
            Host.Assert(Enum.IsDefined(typeof(ErrorFunctionUtils.ErrorFunction), ErrorFunction));
            Host.Assert(Model != null);

            // *** Binary format ***
            // <base>
            // int: _seasonalWindowSize
            // float: _discountFactor
            // byte: _errorFunction
            // bool: _isAdaptive
            // AdaptiveSingularSpectrumSequenceModeler: _model

            base.Save(ctx);
            ctx.Writer.Write(SeasonalWindowSize);
            ctx.Writer.Write(DiscountFactor);
            ctx.Writer.Write((byte)ErrorFunction);
            ctx.Writer.Write(IsAdaptive);
            ctx.SaveModel(Model, "SSA");
        }

        public sealed class State : AnomalyDetectionStateBase
        {
            private ISequenceModeler<Single, Single> _model;
            private SsaAnomalyDetectionBase _parentAnomalyDetector;

            protected override void LearnStateFromDataCore(FixedSizeQueue<Single> data)
            {
                // This method is empty because there is no need to implement a training logic here.
            }

            protected override void InitializeAnomalyDetector()
            {
                _parentAnomalyDetector = (SsaAnomalyDetectionBase)Parent;
                _model = _parentAnomalyDetector.Model.Clone();
                _model.InitState();
            }

            protected override double ComputeRawAnomalyScore(ref Single input, FixedSizeQueue<Single> windowedBuffer, long iteration)
            {
                // Get the prediction for the next point opn the series
                Single expectedValue = 0;
                _model.PredictNext(ref expectedValue);

                // Feed the current point to the model
                _model.Consume(ref input, _parentAnomalyDetector.IsAdaptive);

                // Return the error as the raw anomaly score
                return _parentAnomalyDetector.ErrorFunc(input, expectedValue);
            }
        }
    }
}
