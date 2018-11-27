// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.TimeSeriesProcessing;
using Microsoft.ML.StaticPipe.Runtime;
using System.Collections.Generic;

namespace Microsoft.ML.StaticPipe
{
    using IidBase = Microsoft.ML.Runtime.TimeSeriesProcessing.SequentialAnomalyDetectionTransformBase<float, Microsoft.ML.Runtime.TimeSeriesProcessing.IidAnomalyDetectionBase.State>;
    using SsaBase = Microsoft.ML.Runtime.TimeSeriesProcessing.SequentialAnomalyDetectionTransformBase<float, Microsoft.ML.Runtime.TimeSeriesProcessing.SsaAnomalyDetectionBase.State>;

    /// <summary>
    /// IidChangePoint static API extension methods.
    /// </summary>
    public static class IidChangePointStaticExtensions
    {
        private sealed class OutColumn : Vector<double>
        {
            public PipelineColumn Input { get; }

            public OutColumn(
                Scalar<float> input,
                int confidence,
                int changeHistoryLength,
                IidBase.MartingaleType martingale,
                double eps)
                : base(new Reconciler(confidence, changeHistoryLength, martingale, eps), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly int _confidence;
            private readonly int _changeHistoryLength;
            private readonly IidBase.MartingaleType _martingale;
            private readonly double _eps;

            public Reconciler(
                int confidence,
                int changeHistoryLength,
                IidBase.MartingaleType martingale,
                double eps)
            {
                _confidence = confidence;
                _changeHistoryLength = changeHistoryLength;
                _martingale = martingale;
                _eps = eps;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);
                var outCol = (OutColumn)toOutput[0];
                return new IidChangePointEstimator(env,
                    inputNames[outCol.Input],
                    outputNames[outCol],
                    _confidence,
                    _changeHistoryLength,
                    _martingale,
                    _eps);
            }
        }

        public static Vector<double> IidChangePointDetect(
            this Scalar<float> input,
            int confidence,
            int changeHistoryLength,
            IidBase.MartingaleType martingale = IidBase.MartingaleType.Power,
            double eps = 0.1) => new OutColumn(input, confidence, changeHistoryLength, martingale, eps);
    }

    /// <summary>
    /// Extension methods for the static-pipeline over <see cref="PipelineColumn"/> objects.
    /// </summary>
    public static class IidSpikeDetectorStaticExtensions
    {
        private sealed class OutColumn : Vector<double>
        {
            public PipelineColumn Input { get; }

            public OutColumn(Scalar<float> input,
                int confidence,
                int pvalueHistoryLength,
                IidBase.AnomalySide side)
                : base(new Reconciler(confidence, pvalueHistoryLength, side), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly int _confidence;
            private readonly int _pvalueHistoryLength;
            private readonly IidBase.AnomalySide _side;

            public Reconciler(
                int confidence,
                int pvalueHistoryLength,
                IidBase.AnomalySide side)
            {
                _confidence = confidence;
                _pvalueHistoryLength = pvalueHistoryLength;
                _side = side;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);
                var outCol = (OutColumn)toOutput[0];
                return new IidSpikeEstimator(env,
                    inputNames[outCol.Input],
                    outputNames[outCol],
                    _confidence,
                    _pvalueHistoryLength,
                    _side);
            }
        }
        public static Vector<double> IidSpikeDetect(
            this Scalar<float> input,
            int confidence,
            int pvalueHistoryLength,
            IidBase.AnomalySide side = IidBase.AnomalySide.TwoSided
            ) => new OutColumn(input, confidence, pvalueHistoryLength, side);
    }
    /// <summary>
    /// Extension methods for the static-pipeline over <see cref="PipelineColumn"/> objects.
    /// </summary>
    public static class SsaChangePointStaticExtensions
    {
        private sealed class OutColumn : Vector<double>
        {
            public PipelineColumn Input { get; }

            public OutColumn(Scalar<float> input,
                int confidence,
                int changeHistoryLength,
                int trainingWindowSize,
                int seasonalityWindowSize,
                ErrorFunctionUtils.ErrorFunction errorFunction,
                SsaBase.MartingaleType martingale,
                double eps)
                : base(new Reconciler(confidence, changeHistoryLength, trainingWindowSize, seasonalityWindowSize, errorFunction, martingale, eps), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly int _confidence;
            private readonly int _changeHistoryLength;
            private readonly int _trainingWindowSize;
            private readonly int _seasonalityWindowSize;
            private readonly ErrorFunctionUtils.ErrorFunction _errorFunction;
            private readonly SsaBase.MartingaleType _martingale;
            private readonly double _eps;

            public Reconciler(
                int confidence,
                int changeHistoryLength,
                int trainingWindowSize,
                int seasonalityWindowSize,
                ErrorFunctionUtils.ErrorFunction errorFunction,
                SsaBase.MartingaleType martingale,
                double eps)
            {
                _confidence = confidence;
                _changeHistoryLength = changeHistoryLength;
                _trainingWindowSize = trainingWindowSize;
                _seasonalityWindowSize = seasonalityWindowSize;
                _errorFunction = errorFunction;
                _martingale = martingale;
                _eps = eps;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);
                var outCol = (OutColumn)toOutput[0];
                return new SsaChangePointEstimator(env,
                    inputNames[outCol.Input],
                    outputNames[outCol],
                    _confidence,
                    _changeHistoryLength,
                    _trainingWindowSize,
                    _seasonalityWindowSize,
                    _errorFunction,
                    _martingale,
                    _eps);
            }
        }

        public static Vector<double> SsaChangePointDetect(
            this Scalar<float> input,
            int confidence,
            int changeHistoryLength,
            int trainingWindowSize,
            int seasonalityWindowSize,
            ErrorFunctionUtils.ErrorFunction errorFunction = ErrorFunctionUtils.ErrorFunction.SignedDifference,
            SsaBase.MartingaleType martingale = SsaBase.MartingaleType.Power,
            double eps = 0.1) => new OutColumn(input, confidence, changeHistoryLength, trainingWindowSize, seasonalityWindowSize, errorFunction, martingale, eps);
    }

    /// <summary>
    /// Extension methods for the static-pipeline over <see cref="PipelineColumn"/> objects.
    /// </summary>
    public static class SsaSpikeDetecotStaticExtensions
    {
        private sealed class OutColumn : Vector<double>
        {
            public PipelineColumn Input { get; }

            public OutColumn(Scalar<float> input,
                int confidence,
                int pvalueHistoryLength,
                int trainingWindowSize,
                int seasonalityWindowSize,
                SsaBase.AnomalySide side,
                ErrorFunctionUtils.ErrorFunction errorFunction)
                : base(new Reconciler(confidence, pvalueHistoryLength, trainingWindowSize, seasonalityWindowSize, side, errorFunction), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly int _confidence;
            private readonly int _pvalueHistoryLength;
            private readonly int _trainingWindowSize;
            private readonly int _seasonalityWindowSize;
            private readonly SsaBase.AnomalySide _side;
            private readonly ErrorFunctionUtils.ErrorFunction _errorFunction;

            public Reconciler(
                int confidence,
                int pvalueHistoryLength,
                int trainingWindowSize,
                int seasonalityWindowSize,
                SsaBase.AnomalySide side,
                ErrorFunctionUtils.ErrorFunction errorFunction)
            {
                _confidence = confidence;
                _pvalueHistoryLength = pvalueHistoryLength;
                _trainingWindowSize = trainingWindowSize;
                _seasonalityWindowSize = seasonalityWindowSize;
                _side = side;
                _errorFunction = errorFunction;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);
                var outCol = (OutColumn)toOutput[0];
                return new SsaSpikeEstimator(env,
                    inputNames[outCol.Input],
                    outputNames[outCol],
                    _confidence,
                    _pvalueHistoryLength,
                    _trainingWindowSize,
                    _seasonalityWindowSize,
                    _side,
                    _errorFunction);
            }
        }

    public static Vector<double> SsaSpikeDetect(
        this Scalar<float> input,
        int confidence,
        int changeHistoryLength,
        int trainingWindowSize,
        int seasonalityWindowSize,
        SsaBase.AnomalySide side = SsaBase.AnomalySide.TwoSided,
        ErrorFunctionUtils.ErrorFunction errorFunction = ErrorFunctionUtils.ErrorFunction.SignedDifference
        ) => new OutColumn(input, confidence, changeHistoryLength, trainingWindowSize, seasonalityWindowSize, side, errorFunction);

    }
}
