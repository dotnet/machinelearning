// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.TimeSeries;

namespace Microsoft.ML.StaticPipe
{

    /// <summary>
    /// Static API extension methods for <see cref="IidChangePointEstimator"/>.
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
                MartingaleType martingale,
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
            private readonly MartingaleType _martingale;
            private readonly double _eps;

            public Reconciler(
                int confidence,
                int changeHistoryLength,
                MartingaleType martingale,
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
                return new MLContext().Transforms.DetectIidChangePoint(
                    outputNames[outCol],
                    inputNames[outCol.Input],
                    _confidence,
                    _changeHistoryLength,
                    _martingale,
                    _eps);
            }
        }

        /// <summary>
        /// Perform IID change point detection over a column of time series data. See <see cref="IidChangePointEstimator"/>.
        /// </summary>
        public static Vector<double> DetectIidChangePoint(
            this Scalar<float> input,
            int confidence,
            int changeHistoryLength,
            MartingaleType martingale = MartingaleType.Power,
            double eps = 0.1) => new OutColumn(input, confidence, changeHistoryLength, martingale, eps);
    }

    /// <summary>
    /// Static API extension methods for <see cref="IidSpikeEstimator"/>.
    /// </summary>
    public static class IidSpikeDetectorStaticExtensions
    {
        private sealed class OutColumn : Vector<double>
        {
            public PipelineColumn Input { get; }

            public OutColumn(Scalar<float> input,
                int confidence,
                int pvalueHistoryLength,
                AnomalySide side)
                : base(new Reconciler(confidence, pvalueHistoryLength, side), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly int _confidence;
            private readonly int _pvalueHistoryLength;
            private readonly AnomalySide _side;

            public Reconciler(
                int confidence,
                int pvalueHistoryLength,
                AnomalySide side)
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
                return new MLContext().Transforms.DetectIidSpike(
                    outputNames[outCol],
                    inputNames[outCol.Input],
                    _confidence,
                    _pvalueHistoryLength,
                    _side);
            }
        }

        /// <summary>
        /// Perform IID spike detection over a column of time series data. See <see cref="IidSpikeEstimator"/>.
        /// </summary>
        public static Vector<double> DetectIidSpike(
            this Scalar<float> input,
            int confidence,
            int pvalueHistoryLength,
            AnomalySide side = AnomalySide.TwoSided
            ) => new OutColumn(input, confidence, pvalueHistoryLength, side);
    }

    /// <summary>
    /// Static API extension methods for <see cref="SsaChangePointEstimator"/>.
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
                ErrorFunction errorFunction,
                MartingaleType martingale,
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
            private readonly ErrorFunction _errorFunction;
            private readonly MartingaleType _martingale;
            private readonly double _eps;

            public Reconciler(
                int confidence,
                int changeHistoryLength,
                int trainingWindowSize,
                int seasonalityWindowSize,
                ErrorFunction errorFunction,
                MartingaleType martingale,
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
                return new MLContext().Transforms.DetectChangePointBySsa(
                    outputNames[outCol],
                    inputNames[outCol.Input],
                    _confidence,
                    _changeHistoryLength,
                    _trainingWindowSize,
                    _seasonalityWindowSize,
                    _errorFunction,
                    _martingale,
                    _eps);
            }
        }

        /// <summary>
        /// Perform SSA change point detection over a column of time series data. See <see cref="SsaChangePointEstimator"/>.
        /// </summary>
        public static Vector<double> DetectChangePointBySsa(
            this Scalar<float> input,
            int confidence,
            int changeHistoryLength,
            int trainingWindowSize,
            int seasonalityWindowSize,
            ErrorFunction errorFunction = ErrorFunction.SignedDifference,
            MartingaleType martingale = MartingaleType.Power,
            double eps = 0.1) => new OutColumn(input, confidence, changeHistoryLength, trainingWindowSize, seasonalityWindowSize, errorFunction, martingale, eps);
    }

    /// <summary>
    /// Static API extension methods for <see cref="SsaSpikeEstimator"/>.
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
                AnomalySide side,
                ErrorFunction errorFunction)
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
            private readonly AnomalySide _side;
            private readonly ErrorFunction _errorFunction;

            public Reconciler(
                int confidence,
                int pvalueHistoryLength,
                int trainingWindowSize,
                int seasonalityWindowSize,
                AnomalySide side,
                ErrorFunction errorFunction)
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
                return new MLContext().Transforms.DetectSpikeBySsa(
                    outputNames[outCol],
                    inputNames[outCol.Input],
                    _confidence,
                    _pvalueHistoryLength,
                    _trainingWindowSize,
                    _seasonalityWindowSize,
                    _side,
                    _errorFunction);
            }
        }

        /// <summary>
        /// Perform SSA spike detection over a column of time series data. See <see cref="SsaSpikeEstimator"/>.
        /// </summary>
        public static Vector<double> DetectSpikeBySsa(
            this Scalar<float> input,
            int confidence,
            int changeHistoryLength,
            int trainingWindowSize,
            int seasonalityWindowSize,
            AnomalySide side = AnomalySide.TwoSided,
            ErrorFunction errorFunction = ErrorFunction.SignedDifference
            ) => new OutColumn(input, confidence, changeHistoryLength, trainingWindowSize, seasonalityWindowSize, side, errorFunction);

    }
}
