// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.TimeSeriesProcessing;

[assembly: LoadableClass(typeof(ISequenceModeler<Single, Single>), typeof(AdaptiveSingularSpectrumSequenceModeler), null, typeof(SignatureLoadModel),
    "SSA Sequence Modeler",
    AdaptiveSingularSpectrumSequenceModeler.LoaderSignature)]

namespace Microsoft.ML.Runtime.TimeSeriesProcessing
{
    /// <summary>
    /// This class implements basic Singular Spectrum Analysis (SSA) model for modeling univariate time-series.
    /// For the details of the model, refer to http://arxiv.org/pdf/1206.6910.pdf.
    /// </summary>
    public sealed class AdaptiveSingularSpectrumSequenceModeler : ISequenceModeler<Single, Single>
    {
        public const string LoaderSignature = "SSAModel";

        public enum RankSelectionMethod
        {
            Fixed,
            Exact,
            Fast
        }

        public sealed class SsaForecastResult : ForecastResultBase<Single>
        {
            public VBuffer<Single> ForecastStandardDeviation;
            public VBuffer<Single> UpperBound;
            public VBuffer<Single> LowerBound;
            public Single ConfidenceLevel;

            internal bool CanComputeForecastIntervals;
            internal Single BoundOffset;

            public bool IsVarianceValid { get { return CanComputeForecastIntervals; } }
        }

        public struct GrowthRatio
        {
            private int _timeSpan;
            private Double _growth;

            public int TimeSpan
            {
                get
                {
                    return _timeSpan;
                }
                set
                {
                    Contracts.CheckParam(value > 0, nameof(TimeSpan), "The time span must be strictly positive.");
                    _timeSpan = value;
                }
            }

            public Double Growth
            {
                get
                {
                    return _growth;
                }
                set
                {
                    Contracts.CheckParam(value >= 0, nameof(Growth), "The growth must be non-negative.");
                    _growth = value;
                }
            }

            public GrowthRatio(int timeSpan = 1, double growth = Double.PositiveInfinity)
            {
                Contracts.CheckParam(timeSpan > 0, nameof(TimeSpan), "The time span must be strictly positive.");
                Contracts.CheckParam(growth >= 0, nameof(Growth), "The growth must be non-negative.");

                _growth = growth;
                _timeSpan = timeSpan;
            }

            public Double Ratio { get { return Math.Pow(_growth, 1d / _timeSpan); } }
        }

        public sealed class ModelInfo
        {
            /// <summary>
            /// The singular values of the SSA of the input time-series
            /// </summary>
            public Single[] Spectrum;

            /// <summary>
            /// The roots of the characteristic polynomial after stabilization (meaningful only if the model is stabilized.)
            /// </summary>
            public Complex[] RootsAfterStabilization;

            /// <summary>
            /// The roots of the characteristic polynomial before stabilization (meaningful only if the model is stabilized.)
            /// </summary>
            public Complex[] RootsBeforeStabilization;

            /// <summary>
            /// The rank of the model
            /// </summary>
            public int Rank;

            /// <summary>
            /// The window size used to compute the SSA model
            /// </summary>
            public int WindowSize;

            /// <summary>
            /// The auto-regressive coefficients learned by the model
            /// </summary>
            public Single[] AutoRegressiveCoefficients;

            /// <summary>
            /// The flag indicating whether the model has been trained
            /// </summary>
            public bool IsTrained;

            /// <summary>
            /// The flag indicating a naive model is trained instead of SSA
            /// </summary>
            public bool IsNaiveModelTrained;

            /// <summary>
            /// The flag indicating whether the learned model has an exponential trend (meaningful only if the model is stabilized.)
            /// </summary>
            public bool IsExponentialTrendPresent;

            /// <summary>
            /// The flag indicating whether the learned model has a polynomial trend (meaningful only if the model is stabilized.)
            /// </summary>
            public bool IsPolynomialTrendPresent;

            /// <summary>
            /// The flag indicating whether the learned model has been stabilized
            /// </summary>
            public bool IsStabilized;

            /// <summary>
            /// The flag indicating whether any artificial seasonality (a seasonality with period greater than the window size) is removed
            /// (meaningful only if the model is stabilized.)
            /// </summary>
            public bool IsArtificialSeasonalityRemoved;

            /// <summary>
            /// The exponential trend magnitude (meaningful only if the model is stabilized.)
            /// </summary>
            public Double ExponentialTrendFactor;
        }

        private ModelInfo _info;

        /// <summary>
        /// Returns the meta information about the learned model.
        /// </summary>
        public ModelInfo Info
        {
            get { return _info; }
        }

        private Single[] _alpha;
        private Single[] _state;
        private readonly FixedSizeQueue<Single> _buffer;
        private CpuAlignedVector _x;
        private CpuAlignedVector _xSmooth;
        private int _windowSize;
        private readonly int _seriesLength;
        private readonly RankSelectionMethod _rankSelectionMethod;
        private readonly Single _discountFactor;
        private readonly int _trainSize;
        private int _maxRank;
        private readonly Double _maxTrendRatio;
        private readonly bool _shouldStablize;
        private readonly bool _shouldMaintainInfo;

        private readonly IHost _host;

        private CpuAlignedMatrixRow _wTrans;
        private Single _observationNoiseVariance;
        private Single _observationNoiseMean;
        private Single _autoregressionNoiseVariance;
        private Single _autoregressionNoiseMean;

        private int _rank;
        private CpuAlignedVector _y;
        private Single _nextPrediction;

        /// <summary>
        /// Determines whether the confidence interval required for forecasting.
        /// </summary>
        public bool ShouldComputeForecastIntervals { get; set; }

        /// <summary>
        /// Returns the rank of the subspace used for SSA projection (parameter r).
        /// </summary>
        public int Rank { get { return _rank; } }

        /// <summary>
        /// Returns the smoothed (via SSA projection) version of the last series observation fed to the model.
        /// </summary>
        public Single LastSmoothedValue { get { return _state[_windowSize - 2]; } }

        /// <summary>
        /// Returns the last series observation fed to the model.
        /// </summary>
        public Single LastValue { get { return _buffer.Count > 0 ? _buffer[_buffer.Count - 1] : Single.NaN; } }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SSAMODLR",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(AdaptiveSingularSpectrumSequenceModeler).Assembly.FullName);
        }

        /// <summary>
        /// The constructor for Adaptive SSA model.
        /// </summary>
        /// <param name="env">The exception context.</param>
        /// <param name="trainSize">The length of series from the begining used for training.</param>
        /// <param name="seriesLength">The length of series that is kept in buffer for modeling (parameter N).</param>
        /// <param name="windowSize">The length of the window on the series for building the trajectory matrix (parameter L).</param>
        /// <param name="discountFactor">The discount factor in [0,1] used for online updates (default = 1).</param>
        /// <param name="buffer">The buffer used to keep the series in the memory. If null, an internal buffer is created (default = null).</param>
        /// <param name="rankSelectionMethod">The rank selection method (default = Exact).</param>
        /// <param name="rank">The desired rank of the subspace used for SSA projection (parameter r). This parameter should be in the range in [1, windowSize].
        /// If set to null, the rank is automatically determined based on prediction error minimization. (default = null)</param>
        /// <param name="maxRank">The maximum rank considered during the rank selection process. If not provided (i.e. set to null), it is set to windowSize - 1.</param>
        /// <param name="shouldComputeForecastIntervals">The flag determining whether the confidence bounds for the point forecasts should be computed. (default = true)</param>
        /// <param name="shouldstablize">The flag determining whether the model should be stabilized.</param>
        /// <param name="shouldMaintainInfo">The flag determining whether the meta information for the model needs to be maintained.</param>
        /// <param name="maxGrowth">The maximum growth on the exponential trend</param>
        public AdaptiveSingularSpectrumSequenceModeler(IHostEnvironment env, int trainSize, int seriesLength, int windowSize, Single discountFactor = 1,
            FixedSizeQueue<Single> buffer = null, RankSelectionMethod rankSelectionMethod = RankSelectionMethod.Exact, int? rank = null, int? maxRank = null,
            bool shouldComputeForecastIntervals = true, bool shouldstablize = true, bool shouldMaintainInfo = false, GrowthRatio? maxGrowth = null)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoaderSignature);
            _host.CheckParam(windowSize >= 2, nameof(windowSize), "The window size should be at least 2."); // ...because otherwise we have nothing to autoregress on
            _host.CheckParam(seriesLength > windowSize, nameof(seriesLength), "The series length should be greater than the window size.");
            _host.Check(trainSize > 2 * windowSize, "The input series length for training should be greater than twice the window size.");
            _host.CheckParam(0 <= discountFactor && discountFactor <= 1, nameof(discountFactor), "The discount factor should be in [0,1].");

            if (maxRank != null)
            {
                _maxRank = (int)maxRank;
                _host.CheckParam(1 <= _maxRank && _maxRank < windowSize, nameof(maxRank),
                    "The max rank should be in [1, windowSize).");
            }
            else
                _maxRank = windowSize - 1;

            _rankSelectionMethod = rankSelectionMethod;
            if (_rankSelectionMethod == RankSelectionMethod.Fixed)
            {
                if (rank != null)
                {
                    _rank = (int)rank;
                    _host.CheckParam(1 <= _rank && _rank < windowSize, nameof(rank), "The rank should be in [1, windowSize).");
                }
                else
                    _rank = _maxRank;
            }

            _seriesLength = seriesLength;
            _windowSize = windowSize;
            _trainSize = trainSize;
            _discountFactor = discountFactor;

            if (buffer == null)
                _buffer = new FixedSizeQueue<Single>(seriesLength);
            else
                _buffer = buffer;

            _alpha = new Single[windowSize - 1];
            _state = new Single[windowSize - 1];
            _x = new CpuAlignedVector(windowSize, SseUtils.CbAlign);
            _xSmooth = new CpuAlignedVector(windowSize, SseUtils.CbAlign);
            ShouldComputeForecastIntervals = shouldComputeForecastIntervals;

            _observationNoiseVariance = 0;
            _autoregressionNoiseVariance = 0;
            _observationNoiseMean = 0;
            _autoregressionNoiseMean = 0;
            _shouldStablize = shouldstablize;
            _maxTrendRatio = maxGrowth == null ? Double.PositiveInfinity : ((GrowthRatio)maxGrowth).Ratio;

            _shouldMaintainInfo = shouldMaintainInfo;
            if (_shouldMaintainInfo)
            {
                _info = new ModelInfo();
                _info.WindowSize = _windowSize;
            }
        }

        /// <summary>
        /// The copy constructor
        /// </summary>
        /// <param name="model">An object whose contents are copied to the current object.</param>
        private AdaptiveSingularSpectrumSequenceModeler(AdaptiveSingularSpectrumSequenceModeler model)
        {
            _host = model._host.Register(LoaderSignature);
            _host.Assert(model._windowSize >= 2);
            _host.Assert(model._seriesLength > model._windowSize);
            _host.Assert(model._trainSize > 2 * model._windowSize);
            _host.Assert(0 <= model._discountFactor && model._discountFactor <= 1);
            _host.Assert(1 <= model._rank && model._rank < model._windowSize);

            _rank = model._rank;
            _maxRank = model._maxRank;
            _rankSelectionMethod = model._rankSelectionMethod;
            _seriesLength = model._seriesLength;
            _windowSize = model._windowSize;
            _trainSize = model._trainSize;
            _discountFactor = model._discountFactor;
            ShouldComputeForecastIntervals = model.ShouldComputeForecastIntervals;
            _observationNoiseVariance = model._observationNoiseVariance;
            _autoregressionNoiseVariance = model._autoregressionNoiseVariance;
            _observationNoiseMean = model._observationNoiseMean;
            _autoregressionNoiseMean = model._autoregressionNoiseMean;
            _nextPrediction = model._nextPrediction;
            _maxTrendRatio = model._maxTrendRatio;
            _shouldStablize = model._shouldStablize;
            _shouldMaintainInfo = model._shouldMaintainInfo;
            _info = model._info;
            _buffer = new FixedSizeQueue<Single>(_seriesLength);
            _alpha = new Single[_windowSize - 1];
            Array.Copy(model._alpha, _alpha, _windowSize - 1);
            _state = new Single[_windowSize - 1];
            Array.Copy(model._state, _state, _windowSize - 1);

            _x = new CpuAlignedVector(_windowSize, SseUtils.CbAlign);
            _xSmooth = new CpuAlignedVector(_windowSize, SseUtils.CbAlign);

            if (model._wTrans != null)
            {
                _y = new CpuAlignedVector(_rank, SseUtils.CbAlign);
                _wTrans = new CpuAlignedMatrixRow(_rank, _windowSize, SseUtils.CbAlign);
                _wTrans.CopyFrom(model._wTrans);
            }
        }

        public AdaptiveSingularSpectrumSequenceModeler(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoaderSignature);

            // *** Binary format ***
            // int: _seriesLength
            // int: _windowSize
            // int: _trainSize
            // int: _rank
            // float: _discountFactor
            // RankSelectionMethod: _rankSelectionMethod
            // bool: isWeightSet
            // float[]: _alpha
            // float[]: _state
            // bool: ShouldComputeForecastIntervals
            // float: _observationNoiseVariance
            // float: _autoregressionNoiseVariance
            // float: _observationNoiseMean
            // float: _autoregressionNoiseMean
            // float: _nextPrediction
            // int: _maxRank
            // bool: _shouldStablize
            // bool: _shouldMaintainInfo
            // double: _maxTrendRatio
            // float[]: _wTrans (only if _isWeightSet == true)

            _seriesLength = ctx.Reader.ReadInt32();
            // Do an early check. We'll have the stricter check later.
            _host.CheckDecode(_seriesLength > 2);

            _windowSize = ctx.Reader.ReadInt32();
            _host.CheckDecode(_windowSize >= 2);
            _host.CheckDecode(_seriesLength > _windowSize);

            _trainSize = ctx.Reader.ReadInt32();
            _host.CheckDecode(_trainSize > 2 * _windowSize);

            _rank = ctx.Reader.ReadInt32();
            _host.CheckDecode(1 <= _rank && _rank < _windowSize);

            _discountFactor = ctx.Reader.ReadSingle();
            _host.CheckDecode(0 <= _discountFactor && _discountFactor <= 1);

            byte temp;
            temp = ctx.Reader.ReadByte();
            _rankSelectionMethod = (RankSelectionMethod)temp;
            bool isWeightSet = ctx.Reader.ReadBoolByte();

            _alpha = ctx.Reader.ReadFloatArray();
            _host.CheckDecode(Utils.Size(_alpha) == _windowSize - 1);

            _state = ctx.Reader.ReadFloatArray();
            _host.CheckDecode(Utils.Size(_state) == _windowSize - 1);

            ShouldComputeForecastIntervals = ctx.Reader.ReadBoolByte();

            _observationNoiseVariance = ctx.Reader.ReadSingle();
            _host.CheckDecode(_observationNoiseVariance >= 0);

            _autoregressionNoiseVariance = ctx.Reader.ReadSingle();
            _host.CheckDecode(_autoregressionNoiseVariance >= 0);

            _observationNoiseMean = ctx.Reader.ReadSingle();
            _autoregressionNoiseMean = ctx.Reader.ReadSingle();
            _nextPrediction = ctx.Reader.ReadSingle();

            _maxRank = ctx.Reader.ReadInt32();
            _host.CheckDecode(1 <= _maxRank && _maxRank <= _windowSize - 1);

            _shouldStablize = ctx.Reader.ReadBoolByte();
            _shouldMaintainInfo = ctx.Reader.ReadBoolByte();
            if (_shouldMaintainInfo)
            {
                _info = new ModelInfo();
                _info.AutoRegressiveCoefficients = new Single[_windowSize - 1];
                Array.Copy(_alpha, _info.AutoRegressiveCoefficients, _windowSize - 1);

                _info.IsStabilized = _shouldStablize;
                _info.Rank = _rank;
                _info.WindowSize = _windowSize;
            }

            _maxTrendRatio = ctx.Reader.ReadDouble();
            _host.CheckDecode(_maxTrendRatio >= 0);

            if (isWeightSet)
            {
                var tempArray = ctx.Reader.ReadFloatArray();
                _host.CheckDecode(Utils.Size(tempArray) == _rank * _windowSize);
                _wTrans = new CpuAlignedMatrixRow(_rank, _windowSize, SseUtils.CbAlign);
                int i = 0;
                _wTrans.CopyFrom(tempArray, ref i);
            }

            _buffer = new FixedSizeQueue<Single>(_seriesLength);

            _x = new CpuAlignedVector(_windowSize, SseUtils.CbAlign);
            _xSmooth = new CpuAlignedVector(_windowSize, SseUtils.CbAlign);
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            _host.Assert(_windowSize >= 2);
            _host.Assert(_seriesLength > _windowSize);
            _host.Assert(_trainSize > 2 * _windowSize);
            _host.Assert(0 <= _discountFactor && _discountFactor <= 1);
            _host.Assert(1 <= _rank && _rank < _windowSize);
            _host.Assert(Utils.Size(_alpha) == _windowSize - 1);
            _host.Assert(_observationNoiseVariance >= 0);
            _host.Assert(_autoregressionNoiseVariance >= 0);
            _host.Assert(1 <= _maxRank && _maxRank <= _windowSize - 1);
            _host.Assert(_maxTrendRatio >= 0);

            // *** Binary format ***
            // int: _seriesLength
            // int: _windowSize
            // int: _trainSize
            // int: _rank
            // float: _discountFactor
            // RankSelectionMethod: _rankSelectionMethod
            // bool: _isWeightSet
            // float[]: _alpha
            // float[]: _state
            // bool: ShouldComputeForecastIntervals
            // float: _observationNoiseVariance
            // float: _autoregressionNoiseVariance
            // float: _observationNoiseMean
            // float: _autoregressionNoiseMean
            // float: _nextPrediction
            // int: _maxRank
            // bool: _shouldStablize
            // bool: _shouldMaintainInfo
            // double: _maxTrendRatio
            // float[]: _wTrans (only if _isWeightSet == true)

            ctx.Writer.Write(_seriesLength);
            ctx.Writer.Write(_windowSize);
            ctx.Writer.Write(_trainSize);
            ctx.Writer.Write(_rank);
            ctx.Writer.Write(_discountFactor);
            ctx.Writer.Write((byte)_rankSelectionMethod);
            ctx.Writer.WriteBoolByte(_wTrans != null);
            ctx.Writer.WriteFloatArray(_alpha);
            ctx.Writer.WriteFloatArray(_state);
            ctx.Writer.WriteBoolByte(ShouldComputeForecastIntervals);
            ctx.Writer.Write(_observationNoiseVariance);
            ctx.Writer.Write(_autoregressionNoiseVariance);
            ctx.Writer.Write(_observationNoiseMean);
            ctx.Writer.Write(_autoregressionNoiseMean);
            ctx.Writer.Write(_nextPrediction);
            ctx.Writer.Write(_maxRank);
            ctx.Writer.WriteBoolByte(_shouldStablize);
            ctx.Writer.WriteBoolByte(_shouldMaintainInfo);
            ctx.Writer.Write(_maxTrendRatio);

            if (_wTrans != null)
            {
                // REVIEW: this may not be the most efficient way for serializing an aligned matrix.
                var tempArray = new Single[_rank * _windowSize];
                int iv = 0;
                _wTrans.CopyTo(tempArray, ref iv);
                ctx.Writer.WriteFloatArray(tempArray);
            }
        }

        private static void ReconstructSignal(TrajectoryMatrix tMat, Single[] singularVectors, int rank, Single[] output)
        {
            Contracts.Assert(tMat != null);
            Contracts.Assert(1 <= rank && rank <= tMat.WindowSize);
            Contracts.Assert(Utils.Size(singularVectors) >= tMat.WindowSize * rank);
            Contracts.Assert(Utils.Size(output) >= tMat.SeriesLength);

            var k = tMat.SeriesLength - tMat.WindowSize + 1;
            Contracts.Assert(k > 0);

            var v = new Single[k];
            int i;

            for (i = 0; i < tMat.SeriesLength; ++i)
                output[i] = 0;

            for (i = 0; i < rank; ++i)
            {
                // Reconstructing the i-th eigen triple component and adding it to output
                tMat.MultiplyTranspose(singularVectors, v, false, tMat.WindowSize * i, 0);
                tMat.RankOneHankelization(singularVectors, v, 1, output, true, tMat.WindowSize * i, 0, 0);
            }
        }

        private static void ReconstructSignalTailFast(Single[] series, TrajectoryMatrix tMat, Single[] singularVectors, int rank, Single[] output)
        {
            Contracts.Assert(tMat != null);
            Contracts.Assert(1 <= rank && rank <= tMat.WindowSize);
            Contracts.Assert(Utils.Size(singularVectors) >= tMat.WindowSize * rank);

            int len = 2 * tMat.WindowSize - 1;
            Contracts.Assert(Utils.Size(output) >= len);

            Single v;
            /*var k = tMat.SeriesLength - tMat.WindowSize + 1;
            Contracts.Assert(k > 0);
            var v = new Single[k];*/

            Single temp;
            int i;
            int j;
            int offset1 = tMat.SeriesLength - 2 * tMat.WindowSize + 1;
            int offset2 = tMat.SeriesLength - tMat.WindowSize;

            for (i = 0; i < len; ++i)
                output[i] = 0;

            for (i = 0; i < rank; ++i)
            {
                // Reconstructing the i-th eigen triple component and adding it to output
                v = 0;
                for (j = offset1; j < offset1 + tMat.WindowSize; ++j)
                    v += series[j] * singularVectors[tMat.WindowSize * i - offset1 + j];

                for (j = 0; j < tMat.WindowSize - 1; ++j)
                    output[j] += v * singularVectors[tMat.WindowSize * i + j];

                temp = v * singularVectors[tMat.WindowSize * (i + 1) - 1];

                v = 0;
                for (j = offset2; j < offset2 + tMat.WindowSize; ++j)
                    v += series[j] * singularVectors[tMat.WindowSize * i - offset2 + j];

                for (j = tMat.WindowSize; j < 2 * tMat.WindowSize - 1; ++j)
                    output[j] += v * singularVectors[tMat.WindowSize * (i - 1) + j + 1];

                temp += v * singularVectors[tMat.WindowSize * i];
                output[tMat.WindowSize - 1] += (temp / 2);
            }
        }

        private static void ComputeNoiseMoments(Single[] series, Single[] signal, Single[] alpha, out Single observationNoiseVariance, out Single autoregressionNoiseVariance,
            out Single observationNoiseMean, out Single autoregressionNoiseMean, int startIndex = 0)
        {
            Contracts.Assert(Utils.Size(alpha) > 0);
            Contracts.Assert(Utils.Size(signal) > 2 * Utils.Size(alpha)); // To assure that the autoregression noise variance is unbiased.
            Contracts.Assert(Utils.Size(series) >= Utils.Size(signal) + startIndex);

            var signalLength = Utils.Size(signal);
            var windowSize = Utils.Size(alpha) + 1;
            var k = signalLength - windowSize + 1;
            Contracts.Assert(k > 0);

            var y = new Single[k];
            int i;

            observationNoiseMean = 0;
            observationNoiseVariance = 0;
            autoregressionNoiseMean = 0;
            autoregressionNoiseVariance = 0;

            // Computing the observation noise moments
            for (i = 0; i < signalLength; ++i)
                observationNoiseMean += (series[i + startIndex] - signal[i]);
            observationNoiseMean /= signalLength;

            for (i = 0; i < signalLength; ++i)
                observationNoiseVariance += (series[i + startIndex] - signal[i]) * (series[i + startIndex] - signal[i]);
            observationNoiseVariance /= signalLength;
            observationNoiseVariance -= (observationNoiseMean * observationNoiseMean);

            // Computing the auto-regression noise moments
            TrajectoryMatrix xTM = new TrajectoryMatrix(null, signal, windowSize - 1, signalLength - 1);
            xTM.MultiplyTranspose(alpha, y);

            for (i = 0; i < k; ++i)
                autoregressionNoiseMean += (signal[windowSize - 1 + i] - y[i]);
            autoregressionNoiseMean /= k;

            for (i = 0; i < k; ++i)
            {
                autoregressionNoiseVariance += (signal[windowSize - 1 + i] - y[i] - autoregressionNoiseMean) *
                                               (signal[windowSize - 1 + i] - y[i] - autoregressionNoiseMean);
            }

            autoregressionNoiseVariance /= (k - windowSize + 1);
            Contracts.Assert(autoregressionNoiseVariance >= 0);
        }

        private static int DetermineSignalRank(Single[] series, TrajectoryMatrix tMat, Single[] singularVectors, Single[] singularValues,
            Single[] outputSignal, int maxRank)
        {
            Contracts.Assert(tMat != null);
            Contracts.Assert(Utils.Size(series) >= tMat.SeriesLength);
            Contracts.Assert(Utils.Size(outputSignal) >= tMat.SeriesLength);
            Contracts.Assert(Utils.Size(singularVectors) >= tMat.WindowSize * tMat.WindowSize);
            Contracts.Assert(Utils.Size(singularValues) >= tMat.WindowSize);
            Contracts.Assert(1 <= maxRank && maxRank <= tMat.WindowSize - 1);

            var inputSeriesLength = tMat.SeriesLength;
            var k = inputSeriesLength - tMat.WindowSize + 1;
            Contracts.Assert(k > 0);

            var x = new Single[inputSeriesLength];
            var y = new Single[k];
            var alpha = new Single[tMat.WindowSize - 1];
            var v = new Single[k];

            Single nu = 0;
            Double minErr = Double.MaxValue;
            int minIndex = maxRank - 1;
            int evaluationLength = Math.Min(Math.Max(tMat.WindowSize, 200), k);

            TrajectoryMatrix xTM = new TrajectoryMatrix(null, x, tMat.WindowSize - 1, inputSeriesLength - 1);

            int i;
            int j;
            int n;
            Single temp;
            Double error;
            Double sumSingularVals = 0;
            Single lambda;
            Single observationNoiseMean;

            FixedSizeQueue<Single> window = new FixedSizeQueue<float>(tMat.WindowSize - 1);

            for (i = 0; i < tMat.WindowSize; ++i)
                sumSingularVals += singularValues[i];

            for (i = 0; i < maxRank; ++i)
            {
                // Updating the auto-regressive coefficients
                lambda = singularVectors[tMat.WindowSize * i + tMat.WindowSize - 1];
                for (j = 0; j < tMat.WindowSize - 1; ++j)
                    alpha[j] += lambda * singularVectors[tMat.WindowSize * i + j];

                // Updating nu
                nu += lambda * lambda;

                // Reconstructing the i-th eigen triple component and adding it to x
                tMat.MultiplyTranspose(singularVectors, v, false, tMat.WindowSize * i, 0);
                tMat.RankOneHankelization(singularVectors, v, 1, x, true, tMat.WindowSize * i, 0, 0);

                observationNoiseMean = 0;
                for (j = 0; j < inputSeriesLength; ++j)
                    observationNoiseMean += (series[j] - x[j]);
                observationNoiseMean /= inputSeriesLength;

                for (j = inputSeriesLength - evaluationLength - tMat.WindowSize + 1; j < inputSeriesLength - evaluationLength; ++j)
                    window.AddLast(x[j]);

                error = 0;
                for (j = inputSeriesLength - evaluationLength; j < inputSeriesLength; ++j)
                {
                    temp = 0;
                    for (n = 0; n < tMat.WindowSize - 1; ++n)
                        temp += alpha[n] * window[n];

                    temp /= (1 - nu);
                    temp += observationNoiseMean;
                    window.AddLast(temp);
                    error += Math.Abs(series[j] - temp);
                    if (error > minErr)
                        break;
                }

                if (error < minErr)
                {
                    minErr = error;
                    minIndex = i;
                    Array.Copy(x, outputSignal, inputSeriesLength);
                }
            }

            return minIndex + 1;
        }

        public void InitState()
        {
            for (int i = 0; i < _windowSize - 2; ++i)
                _state[i] = 0;

            _buffer.Clear();
        }

        private static int DetermineSignalRankFast(Single[] series, TrajectoryMatrix tMat, Single[] singularVectors, Single[] singularValues, int maxRank)
        {
            Contracts.Assert(tMat != null);
            Contracts.Assert(Utils.Size(series) >= tMat.SeriesLength);
            Contracts.Assert(Utils.Size(singularVectors) >= tMat.WindowSize * tMat.WindowSize);
            Contracts.Assert(Utils.Size(singularValues) >= tMat.WindowSize);
            Contracts.Assert(1 <= maxRank && maxRank <= tMat.WindowSize - 1);

            var inputSeriesLength = tMat.SeriesLength;
            var k = inputSeriesLength - tMat.WindowSize + 1;
            Contracts.Assert(k > 0);

            var x = new Single[tMat.WindowSize - 1];
            var alpha = new Single[tMat.WindowSize - 1];
            Single v;

            Single nu = 0;
            Double minErr = Double.MaxValue;
            int minIndex = maxRank - 1;
            int evaluationLength = Math.Min(Math.Max(tMat.WindowSize, 200), k);

            int i;
            int j;
            int n;
            int offset;
            Single temp;
            Double error;
            Single lambda;
            Single observationNoiseMean;

            FixedSizeQueue<Single> window = new FixedSizeQueue<float>(tMat.WindowSize - 1);

            for (i = 0; i < maxRank; ++i)
            {
                // Updating the auto-regressive coefficients
                lambda = singularVectors[tMat.WindowSize * i + tMat.WindowSize - 1];
                for (j = 0; j < tMat.WindowSize - 1; ++j)
                    alpha[j] += lambda * singularVectors[tMat.WindowSize * i + j];

                // Updating nu
                nu += lambda * lambda;

                // Reconstructing the i-th eigen triple component and adding it to x
                v = 0;
                offset = inputSeriesLength - evaluationLength - tMat.WindowSize + 1;

                for (j = offset; j <= inputSeriesLength - evaluationLength; ++j)
                    v += series[j] * singularVectors[tMat.WindowSize * i - offset + j];

                for (j = 0; j < tMat.WindowSize - 1; ++j)
                    x[j] += v * singularVectors[tMat.WindowSize * i + j];

                // Computing the empirical observation noise mean
                observationNoiseMean = 0;
                for (j = offset; j < inputSeriesLength - evaluationLength; ++j)
                    observationNoiseMean += (series[j] - x[j - offset]);
                observationNoiseMean /= (tMat.WindowSize - 1);

                for (j = 0; j < tMat.WindowSize - 1; ++j)
                    window.AddLast(x[j]);

                error = 0;
                for (j = inputSeriesLength - evaluationLength; j < inputSeriesLength; ++j)
                {
                    temp = 0;
                    for (n = 0; n < tMat.WindowSize - 1; ++n)
                        temp += alpha[n] * window[n];

                    temp /= (1 - nu);
                    temp += observationNoiseMean;
                    window.AddLast(temp);
                    error += Math.Abs(series[j] - temp);
                    if (error > minErr)
                        break;
                }

                if (error < minErr)
                {
                    minErr = error;
                    minIndex = i;
                }
            }

            return minIndex + 1;
        }

        private class SignalComponent
        {
            public Double Phase;
            public int Index;
            public int Cluster;

            public SignalComponent(Double phase, int index)
            {
                Phase = phase;
                Index = index;
            }
        }

        private bool Stabilize()
        {
            if (Utils.Size(_alpha) == 1)
            {
                if (_shouldMaintainInfo)
                    _info.RootsBeforeStabilization = new[] { new Complex(_alpha[0], 0) };

                if (_alpha[0] > 1)
                    _alpha[0] = 1;
                else if (_alpha[0] < -1)
                    _alpha[0] = -1;

                if (_shouldMaintainInfo)
                {
                    _info.IsStabilized = true;
                    _info.RootsAfterStabilization = new[] { new Complex(_alpha[0], 0) };
                    _info.IsExponentialTrendPresent = false;
                    _info.IsPolynomialTrendPresent = false;
                    _info.ExponentialTrendFactor = Math.Abs(_alpha[0]);
                }
                return true;
            }

            var coeff = new Double[_windowSize - 1];
            Complex[] roots = null;
            bool trendFound = false;
            bool polynomialTrendFound = false;
            Double maxTrendMagnitude = Double.NegativeInfinity;
            Double maxNonTrendMagnitude = Double.NegativeInfinity;
            Double eps = 1e-9;
            Double highFrequenceyBoundry = Math.PI / 2;
            var sortedComponents = new List<SignalComponent>();
            var trendComponents = new List<int>();
            int i;

            // Computing the roots of the characteristic polynomial
            for (i = 0; i < _windowSize - 1; ++i)
                coeff[i] = -_alpha[i];

            if (!PolynomialUtils.FindPolynomialRoots(coeff, ref roots))
                return false;

            if (_shouldMaintainInfo)
            {
                _info.RootsBeforeStabilization = new Complex[_windowSize - 1];
                Array.Copy(roots, _info.RootsBeforeStabilization, _windowSize - 1);
            }

            // Detecting trend components
            for (i = 0; i < _windowSize - 1; ++i)
            {
                if (roots[i].Magnitude > 1 && (Math.Abs(roots[i].Phase) <= eps || Math.Abs(Math.Abs(roots[i].Phase) - Math.PI) <= eps))
                    trendComponents.Add(i);
            }

            // Removing unobserved seasonalities and consequently introducing polynomial trend
            for (i = 0; i < _windowSize - 1; ++i)
            {
                if (roots[i].Phase != 0)
                {
                    if (roots[i].Magnitude >= 1 && 2 * Math.PI / Math.Abs(roots[i].Phase) > _windowSize)
                    {
                        /*if (roots[i].Real > 1)
                        {
                            polynomialTrendFound = true;
                            roots[i] = new Complex(Math.Max(1, roots[i].Magnitude), 0);
                            maxPolynomialTrendMagnitude = Math.Max(maxPolynomialTrendMagnitude, roots[i].Magnitude);
                        }
                        else
                            roots[i] = Complex.FromPolarCoordinates(1, 0);
                        //roots[i] = Complex.FromPolarCoordinates(0.99, roots[i].Phase);*/

                        /* if (_maxTrendRatio > 1)
                         {
                             roots[i] = new Complex(roots[i].Real, 0);
                             polynomialTrendFound = true;
                         }
                         else
                             roots[i] = roots[i].Imaginary > 0 ? new Complex(roots[i].Real, 0) : new Complex(1, 0);*/

                        roots[i] = new Complex(roots[i].Real, 0);
                        polynomialTrendFound = true;

                        if (_shouldMaintainInfo)
                            _info.IsArtificialSeasonalityRemoved = true;
                    }
                    else if (roots[i].Magnitude > 1)
                        sortedComponents.Add(new SignalComponent(roots[i].Phase, i));
                }
            }

            if (_maxTrendRatio > 1)
            {
                // Combining the close exponential-seasonal components
                if (sortedComponents.Count > 1 && polynomialTrendFound)
                {
                    sortedComponents.Sort((a, b) => a.Phase.CompareTo(b.Phase));
                    var clusterNum = 0;

                    for (i = 0; i < sortedComponents.Count - 1; ++i)
                    {
                        if ((sortedComponents[i].Phase < 0 && sortedComponents[i + 1].Phase < 0) ||
                            (sortedComponents[i].Phase > 0 && sortedComponents[i + 1].Phase > 0))
                        {
                            sortedComponents[i].Cluster = clusterNum;
                            if (Math.Abs(sortedComponents[i + 1].Phase - sortedComponents[i].Phase) > 0.05)
                                clusterNum++;
                            sortedComponents[i + 1].Cluster = clusterNum;
                        }
                        else
                            clusterNum++;
                    }

                    int start = 0;
                    bool polynomialSeasonalityFound = false;
                    Double largestSeasonalityPhase = 0;
                    for (i = 1; i < sortedComponents.Count; ++i)
                    {
                        if (sortedComponents[i].Cluster != sortedComponents[i - 1].Cluster)
                        {
                            if (i - start > 1) // There are more than one point in the cluster
                            {
                                Double avgPhase = 0;
                                Double avgMagnitude = 0;

                                for (var j = start; j < i; ++j)
                                {
                                    avgPhase += sortedComponents[j].Phase;
                                    avgMagnitude += roots[sortedComponents[j].Index].Magnitude;
                                }
                                avgPhase /= (i - start);
                                avgMagnitude /= (i - start);

                                for (var j = start; j < i; ++j)
                                    roots[sortedComponents[j].Index] = Complex.FromPolarCoordinates(avgMagnitude,
                                        avgPhase);

                                if (!polynomialSeasonalityFound && avgPhase > 0)
                                {
                                    largestSeasonalityPhase = avgPhase;
                                    polynomialSeasonalityFound = true;
                                }
                            }

                            start = i;
                        }
                    }
                }

                // Combining multiple exponential trends into polynomial ones
                if (!polynomialTrendFound)
                {
                    var ind1 = -1;
                    var ind2 = -1;

                    foreach (var ind in trendComponents)
                    {
                        if (Math.Abs(roots[ind].Phase) <= eps)
                        {
                            ind1 = ind;
                            break;
                        }
                    }

                    for (i = 0; i < _windowSize - 1; ++i)
                    {
                        if (Math.Abs(roots[i].Phase) <= eps && 0.9 <= roots[i].Magnitude && i != ind1)
                        {
                            ind2 = i;
                            break;
                        }
                    }

                    if (ind1 >= 0 && ind2 >= 0 && ind1 != ind2)
                    {
                        roots[ind1] = Complex.FromPolarCoordinates(1, 0);
                        roots[ind2] = Complex.FromPolarCoordinates(1, 0);
                        polynomialTrendFound = true;
                    }
                }
            }

            if (polynomialTrendFound) // Suppress the exponential trend
            {
                maxTrendMagnitude = Math.Min(1, _maxTrendRatio);
                foreach (var ind in trendComponents)
                    roots[ind] = Complex.FromPolarCoordinates(0.99, roots[ind].Phase);
            }
            else
            {
                // Spotting the exponential trend components and finding the max trend magnitude
                for (i = 0; i < _windowSize - 1; ++i)
                {
                    if (roots[i].Magnitude > 1 && Math.Abs(roots[i].Phase) <= eps)
                    {
                        trendFound = true;
                        maxTrendMagnitude = Math.Max(maxTrendMagnitude, roots[i].Magnitude);
                    }
                    else
                        maxNonTrendMagnitude = Math.Max(maxNonTrendMagnitude, roots[i].Magnitude);
                }

                if (!trendFound)
                    maxTrendMagnitude = 1;

                maxTrendMagnitude = Math.Min(maxTrendMagnitude, _maxTrendRatio);
            }

            // Squeezing all components below the maximum trend magnitude
            var smallTrendMagnitude = Math.Min(maxTrendMagnitude, (maxTrendMagnitude + 1) / 2);
            for (i = 0; i < _windowSize - 1; ++i)
            {
                if (roots[i].Magnitude >= maxTrendMagnitude)
                {
                    if ((highFrequenceyBoundry < roots[i].Phase && roots[i].Phase < Math.PI - eps) ||
                        (-Math.PI + eps < roots[i].Phase && roots[i].Phase < -highFrequenceyBoundry))
                        roots[i] = Complex.FromPolarCoordinates(smallTrendMagnitude, roots[i].Phase);
                    else
                        roots[i] = Complex.FromPolarCoordinates(maxTrendMagnitude, roots[i].Phase);
                }
            }

            // Correcting all the other trend components
            for (i = 0; i < _windowSize - 1; ++i)
            {
                var phase = roots[i].Phase;
                if (Math.Abs(phase) <= eps)
                    roots[i] = new Complex(roots[i].Magnitude, 0);
                else if (Math.Abs(phase - Math.PI) <= eps)
                    roots[i] = new Complex(-roots[i].Magnitude, 0);
                else if (Math.Abs(phase + Math.PI) <= eps)
                    roots[i] = new Complex(-roots[i].Magnitude, 0);
            }

            // Computing the characteristic polynomial from the modified roots
            try
            {
                if (!PolynomialUtils.FindPolynomialCoefficients(roots, ref coeff))
                    return false;
            }
            catch (OverflowException)
            {
                return false;
            }

            // Updating alpha
            for (i = 0; i < _windowSize - 1; ++i)
                _alpha[i] = (Single)(-coeff[i]);

            if (_shouldMaintainInfo)
            {
                _info.RootsAfterStabilization = roots;
                _info.IsStabilized = true;
                _info.IsPolynomialTrendPresent = polynomialTrendFound;
                _info.IsExponentialTrendPresent = maxTrendMagnitude > 1;
                _info.ExponentialTrendFactor = maxTrendMagnitude;
            }

            return true;
        }

        /// <summary>
        /// Feeds the next observation on the series to the model and as a result changes the state of the model.
        /// </summary>
        /// <param name="input">The next observation on the series.</param>
        /// <param name="updateModel">Determines whether the model parameters also need to be updated upon consuming the new observation (default = false).</param>
        public void Consume(ref Single input, bool updateModel = false)
        {
            if (Single.IsNaN(input))
                return;

            int i;

            if (_wTrans == null)
            {
                _y = new CpuAlignedVector(_rank, SseUtils.CbAlign);
                _wTrans = new CpuAlignedMatrixRow(_rank, _windowSize, SseUtils.CbAlign);
                Single[] vecs = new Single[_rank * _windowSize];

                for (i = 0; i < _rank; ++i)
                    vecs[(_windowSize + 1) * i] = 1;

                i = 0;
                _wTrans.CopyFrom(vecs, ref i);
            }

            // Forming vector x

            if (_buffer.Count == 0)
            {
                for (i = 0; i < _windowSize - 1; ++i)
                    _buffer.AddLast(_state[i]);
            }

            int len = _buffer.Count;
            for (i = 0; i < _windowSize - len - 1; ++i)
                _x[i] = 0;
            for (i = Math.Max(0, len - _windowSize + 1); i < len; ++i)
                _x[i - len + _windowSize - 1] = _buffer[i];
            _x[_windowSize - 1] = input;

            // Computing y: Eq. (11) in https://hal-institut-mines-telecom.archives-ouvertes.fr/hal-00479772/file/twocolumns.pdf
            CpuAligenedMathUtils<CpuAlignedMatrixRow>.MatTimesSrc(_wTrans, _x, _y);

            // Updating the state vector
            CpuAligenedMathUtils<CpuAlignedMatrixRow>.MatTranTimesSrc(_wTrans, _y, _xSmooth);

            _nextPrediction = _autoregressionNoiseMean + _observationNoiseMean;
            for (i = 0; i < _windowSize - 2; ++i)
            {
                _state[i] = ((_windowSize - 2 - i) * _state[i + 1] + _xSmooth[i + 1]) / (_windowSize - 1 - i);
                _nextPrediction += _state[i] * _alpha[i];
            }
            _state[_windowSize - 2] = _xSmooth[_windowSize - 1];
            _nextPrediction += _state[_windowSize - 2] * _alpha[_windowSize - 2];

            if (updateModel)
            {
                // REVIEW: to be implemented in the next version based on the FAPI algorithm
                // in https://hal-institut-mines-telecom.archives-ouvertes.fr/hal-00479772/file/twocolumns.pdf.
            }

            _buffer.AddLast(input);
        }

        /// <summary>
        /// Train the model parameters based on a training series.
        /// </summary>
        /// <param name="data">The training time-series.</param>
        public void Train(FixedSizeQueue<Single> data)
        {
            _host.CheckParam(data != null, nameof(data), "The input series for training cannot be null.");
            _host.CheckParam(data.Count >= _trainSize, nameof(data), "The input series for training does not have enough points for training.");

            Single[] dataArray = new Single[_trainSize];

            int i;
            int count;
            for (i = 0, count = 0; count < _trainSize && i < data.Count; ++i)
                if (!Single.IsNaN(data[i]))
                    dataArray[count++] = data[i];

            if (_shouldMaintainInfo)
            {
                _info = new ModelInfo();
                _info.WindowSize = _windowSize;
            }

            if (count <= 2 * _windowSize)
            {
#if !TLCSSA
                using (var ch = _host.Start("Train"))
                    ch.Warning(
                        "Training cannot be completed because the input series for training does not have enough points.");
#endif
            }
            else
            {
                if (count != _trainSize)
                    Array.Resize(ref dataArray, count);

                TrainCore(dataArray, count);
            }
        }

#if !TLCSSA
        /// <summary>
        /// Train the model parameters based on a training series.
        /// </summary>
        /// <param name="data">The training time-series.</param>
        public void Train(RoleMappedData data)
        {
            _host.CheckParam(data != null, nameof(data), "The input series for training cannot be null.");
            if (data.Schema.Feature.Type != NumberType.Float)
                throw _host.ExceptUserArg(nameof(data.Schema.Feature.Name), "The feature column has  type '{0}', but must be a float.", data.Schema.Feature.Type);

            Single[] dataArray = new Single[_trainSize];
            int col = data.Schema.Feature.Index;

            int count = 0;
            using (var cursor = data.Data.GetRowCursor(c => c == col))
            {
                var getVal = cursor.GetGetter<Single>(col);
                Single val = default(Single);
                while (cursor.MoveNext() && count < _trainSize)
                {
                    getVal(ref val);
                    if (!Single.IsNaN(val))
                        dataArray[count++] = val;
                }
            }

            if (_shouldMaintainInfo)
            {
                _info = new ModelInfo();
                _info.WindowSize = _windowSize;
            }

            if (count <= 2 * _windowSize)
            {
                using (var ch = _host.Start("Train"))
                    ch.Warning("Training cannot be completed because the input series for training does not have enough points.");
            }
            else
            {
                if (count != _trainSize)
                    Array.Resize(ref dataArray, count);

                TrainCore(dataArray, count);
            }
        }
#endif

        private void TrainCore(Single[] dataArray, int originalSeriesLength)
        {
            _host.Assert(Utils.Size(dataArray) > 0);
            Single[] singularVals;
            Single[] leftSingularVecs;
            var learnNaiveModel = false;

            var signalLength = _rankSelectionMethod == RankSelectionMethod.Exact ? originalSeriesLength : 2 * _windowSize - 1;//originalSeriesLength;
            var signal = new Single[signalLength];

            int i;
            // Creating the trajectory matrix for the series
            TrajectoryMatrix tMat = new TrajectoryMatrix(_host, dataArray, _windowSize, originalSeriesLength);

            // Computing the SVD of the trajectory matrix
            if (!tMat.ComputeSvd(out singularVals, out leftSingularVecs))
                learnNaiveModel = true;
            else
            {
                for (i = 0; i < _windowSize * _maxRank; ++i)
                {
                    if (Single.IsNaN(leftSingularVecs[i]))
                    {
                        learnNaiveModel = true;
                        break;
                    }
                }
            }

            // Checking for standard eigenvectors, if found reduce the window size and reset training.
            if (!learnNaiveModel)
            {
                for (i = 0; i < _windowSize; ++i)
                {
                    var v = leftSingularVecs[(i + 1) * _windowSize - 1];
                    if (v * v == 1)
                    {
                        if (_windowSize > 2)
                        {
                            _windowSize--;
                            _maxRank = _windowSize / 2;
                            _alpha = new Single[_windowSize - 1];
                            _state = new Single[_windowSize - 1];
                            _x = new CpuAlignedVector(_windowSize, SseUtils.CbAlign);
                            _xSmooth = new CpuAlignedVector(_windowSize, SseUtils.CbAlign);

                            TrainCore(dataArray, originalSeriesLength);
                            return;
                        }
                        else
                        {
                            learnNaiveModel = true;
                            break;
                        }
                    }
                }
            }

            // Learn the naive (averaging) model in case the eigen decomposition is not possible
            if (learnNaiveModel)
            {
#if !TLCSSA
                using (var ch = _host.Start("Train"))
                    ch.Warning("The precise SSA model cannot be trained.");
#endif

                _rank = 1;
                var temp = (Single)(1f / Math.Sqrt(_windowSize));
                for (i = 0; i < _windowSize; ++i)
                    leftSingularVecs[i] = temp;
            }
            else
            {
                // Computing the signal rank
                if (_rankSelectionMethod == RankSelectionMethod.Exact)
                    _rank = DetermineSignalRank(dataArray, tMat, leftSingularVecs, singularVals, signal, _maxRank);
                else if (_rankSelectionMethod == RankSelectionMethod.Fast)
                    _rank = DetermineSignalRankFast(dataArray, tMat, leftSingularVecs, singularVals, _maxRank);
            }

            // Setting the the y vector
            _y = new CpuAlignedVector(_rank, SseUtils.CbAlign);

            // Setting the weight matrix
            _wTrans = new CpuAlignedMatrixRow(_rank, _windowSize, SseUtils.CbAlign);
            i = 0;
            _wTrans.CopyFrom(leftSingularVecs, ref i);

            // Setting alpha
            Single nu = 0;
            for (i = 0; i < _rank; ++i)
            {
                _y[i] = leftSingularVecs[_windowSize * (i + 1) - 1];
                nu += _y[i] * _y[i];
            }

            CpuAligenedMathUtils<CpuAlignedMatrixRow>.MatTranTimesSrc(_wTrans, _y, _xSmooth);
            for (i = 0; i < _windowSize - 1; ++i)
                _alpha[i] = _xSmooth[i] / (1 - nu);

            // Stabilizing the model
            if (_shouldStablize && !learnNaiveModel)
            {
                if (!Stabilize())
                {
#if !TLCSSA
                    using (var ch = _host.Start("Train"))
                        ch.Warning("The trained model cannot be stablized.");
#endif
                }
            }

            // Computing the noise moments
            if (ShouldComputeForecastIntervals)
            {
                if (_rankSelectionMethod != RankSelectionMethod.Exact)
                    ReconstructSignalTailFast(dataArray, tMat, leftSingularVecs, _rank, signal);

                ComputeNoiseMoments(dataArray, signal, _alpha, out _observationNoiseVariance, out _autoregressionNoiseVariance,
                    out _observationNoiseMean, out _autoregressionNoiseMean, originalSeriesLength - signalLength);
                _observationNoiseMean = 0;
                _autoregressionNoiseMean = 0;
            }

            // Setting the state
            _nextPrediction = _autoregressionNoiseMean + _observationNoiseMean;

            if (_buffer.Count > 0) // Use the buffer to set the state when there are data points pushed into the buffer using the Consume() method
            {
                int len = _buffer.Count;
                for (i = 0; i < _windowSize - len; ++i)
                    _x[i] = 0;
                for (i = Math.Max(0, len - _windowSize); i < len; ++i)
                    _x[i - len + _windowSize] = _buffer[i];
            }
            else // use the training data points otherwise
            {
                for (i = originalSeriesLength - _windowSize; i < originalSeriesLength; ++i)
                    _x[i - originalSeriesLength + _windowSize] = dataArray[i];
            }

            CpuAligenedMathUtils<CpuAlignedMatrixRow>.MatTimesSrc(_wTrans, _x, _y);
            CpuAligenedMathUtils<CpuAlignedMatrixRow>.MatTranTimesSrc(_wTrans, _y, _xSmooth);

            for (i = 1; i < _windowSize; ++i)
            {
                _state[i - 1] = _xSmooth[i];
                _nextPrediction += _state[i - 1] * _alpha[i - 1];
            }

            if (_shouldMaintainInfo)
            {
                _info.IsTrained = true;
                _info.WindowSize = _windowSize;
                _info.AutoRegressiveCoefficients = new Single[_windowSize - 1];
                Array.Copy(_alpha, _info.AutoRegressiveCoefficients, _windowSize - 1);
                _info.Rank = _rank;
                _info.IsNaiveModelTrained = learnNaiveModel;
                _info.Spectrum = singularVals;
            }
        }

        /// <summary>
        /// Forecasts the future values of the series up to the given horizon.
        /// </summary>
        /// <param name="result">The forecast result.</param>
        /// <param name="horizon">The forecast horizon.</param>
        public void Forecast(ref ForecastResultBase<Single> result, int horizon = 1)
        {
            _host.CheckParam(horizon >= 1, nameof(horizon), "The horizon parameter should be greater than 0.");
            if (result == null)
                result = new SsaForecastResult();

            var str = "The result argument must be of type " + typeof(SsaForecastResult).ToString();
            _host.CheckParam(result is SsaForecastResult, nameof(result), str);

            var output = result as SsaForecastResult;

            var res = result.PointForecast.Values;
            if (Utils.Size(res) < horizon)
                res = new Single[horizon];

            int i;
            int j;
            int k;

            // Computing the point forecasts
            res[0] = _nextPrediction;
            for (i = 1; i < horizon; ++i)
            {
                k = 0;
                res[i] = _autoregressionNoiseMean + _observationNoiseMean;
                for (j = i; j < _windowSize - 1; ++j, ++k)
                    res[i] += _state[j] * _alpha[k];

                for (j = Math.Max(0, i - _windowSize + 1); j < i; ++j, ++k)
                    res[i] += res[j] * _alpha[k];
            }

            // Computing the forecast variances
            if (ShouldComputeForecastIntervals)
            {
                var sd = output.ForecastStandardDeviation.Values;
                if (Utils.Size(sd) < horizon)
                    sd = new Single[horizon];

                var lastCol = new FixedSizeQueue<Single>(_windowSize - 1);

                for (i = 0; i < _windowSize - 3; ++i)
                    lastCol.AddLast(0);
                lastCol.AddLast(1);
                lastCol.AddLast(_alpha[_windowSize - 2]);
                sd[0] = _autoregressionNoiseVariance + _observationNoiseVariance;

                for (i = 1; i < horizon; ++i)
                {
                    Single temp = 0;
                    for (j = 0; j < _windowSize - 1; ++j)
                        temp += _alpha[j] * lastCol[j];
                    lastCol.AddLast(temp);

                    sd[i] = sd[i - 1] + _autoregressionNoiseVariance * temp * temp;
                }

                for (i = 0; i < horizon; ++i)
                    sd[i] = (float)Math.Sqrt(sd[i]);

                output.ForecastStandardDeviation = new VBuffer<Single>(horizon, sd, output.ForecastStandardDeviation.Indices);
            }

            result.PointForecast = new VBuffer<Single>(horizon, res, result.PointForecast.Indices);
            output.CanComputeForecastIntervals = ShouldComputeForecastIntervals;
            output.BoundOffset = 0;
        }

        /// <summary>
        /// Predicts the next value on the series.
        /// </summary>
        /// <param name="output">The prediction result.</param>
        public void PredictNext(ref Single output)
        {
            output = _nextPrediction;
        }

        public ISequenceModeler<Single, Single> Clone()
        {
            return new AdaptiveSingularSpectrumSequenceModeler(this);
        }

        /// <summary>
        /// Computes the forecast intervals for the input forecast object at the given confidence level. The results are stored in the forecast object.
        /// </summary>
        /// <param name="forecast">The input forecast object</param>
        /// <param name="confidenceLevel">The confidence level in [0, 1)</param>
        public static void ComputeForecastIntervals(ref SsaForecastResult forecast, Single confidenceLevel = 0.95f)
        {
            Contracts.CheckParam(0 <= confidenceLevel && confidenceLevel < 1, nameof(confidenceLevel), "The confidence level must be in [0, 1).");
            Contracts.CheckValue(forecast, nameof(forecast));
            Contracts.Check(forecast.CanComputeForecastIntervals, "The forecast intervals cannot be computed for this forecast object.");

            var horizon = Utils.Size(forecast.PointForecast.Values);
            Contracts.Check(Utils.Size(forecast.ForecastStandardDeviation.Values) >= horizon, "The forecast standard deviation values are not available.");

            forecast.ConfidenceLevel = confidenceLevel;
            if (horizon == 0)
                return;

            var upper = forecast.UpperBound.Values;
            if (Utils.Size(upper) < horizon)
                upper = new Single[horizon];

            var lower = forecast.LowerBound.Values;
            if (Utils.Size(lower) < horizon)
                lower = new Single[horizon];

            var z = ProbabilityFunctions.Probit(0.5 + confidenceLevel / 2.0);
            var meanForecast = forecast.PointForecast.Values;
            var sdForecast = forecast.ForecastStandardDeviation.Values;
            double temp;

            for (int i = 0; i < horizon; ++i)
            {
                temp = z * sdForecast[i];
                upper[i] = (Single)(meanForecast[i] + forecast.BoundOffset + temp);
                lower[i] = (Single)(meanForecast[i] + forecast.BoundOffset - temp);
            }

            forecast.UpperBound = new VBuffer<Single>(horizon, upper, forecast.UpperBound.Indices);
            forecast.LowerBound = new VBuffer<Single>(horizon, lower, forecast.LowerBound.Indices);
        }
    }
}
