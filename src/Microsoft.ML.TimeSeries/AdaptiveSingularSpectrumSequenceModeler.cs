// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
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
    internal sealed class AdaptiveSingularSpectrumSequenceModeler : ISequenceModeler<Single, Single>
    {
        public const string LoaderSignature = "SSAModel";

        internal sealed class SsaForecastResult : ForecastResultBase<Single>
        {
            // This class is empty because there is no extra member needed for the forecast result at this point.
        }

        private readonly Single[] _alpha;
        private readonly Single[] _state;
        private readonly FixedSizeQueue<Single> _buffer;
        private readonly CpuAlignedVector _x;
        private readonly CpuAlignedVector _xSmooth;
        private readonly int _windowSize;
        private readonly int _seriesLength;
        private readonly bool _shouldComputeRank;
        private readonly bool _implementOwnBuffer;
        private readonly Single _discountFactor;
        private readonly int _trainSize;

        private readonly IHost _host;

        private CpuAlignedMatrixRow _wTrans;

        private int _rank;
        private CpuAlignedVector _y;
        private Single _nextPrediction;

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
                loaderSignature: LoaderSignature);
        }

        /// <summary>
        /// The constructor for Adaptive SSA model.
        /// </summary>
        /// <param name="env">The exception context.</param>
        /// <param name="trainSize"></param>
        /// <param name="seriesLength">The length of series that is kept in buffer for modeling (parameter N).</param>
        /// <param name="windowSize">The length of the window on the series for building the trajectory matrix (parameter L).</param>
        /// <param name="discountFactor">The discount factor in [0,1] used for online updates (default = 1).</param>
        /// <param name="buffer">The buffer used to keep the series in the memory. If null, an internal buffer is created (default = null).</param>
        /// <param name="rank">The desired rank of the subspace used for SSA projection (parameter r). This parameter should be in the range in [1, windowSize].
        /// If set to null, the rank is automatically determined based on prediction error minimization. (default = null)</param>
        public AdaptiveSingularSpectrumSequenceModeler(IHostEnvironment env, int trainSize, int seriesLength, int windowSize, Single discountFactor = 1,
            FixedSizeQueue<Single> buffer = null, int? rank = null)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoaderSignature);
            _host.CheckParam(windowSize >= 2, nameof(windowSize), "Should be at least 2."); // ...because otherwise we have nothing to autoregress on
            _host.CheckParam(seriesLength > windowSize, nameof(seriesLength), "Should be greater than the window size.");
            _host.CheckParam(trainSize > 2 * windowSize, nameof(trainSize), "Should be greater than twice the window size.");
            _host.CheckParam(0 <= discountFactor && discountFactor <= 1, nameof(discountFactor), "Should be in [0,1].");

            if (rank != null)
            {
                _rank = (int)rank;
                _host.CheckParam(1 <= _rank && _rank < windowSize, nameof(rank), "Should be in [1, windowSize).");
            }
            else
            {
                _shouldComputeRank = true;
                _rank = windowSize - 1;
            }

            _seriesLength = seriesLength;
            _windowSize = windowSize;
            _trainSize = trainSize;
            _discountFactor = discountFactor;

            if (buffer == null)
            {
                _buffer = new FixedSizeQueue<Single>(seriesLength);
                _implementOwnBuffer = true;
            }
            else
                _buffer = buffer;

            _alpha = new Single[windowSize - 1];
            _state = new Single[windowSize - 1];
            _x = new CpuAlignedVector(windowSize, SseUtils.CbAlign);
            _xSmooth = new CpuAlignedVector(windowSize, SseUtils.CbAlign);
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
            _shouldComputeRank = model._shouldComputeRank;
            _seriesLength = model._seriesLength;
            _windowSize = model._windowSize;
            _trainSize = model._trainSize;
            _discountFactor = model._discountFactor;

            _buffer = new FixedSizeQueue<Single>(_seriesLength);
            _implementOwnBuffer = true;

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
            // bool: _shouldComputeRank
            // bool: isWeightSet
            // float[]: _alpha
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

            _shouldComputeRank = ctx.Reader.ReadBoolByte();
            bool isWeightSet = ctx.Reader.ReadBoolByte();

            _alpha = ctx.Reader.ReadFloatArray();
            _host.CheckDecode(Utils.Size(_alpha) == _windowSize - 1);

            if (isWeightSet)
            {
                var tempArray = ctx.Reader.ReadFloatArray();
                _host.CheckDecode(Utils.Size(tempArray) == _rank * _windowSize);
                _wTrans = new CpuAlignedMatrixRow(_rank, _windowSize, SseUtils.CbAlign);
                int i = 0;
                _wTrans.CopyFrom(tempArray, ref i);
            }

            _buffer = new FixedSizeQueue<Single>(_seriesLength);
            _implementOwnBuffer = true;

            _state = new Single[_windowSize - 1];

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

            // *** Binary format ***
            // int: _seriesLength
            // int: _windowSize
            // int: _trainSize
            // int: _rank
            // float: _discountFactor
            // bool: _shouldComputeRank
            // bool: _isWeightSet
            // float[]: _alpha
            // float[]: _wTrans (only if _isWeightSet == true)

            ctx.Writer.Write(_seriesLength);
            ctx.Writer.Write(_windowSize);
            ctx.Writer.Write(_trainSize);
            ctx.Writer.Write(_rank);
            ctx.Writer.Write(_discountFactor);
            ctx.Writer.WriteBoolByte(_shouldComputeRank);
            ctx.Writer.WriteBoolByte(_wTrans != null);
            ctx.Writer.WriteFloatArray(_alpha);

            if (_wTrans != null)
            {
                // This may not be the most efficient way for serializing an aligned matrix.
                var tempArray = new Single[_rank * _windowSize];
                int iv = 0;
                _wTrans.CopyTo(tempArray, ref iv);
                ctx.Writer.WriteFloatArray(tempArray);
            }
        }

        private int DetermineSignalRank(Single[] series, TrajectoryMatrix tMat, Single[] singularVectors, Single[] singularValues)
        {
            var inputSeriesLength = Utils.Size(series);
            var k = inputSeriesLength - _windowSize + 1;
            _host.Assert(k > 0);
            var x = new Single[inputSeriesLength];
            var y = new Single[k];
            var alpha = new Single[_windowSize - 1];
            var v = new Single[k];

            Single nu = 0;
            Double minErr = Double.MaxValue;
            int minIndex = _windowSize - 2;

            TrajectoryMatrix xTM = new TrajectoryMatrix(_host, x, _windowSize - 1, inputSeriesLength - 1);

            int i;
            int j;
            Double error;
            Double sumSingularVals = 0;
            Double a;
            Single lambda;

            for (i = 0; i < _windowSize; ++i)
                sumSingularVals += singularValues[i];

            for (i = 0; i < _windowSize - 1; ++i)
            {
                // Updating the auto-regressive coefficients
                lambda = singularVectors[_windowSize * i + _windowSize - 1];
                for (j = 0; j < _windowSize - 1; ++j)
                    alpha[j] += lambda * singularVectors[_windowSize * i + j];

                // Updating nu
                nu += lambda * lambda;

                // Reconstructing the i-th eigen triple component and adding it to x
                tMat.MultiplyTranspose(singularVectors, v, false, _windowSize * i, 0);
                tMat.RankOneHankelization(singularVectors, v, 1, x, true, _windowSize * i, 0, 0);

                // Computing the predicted series
                xTM.SetSeries(x);
                xTM.MultiplyTranspose(alpha, y);

                // Computing the error
                error = 0;
                for (j = 0; j < k; ++j)
                {
                    a = y[j] / (1 - nu) - series[_windowSize - 1 + j];
                    error += a * a;
                }

                sumSingularVals -= singularValues[i];
                error /= sumSingularVals;

                if (error < minErr)
                {
                    minErr = error;
                    minIndex = i;
                }
            }

            return minIndex + 1;
        }

        public void InitState()
        {
            for (int i = 0; i < _windowSize - 2; ++i)
                _state[i] = 0;
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
            int len = _buffer.Count;
            for (i = 0; i < _windowSize - len - 1; ++i)
                _x[i] = 0;
            for (i = Math.Max(0, len - _windowSize + 1); i < len; ++i)
                _x[i - len + _windowSize - 1] = _buffer[i];
            _x[_windowSize - 1] = input;

            // Computing y: Eq. (11) in https://hal-institut-mines-telecom.archives-ouvertes.fr/hal-00479772/file/twocolumns.pdf
            CpuAligenedMathUtils<CpuAlignedMatrixRow>.MatTimesSrc(false, _wTrans, _x, _y);

            // Updating the state vector
            CpuAligenedMathUtils<CpuAlignedMatrixRow>.MatTranTimesSrc(false, _wTrans, _y, _xSmooth);

            _nextPrediction = 0;
            for (i = 0; i < _windowSize - 2; ++i)
            {
                _state[i] = ((_windowSize - 2 - i) * _state[i + 1] + _xSmooth[i + 1]) / (_windowSize - 1 - i);
                _nextPrediction += _state[i] * _alpha[i];
            }
            _state[_windowSize - 2] = _xSmooth[_windowSize - 1];
            _nextPrediction += _state[_windowSize - 2] * _alpha[_windowSize - 2];

            if (updateModel)
            {
                // To be implemented in the next version based on the FAPI algorithm
                // in https://hal-institut-mines-telecom.archives-ouvertes.fr/hal-00479772/file/twocolumns.pdf.
            }

            if (_implementOwnBuffer)
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

            if (count <= 2 * _windowSize)
            {
                using (var ch = _host.Start("Train"))
                    ch.Warning(
                        "Training cannot be completed because the input series for training does not have enough points.");
            }
            else
            {
                if (count != _trainSize)
                    Array.Resize(ref dataArray, count);

                TrainCore(dataArray, count);
            }
        }

        /// <summary>
        /// Train the model parameters based on a training series.
        /// </summary>
        /// <param name="data">The training time-series.</param>
        public void Train(RoleMappedData data)
        {
            _host.CheckParam(data != null, nameof(data), "The input series for training cannot be null.");
            if (data.Schema.Feature.Type != NumberType.Float)
                throw _host.ExceptParam(nameof(data), "The feature column has  type '{0}', but must be a float.", data.Schema.Feature.Type);

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

        private void TrainCore(Single[] dataArray, int originalSeriesLength)
        {
            _host.Assert(Utils.Size(dataArray) > 0);
            Single[] singularVals;
            Single[] leftSingularVecs;

            int i;
            // Creating the trajectory matrix for the series
            TrajectoryMatrix tMat = new TrajectoryMatrix(_host, dataArray, _windowSize, originalSeriesLength);

            // Computing the SVD of the trajectory matrix
            tMat.ComputeSvd(out singularVals, out leftSingularVecs);

            // Computing the signal rank
            if (_shouldComputeRank)
                _rank = DetermineSignalRank(dataArray, tMat, leftSingularVecs, singularVals);

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

            CpuAligenedMathUtils<CpuAlignedMatrixRow>.MatTranTimesSrc(false, _wTrans, _y, _xSmooth);
            for (i = 0; i < _windowSize - 1; ++i)
                _alpha[i] = _xSmooth[i] / (1 - nu);

            // Setting the state
            int len = _buffer.Count;
            for (i = 0; i < _windowSize - len; ++i)
                _x[i] = 0;
            for (i = Math.Max(0, len - _windowSize); i < len; ++i)
                _x[i - len + _windowSize] = _buffer[i];

            CpuAligenedMathUtils<CpuAlignedMatrixRow>.MatTimesSrc(false, _wTrans, _x, _y);
            CpuAligenedMathUtils<CpuAlignedMatrixRow>.MatTranTimesSrc(false, _wTrans, _y, _xSmooth);

            _nextPrediction = 0;
            for (i = 1; i < _windowSize; ++i)
            {
                _state[i - 1] = _xSmooth[i];
                _nextPrediction += _state[i - 1] * _alpha[i - 1];
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

            _host.CheckParam(result is SsaForecastResult, nameof(result),
                "The result argument must be of type " + nameof(SsaForecastResult) + ".");

            var res = result.PointForecast.Values;
            if (Utils.Size(res) < horizon)
                res = new Single[horizon];

            res[0] = _nextPrediction;

            int i;
            int j;
            int k;

            for (i = 1; i < horizon; ++i)
            {
                k = 0;
                res[i] = 0;
                for (j = i; j < _windowSize - 1; ++j, ++k)
                    res[i] += _state[j] * _alpha[k];

                for (j = Math.Max(0, i - _windowSize + 1); j < i; ++j, ++k)
                    res[i] += res[j] * _alpha[k];
            }

            result.PointForecast = new VBuffer<Single>(horizon, res, result.PointForecast.Indices);
        }

        public void PredictNext(ref Single output)
        {
            output = _nextPrediction;
        }

        public ISequenceModeler<Single, Single> Clone()
        {
            return new AdaptiveSingularSpectrumSequenceModeler(this);
        }
    }
}
