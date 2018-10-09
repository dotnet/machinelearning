// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.TimeSeriesProcessing
{
    /// <summary>
    /// This class encapsulates the trajectory matrix of a time-series used in Singular Spectrum Analysis (SSA).
    /// In particular, for a given series of length N and the window size of L, (such that N > L):
    ///
    /// x(1), x(2), x(3), ... , x(N)
    ///
    /// The trajectory matrix H is defined in the explicit form as:
    ///
    ///     [x(1)  x(2)   x(3)   ...  x(N - L + 1)]
    ///     [x(2)  x(3)   x(4)   ...  x(N - L + 2)]
    /// H = [x(3)  x(4)   x(5)   ...  x(N - L + 3)]
    ///     [ .     .      .               .      ]
    ///     [ .     .      .               .      ]
    ///     [x(L)  x(L+1) x(L+2) ...      x(N)    ]
    ///
    /// of size L * K, where K = N - L + 1.
    ///
    /// This class does not explicitly store the trajectory matrix though. Furthermore, since the trajectory matrix is
    /// a Hankel matrix, its multiplication by an arbitrary vector is implemented efficiently using the Discrete Fast Fourier Transform.
    /// </summary>
    public sealed class TrajectoryMatrix
    {
        /// <summary>
        /// The time series data
        /// </summary>
        private Single[] _data;

        /// <summary>
        /// The window length L
        /// </summary>
        private readonly int _windowSize;

        /// <summary>
        /// The series length N
        /// </summary>
        private readonly int _seriesLength;

        /// <summary>
        /// The real part of the Fourier transform of the input series.
        /// </summary>
        private Double[] _cachedSeriesFftRe;

        /// <summary>
        /// The imaginary part of the Fourier transform of the input series.
        /// </summary>
        private Double[] _cachedSeriesFftIm;

        private Double[] _allZerosIm;
        private Double[] _inputRe;
        private Double[] _outputRe;
        private Double[] _outputIm;

        private bool _isSeriesFftCached;
        private readonly bool _shouldFftUsed;
        private IExceptionContext _ectx;
        private readonly int _k;

        private void ComputeBoundryIndices(int start, int end, out int us, out int ue, out int vs, out int ve)
        {
            _ectx.Assert(0 <= end && end < _seriesLength, "The end index must be in [0, seriesLength).");
            _ectx.Assert(0 <= start && start <= end, "The start index must be in [0, end index].");

            if (start < _k)
            {
                us = 0;
                vs = start;
            }
            else
            {
                us = start - _k + 1;
                vs = _k - 1;
            }

            if (end < _windowSize)
            {
                ue = end;
                ve = 0;
            }
            else
            {
                ue = _windowSize - 1;
                ve = end - _windowSize + 1;
            }
        }

        /// <summary>
        /// Returns the length of the time-series represented by this trajectory matrix.
        /// </summary>
        public int SeriesLength { get { return _seriesLength; } }

        /// <summary>
        /// Returns the window size (L) used for building this trajectory matrix.
        /// </summary>
        public int WindowSize { get { return _windowSize; } }

        /// <summary>
        /// Constructs a trajectory matrix from the input series given the window length (L)
        /// </summary>
        /// <param name="ectx">The exception context</param>
        /// <param name="data">The input series</param>
        /// <param name="windowSize">The window size L</param>
        /// <param name="seriesLength">The number of elements from the beginning of the input array to be used for building the trajectory matrix</param>
        public TrajectoryMatrix(IExceptionContext ectx, Single[] data, int windowSize, int seriesLength)
        {
            Contracts.CheckValueOrNull(ectx);
            _ectx = ectx;

            _ectx.Check(windowSize > 0, "The window length should be greater than 0.");
            _ectx.CheckValue(data, nameof(data), "The input data cannot be null.");
            _ectx.Check(data.Length >= seriesLength, "The series length cannot be greater than the data length.");

            _seriesLength = seriesLength;
            _ectx.Check(windowSize <= _seriesLength, "The length of the window should be less than or equal to the length of the data.");

            _data = data;
            _windowSize = windowSize;
            _k = _seriesLength - _windowSize + 1;
            _shouldFftUsed = _windowSize * _k > (3 + 3 * Math.Log(_seriesLength)) * _seriesLength;
        }

        /// <summary>
        /// Sets the value of the underlying series to new values.
        /// </summary>
        /// <param name="data">The new series</param>
        public void SetSeries(Single[] data)
        {
            _ectx.Check(Utils.Size(data) >= _seriesLength, "The length of the input series cannot be less than that of the original series.");

            _data = data;
            if (_isSeriesFftCached)
            {
                int i;

                for (i = _k - 1; i < _seriesLength; ++i)
                    _inputRe[i - _k + 1] = _data[i];

                for (i = 0; i < _k - 1; ++i)
                    _inputRe[_windowSize + i] = _data[i];

                FftUtils.ComputeForwardFft(_inputRe, _allZerosIm, _cachedSeriesFftRe, _cachedSeriesFftIm, _inputRe.Length);
            }
        }

        private static Single RoundUpToReal(Double re, Double im, Double coeff = 1)
        {
            return (Single)(coeff * Math.Sign(re) * Math.Sqrt(re * re + im * im));
        }

        private void CacheInputSeriesFft()
        {
            int i;

            _cachedSeriesFftRe = new Double[_seriesLength];
            _cachedSeriesFftIm = new Double[_seriesLength];
            _allZerosIm = new Double[_seriesLength];
            _inputRe = new Double[_seriesLength];
            _outputIm = new Double[_seriesLength];
            _outputRe = new Double[_seriesLength];

            for (i = _k - 1; i < _seriesLength; ++i)
                _inputRe[i - _k + 1] = _data[i];

            for (i = 0; i < _k - 1; ++i)
                _inputRe[_windowSize + i] = _data[i];

            FftUtils.ComputeForwardFft(_inputRe, _allZerosIm, _cachedSeriesFftRe, _cachedSeriesFftIm, _inputRe.Length);
            _isSeriesFftCached = true;
        }

        /// <summary>
        /// This function computes the unnormalized covariance of the trajectory matrix (which is a Hankel matrix of size L*K).
        /// In particular, if H is the trajectory matrix of size L*K on the input series, this method computes H * H' (of size L*L).
        /// This function does not form the trajectory matrix H explicitly.
        /// Let k = N - L + 1 be the number of columns of the trajectory matrix.
        /// In most applications, we have L smaller than K, though this is not a strict constraint.
        /// The naive computational complexity for computing H * H' is O(L*L*K) while the naive memory complexity is O(K*L + L*L).
        /// However, this function computes H * H' in O(L*L + M) time, where M = min(L*K, (L + K)*Log(L + K)) and O(L*L) memory.
        /// </summary>
        /// <param name="cov">The output row-major vectorized covariance matrix of size L*L</param>
        public void ComputeUnnormalizedTrajectoryCovarianceMat(Single[] cov)
        {
            _ectx.Assert(Utils.Size(cov) >= _windowSize * _windowSize);

            int i;
            int j;

            // Computing the first row of the covariance matrix
            var temp = new Single[_k];

            for (i = 0; i < _k; ++i)
                temp[i] = _data[i];
            Multiply(temp, cov);

            // Computing the rest of the rows
            for (i = 1; i < _windowSize; ++i)
            {
                // Copying the symmetric part
                for (j = 0; j < i; ++j)
                    cov[i * _windowSize + j] = cov[j * _windowSize + i];

                // Computing the novel part
                for (j = i; j < _windowSize; ++j)
                    cov[i * _windowSize + j] = (float)((double)cov[(i - 1) * _windowSize + j - 1] - (double)_data[i - 1] * _data[j - 1] + (double)_data[i + _k - 1] * _data[j + _k - 1]);
            }
        }

        /// <summary>
        /// This function computes the singular value decomposition of the trajectory matrix.
        /// This function only computes the singular values and the left singular vectors.
        /// </summary>
        /// <param name="singularValues">The output singular values of size L</param>
        /// <param name="leftSingularvectors">The output singular vectors of size L*L</param>
        public bool ComputeSvd(out Single[] singularValues, out Single[] leftSingularvectors)
        {
            Single[] covariance = new Single[_windowSize * _windowSize];
            Single[] sVal;
            Single[] sVec;
            singularValues = new Single[_windowSize];
            leftSingularvectors = new Single[_windowSize * _windowSize];

            // Computing the covariance matrix of the trajectory matrix on the input series
            ComputeUnnormalizedTrajectoryCovarianceMat(covariance);

            // Computing the eigen decomposition of the covariance matrix
            //EigenUtils.EigenDecomposition(covariance, out sVal, out sVec);
            EigenUtils.MklSymmetricEigenDecomposition(covariance, _windowSize, out sVal, out sVec);

            var ind = new int[_windowSize];
            int i;

            for (i = 0; i < _windowSize; ++i)
                ind[i] = i;

            Array.Sort(ind, (a, b) => sVal[b].CompareTo(sVal[a]));
            for (i = 0; i < _windowSize; ++i)
            {
                singularValues[i] = sVal[ind[i]];
                Array.Copy(sVec, _windowSize * ind[i], leftSingularvectors, _windowSize * i, _windowSize);
            }

            return true;
        }

        /// <summary>
        /// This function computes the naive multiplication of the trajectory matrix H by an arbitrary vector v, i.e. H * v.
        /// </summary>
        /// <param name="vector">The input vector</param>
        /// <param name="result">The output vector allocated by the caller</param>
        /// <param name="add">Whether the multiplication result should be added to the current value in result</param>
        /// <param name="srcIndex">The starting index for the vector argument</param>
        /// <param name="dstIndex">The starting index for the result</param>
        private void NaiveMultiply(Single[] vector, Single[] result, bool add = false, int srcIndex = 0, int dstIndex = 0)
        {
            _ectx.Assert(srcIndex >= 0);
            _ectx.Assert(dstIndex >= 0);
            _ectx.Assert(Utils.Size(vector) >= _k + srcIndex);
            _ectx.Assert(Utils.Size(result) >= _windowSize + dstIndex);

            int i;
            int j;

            for (j = 0; j < _windowSize; ++j)
            {
                if (!add)
                    result[j + dstIndex] = 0;
                for (i = 0; i < _k; ++i)
                    result[j + dstIndex] += (vector[i + srcIndex] * _data[i + j]);
            }
        }

        /// <summary>
        /// This function computes the efficient multiplication of the trajectory matrix H by an arbitrary vector v, i.e. H * v.
        /// Since the trajectory matrix is a Hankel matrix, using the Discrete Fourier Transform,
        /// the multiplication is carried out in O(N.log(N)) instead of O(N^2), wheere N is the series length.
        /// For details, refer to Algorithm 2 in http://arxiv.org/pdf/0911.4498.pdf.
        /// </summary>
        /// <param name="vector">The input vector</param>
        /// <param name="result">The output vector allocated by the caller</param>
        /// <param name="add">Whether the multiplication result should be added to the current value in result</param>
        /// <param name="srcIndex">The starting index for the vector argument</param>
        /// <param name="dstIndex">The starting index for the result</param>
        private void FftMultiply(Single[] vector, Single[] result, bool add = false, int srcIndex = 0, int dstIndex = 0)
        {
            _ectx.Assert(srcIndex >= 0);
            _ectx.Assert(dstIndex >= 0);
            _ectx.Assert(Utils.Size(vector) >= _k + srcIndex);
            _ectx.Assert(Utils.Size(result) >= _windowSize + dstIndex);

            int i;

            // Computing the FFT of the trajectory matrix
            if (!_isSeriesFftCached)
                CacheInputSeriesFft();

            // Computing the FFT of the input vector
            for (i = 0; i < _k; ++i)
                _inputRe[i] = vector[_k - i - 1 + srcIndex];

            for (i = _k; i < _seriesLength; ++i)
                _inputRe[i] = 0;

            FftUtils.ComputeForwardFft(_inputRe, _allZerosIm, _outputRe, _outputIm, _inputRe.Length);

            // Computing the element-by-element product in the Fourier space
            double re;
            double im;
            for (i = 0; i < _seriesLength; ++i)
            {
                re = _outputRe[i];
                im = _outputIm[i];

                _outputRe[i] = _cachedSeriesFftRe[i] * re - _cachedSeriesFftIm[i] * im;
                _outputIm[i] = _cachedSeriesFftRe[i] * im + _cachedSeriesFftIm[i] * re;
            }

            // Computing the inverse FFT of the result
            FftUtils.ComputeBackwardFft(_outputRe, _outputIm, _outputRe, _outputIm, _inputRe.Length);

            // Generating the output
            if (add)
            {
                for (i = 0; i < _windowSize; ++i)
                    result[i + dstIndex] += RoundUpToReal(_outputRe[i], _outputIm[i]);
            }
            else
            {
                for (i = 0; i < _windowSize; ++i)
                    result[i + dstIndex] = RoundUpToReal(_outputRe[i], _outputIm[i]);
            }
        }

        /// <summary>
        /// This function efficiently computes the multiplication of the trajectory matrix H by an arbitrary vector v, i.e. H * v.
        /// </summary>
        /// <param name="vector">The input vector</param>
        /// <param name="result">The output vector allocated by the caller</param>
        /// <param name="add">Whether the multiplication result should be added to the current value in result</param>
        /// <param name="srcIndex">The starting index for the vector argument</param>
        /// <param name="dstIndex">The starting index for the result</param>
        public void Multiply(Single[] vector, Single[] result, bool add = false, int srcIndex = 0, int dstIndex = 0)
        {
            if (_shouldFftUsed)
                FftMultiply(vector, result, add, srcIndex, dstIndex);
            else
                NaiveMultiply(vector, result, add, srcIndex, dstIndex);
        }

        /// <summary>
        /// This function computes the naive multiplication of the transpose of the trajectory matrix H by an arbitrary vector v, i.e. H' * v.
        /// </summary>
        /// <param name="vector">The input vector</param>
        /// <param name="result">The output vector allocated by the caller</param>
        /// <param name="add">Whether the multiplication result should be added to the current value in result</param>
        /// <param name="srcIndex">The starting index for the vector argument</param>
        /// <param name="dstIndex">The starting index for the result</param>
        private void NaiveMultiplyTranspose(Single[] vector, Single[] result, bool add = false, int srcIndex = 0, int dstIndex = 0)
        {
            _ectx.Assert(srcIndex >= 0);
            _ectx.Assert(dstIndex >= 0);
            _ectx.Assert(Utils.Size(vector) >= _windowSize + srcIndex);
            _ectx.Assert(Utils.Size(result) >= _k + dstIndex);

            int i;
            int j;

            for (j = 0; j < _k; ++j)
            {
                if (!add)
                    result[j + dstIndex] = 0;
                for (i = 0; i < _windowSize; ++i)
                    result[j + dstIndex] += (vector[i + srcIndex] * _data[i + j]);
            }
        }

        /// <summary>
        /// This function computes the the multiplication of the transpose of the trajectory matrix H by an arbitrary vector v, i.e. H' * v.
        /// Since the trajectory matrix is a Hankel matrix, using the Discrete Fourier Transform,
        /// the multiplication is carried out in O(N.log(N)) instead of O(N^2), wheere N is the series length.
        /// For details, refer to Algorithm 3 in http://arxiv.org/pdf/0911.4498.pdf.
        /// </summary>
        /// <param name="vector">The input vector</param>
        /// <param name="result">The output vector allocated by the caller</param>
        /// <param name="add">Whether the multiplication result should be added to the current value in result</param>
        /// <param name="srcIndex">The starting index for the vector argument</param>
        /// <param name="dstIndex">The starting index for the result</param>
        private void FftMultiplyTranspose(Single[] vector, Single[] result, bool add = false, int srcIndex = 0, int dstIndex = 0)
        {
            _ectx.Assert(srcIndex >= 0);
            _ectx.Assert(dstIndex >= 0);
            _ectx.Assert(Utils.Size(vector) >= _windowSize + srcIndex);
            _ectx.Assert(Utils.Size(result) >= _k + dstIndex);

            int i;

            // Computing the FFT of the trajectory matrix
            if (!_isSeriesFftCached)
                CacheInputSeriesFft();

            // Computing the FFT of the input vector
            for (i = 0; i < _k - 1; ++i)
                _inputRe[i] = 0;

            for (i = _k - 1; i < _seriesLength; ++i)
                _inputRe[i] = vector[_seriesLength - i - 1 + srcIndex];

            FftUtils.ComputeForwardFft(_inputRe, _allZerosIm, _outputRe, _outputIm, _inputRe.Length);

            // Computing the element-by-element product in the Fourier space
            double re;
            double im;
            for (i = 0; i < _seriesLength; ++i)
            {
                re = _outputRe[i];
                im = _outputIm[i];

                _outputRe[i] = _cachedSeriesFftRe[i] * re - _cachedSeriesFftIm[i] * im;
                _outputIm[i] = _cachedSeriesFftRe[i] * im + _cachedSeriesFftIm[i] * re;
            }

            // Computing the inverse FFT of the result
            FftUtils.ComputeBackwardFft(_outputRe, _outputIm, _outputRe, _outputIm, _inputRe.Length);

            // Generating the output
            if (add)
            {
                for (i = 0; i < _k; ++i)
                    result[i + dstIndex] += RoundUpToReal(_outputRe[_windowSize - 1 + i], _outputIm[_windowSize - 1 + i]);
            }
            else
            {
                for (i = 0; i < _k; ++i)
                    result[i + dstIndex] = RoundUpToReal(_outputRe[_windowSize - 1 + i], _outputIm[_windowSize - 1 + i]);
            }
        }

        /// <summary>
        /// This function efficiently computes the multiplication of the transpose of the trajectory matrix H by an arbitrary vector v, i.e. H' * v.
        /// </summary>
        /// <param name="vector">The input vector</param>
        /// <param name="result">The output vector allocated by the caller</param>
        /// <param name="add">Whether the multiplication result should be added to the current value in result</param>
        /// <param name="srcIndex">The starting index for the vector argument</param>
        /// <param name="dstIndex">The starting index for the result</param>
        public void MultiplyTranspose(Single[] vector, Single[] result, bool add = false, int srcIndex = 0, int dstIndex = 0)
        {
            if (_shouldFftUsed)
                FftMultiplyTranspose(vector, result, add, srcIndex, dstIndex);
            else
                NaiveMultiplyTranspose(vector, result, add, srcIndex, dstIndex);
        }

        /// <summary>
        /// This function computes the naive Hankelization of the matrix sigma * u * v' in O(L * K).
        /// </summary>
        /// <param name="u">The u vector</param>
        /// <param name="v">The v vector</param>
        /// <param name="sigma">The scalar coefficient</param>
        /// <param name="result">The output series</param>
        /// <param name="add">Whether the hankelization result should be added to the current value in result</param>
        /// <param name="uIndex">The starting index for the u vector argument</param>
        /// <param name="vIndex">The starting index for the v vector argument</param>
        /// <param name="dstIndex">The starting index for the result</param>
        /// <param name="start">The staring index of the series to be reconstructed (by default zero)</param>
        /// <param name="end">The ending index of the series to be reconstructed (by default series length)</param>
        private void NaiveRankOneHankelization(Single[] u, Single[] v, Single sigma, Single[] result, bool add = false,
            int uIndex = 0, int vIndex = 0, int dstIndex = 0, int? start = null, int? end = null)
        {
            int s;
            int e;
            int us;
            int ue;
            int vs;
            int ve;

            s = start ?? 0;
            e = end ?? _seriesLength - 1;

            ComputeBoundryIndices(s, e, out us, out ue, out vs, out ve);
            _ectx.Assert(0 <= ue && ue < _windowSize);
            _ectx.Assert(0 <= us && us <= ue);
            _ectx.Assert(0 <= ve && ve < _k);
            _ectx.Assert(0 <= vs && vs <= ve);

            var len = e - s + 1;
            var uLen = ue - us + 1;
            var vLen = ve - vs + 1;

            _ectx.Assert(uIndex >= 0);
            _ectx.Assert(vIndex >= 0);
            _ectx.Assert(dstIndex >= 0);
            _ectx.Assert(Utils.Size(u) >= _windowSize + uIndex);
            _ectx.Assert(Utils.Size(v) >= _k + vIndex);
            _ectx.Assert(Utils.Size(result) >= len + dstIndex);
            _ectx.Assert(!Single.IsNaN(sigma));
            _ectx.Assert(!Single.IsInfinity(sigma));

            int i;
            int j;
            int a;
            int b;
            int c;
            Single temp;

            if (!add)
            {
                for (i = 0; i < len; ++i)
                    result[i + dstIndex] = 0;
            }

            for (i = 0; i < len; ++i)
            {
                b = Math.Min(uLen, i + 1) - 1;
                a = i >= Math.Max(uLen, vLen) ? len - i : b + 1;
                c = Math.Max(0, i - uLen + 1);
                temp = 0;
                for (j = 0; j < a; ++j)
                    temp += u[us + b - j + uIndex] * v[vs + c + j + vIndex];

                result[i + dstIndex] += (temp * sigma / a);
            }
        }

        /// <summary>
        /// This function computes the efficient Hankelization of the matrix sigma * u * v' using Fast Fourier Transform in in O((L + K) * log(L + K)).
        /// For details, refer to Algorithm 4 in http://arxiv.org/pdf/0911.4498.pdf.
        /// </summary>
        /// <param name="u">The u vector</param>
        /// <param name="v">The v vector</param>
        /// <param name="sigma">The scalar coefficient</param>
        /// <param name="result">The output series</param>
        /// <param name="add">Whether the hankelization result should be added to the current value in result</param>
        /// <param name="uIndex">The starting index for the u vector argument</param>
        /// <param name="vIndex">The starting index for the v vector argument</param>
        /// <param name="dstIndex">The starting index for the result</param>
        /// <param name="start">The staring index of the series to be reconstructed (by default zero)</param>
        /// <param name="end">The ending index of the series to be reconstructed (by default series length)</param>
        private void FftRankOneHankelization(Single[] u, Single[] v, Single sigma, Single[] result, bool add = false,
            int uIndex = 0, int vIndex = 0, int dstIndex = 0, int? start = null, int? end = null)
        {
            int s;
            int e;
            int us;
            int ue;
            int vs;
            int ve;
            int i;

            s = start ?? 0;
            e = end ?? _seriesLength - 1;

            ComputeBoundryIndices(s, e, out us, out ue, out vs, out ve);
            _ectx.Assert(0 <= ue && ue < _windowSize);
            _ectx.Assert(0 <= us && us <= ue);
            _ectx.Assert(0 <= ve && ve < _k);
            _ectx.Assert(0 <= vs && vs <= ve);

            var len = e - s + 1;

            _ectx.Assert(uIndex >= 0);
            _ectx.Assert(vIndex >= 0);
            _ectx.Assert(dstIndex >= 0);
            _ectx.Assert(Utils.Size(u) >= _windowSize + uIndex);
            _ectx.Assert(Utils.Size(v) >= _k + vIndex);
            _ectx.Assert(Utils.Size(result) >= len + dstIndex);
            _ectx.Assert(!Single.IsNaN(sigma));
            _ectx.Assert(!Single.IsInfinity(sigma));

            if (!_isSeriesFftCached)
                CacheInputSeriesFft();

            // Computing the FFT of u
            for (i = us; i <= ue; ++i)
                _inputRe[i - us] = u[i + uIndex];

            for (i = ue + 1; i < len + us; ++i)
                _inputRe[i - us] = 0;

            FftUtils.ComputeForwardFft(_inputRe, _allZerosIm, _outputRe, _outputIm, len);

            // Computing the FFT of v
            for (i = vs; i <= ve; ++i)
                _inputRe[i - vs] = v[i + vIndex];

            for (i = ve + 1; i < len + vs; ++i)
                _inputRe[i - vs] = 0;

            FftUtils.ComputeForwardFft(_inputRe, _allZerosIm, _inputRe, _allZerosIm, len);

            // Computing the element-by-element product in the Fourier space
            double re;
            double im;
            for (i = 0; i < len; ++i)
            {
                re = _outputRe[i];
                im = _outputIm[i];

                _outputRe[i] = _inputRe[i] * re - _allZerosIm[i] * im;
                _outputIm[i] = _inputRe[i] * im + _allZerosIm[i] * re;
            }

            // Setting _allZerosIm to 0's again
            for (i = 0; i < _seriesLength; ++i)
                _allZerosIm[i] = 0;

            // Computing the inverse FFT of the result
            FftUtils.ComputeBackwardFft(_outputRe, _outputIm, _outputRe, _outputIm, len);

            // Generating the output
            int a = Math.Min(ue - us + 1, ve - vs + 1);

            if (add)
            {
                for (i = 0; i < a; ++i)
                    result[i + dstIndex] += RoundUpToReal(_outputRe[i], _outputIm[i], sigma / (i + 1));

                for (i = a; i < len - a + 1; ++i)
                    result[i + dstIndex] += RoundUpToReal(_outputRe[i], _outputIm[i], sigma / a);

                for (i = len - a + 1; i < len; ++i)
                    result[i + dstIndex] += RoundUpToReal(_outputRe[i], _outputIm[i], sigma / (len - i));
            }
            else
            {
                for (i = 0; i < a; ++i)
                    result[i + dstIndex] = RoundUpToReal(_outputRe[i], _outputIm[i], sigma / (i + 1));

                for (i = a; i < len - a + 1; ++i)
                    result[i + dstIndex] = RoundUpToReal(_outputRe[i], _outputIm[i], sigma / a);

                for (i = len - a + 1; i < len; ++i)
                    result[i + dstIndex] = RoundUpToReal(_outputRe[i], _outputIm[i], sigma / (len - i));
            }
        }

        /// <summary>
        /// This function efficiently computes the  Hankelization of the matrix sigma * u * v'.
        /// </summary>
        /// <param name="u">The u vector</param>
        /// <param name="v">The v vector</param>
        /// <param name="sigma">The scalar coefficient</param>
        /// <param name="result">The output series</param>
        /// <param name="add">Whether the hankelization result should be added to the current value in result</param>
        /// <param name="uIndex">The starting index for the u vector argument</param>
        /// <param name="vIndex">The starting index for the v vector argument</param>
        /// <param name="dstIndex">The starting index for the result</param>
        /// <param name="start">The staring index of the series to be reconstructed (by default zero)</param>
        /// <param name="end">The ending index of the series to be reconstructed (by default series length)</param>
        public void RankOneHankelization(Single[] u, Single[] v, Single sigma, Single[] result, bool add = false,
            int uIndex = 0, int vIndex = 0, int dstIndex = 0, int? start = null, int? end = null)
        {
            if (_shouldFftUsed)
                FftRankOneHankelization(u, v, sigma, result, add, uIndex, vIndex, dstIndex, start, end);
            else
                NaiveRankOneHankelization(u, v, sigma, result, add, uIndex, vIndex, dstIndex, start, end);
        }
    }
}
