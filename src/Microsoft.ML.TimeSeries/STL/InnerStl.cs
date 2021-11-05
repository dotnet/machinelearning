// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.TimeSeries
{
    internal class InnerStl
    {
        private readonly bool _isTemporal;
        private double[] _seasonalComponent;
        private double[] _trendComponent;
        private double[] _residual;

        // Arrays for intermediate results
        private List<double>[] _cycleSubSeries;
        private List<double>[] _smoothedSubseries;

        private double[] _s;
        private double[] _t;
        private double[] _detrendedY;
        private double[] _c;
        private double[] _deseasonSeries;

        /// <summary>
        /// The minimum length of a valid time series. A time series with length equals 2 is so trivial and meaningless less than 2.
        /// </summary>
        public const int MinTimeSeriesLength = 3;

        /// <summary>
        /// The smoothing parameter for the seasonal component.
        /// This parameter should be odd, and at least 7.
        /// </summary>
        private const int Ns = 9;

        /// <summary>
        /// The number of passes through the inner loop. This parameter is set to 2, which works for many cases.
        /// </summary>
        private const int Ni = 2;

        /// <summary>
        /// The number of robustness iterations of the outer loop. This parameter is not used in this implementation as we simplify the implementation.
        /// Keep this parameter here as it is listed as one of the six parameters described in the original paper.
        /// </summary>
        private const int No = 10;

        /// <summary>
        /// The smoothing parameter for the low-pass filter.
        /// This parameter should be the least odd integer greater than or equal to np.
        /// It will preventing the trend and seasonal components from competing for the same variation in the data.
        /// </summary>
        private int Nl(int np)
        {
            if (np % 2 == 0)
                return np + 1;
            return np;
        }

        /// <summary>
        /// The smoothing parameter for the trend component.
        /// In order to avoid the trend ans seasonal components compete for variation in the data, the nt should be chosen
        /// S.t., satisty the following inequality.
        /// </summary>
        private int Nt(int np)
        {
            double value = 1.5 * np / (1.0 - 1.5 / Ns);
            int result = (int)value + 1;
            if (result % 2 == 0)
                result++;
            return result;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="InnerStl"/> class.
        /// For a time series, only with y values. assume the x-values are 0, 1, 2, ...
        /// Since this method supports decompose seasonal signal, which requires the equal-space of the input x-axis values.
        /// Otherwise, the smoothing on seasonal component will be very complicated.
        /// </summary>
        /// <param name="isTemporal">If the regression is considered to take temporal information into account. In general, this is true if we are regressing a time series, and false if we are regressing scatter plot data</param>
        public InnerStl(bool isTemporal)
        {
            _isTemporal = isTemporal;
        }

        /// <summary>
        /// The seasonal component
        /// </summary>
        public IReadOnlyList<double> SeasonalComponent
        {
            get { return _seasonalComponent; }
        }

        /// <summary>
        /// The trend component
        /// </summary>
        public IReadOnlyList<double> TrendComponent
        {
            get { return _trendComponent; }
        }

        /// <summary>
        /// The left component after seasonal and trend are eliminated.
        /// </summary>
        public IReadOnlyList<double> Residual
        {
            get { return _residual; }
        }

        /// <summary>
        /// The core for the robust trend-seasonal decomposition. See the ref: http://www.wessa.net/download/stl.pdf,
        /// See section 2 and 3. especially section 2.
        /// </summary>
        /// <returns>Return true if the process goes successfully. Otherwise, return false.</returns>
        public bool Decomposition(IReadOnlyList<double> yValues, int np)
        {
            Contracts.CheckValue(yValues, nameof(yValues));
            Contracts.CheckParam(np > 0, nameof(np));

            if (yValues.Count < MinTimeSeriesLength)
                throw Contracts.Except(string.Format("input time series length for InnerStl is below {0}", MinTimeSeriesLength));

            int length = yValues.Count;
            Array.Resize(ref _seasonalComponent, length);
            Array.Resize(ref _trendComponent, length);
            Array.Resize(ref _residual, length);

            Array.Resize(ref _s, length);
            Array.Resize(ref _t, length);
            Array.Resize(ref _detrendedY, length);
            Array.Resize(ref _c, length + np * 2);
            Array.Resize(ref _deseasonSeries, length);

            Array.Resize(ref _cycleSubSeries, np);
            Array.Resize(ref _smoothedSubseries, np);

            for (int i = 0; i < length; ++i)
            {
                _t[i] = 0;
            }

            for (int iter = 0; iter < Ni; iter++)
            {
                // step1: detrending
                Detrending(yValues, _t, _detrendedY);

                // step2: cycle-subseries smoothing
                bool success = CycleSubseriesSmooth(_detrendedY, np, _c);
                if (!success)
                {
                    return false;
                }

                // step3: low-pass filtering of smoothed cycle-subseries
                var lowPass = LowPassFiltering(_c, np);

                // step4: detrending of smoothed cycle-subseries
                SmoothedCycleSubseriesDetrending(_c, lowPass, _s);

                // step5: deseasonalizing
                Deseasonalizing(yValues, _s, _deseasonSeries);

                // step6: trend smoothing
                TrendSmooth(_deseasonSeries, np, _t);
            }

            for (int i = 0; i < _s.Length; i++)
            {
                _seasonalComponent[i] = _s[i];
                _trendComponent[i] = _t[i];
                _residual[i] = yValues[i] - _s[i] - _t[i];
            }

            return true;
        }

        private void Detrending(IReadOnlyList<double> y, IReadOnlyList<double> t, double[] detrendedY)
        {
            for (int i = 0; i < y.Count; i++)
                detrendedY[i] = y[i] - t[i];
        }

        private bool CycleSubseriesSmooth(double[] detrendedY, int np, double[] c)
        {
            for (int i = 0; i < np; i++)
            {
                _cycleSubSeries[i] = new List<double>();
                _smoothedSubseries[i] = new List<double>();
            }

            // obtain all the subseries
            for (int i = 0; i < detrendedY.Length; i++)
            {
                int cycleIndex = i % np;
                _cycleSubSeries[cycleIndex].Add(detrendedY[i]);
            }

            // smoothing on each subseries
            for (int i = 0; i < _cycleSubSeries.Length; i++)
            {
                List<double> virtualXValues = VirtualXValuesProvider.GetXValues(_cycleSubSeries[i].Count);

                FastLoess model = new FastLoess(virtualXValues, _cycleSubSeries[i], _isTemporal, Ns);
                model.Estimate();

                // add a prior point
                _smoothedSubseries[i].Add(model.EstimateY(-1.0));
                _smoothedSubseries[i].AddRange(model.Y);

                // add a after point
                _smoothedSubseries[i].Add(model.EstimateY(_cycleSubSeries[i].Count * 1.0));
            }

            // c is the smoothed series, with _length + 2Np points.
            int index = 0;
            for (int i = 0; i < _smoothedSubseries[0].Count; i++)
            {
                for (int j = 0; j < _smoothedSubseries.Length; j++)
                {
                    if (_smoothedSubseries[j].Count <= i)
                        break;
                    if (_smoothedSubseries[j][i].Equals(double.NaN))
                    {
                        return false;
                    }
                    c[index] = (_smoothedSubseries[j][i]);
                    ++index;
                }
            }

            return true;
        }

        private FastLoess LowPassFiltering(double[] c, int np)
        {
            List<double> c1 = MovingAverage(c, np);
            List<double> c2 = MovingAverage(c1, np);
            List<double> c3 = MovingAverage(c2, 3);
            List<double> virtualC3XValues = VirtualXValuesProvider.GetXValues(c3.Count);
            FastLoess lowPass = new FastLoess(virtualC3XValues, c3, _isTemporal, Nl(np));
            lowPass.Estimate();

            return lowPass;
        }

        private void SmoothedCycleSubseriesDetrending(double[] c, FastLoess lowPass, double[] s)
        {
            for (int i = 0; i < s.Length; i++)
            {
                s[i] = c[i] - lowPass.Y[i];
            }
        }

        private void Deseasonalizing(IReadOnlyList<double> y, double[] s, double[] deseasonSeries)
        {
            for (int i = 0; i < y.Count; i++)
            {
                deseasonSeries[i] = y[i] - s[i];
            }
        }

        private void TrendSmooth(double[] deseasonSeries, int np, double[] t)
        {
            List<double> virtualDeseasonSeries = VirtualXValuesProvider.GetXValues(deseasonSeries.Length);
            FastLoess trender = new FastLoess(virtualDeseasonSeries, deseasonSeries, _isTemporal, Nt(np));
            trender.Estimate();
            for (int i = 0; i < deseasonSeries.Length; i++)
            {
                t[i] = trender.Y[i];
            }
        }

        /// <summary>
        /// This class provides the virtual x values for multi object usage.
        /// The cache mechanism is used for performance consideration.
        /// </summary>
        internal class VirtualXValuesProvider
        {
            private static readonly Dictionary<int, List<double>> _xValuesPool;

            static VirtualXValuesProvider()
            {
                _xValuesPool = new Dictionary<int, List<double>>();
            }

            /// <summary>
            /// Get a list of virtual x-axis values. the values are from 0 to length - 1.
            /// </summary>
            /// <param name="length">Specify the length you want to create the x values.</param>
            /// <returns>If the input is cached, return the cached output directly. Otherwise, create a new list and return</returns>
            internal static List<double> GetXValues(int length)
            {
                lock (_xValuesPool)
                {
                    List<double> xValues;
                    if (_xValuesPool.TryGetValue(length, out xValues))
                        return xValues;

                    var newXValues = new List<double>(length);
                    for (int i = 0; i < length; i++)
                        newXValues.Add(i);

                    _xValuesPool.Add(length, newXValues);
                    return newXValues;
                }
            }
        }

        private static List<double> MovingAverage(IReadOnlyList<double> s, int length)
        {
            List<double> results = new List<double>(s.Count);
            double partialSum = 0;
            for (int i = 0; i < length; ++i)
            {
                partialSum += s[i];
            }

            for (int i = length; i < s.Count; ++i)
            {
                results.Add(partialSum / length);
                partialSum = partialSum - s[i - length] + s[i];
            }
            results.Add(partialSum / length);

            return results;
        }
    }
}
