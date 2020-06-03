using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.TimeSeries
{
    internal class InnerStl
    {
        private readonly IReadOnlyList<double> _x;
        private readonly IReadOnlyList<double> _y;
        private readonly int _length;
        private readonly bool _isTemporal;
        private readonly StlConfiguration _config;

        private readonly double[] _seasonalComponent;
        private readonly double[] _trendComponent;
        private readonly double[] _residual;
        private readonly int[] _outlierIndexes;
        private readonly double[] _outlierSeverity;

        /// <summary>
        /// Initializes a new instance of the <see cref="InnerStl"/> class.
        /// for a time series, only with y values. assume the x-values are 0, 1, 2, ...
        /// since this method supports decompose seasonal signal, which requires the equal-space of the input x-axis values.
        /// otherwise, the smoothing on seasonal component will be very complicated.
        /// </summary>
        /// <param name="yValues">the y-axis values</param>
        /// <param name="config">the configuration for applying regression</param>
        /// <param name="isTemporal">if the regression is considered to take temporal information into account. in general, this is true if we are regressing a time series, and false if we are regressing scatter plot data</param>
        public InnerStl(IReadOnlyList<double> yValues, StlConfiguration config, bool isTemporal)
        {
            //ExtendedDiagnostics.EnsureArgumentNotNull(yValues, nameof(yValues));
            //ExtendedDiagnostics.EnsureArgumentNotNull(config, nameof(config));

            if (yValues.Count == 0)
                throw new Exception("input data structure cannot be 0-length: innerSTL");

            _y = yValues;
            _length = _y.Count;
            _isTemporal = isTemporal;
            _x = VirtualXValuesProvider.GetXValues(_length);
            _config = config;

            _seasonalComponent = new double[_length];
            _trendComponent = new double[_length];
            _residual = new double[_length];
            _outlierIndexes = new int[_length];
            _outlierSeverity = new double[_length];
        }

        /// <summary>
        /// the seasonal component
        /// </summary>
        public IReadOnlyList<double> SeasonalComponent
        {
            get { return _seasonalComponent; }
        }

        /// <summary>
        /// the trend component
        /// </summary>
        public IReadOnlyList<double> TrendComponent
        {
            get { return _trendComponent; }
        }

        /// <summary>
        /// the left component after seasonal and trend are eliminated.
        /// </summary>
        public IReadOnlyList<double> Residual
        {
            get { return _residual; }
        }

        /// <summary>
        /// this field is used to indicate which data point is an outlier.
        /// the definition is outlier is, not contribute for the local regression.
        /// </summary>
        public IReadOnlyList<int> OutlierIndexeses
        {
            get { return _outlierIndexes; }
        }

        /// <summary>
        /// [0, infinite] indicate the severity of each outlier
        /// </summary>
        public IReadOnlyList<double> OutlierSeverity
        {
            get { return _outlierSeverity; }
        }

        /// <summary>
        /// calculate the slope of the trend component
        /// </summary>
        public double Slope
        {
            get;
            private set;
        }

        /// <summary>
        /// the mean residual squares. here the outliers are not taken into account.
        /// </summary>
        public double Mrs
        {
            get;
            private set;
        }

        /// <summary>
        /// the core for the robust trend-seasonal decomposition. see the ref: http://www.wessa.net/download/stl.pdf,
        /// see section 2 and 3. especially section 2.
        /// </summary>
        /// <returns>return true if the process goes successfully. otherwise, return false.</returns>
        public bool Decomposition()
        {
            double[] s = new double[_length];
            double[] t = new double[_length];
            for (int iter = 0; iter < StlConfiguration.Ni; iter++)
            {
                // step1: detrending
                double[] detrendedY = new double[_length];
                for (int i = 0; i < _length; i++)
                    detrendedY[i] = _y[i] - t[i];

                // step2: cycle-subseries smoothing
                List<double>[] cycleSubSeries = new List<double>[_config.Np];
                List<double>[] smoothedSubseries = new List<double>[_config.Np];
                for (int i = 0; i < _config.Np; i++)
                {
                    cycleSubSeries[i] = new List<double>();
                    smoothedSubseries[i] = new List<double>();
                }

                // obtain all the subseries
                for (int i = 0; i < _length; i++)
                {
                    int cycleIndex = i % _config.Np;
                    cycleSubSeries[cycleIndex].Add(detrendedY[i]);
                }

                // smoothing on each subseries
                for (int i = 0; i < cycleSubSeries.Length; i++)
                {
                    List<double> virtualXValues = VirtualXValuesProvider.GetXValues(cycleSubSeries[i].Count);

                    FastLoess model = new FastLoess(virtualXValues, cycleSubSeries[i], _isTemporal, StlConfiguration.Ns);
                    model.Estimate();

                    // add a prior point
                    smoothedSubseries[i].Add(model.EstimateY(-1.0));
                    smoothedSubseries[i].AddRange(model.Y);

                    // add a after point
                    smoothedSubseries[i].Add(model.EstimateY(cycleSubSeries[i].Count * 1.0));
                }

                // c is the smoothed series, with _length+2Np points.
                List<double> c = new List<double>();
                for (int i = 0; i < smoothedSubseries[0].Count; i++)
                {
                    for (int j = 0; j < smoothedSubseries.Length; j++)
                    {
                        if (smoothedSubseries[j].Count <= i)
                            break;
                        if (smoothedSubseries[j][i].Equals(double.NaN))
                        {
                            return false;
                        }
                        c.Add(smoothedSubseries[j][i]);
                    }
                }

                // step3: low-pass filtering of smoothed cycle-subseries
                List<double> c1 = MovingAverage.MA(c, _config.Np);
                List<double> c2 = MovingAverage.MA(c1, _config.Np);
                List<double> c3 = MovingAverage.MA(c2, 3);
                List<double> virtualC3XValues = VirtualXValuesProvider.GetXValues(c3.Count);
                FastLoess lowPass = new FastLoess(virtualC3XValues, c3, _isTemporal, _config.Nl);
                lowPass.Estimate();

                // step4: detrending of smoothed cycle-subseries
                for (int i = 0; i < _length; i++)
                {
                    s[i] = c[i] - lowPass.Y[i];
                }

                // step5: deseasonalizing
                List<double> deseasonSeries = new List<double>();
                for (int i = 0; i < _length; i++)
                {
                    deseasonSeries.Add(_y[i] - s[i]);
                }

                // step6: trend smoothing
                List<double> virtualDeseasonSeries = VirtualXValuesProvider.GetXValues(deseasonSeries.Count);
                FastLoess trender = new FastLoess(virtualDeseasonSeries, deseasonSeries, _isTemporal, _config.Nt);
                trender.Estimate();
                for (int i = 0; i < _length; i++)
                {
                    t[i] = trender.Y[i];
                }
            }

            for (int i = 0; i < s.Length; i++)
            {
                _seasonalComponent[i] = s[i];
                _trendComponent[i] = t[i];
            }

            // the slope is still based on the regression models.
            Slope = (_trendComponent[_length - 1] - _seasonalComponent[0]) / (_length - 1);

            var absResiduals = new List<double>(_residual);
            for (int i = 0; i < _y.Count; i++)
            {
                _residual[i] = _y[i] - _seasonalComponent[i] - _trendComponent[i];
                absResiduals.Add(Math.Abs(_y[i] - _seasonalComponent[i] - _trendComponent[i]));
            }

            // identify the outliers and corresponding mean residual squares (Mrs)
            //double median = MathUtility.QuickSelect(absResiduals, absResiduals.Count / 2);
            double median = 0;

            // when median is very close to 0, which means the regularity of the serial is strong, so that no data points is outlier.
            Mrs = 0;
            int nonOutlierCount = 0;
            if (median < 0.0001)
            {
                // the curve fitting is perfect, so Mrs remains 0. no update.
                for (int i = 0; i < _length; i++)
                {
                    _outlierIndexes[i] = 0;
                    _outlierSeverity[i] = 0;
                }
            }
            else
            {
                for (int i = 0; i < _length; i++)
                {
                    double severity = Math.Abs(_residual[i]) / median;

                    // this is the key criteria
                    if (severity > 6)
                    {
                        _outlierIndexes[i] = 1;
                        _outlierSeverity[i] = severity;
                    }
                    else
                    {
                        nonOutlierCount++;
                        Mrs += _residual[i] * _residual[i];
                        _outlierIndexes[i] = 0;
                        _outlierSeverity[i] = 0;
                    }
                }
                Mrs /= nonOutlierCount;
            }
            return true;
        }

        public bool DecompositionSimple()
        {
            if (_config.Np <= 0)
            {
                for (int i = 0; i < _y.Count; ++i)
                {
                    _residual[i] = _y[i];
                }
            } else
            {
                double[] sum = new double[_config.Np];
                for (int i = 0; i < _y.Count; i++)
                {
                    var indexInPeriod = i % _config.Np;
                    sum[indexInPeriod] += _y[i];
                }
                double[] averages = sum.Select((s, i) => s / (_y.Count / _config.Np)).ToArray();
                for (int i = 0; i < _y.Count; ++i)
                {
                    _residual[i] = _y[i] - averages[i % _config.Np];
                }
            }

            return true;
        }

        /// <summary>
        /// this class provides the virtual x values for multi object usage.
        /// the cache mechanism is used for performance consideration.
        /// </summary>
        internal class VirtualXValuesProvider
        {
            private static Dictionary<int, List<double>> _xValuesPool;

            static VirtualXValuesProvider()
            {
                _xValuesPool = new Dictionary<int, List<double>>();
            }

            /// <summary>
            /// get a list of virtual x-axis values. the values are from 0 to length - 1.
            /// </summary>
            /// <param name="length">specify the length you want to create the x values.</param>
            /// <returns>if this is cached, return directly. otherwise, create a new list and return</returns>
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
    }
}
