using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.TimeSeries
{
    /// <summary>
    /// this is the fast version of Loess. there are several alternatives to improve the performance. this one is an approximation approach.
    /// the smoothing is conducted on a sample set, and then the values on the left points are assigned directly.
    /// </summary>
    public class FastLoess
    {
        /// <summary>
        /// this class is a sampling based method, so here specifies the sample size.
        /// </summary>
        private static int _sampleSize = 100;

        private readonly IReadOnlyList<double> _x;
        private readonly IReadOnlyList<double> _y;
        private readonly int _length;

        private readonly Loess _smoother;

        /// <summary>
        /// Initializes a new instance of the <see cref="FastLoess"/> class.
        /// the fast version of the Loess method. when the time series is too long, the sampling will be conducted first
        /// to improve the performance.
        /// </summary>
        /// <param name="xValues">the input x-axis values</param>
        /// <param name="yValues">the input y-axis values</param>
        /// <param name="isTemporal">if the regression is considered to take temporal information into account. in general, this is true if we are regressing a time series, and false if we are regressing scatter plot data</param>
        /// <param name="r">this method will provide default smoothing ratio if user did not specify</param>
        public FastLoess(IReadOnlyList<double> xValues, IReadOnlyList<double> yValues, bool isTemporal = true, int r = -1)
        {
            Contracts.CheckValue(xValues, nameof(xValues));
            Contracts.CheckValue(yValues, nameof(yValues));
            Y = new List<double>();

            if (yValues.Count < BasicParameters.MinTimeSeriesLength)
                throw new Exception("input data structure cannot be 0-length: lowess");

            _x = xValues;
            _y = yValues;
            _length = _y.Count;

            // the sampling is not neccessary
            if (_length <= FastLoess._sampleSize)
            {
                if (r == -1)
                    _smoother = new Loess(_x, _y, isTemporal);
                else
                    _smoother = new Loess(_x, _y, r, isTemporal);
            }
            else
            {
                // conduct sampling based strategy, to boost the performance.
                double step = _length * 1.0 / FastLoess._sampleSize;
                var sampleX = new double[FastLoess._sampleSize];
                var sampleY = new double[FastLoess._sampleSize];
                for (int i = 0; i < FastLoess._sampleSize; i++)
                {
                    int index = (int)(i * step);
                    sampleX[i] = _x[index];
                    sampleY[i] = _y[index];
                }
                if (r == -1)
                    _smoother = new Loess(sampleX, sampleY, isTemporal);
                else
                    _smoother = new Loess(sampleX, sampleY, r, isTemporal);
            }
        }

        /// <summary>
        /// the estimated y values. this is the very cool smoothing method.
        /// </summary>
        public List<double> Y { get; }

        /// <summary>
        /// assign the smoothing values to all the data points, not only on the sample size.
        /// </summary>
        public void Estimate()
        {
            for (int i = 0; i < _length; i++)
            {
                double yValue = _smoother.EstimateY(_x[i]);
                Y.Add(yValue);
            }
        }

        /// <summary>
        /// estimate any y value by given any x value, event the x value is not one of the input points.
        /// </summary>
        public double EstimateY(double xValue)
        {
            return _smoother.EstimateY(xValue);
        }
    }
}
