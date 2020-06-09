using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.TimeSeries
{
    /// <summary>
    /// Loess is short for Robust Locally Weighted Regression and Smoothing Scatterplots.
    /// </summary>
    public class Loess
    {

        private const double NumericalThreshold = 1.0e-10;

        /// <summary>
        /// the ratio to determine the local region
        /// </summary>
        private readonly int _r;

        private readonly bool _isTemporal;

        /// <summary>
        /// key is the index of the given point, value is the corresponding neighbors of the given point.
        /// </summary>
        private Dictionary<int, LocalRegression> _neighbors;

        private IReadOnlyList<double> _x;
        private IReadOnlyList<double> _y;
        private int _length;

        /// <summary>
        /// Initializes a new instance of the <see cref="Loess"/> class.
        /// constructing the least square algorithm
        /// </summary>
        /// <param name="xValues">the corresponding x-axis value</param>
        /// <param name="yValues">the corresponding y-axis value</param>
        /// <param name="isTemporal">if the regression is considered to take temporal information into account. in general, this is true if we are regressing a time series, and false if we are regressing scatter plot data</param>
        public Loess(IReadOnlyList<double> xValues, IReadOnlyList<double> yValues, bool isTemporal)
        {
            Contracts.CheckValue(xValues, nameof(xValues));
            Contracts.CheckValue(yValues, nameof(yValues));

            if (xValues.Count < BasicParameters.MinTimeSeriesLength || yValues.Count < BasicParameters.MinTimeSeriesLength)
                throw new Exception("input data structure cannot be 0-length: lowess");

            if (xValues.Count != yValues.Count)
                throw new Exception("the x-axis length should be equal to y-axis length!: lowess");

            _neighbors = new Dictionary<int, LocalRegression>();

            _length = xValues.Count;
            _isTemporal = isTemporal;

            _r = (int)(_length * LoessConfiguration.F);

            // r cannot be equal to length.
            if (_r >= _length)
                _r = _length - 1;
            else if (_r < LoessConfiguration.MinimumNeighborCount) // the neighbors should be at least 2, or the matrix operations would encounter issues.
                _r = LoessConfiguration.MinimumNeighborCount;

            // DEBUG
            // control the performance
            if (_r >= LoessConfiguration.MaximumNeighborCount)
            {
                _r = LoessConfiguration.MaximumNeighborCount;
            }
            Init(xValues, yValues);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Loess"/> class.
        /// constructing the least square algorithm. specified with the # of neighbors
        /// </summary>
        /// <param name="xValues">the corresponding x-axis value</param>
        /// <param name="yValues">the corresponding y-axis value</param>
        /// <param name="r">the smoothing range is not determined by the ratio, but be specified externally. (which can exceed the length of the list)</param>
        /// <param name="isTemporal">if the regression is considered to take temporal information into account. in general, this is true if we are regressing a time series, and false if we are regressing scatter plot data</param>
        public Loess(IReadOnlyList<double> xValues, IReadOnlyList<double> yValues, int r, bool isTemporal)
        {
            Contracts.CheckValue(xValues, nameof(xValues));
            Contracts.CheckValue(yValues, nameof(yValues));

            if (xValues.Count < BasicParameters.MinTimeSeriesLength || yValues.Count < BasicParameters.MinTimeSeriesLength)
                throw new Exception("input data structure cannot be 0-length: lowess");

            if (xValues.Count != yValues.Count)
                throw new Exception("the x-axis length should be equal to y-axis length!: lowess");

            _neighbors = new Dictionary<int, LocalRegression>();

            _length = xValues.Count;
            _isTemporal = isTemporal;

            _r = r;
            if (_r < LoessConfiguration.MinimumNeighborCount) // the neighbors should be at least 2, or the matrix operations would encounter issues.
                _r = LoessConfiguration.MinimumNeighborCount;
            Init(xValues, yValues);
        }

        /// <summary>
        /// initialize the signal with basic checking
        /// </summary>
        /// <param name="xValues">the input x-axis values</param>
        /// <param name="yValues">the input y-axis values</param>
        private void Init(IReadOnlyList<double> xValues, IReadOnlyList<double> yValues)
        {
            _x = xValues;
            _y = yValues;
            for (int i = 0; i < _length; i++)
            {
                LocalRegression neighbor = new LocalRegression(_x, _y, i, _r, _isTemporal);
                _neighbors.Add(i, neighbor);
            }
        }

        /// <summary>
        /// estimate any y value by given any x value, even the x value is not one of the input points.
        /// when the x value is not one of the input points, find the closed one from input points, and use its model.
        /// </summary>
        /// <param name="xValue">find the index with value closest to the input x value.</param>
        public double EstimateY(double xValue)
        {
            // find the closest point in x to the xValue
            int start = 0;
            int end = _length - 1;
            while (end - start > 1)
            {
                int mid = (start + end) / 2;
                if (_x[mid] > xValue)
                {
                    end = mid;
                }
                else
                {
                    start = mid;
                }
            }
            double distanceDiff = (_x[end] - xValue) - (xValue - _x[start]);

            int index = distanceDiff > -NumericalThreshold ? start : end;
            return _neighbors[index].Y(xValue);
        }
    }

    /// <summary>
    /// this class is used to define a set of weight functions. these functions are useful for various purposes for smoothing.
    /// i.e., the weighted least squares.
    /// </summary>
    public class WeightMethod
    {
        /// <summary>
        /// this is used for robust weight, it is one iteration step of loess.
        /// </summary>
        public static double BisquareWeight(double value)
        {
            double abs = Math.Abs(value);
            if (abs >= 1)
                return 0;
            double temp = 1 - abs * abs;
            return temp * temp;
        }

        /// <summary>
        /// a famous weight function, since it enhances a chi-squared distributional approximation of an estimated of the error variance.
        /// tricube should provide an adequate smooth in almost all situations. /ref
        /// </summary>
        public static double Tricube(double value)
        {
            double abs = Math.Abs(value);
            if (abs >= 1)
                return 0;
            double temp = 1 - abs * abs * abs;
            return temp * temp * temp;
        }
    }

    /// <summary>
    /// this class is used to store the parameters which are needed for lowess algorithm.
    /// the name of these constansts are compliant with the original terms in paper.
    /// </summary>
    public class LoessConfiguration
    {
        /// <summary>
        /// this value is used for performance concern. when the length of the series goes large, a ratio of neighbors will be significant,
        /// which leads to unsatisfied slow. so this value is used to bound the maximum # of neighbors one epoch can have.
        /// </summary>
        public const int MaximumNeighborCount = 100;

        /// <summary>
        /// minumum number of neighbor counts, to apply underlying regression analysis.
        /// this number should be even, so that neighbors on left/right side of a given data point is balanced. unbalanced neighbors would make the local-weighted regression biased noticeably at corner cases.
        /// </summary>
        public const int MinimumNeighborCount = 4;

        /// <summary>
        /// (0, 1], a smooth range ratio. let fn be the number of neighbors of a specific point.
        /// </summary>
        public static readonly double F = 0.3;

        /// <summary>
        /// the number of iterations for robust regression.
        /// </summary>
        public static readonly int T = 2;
    }
}
