// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.TimeSeries
{
    /// <summary>
    /// Loess is short for Robust Locally Weighted Regression and Smoothing Scatterplots.
    /// </summary>
    internal class Loess
    {
        /// <summary>
        /// The minimum length of a valid time series. A time series with length equals 2 is so trivial and meaningless less than 2.
        /// </summary>
        public const int MinTimeSeriesLength = 3;

        private const double NumericalThreshold = 1.0e-10;

        /// <summary>
        /// The ratio to determine the local region
        /// </summary>
        private readonly int _r;

        private readonly bool _isTemporal;

        /// <summary>
        /// Key is the index of the given point, value is the corresponding neighbors of the given point.
        /// </summary>
        private readonly Dictionary<int, LocalRegression> _neighbors;

        private IReadOnlyList<double> _x;
        private IReadOnlyList<double> _y;
        private readonly int _length;

        /// <summary>
        /// Initializes a new instance of the <see cref="Loess"/> class.
        /// Construct the least square algorithm specified with the number of neighbors
        /// </summary>
        /// <param name="xValues">The corresponding x-axis value</param>
        /// <param name="yValues">The corresponding y-axis value</param>
        /// <param name="isTemporal">If the regression is considered to take temporal information into account. In general, this is true if we are regressing a time series, and false if we are regressing scatter plot data</param>
        /// <param name="r">The smoothing range, if it is not specified, the algorithm will estimate the value of r by ratio.</param>
        public Loess(IReadOnlyList<double> xValues, IReadOnlyList<double> yValues, bool isTemporal, int? r = null)
        {
            Contracts.CheckValue(xValues, nameof(xValues));
            Contracts.CheckValue(yValues, nameof(yValues));

            if (xValues.Count < MinTimeSeriesLength || yValues.Count < MinTimeSeriesLength)
                throw Contracts.Except(string.Format("input time series length for Loess is below {0}", MinTimeSeriesLength));

            if (xValues.Count != yValues.Count)
                throw Contracts.Except("the x-axis length should be equal to y-axis length!: lowess");

            _neighbors = new Dictionary<int, LocalRegression>();

            _length = xValues.Count;
            _isTemporal = isTemporal;

            if (r == null)
            {
                _r = (int)(_length * LoessConfiguration.F);
            }
            else
            {
                _r = (int)r;
            }

            if (_r >= _length)
                _r = _length - 1;
            else if (_r < LoessConfiguration.MinimumNeighborCount) // the neighbors should be at least 2, or the matrix operations would encounter issues.
                _r = LoessConfiguration.MinimumNeighborCount;

            Init(xValues, yValues);
        }

        /// <summary>
        /// Initialize the signal with basic checking
        /// </summary>
        /// <param name="xValues">The input x-axis values</param>
        /// <param name="yValues">The input y-axis values</param>
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
        /// Estimate a y value by giving an x value, even if the x value is not one of the input points.
        /// When the x value is not one of the input points, find the closed one from input points, and use its model.
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
    /// This class is used to define a set of weight functions. These functions are useful for various purposes for smoothing,
    /// i.e., the weighted least squares.
    /// </summary>
    internal class WeightMethod
    {
        /// <summary>
        /// This is used for robust weight, It is one iteration step of loess.
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
        /// A famous weight function, since it enhances a chi-squared distributional approximation f an estimated of the error variance.
        /// Tricube should provide an adequate smooth in almost all situations.
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
    /// This class is used to store the parameters which are needed for lowess algorithm.
    /// The name of these constansts are compliant with the original terms in paper.
    /// </summary>
    internal class LoessConfiguration
    {
        /// <summary>
        /// Minumum number of neighbor counts, to apply underlying regression analysis.
        /// This number should be even, so that neighbors on left/right side of a given data point is balanced. Unbalanced neighbors would make the local-weighted regression biased noticeably at corner cases.
        /// </summary>
        public const int MinimumNeighborCount = 4;

        /// <summary>
        /// (0, 1], a smooth range ratio. Let fn be the number of neighbors of a specific point.
        /// </summary>
        public const double F = 0.3;

        /// <summary>
        /// The number of iterations for robust regression.
        /// </summary>
        public const int T = 2;
    }
}
