// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Internal.CpuMath.Core;

namespace Microsoft.ML.TimeSeries
{
    /// <summary>
    /// This class is used to maintain the neighbors of a given particular point.
    /// </summary>
    internal class LocalRegression
    {
        private const double NumericalThreshold = 1.0e-10;
        private readonly IReadOnlyList<double> _x;
        private readonly IReadOnlyList<double> _y;
        private readonly int _length;

        /// <summary>
        /// The model is learned by several iterations of local weighted regression.
        /// </summary>
        private AbstractPolynomialModel _model;

        /// <summary>
        /// Initializes a new instance of the <see cref="LocalRegression"/> class.
        /// Construct the neighborhood information of a given point. note that the input series will not be copies again, due to
        /// memory usage concern.
        /// </summary>
        /// <param name="x">The complete values of x-axis</param>
        /// <param name="y">The complete values of y-axis</param>
        /// <param name="selfIndex">The index of the current point</param>
        /// <param name="r">Number of neighbors, usually should be less then n. if it is equal/larger than n, the weight has slight change.</param>
        /// <param name="isTemporal">If the regression is considered to take temporal information into account. In general, this is true if we are regressing a time series, and false if we are regressing scatter plot data</param>
        public LocalRegression(IReadOnlyList<double> x, IReadOnlyList<double> y, int selfIndex, int r, bool isTemporal = true)
        {
            Contracts.CheckValue(x, nameof(x));
            Contracts.CheckValue(y, nameof(y));

            if (x.Count <= 1 || x.Count != y.Count)
                throw Contracts.Except("cannot accomplish neighbors obtaining");

            _model = null;

            _x = x;
            _y = y;
            _length = _x.Count;
            SelfIndex = selfIndex;

            int startIndex = selfIndex;
            int endIndex = selfIndex;
            double selfValue = _x[SelfIndex];

            // The farthest neighbor is contained in the list. This is the normal case.
            if (r < _length)
            {
                int left = r;
                while (left > 0)
                {
                    if (startIndex == 0)
                    {
                        endIndex += left;
                        break;
                    }
                    if (endIndex == _length - 1)
                    {
                        startIndex -= left;
                        break;
                    }
                    double startV = _x[startIndex];
                    double endV = _x[endIndex];

                    // the left point is closer to the current
                    // bug fix: avoid potential inconsistent index assignment due to numerical precision.
                    double distanceDiff = (selfValue - startV) - (endV - selfValue);

                    if (distanceDiff < NumericalThreshold)
                    {
                        startIndex--;
                    }
                    else
                    {
                        endIndex++;
                    }
                    left--;
                }
                StartIndex = startIndex;
                EndIndex = endIndex;

                var neighborsCount = EndIndex - StartIndex + 1;
                NeighborsX = new double[neighborsCount];
                NeighborsY = new double[neighborsCount];
                Weights = new double[neighborsCount];

                double leftRange = selfValue - _x[startIndex];
                double rightRange = _x[endIndex] - selfValue;
                double range = Math.Max(leftRange, rightRange);

                if (isTemporal)
                {
                    for (int i = StartIndex; i <= EndIndex; i++)
                    {
                        NeighborsX[i - StartIndex] = _x[i];
                        NeighborsY[i - StartIndex] = _y[i];
                        Weights[i - StartIndex] = WeightMethod.Tricube((_x[i] - selfValue) / range);
                    }
                }
                else
                {
                    for (int i = StartIndex; i <= EndIndex; i++)
                    {
                        NeighborsX[i - StartIndex] = _x[i];
                        NeighborsY[i - StartIndex] = _y[i];

                        // since we do not consider the local/temporal information, all the neighbors share same weight for further weighted regression
                        Weights[i - StartIndex] = 1.0;
                    }
                }
            }
            else
            {
                // when the r is equal/larger than n
                StartIndex = 0;
                EndIndex = _length - 1;

                double leftRange = selfValue - _x[StartIndex];
                double rightRange = _x[EndIndex] - selfValue;
                double range = Math.Max(leftRange, rightRange);

                var neighborsCount = EndIndex - StartIndex + 1;
                NeighborsX = new double[neighborsCount];
                NeighborsY = new double[neighborsCount];
                Weights = new double[neighborsCount];

                // this is the slight modification of the weighting calculation
                range = range * r / (_length - 1);

                if (isTemporal)
                {
                    for (int i = StartIndex; i <= EndIndex; i++)
                    {
                        NeighborsX[i - StartIndex] = _x[i];
                        NeighborsY[i - StartIndex] = _y[i];
                        Weights[i - StartIndex] = WeightMethod.Tricube((_x[i] - selfValue) / range);
                    }
                }
                else
                {
                    for (int i = StartIndex; i <= EndIndex; i++)
                    {
                        NeighborsX[i - StartIndex] = _x[i];
                        NeighborsY[i - StartIndex] = _y[i];

                        // since we do not consider the local/temporal information, all the neighbors share same weight for further weighted regression
                        Weights[i - StartIndex] = 1.0;
                    }
                }
            }
        }

        /// <summary>
        /// The values of the y-axis of the neighbors (include the self point)
        /// </summary>
        public double[] NeighborsY { get; private set; }

        /// <summary>
        /// The values of the x-axis of the neighbors (include the self point)
        /// </summary>
        public double[] NeighborsX { get; private set; }

        /// <summary>
        /// The weights for each neighbor. This is used for weighted least squares.
        /// </summary>
        public double[] Weights { get; private set; }

        /// <summary>
        /// The start index of the neighbors (inclusive)
        /// </summary>
        public int StartIndex { get; private set; }

        /// <summary>
        /// The end index of the neighbors (inclusive)
        /// </summary>
        public int EndIndex { get; private set; }

        /// <summary>
        /// The index of the self point. The index is on the complete series, not only on the neighbor series.
        /// </summary>
        public int SelfIndex { get; private set; }

        private void Estimate()
        {
            for (int iter = 0; iter < LoessConfiguration.T; iter++)
            {
                _model = Regression();

                // calculate the errors
                var errors = new double[NeighborsX.Length];
                var absErrors = new double[NeighborsX.Length];
                for (int i = 0; i < NeighborsX.Length; i++)
                {
                    double error = NeighborsY[i] - _model.Y(NeighborsX[i]);
                    errors[i] = error;
                    absErrors[i] = Math.Abs(error);
                }

                Array.Sort(absErrors);

                double median = absErrors[absErrors.Length / 2];
                if (median == 0) // a very subtle bug! sometimes, when the input data is very clean, so that the median could be 0!
                    median = double.Epsilon;

                // calculate the gain for new weights. the outliers will get much less weight
                var deltas = new double[errors.Length];
                for (int i = 0; i < errors.Length; i++)
                {
                    deltas[i] = WeightMethod.BisquareWeight(errors[i] / 6.0 / median);
                }

                // update new weights.
                for (int i = 0; i < Weights.Length; i++)
                {
                    Weights[i] *= deltas[i];
                }
            }
        }

        /// <summary>
        /// Get the best estimated y for the current value.
        /// </summary>
        public double Y()
        {
            if (_model == null)
            {
                Estimate();
            }
            return _model.Y(_x[SelfIndex]);
        }

        /// <summary>
        /// Get the best estimated y for any given x-value, event not one of the observed point
        /// </summary>
        /// <param name="xValue">Any given x value</param>
        public double Y(double xValue)
        {
            if (_model == null)
            {
                Estimate();
            }
            return _model.Y(xValue);
        }

        private AbstractPolynomialModel Regression()
        {
            LeastSquares ls = new LeastSquares(NeighborsX, NeighborsY);
            return ls.RegressionDegreeOneWeighted(Weights);
        }
    }
}
