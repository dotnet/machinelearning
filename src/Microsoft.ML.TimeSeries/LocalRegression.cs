using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.TimeSeries
{
    /// <summary>
    /// this class is used to maintain the neighbors of a given particular point.
    /// </summary>
    internal class LocalRegression
    {
        private const double NumericalThreshold = 1.0e-10;
        private readonly IReadOnlyList<double> _x;
        private readonly IReadOnlyList<double> _y;
        private int _length;

        /// <summary>
        /// the model is learned by several iterations of local weighted regression.
        /// </summary>
        private PolynomialModel _model = null;

        /// <summary>
        /// Initializes a new instance of the <see cref="LocalRegression"/> class.
        /// construct the neighborhood information of a given point. note that the input series will not be copies again, due to
        /// memory usage concern.
        /// </summary>
        /// <param name="x">the complete values of x-axis</param>
        /// <param name="y">the complete values of y-axis</param>
        /// <param name="selfIndex">the index of the current point</param>
        /// <param name="r">number of neighbors, usually should be less then n. if it is equal/larger than n, the weight has slight change.</param>
        /// <param name="isTemporal">if the regression is considered to take temporal information into account. in general, this is true if we are regressing a time series, and false if we are regressing scatter plot data</param>
        internal LocalRegression(IReadOnlyList<double> x, IReadOnlyList<double> y, int selfIndex, int r, bool isTemporal = true)
        {
            //ExtendedDiagnostics.EnsureArgumentNotNull(x, nameof(x));
            //ExtendedDiagnostics.EnsureArgumentNotNull(y, nameof(y));

            if (x.Count <= 1 || x.Count != y.Count)
                throw new Exception("cannot accomplish neighbors obtaining");

            _x = x;
            _y = y;
            _length = _x.Count;
            SelfIndex = selfIndex;

            NeighborsX = new List<double>();
            NeighborsY = new List<double>();
            Weights = new List<double>();

            int startIndex = selfIndex;
            int endIndex = selfIndex;
            double selfValue = _x[SelfIndex];

            // the normal case, the farthest neighbor is contained in the list.
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

                double leftRange = selfValue - _x[startIndex];
                double rightRange = _x[endIndex] - selfValue;
                double range = Math.Max(leftRange, rightRange);

                if (isTemporal)
                {
                    for (int i = StartIndex; i <= EndIndex; i++)
                    {
                        NeighborsX.Add(_x[i]);
                        NeighborsY.Add(_y[i]);
                        Weights.Add(WeightMethod.Tricube((_x[i] - selfValue) / range));
                    }
                }
                else
                {
                    for (int i = StartIndex; i <= EndIndex; i++)
                    {
                        NeighborsX.Add(_x[i]);
                        NeighborsY.Add(_y[i]);

                        // since we do not consider the local/temporal information, all the neighbors share same weight for further weighted regression
                        Weights.Add(1.0);
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

                // this is the slight modification of the weighting calculation
                range = range * r / (_length - 1);

                if (isTemporal)
                {
                    for (int i = StartIndex; i <= EndIndex; i++)
                    {
                        NeighborsX.Add(_x[i]);
                        NeighborsY.Add(_y[i]);
                        Weights.Add(WeightMethod.Tricube((_x[i] - selfValue) / range));
                    }
                }
                else
                {
                    for (int i = StartIndex; i <= EndIndex; i++)
                    {
                        NeighborsX.Add(_x[i]);
                        NeighborsY.Add(_y[i]);

                        // since we do not consider the local/temporal information, all the neighbors share same weight for further weighted regression
                        Weights.Add(1.0);
                    }
                }
            }
        }

        /// <summary>
        /// the values of the y-axis of the neighbors (include the self point)
        /// </summary>
        public List<double> NeighborsY { get; private set; }

        /// <summary>
        /// the values of the x-axis of the neighbors (include the self point)
        /// </summary>
        public List<double> NeighborsX { get; private set; }

        /// <summary>
        /// the weights for each neighbor. this is used for weighted least squares.
        /// </summary>
        public List<double> Weights { get; private set; }

        /// <summary>
        /// the start index of the neighbors (inclusive)
        /// </summary>
        public int StartIndex { get; private set; }

        /// <summary>
        /// the end index of the neighbors (inclusive)
        /// </summary>
        public int EndIndex { get; private set; }

        /// <summary>
        /// the index of the self point. the index is on the complete series, not only on the neighbor series.
        /// </summary>
        public int SelfIndex { get; private set; }

        private void Estimate()
        {
            for (int iter = 0; iter < LoessConfiguration.T; iter++)
            {
                _model = Regression();

                // calculate the errors
                var errors = new double[NeighborsX.Count];
                var absErrors = new double[NeighborsX.Count];
                for (int i = 0; i < NeighborsX.Count; i++)
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
                for (int i = 0; i < Weights.Count; i++)
                {
                    Weights[i] *= deltas[i];
                }
            }
        }

        /// <summary>
        /// get the best estimated y for the current value.
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
        /// get the best estimated y for any given x-value, event not one of the observed point
        /// </summary>
        /// <param name="xValue">any given x value</param>
        public double Y(double xValue)
        {
            if (_model == null)
            {
                Estimate();
            }
            return _model.Y(xValue);
        }

        private PolynomialModel Regression()
        {
            PolynomialModel result = null;
            LeastSquares ls = new LeastSquares(NeighborsX, NeighborsY);
            switch (LoessConfiguration.ModelType)
            {
                case RegressionModelType.One:
                    result = ls.RegressionDegreeOneWeighted(Weights);
                    break;
                case RegressionModelType.Two:
                    result = ls.RegressionDegreeTwoWeighted(Weights);
                    break;
                default:
                    result = ls.RegressionDegreeOneWeighted(Weights);
                    break;
            }
            return result;
        }
    }
}
