using System.Collections.Generic;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.TimeSeries
{
    /// <summary>
    /// this class is used to calculate the least squares of the scatterplots.
    /// please check http://en.wikipedia.org/wiki/Least_squares for more details.
    /// </summary>
    internal class LeastSquares
    {
        private readonly List<double> _x;
        private readonly List<double> _y;
        private readonly int _length;

        /// <summary>
        /// Initializes a new instance of the <see cref="LeastSquares"/> class.
        /// constructing the least square algorithm. the input will be consumed directly without any copy, due to memory usage concern.
        /// </summary>
        /// <param name="x">the corresponding x-axis value</param>
        /// <param name="y">the corresponding y-axis value</param>
        public LeastSquares(List<double> x, List<double> y)
        {
            Contracts.CheckValue(x, nameof(x));
            Contracts.CheckValue(y, nameof(y));

            if (x.Count == 0 || y.Count == 0)
                throw Contracts.Except("input data structure cannot be 0-length");
            if (x.Count != y.Count)
                throw Contracts.Except("the x-axis length should be equal to y-axis length!");
            _x = x;
            _y = y;
            _length = _x.Count;
        }

        /// <summary>
        /// y=b0+b1x, but the penalty is weighted
        /// </summary>
        /// <param name="weights">the weighted least squares. note that the weight should be non-negative, and equal length to data </param>
        public AbstractPolynomialModel RegressionDegreeOneWeighted(List<double> weights)
        {
            Contracts.CheckValue(weights, nameof(weights));

            Contracts.CheckParam(weights.Count == _length, nameof(weights));
            if (weights.Count != _length)
                throw Contracts.Except("the weight vector is not equal length to the data points");

            // This part unfolds the matrix calculation of [sqrt(W), sqrt(W) .* X]^T * [sqrt(W), sqrt(W) .* X]
            double sum00 = 0;
            double sum01 = 0;
            double sum10 = 0;
            double sum11 = 0;
            for (int k = 0; k < _length; k++)
            {
                double temp = weights[k];
                sum00 += temp;
                temp *= _x[k];
                sum01 += temp;
                sum10 += temp;
                temp *= _x[k];
                sum11 += temp;
            }

            /* calculate the reverse of a 2X2 matrix is simple, because suppose the matrix is [a,b;c,d], then its reverse is
             * [x1,x2;x3,x4] where x1 = d/K, x2 = -c/K, x3 = -b/K, x4 = a/K, where K = ad-bc.
             */
            double a = sum00;
            double b = sum01;
            double c = sum10;
            double d = sum11;
            double divider = a * d - b * c;
            double reverseS00 = d / divider;
            double reverseS01 = -c / divider;
            double reverseS10 = -b / divider;
            double reverseS11 = a / divider;

            // This part unfolds the matrix calculation of [sqrt(W), sqrt(W) .* X]^T * [sqrt(W) .* Y]
            double fy0 = 0;
            double fy1 = 0;
            for (int i = 0; i < _length; i++)
            {
                double temp = weights[i] * _y[i];
                fy0 += temp;
                fy1 += temp * _x[i];
            }

            double b0 = reverseS00 * fy0 + reverseS01 * fy1;
            double b1 = reverseS10 * fy0 + reverseS11 * fy1;

            double[] results = new double[2] { b0, b1 };

            return new LinearModel(results);
        }
    }
}
