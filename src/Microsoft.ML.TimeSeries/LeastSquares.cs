using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;

namespace Microsoft.ML.TimeSeries
{
    /// <summary>
    /// this class is used to calculate the least squares of the scatterplots.
    /// please check http://en.wikipedia.org/wiki/Least_squares for more details.
    /// </summary>
    public class LeastSquares
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
            //ExtendedDiagnostics.EnsureArgumentNotNull(x, nameof(x));
            //ExtendedDiagnostics.EnsureArgumentNotNull(y, nameof(y));

            if (x.Count == 0 || y.Count == 0)
                throw new Exception("input data structure cannot be 0-length");
            if (x.Count != y.Count)
                throw new Exception("the x-axis length should be equal to y-axis length!");
            _x = x;
            _y = y;
            _length = _x.Count;
        }

        /// <summary>
        /// y=b0+b1x, but the penalty is weighted
        /// </summary>
        /// <param name="weights">the weighted least squares. note that the weight should be non-negative, and equal length to data </param>
        public PolynomialModel RegressionDegreeOneWeighted(List<double> weights)
        {
            //ExtendedDiagnostics.EnsureArgumentNotNull(weights, nameof(weights));

            if (weights.Count != _length)
                throw new Exception("the weight vector is not equal length to the data points");

            foreach (double value in weights)
            {
                if (value < 0)
                    throw new Exception("the value in weights should be non-negative!");
            }

            double[] buffer1 = new double[_length];
            double[] buffer2 = new double[_length];
            double[] buffer3 = new double[_length];

            double[] w = buffer1;
            for (int i = 0; i < _length; i++)
            {
                w[i] = Math.Sqrt(weights[i]);
            }

            double[] kernelMatrixR0 = buffer2;
            double[] kernelMatrixR1 = buffer3;
            for (int i = 0; i < _length; i++)
            {
                kernelMatrixR0[i] = 1;
                kernelMatrixR1[i] = _x[i];
            }
            double[] kernelMatrix1R0 = buffer2;
            double[] kernelMatrix1R1 = buffer3;
            for (int i = 0; i < _length; i++)
            {
                kernelMatrix1R0[i] = w[i] * kernelMatrixR0[i];
                kernelMatrix1R1[i] = w[i] * kernelMatrixR1[i];
            }
            double[] y1 = buffer1;
            for (int i = 0; i < _length; i++)
                y1[i] = w[i] * _y[i];

            double sum00 = 0;
            double sum01 = 0;
            double sum10 = 0;
            double sum11 = 0;
            for (int k = 0; k < _length; k++)
            {
                sum00 += kernelMatrix1R0[k] * kernelMatrix1R0[k];
                sum01 += kernelMatrix1R0[k] * kernelMatrix1R1[k];
                sum10 += kernelMatrix1R1[k] * kernelMatrix1R0[k];
                sum11 += kernelMatrix1R1[k] * kernelMatrix1R1[k];
            }

            /* calculating the reverse of a 2X2 matrix is simple, because suppose the matrix is [a,b;c,d], then its reverse is
             * [x1,x2;x3,x4] where x1 = d/K, x2 = -c/K, x3 = -b/K, x4 = a/K, where K = ad-bc.
             */
            double a = sum00;
            double b = sum01;
            double c = sum10;
            double d = sum11;
            double divider = a * d - b * c;
            double[,] reverseS = new double[2, 2];
            reverseS[0, 0] = d / divider;
            reverseS[0, 1] = -c / divider;
            reverseS[1, 0] = -b / divider;
            reverseS[1, 1] = a / divider;

            // double[,] reverseS = MatrixEx.ReverseMatrix(S);

            double fy0 = 0;
            double fy1 = 0;
            for (int i = 0; i < _length; i++)
            {
                fy0 += kernelMatrix1R0[i] * y1[i];
                fy1 += kernelMatrix1R1[i] * y1[i];
            }

            double b0 = reverseS[0, 0] * fy0 + reverseS[0, 1] * fy1;
            double b1 = reverseS[1, 0] * fy0 + reverseS[1, 1] * fy1;

            List<double> results = new List<double>();
            results.Add(b0);
            results.Add(b1);

            return new PolynomialModel(results);
        }
    }

    /// <summary>
    /// indicate a specific polynomial model
    /// </summary>
    public class PolynomialModel
    {
        private readonly List<double> _coeffs;

        public PolynomialModel(ICollection<double> coeffs)
        {
            //ExtendedDiagnostics.EnsureCollectionNotNullOrEmpty(coeffs, nameof(coeffs));

            _coeffs = new List<double>(coeffs);
        }

        /// <summary>
        /// calculate the y value by given the x value, under this model
        /// </summary>
        /// <param name="x">the specific x value</param>
        public double Y(double x)
        {
            double result = _coeffs[0];
            double p = 1.0;
            for (int i = 1; i < _coeffs.Count; i++)
            {
                p = p * x;
                result += _coeffs[i] * p;
            }
            return result;
        }
    }
}
