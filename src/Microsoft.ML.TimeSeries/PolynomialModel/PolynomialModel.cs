using System.Collections.Generic;

namespace Microsoft.ML.TimeSeries
{
    /// <summary>
    /// indicate a general polynomial model
    /// </summary>
    internal class PolynomialModel : AbstractPolynomialModel
    {
        public PolynomialModel(IReadOnlyList<double> coeffs)
            : base(coeffs)
        {
        }

        /// <summary>
        /// calculate the y value by given the x value, under this model
        /// </summary>
        /// <param name="x">the specific x value</param>
        public override double Y(double x)
        {
            double result = Coeffs[0];
            double p = 1.0;
            for (int i = 1; i < Coeffs.Count; i++)
            {
                p *= x;
                result += Coeffs[i] * p;
            }
            return result;
        }
    }
}
